from __future__ import annotations

from unittest import mock

import pytest

from raylight.text_encoder_parallel.lifecycle import (
    DualGpuLeaseCoordinator,
    WorkerGpuLifecycle,
)
from raylight.distributed_worker import sampling_lifecycle


def test_worker_lease_requires_idle_and_reports_free_bytes() -> None:
    lifecycle = WorkerGpuLifecycle(rank=0)
    cleanup = mock.Mock()

    lease = lifecycle.acquire_external_lease(
        "operation-1", "minimax-h3-tp2", cleanup=cleanup, free_bytes=lambda: 123456
    )

    cleanup.assert_called_once_with()
    assert lease.operation_id == "operation-1"
    assert lease.owner == "minimax-h3-tp2"
    assert lease.free_bytes == 123456
    assert lifecycle.snapshot()["state"] == "leased_for_text_encoder"


@pytest.mark.parametrize("activity", ["loading_diffusion", "loading_lora", "sampling"])
def test_worker_rejects_external_lease_while_busy(activity: str) -> None:
    lifecycle = WorkerGpuLifecycle(rank=1)

    with lifecycle.activity(activity):
        with pytest.raises(RuntimeError, match=activity):
            lifecycle.acquire_external_lease(
                "operation-2", "minimax-h3-tp2", cleanup=lambda: None, free_bytes=lambda: 1
            )

    assert lifecycle.snapshot()["state"] == "idle"


def test_worker_rejects_sampling_while_text_encoder_owns_gpu() -> None:
    lifecycle = WorkerGpuLifecycle(rank=0)
    lifecycle.acquire_external_lease(
        "operation-3", "minimax-h3-tp2", cleanup=lambda: None, free_bytes=lambda: 10
    )

    with pytest.raises(RuntimeError, match="minimax-h3-tp2"):
        with lifecycle.activity("sampling"):
            pass


def test_worker_release_is_idempotent_for_same_operation_and_rejects_wrong_owner() -> None:
    lifecycle = WorkerGpuLifecycle(rank=0)
    lifecycle.acquire_external_lease(
        "operation-4", "minimax-h3-tp2", cleanup=lambda: None, free_bytes=lambda: 10
    )

    with pytest.raises(RuntimeError, match="does not own"):
        lifecycle.release_external_lease("operation-4", "other-owner")
    first = lifecycle.release_external_lease("operation-4", "minimax-h3-tp2")
    second = lifecycle.release_external_lease("operation-4", "minimax-h3-tp2")

    assert first["released"] is True
    assert second["released"] is False
    assert lifecycle.snapshot()["state"] == "idle"


def _actor(rank: int):
    actor = mock.Mock(name=f"actor-{rank}")
    actor.enter_external_gpu_lease.remote.return_value = f"enter-{rank}"
    actor.exit_external_gpu_lease.remote.return_value = f"exit-{rank}"
    return actor


def _lease(rank: int, operation_id="operation-5", owner="minimax-h3-tp2"):
    return {
        "operation_id": operation_id,
        "owner": owner,
        "free_bytes": 1000 + rank,
        "worker_state": "leased_for_text_encoder",
        "rank": rank,
    }


def test_coordinator_acquires_and_releases_exactly_two_workers() -> None:
    actors = [_actor(0), _actor(1)]
    ray_module = mock.Mock()
    ray_module.get.side_effect = [[_lease(0), _lease(1)], [{"released": True}, {"released": True}]]
    coordinator = DualGpuLeaseCoordinator(actors, ray_module=ray_module)

    leases = coordinator.acquire("operation-5", "minimax-h3-tp2")
    released = coordinator.release("operation-5", "minimax-h3-tp2")

    assert [lease.rank for lease in leases] == [0, 1]
    assert released == [{"released": True}, {"released": True}]
    for actor in actors:
        actor.enter_external_gpu_lease.remote.assert_called_once_with(
            "operation-5", "minimax-h3-tp2"
        )
        actor.exit_external_gpu_lease.remote.assert_called_once_with(
            "operation-5", "minimax-h3-tp2"
        )


def test_coordinator_rolls_back_both_workers_when_one_rank_fails() -> None:
    actors = [_actor(0), _actor(1)]
    ray_module = mock.Mock()
    ray_module.get.side_effect = [RuntimeError("rank 1 refused"), [{"released": True}, {"released": False}]]
    coordinator = DualGpuLeaseCoordinator(actors, ray_module=ray_module)

    with pytest.raises(RuntimeError, match="rank 1 refused"):
        coordinator.acquire("operation-6", "minimax-h3-tp2")

    for actor in actors:
        actor.exit_external_gpu_lease.remote.assert_called_once_with(
            "operation-6", "minimax-h3-tp2"
        )


def test_coordinator_context_releases_on_encode_error_or_cancellation() -> None:
    actors = [_actor(0), _actor(1)]
    ray_module = mock.Mock()
    ray_module.get.side_effect = [[_lease(0, "operation-7"), _lease(1, "operation-7")], [{}, {}]]
    coordinator = DualGpuLeaseCoordinator(actors, ray_module=ray_module)

    with pytest.raises(KeyboardInterrupt):
        with coordinator.lease("operation-7", "minimax-h3-tp2"):
            raise KeyboardInterrupt()

    for actor in actors:
        actor.exit_external_gpu_lease.remote.assert_called_once()


def test_sampling_lifecycle_exposes_same_all_or_nothing_lease_context() -> None:
    actors = [_actor(0), _actor(1)]
    context = mock.MagicMock()
    context.__enter__.return_value = ("rank0", "rank1")
    coordinator = mock.Mock()
    coordinator.lease.return_value = context

    with mock.patch.object(
        sampling_lifecycle, "DualGpuLeaseCoordinator", return_value=coordinator
    ) as coordinator_type:
        with sampling_lifecycle.external_text_encoder_gpu_lease(
            {"workers": actors}, "operation-8", "minimax-h3-tp2"
        ) as leases:
            assert leases == ("rank0", "rank1")

    coordinator_type.assert_called_once_with(actors, ray_module=sampling_lifecycle.ray)
    coordinator.lease.assert_called_once_with("operation-8", "minimax-h3-tp2")
