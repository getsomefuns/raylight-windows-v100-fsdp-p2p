from pathlib import Path
import sys
from unittest import mock

import pytest

from tests.path_helpers import comfy_root


COMFY_ROOT = comfy_root()
RAYLIGHT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(COMFY_ROOT), str(RAYLIGHT_SRC)]
try:
    from raylight.distributed_worker import ray_worker
finally:
    del sys.path[:2]


def actor():
    return mock.Mock()


def test_wait_for_workers_keeps_timeout_pending_until_actor_dies():
    class TimeoutError(Exception):
        pass

    class ActorDeadError(Exception):
        pass

    worker = actor()
    with (
        mock.patch.object(ray_worker.ray.exceptions, "GetTimeoutError", TimeoutError),
        mock.patch.object(ray_worker, "RayActorError", ActorDeadError),
        mock.patch.object(
            ray_worker.ray,
            "get",
            side_effect=[TimeoutError(), ActorDeadError()],
        ) as get,
        mock.patch.object(ray_worker.time, "monotonic", return_value=0.0),
        mock.patch.object(ray_worker.time, "sleep"),
    ):
        ray_worker.wait_for_ray_workers_exit([worker], timeout_seconds=1.0)

    assert get.call_count == 2


def test_wait_for_workers_times_out_while_actor_remains_alive():
    class TimeoutError(Exception):
        pass

    worker = actor()
    with (
        mock.patch.object(ray_worker.ray.exceptions, "GetTimeoutError", TimeoutError),
        mock.patch.object(ray_worker.ray, "get", side_effect=TimeoutError()),
        mock.patch.object(ray_worker.time, "monotonic", side_effect=[0.0, 0.0, 0.2]),
        pytest.raises(RuntimeError, match="Timed out waiting"),
    ):
        ray_worker.wait_for_ray_workers_exit([worker], timeout_seconds=0.1)


def test_grouped_worker_names_cover_every_replica_and_shard():
    assert ray_worker.ray_worker_actor_names(
        4,
        {
            "dp_degree": 2,
            "shard_size": 2,
            "use_group_process_group": True,
        },
    ) == [
        "RayWorker:0_0",
        "RayWorker:0_1",
        "RayWorker:1_0",
        "RayWorker:1_1",
    ]


def test_ensure_fresh_factory_failure_cleans_partial_named_workers():
    old_workers = [actor(), actor()]
    partial_worker = actor()
    factory = mock.Mock(side_effect=RuntimeError("Gloo init failed"))
    factory.raylight_world_size = 2
    factory.raylight_parallel_dict = {"dp_degree": 1}
    payload = [{"workers": old_workers}, factory]

    with (
        mock.patch.object(ray_worker.ray, "get", side_effect=RuntimeError("rank dead")),
        mock.patch.object(ray_worker.ray, "kill"),
        mock.patch.object(ray_worker, "wait_for_ray_workers_exit") as wait,
        mock.patch.object(
            ray_worker,
            "cleanup_named_ray_workers",
            return_value=[partial_worker],
        ) as cleanup,
        pytest.raises(RuntimeError, match="Gloo init failed"),
    ):
        ray_worker.ensure_fresh_actors(payload)

    cleanup.assert_called_once_with(2, {"dp_degree": 1})
    assert wait.call_count == 2
    assert payload[0] == {"workers": [], "world_size": 2}
