from pathlib import Path
import sys
from unittest import mock

import pytest

from tests.path_helpers import comfy_root


COMFY_ROOT = comfy_root()
RAYLIGHT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(COMFY_ROOT), str(RAYLIGHT_SRC)]
try:
    from raylight import nodes
    from raylight.distributed_worker import ray_worker
finally:
    del sys.path[:2]


def actor():
    return mock.Mock()


def test_fsdp_reuse_requires_every_rank_to_match():
    workers = [actor(), actor()]
    with mock.patch.object(
        nodes.ray,
        "get",
        side_effect=[[True, False], [True, True]],
    ):
        transition = nodes._fsdp_actor_model_transition(
            workers,
            "model.safetensors",
            {"dtype": "fp32"},
        )

    assert transition == "recycle"


def test_fsdp_reuses_only_when_every_rank_matches():
    workers = [actor(), actor()]
    with mock.patch.object(
        nodes.ray,
        "get",
        side_effect=[[True, True], [True, True]],
    ):
        transition = nodes._fsdp_actor_model_transition(
            workers,
            "model.safetensors",
            {"dtype": "fp32"},
        )

    assert transition == "reuse"


def test_fsdp_actor_query_failure_forces_recycle():
    workers = [actor(), actor()]
    with mock.patch.object(nodes.ray, "get", side_effect=RuntimeError("rank 1 dead")):
        transition = nodes._fsdp_actor_model_transition(
            workers,
            "model.safetensors",
            {"dtype": "fp32"},
        )

    assert transition == "recycle"


def test_recycle_factory_failure_invalidates_cached_payload():
    old_workers = [actor(), actor()]
    factory = mock.Mock(side_effect=RuntimeError("P2P health failed"))
    payload = [{"workers": old_workers}, factory]
    nodes._cache_active_ray_session("session", payload)

    with (
        mock.patch.object(nodes.ray, "kill"),
        mock.patch.object(nodes, "_wait_for_ray_workers_exit"),
        mock.patch.object(nodes, "_cleanup_named_ray_workers") as cleanup,
        pytest.raises(RuntimeError, match="P2P health failed"),
    ):
        nodes._recycle_ray_actor_payload(payload)

    assert payload[0] == {"workers": []}
    assert nodes._ACTIVE_RAY_SESSION is None
    cleanup.assert_called_once_with(factory, 2)


def test_ensure_fresh_actors_rebuilds_when_one_rank_probe_fails():
    old_workers = [actor(), actor()]
    new_workers = [actor(), actor()]
    new_actors = {"workers": new_workers}
    factory = mock.Mock(return_value=new_actors)
    payload = [{"workers": old_workers}, factory]
    parallel = {"is_fsdp": True}

    with (
        mock.patch.object(
            ray_worker.ray,
            "get",
            side_effect=[RuntimeError("rank 1 dead"), [parallel, parallel]],
        ),
        mock.patch.object(ray_worker, "wait_for_ray_workers_exit"),
        mock.patch.object(ray_worker.ray, "kill") as kill,
    ):
        ray_actors, gpu_actors, parallel_dict = ray_worker.ensure_fresh_actors(payload)

    assert ray_actors is new_actors
    assert gpu_actors == new_workers
    assert parallel_dict == parallel
    assert payload[0] is new_actors
    assert kill.call_count == 2


def test_ensure_fresh_actors_keeps_healthy_loaded_workers():
    workers = [actor(), actor()]
    factory = mock.Mock()
    payload = [{"workers": workers}, factory]
    parallel = {"is_fsdp": True}

    with (
        mock.patch.object(ray_worker.ray, "get", return_value=[parallel, parallel]),
        mock.patch.object(ray_worker.ray, "kill") as kill,
    ):
        ray_actors, gpu_actors, parallel_dict = ray_worker.ensure_fresh_actors(payload)

    assert ray_actors is payload[0]
    assert gpu_actors == workers
    assert parallel_dict == parallel
    factory.assert_not_called()
    kill.assert_not_called()
