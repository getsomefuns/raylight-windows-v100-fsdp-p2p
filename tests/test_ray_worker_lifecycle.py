from pathlib import Path
import sys
from unittest import mock

import pytest
from types import MethodType

from .path_helpers import comfy_root


COMFY_ROOT = comfy_root()
RAYLIGHT_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path[:0] = [str(COMFY_ROOT), str(RAYLIGHT_SRC)]
try:
    from raylight import nodes
    from raylight.distributed_worker import ray_worker
    from raylight.text_encoder_parallel.lifecycle import WorkerGpuLifecycle
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


def _lease_worker(rank=0):
    worker = object.__new__(ray_worker.RayWorker)
    worker.local_rank = rank
    worker.device = mock.Mock()
    worker._gpu_lifecycle = WorkerGpuLifecycle(rank=rank)
    worker.clear_sampling_vram = mock.Mock(return_value=True)
    return worker


def test_ray_worker_external_lease_cleans_vram_and_reports_state():
    worker = _lease_worker()
    with mock.patch.object(ray_worker.torch.cuda, "mem_get_info", return_value=(987654, 1234567)):
        lease = worker.enter_external_gpu_lease("operation-1", "minimax-h3-tp2")

    worker.clear_sampling_vram.assert_called_once_with()
    assert lease == {
        "operation_id": "operation-1",
        "owner": "minimax-h3-tp2",
        "free_bytes": 987654,
        "worker_state": "leased_for_text_encoder",
        "rank": 0,
    }
    assert worker.get_external_gpu_lease()["state"] == "leased_for_text_encoder"
    assert worker.exit_external_gpu_lease("operation-1", "minimax-h3-tp2")["released"] is True
    assert worker.get_external_gpu_lease()["state"] == "idle"


def test_worker_activity_decorator_rejects_sampling_during_text_encoder_lease():
    worker = _lease_worker()
    with mock.patch.object(ray_worker.torch.cuda, "mem_get_info", return_value=(1, 2)):
        worker.enter_external_gpu_lease("operation-2", "minimax-h3-tp2")

    @ray_worker.worker_gpu_activity("sampling")
    def sample(self):
        return "should not run"

    with pytest.raises(RuntimeError, match="minimax-h3-tp2"):
        sample(worker)


def test_worker_activity_decorator_restores_idle_after_loading_error():
    worker = _lease_worker()

    @ray_worker.worker_gpu_activity("loading_diffusion")
    def load(self):
        raise ValueError("injected load failure")

    with pytest.raises(ValueError, match="injected load failure"):
        load(worker)

    assert worker.get_external_gpu_lease()["state"] == "idle"


def test_external_lease_preserves_cached_diffusion_identity_and_can_resume_activity():
    worker = _lease_worker()
    model = mock.Mock()
    model.offload_device = "cpu"
    worker.model = model
    worker.cached_base_model = model
    worker.cached_base_key = ("checkpoint", "mmap")
    worker.active_request_key = ("checkpoint", "mmap", "lora")
    worker.clear_sampling_vram = MethodType(ray_worker.RayWorker.clear_sampling_vram, worker)

    with (
        mock.patch("comfy.model_management.unload_all_models"),
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
        mock.patch.object(ray_worker.torch.cuda, "is_available", return_value=True),
        mock.patch.object(ray_worker.torch.cuda, "synchronize"),
        mock.patch.object(ray_worker.torch.cuda, "empty_cache"),
        mock.patch.object(ray_worker.torch.cuda, "ipc_collect"),
        mock.patch.object(ray_worker.torch.cuda, "mem_get_info", return_value=(100, 200)),
    ):
        worker.enter_external_gpu_lease("operation-3", "minimax-h3-tp2")
        worker.exit_external_gpu_lease("operation-3", "minimax-h3-tp2")

    assert worker.model is model
    assert worker.cached_base_model is model
    assert worker.cached_base_key == ("checkpoint", "mmap")
    assert worker.active_request_key == ("checkpoint", "mmap", "lora")
    model.unpatch_model.assert_called_once_with(device_to="cpu")
    model.cleanup.assert_called_once_with()

    @ray_worker.worker_gpu_activity("sampling")
    def sample(self):
        return self.active_request_key

    assert sample(worker) == ("checkpoint", "mmap", "lora")
