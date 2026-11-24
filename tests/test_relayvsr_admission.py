"""Real lease-state regressions; only Ray RPC and hardware probes are doubled."""
import importlib
from types import SimpleNamespace

import pytest

from raylight.text_encoder_parallel.lifecycle import WorkerGpuLifecycle


class Remote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args):
        return lambda: self.function(*args)


class FakeRay:
    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_runtime_context():
        return SimpleNamespace(get_job_id=lambda: "job", get_node_id=lambda: "node")

    @staticmethod
    def get(refs, **kwargs):
        if not isinstance(refs, (tuple, list)):
            return refs()
        values, errors = [], []
        for ref in refs:
            try:
                values.append(ref())
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        return values


class Worker:
    def __init__(self, rank, events):
        self.lifecycle = WorkerGpuLifecycle(rank=rank)
        self.identity = {"rank": rank, "job_id": "job", "node_id": "node",
                         "actor_id": f"actor{rank}", "generation": f"generation{rank}",
                         "gpu_uuid": f"gpu{rank}"}
        self._actor_id = f"actor{rank}"
        self.fail = False
        self.events = events
        self.get_relay_vsr_identity = Remote(lambda: dict(self.identity))
        self.enter_relay_vsr_gpu_lease = Remote(self.enter)
        self.exit_external_gpu_lease = Remote(self.exit)
        self.get_external_gpu_lease = Remote(self.lifecycle.snapshot)

    def enter(self, operation, owner):
        def cleanup():
            if self.fail:
                raise RuntimeError("controlled worker cleanup failure")
            self.events.append(f"offload{self.identity['rank']}")
        return self.lifecycle.acquire_external_lease(operation, owner, cleanup=cleanup,
            free_bytes=lambda: 10 * 2**30).to_mapping()

    def exit(self, operation, owner):
        self.events.append(f"release{self.identity['rank']}")
        return self.lifecycle.release_external_lease(operation, owner)


@pytest.fixture
def setup():
    events = []
    return [Worker(0, events), Worker(1, events)], events


def authorization(workers, cleanup):
    module = importlib.import_module("raylight.relayvsr.admission")
    return module.authorize_flash(workers, cleanup=cleanup, ray_module=FakeRay,
                                  device_uuids=("gpu0", "gpu1"))


@pytest.mark.parametrize("field,value", [("job_id", "foreign"), ("node_id", "remote"),
                                          ("gpu_uuid", "wrong"), ("actor_id", "stale")])
def test_foreign_or_remapped_workers_rejected_before_offload(setup, field, value):
    workers, events = setup
    workers[1].identity[field] = value
    with pytest.raises(RuntimeError):
        with authorization(workers, lambda: events.append("cleanup")):
            pytest.fail("Foreign worker admitted")
    assert events == []


def test_cancel_cleans_vsr_then_returns_both_leases_and_allows_reuse(setup):
    workers, events = setup
    with pytest.raises(ValueError, match="cancel"):
        with authorization(workers, lambda: events.append("cleanup")) as token:
            token.validate()
            assert all(worker.lifecycle.snapshot()["lease"] for worker in workers)
            raise ValueError("cancel")
    assert events[-3:] == ["cleanup", "release0", "release1"]
    assert all(worker.lifecycle.snapshot()["state"] == "idle" for worker in workers)
    with authorization(workers, lambda: None) as token:
        token.validate()


def test_partial_worker_cleanup_failure_rolls_back_other_rank(setup):
    workers, _ = setup
    workers[1].fail = True
    with pytest.raises(RuntimeError, match="controlled"):
        with authorization(workers, lambda: None):
            pytest.fail("Incomplete lease admitted")
    assert all(worker.lifecycle.snapshot()["state"] == "idle" for worker in workers)


def test_generation_change_revokes_token_before_local_model_unload(setup):
    workers, _ = setup
    with authorization(workers, lambda: None) as token:
        workers[1].identity["generation"] = "restarted"
        with pytest.raises(RuntimeError):
            token.validate()


def test_vsr_cleanup_failure_keeps_workers_locked(setup):
    workers, events = setup
    def fail():
        raise RuntimeError("VSR cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup"):
        with authorization(workers, fail):
            pass
    assert all(worker.lifecycle.snapshot()["lease"] for worker in workers)
    assert not any(event.startswith("release") for event in events)


def test_strict_worker_cleanup_reports_unpatch_failure(monkeypatch):
    import comfy.model_management as mm
    import torch
    from raylight.distributed_worker.ray_worker import RayWorker
    def fail(**kwargs):
        raise RuntimeError("unpatch failed")
    worker = SimpleNamespace(local_rank=0, model=SimpleNamespace(unpatch_model=fail, cleanup=lambda: None),
                             cached_base_model=None)
    monkeypatch.setattr(mm, "unload_all_models", lambda: None)
    monkeypatch.setattr(mm, "soft_empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unpatch"):
        RayWorker.clear_sampling_vram(worker, strict=True)


def test_failed_pool_close_retains_owned_resources_for_cleanup_retry(monkeypatch):
    from raylight.relayvsr import local
    def fail():
        raise RuntimeError("pool close failed")
    pool = SimpleNamespace(close=fail)
    monkeypatch.setattr(local, "_POOL", pool)
    with pytest.raises(RuntimeError, match="pool close"):
        local.release_local_pool()
    assert local._POOL is pool


def test_local_executor_requires_live_lease_not_a_past_authorization(setup, monkeypatch):
    import ray
    import torch
    from pathlib import Path
    from raylight.relayvsr import local
    workers, events = setup
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(local, "_checkpoint_scale", lambda path: 2)
    def pool(path, count):
        return SimpleNamespace(key=(str(Path(path).resolve()), count), decoders=[],
                               close=lambda: events.append("close"))
    monkeypatch.setattr(local, "_POOL", None)
    monkeypatch.setattr(local, "_LocalPool", pool)
    monkeypatch.setattr(local, "dispatch_window_jobs", lambda *args: args[5].zero_())
    latents, images = torch.zeros(1, 48, 1, 8, 8), torch.zeros(1, 64, 64, 3)
    with authorization(workers, local.release_local_pool):
        result = local.run_local_flash("unused", latents, images, 1.)
        assert result.shape == (1, 128, 128, 3)
    assert events.index("close") < events.index("release0")
    with pytest.raises(RuntimeError, match="Ray"):
        local.run_local_flash("unused", latents, images, 1.)
