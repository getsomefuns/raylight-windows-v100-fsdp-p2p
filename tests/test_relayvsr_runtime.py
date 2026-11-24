"""Contracts: bounded real work, correct pixels, capacity admission, owned cleanup."""
import importlib
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from types import SimpleNamespace

import pytest
import torch

from raylight.relayvsr.planning import plan_relay_windows


def load_runtime():
    try:
        return importlib.import_module("raylight.relayvsr.runtime")
    except ModuleNotFoundError:
        pytest.fail("RelayVSR runtime is not implemented")


def load_worker():
    try:
        return importlib.import_module("raylight.relayvsr.worker")
    except ModuleNotFoundError:
        pytest.fail("RelayVSR decoder is not implemented")


@pytest.mark.parametrize("height,width,scale,expected", [
    (864, 480, 2, 1*2**30),
    (864, 480, 4, 3*2**30),
    (1376, 768, 2, 9*2**28),
    (1376, 768, 4, 27*2**28),
])
def test_store_capacity_covers_two_full_results(height, width, scale, expected):
    runtime = load_runtime()
    assert runtime.required_store_bytes(height, width, scale, 2) == expected


def test_insufficient_shared_store_rejected_before_submission():
    runtime = load_runtime()
    required = runtime.required_store_bytes(864, 480, 4, 2)
    with pytest.raises(MemoryError, match="store|capacity"):
        runtime.validate_store_budget(required, 64*2**30, existing_capacity=2**30)


def test_host_budget_rejects_unbounded_store_growth():
    runtime = load_runtime()
    with pytest.raises(MemoryError, match="RAM|budget"):
        runtime.validate_store_budget(3*2**30, 8*2**30)


def test_bounded_dispatch_handles_real_out_of_order_completions():
    runtime = load_runtime()
    jobs = plan_relay_windows(61)
    output = torch.empty(61, 2, 2, 3)
    seen = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        def execute(worker, job):
            time.sleep(0.03 if job.index == 0 else 0.003)
            seen.append((worker, job.index))
            return torch.arange(job.pixel_start, job.pixel_stop).float()[:, None, None, None].expand(-1, 2, 2, 3)

        def submit(worker, job):
            return executor.submit(execute, worker, job)

        def poll(refs):
            return list(wait(refs, timeout=0.1, return_when=FIRST_COMPLETED).done)

        result = runtime.dispatch_window_jobs(jobs, 2, submit, poll, lambda ref: ref.result(), output)
    assert result is output
    assert torch.equal(result[:, 0, 0, 0], torch.arange(61).float())
    assert {worker for worker, _ in seen} == {0, 1}
    assert seen[0][1] != 0
    assert sorted(index for _, index in seen) == [0, 1, 2, 3]


def test_failed_window_never_returns_a_partial_video():
    runtime = load_runtime()
    jobs = plan_relay_windows(31)
    def receive(ref):
        raise RuntimeError("controlled worker failure")
    with pytest.raises(RuntimeError, match="controlled worker failure"):
        runtime.dispatch_window_jobs(jobs, 2, lambda w,j: j.index,
                                     lambda refs: [refs[0]], receive, torch.empty(31, 2, 2, 3))


def test_interrupt_is_checked_while_workers_have_not_returned():
    runtime = load_runtime()
    def interrupted():
        raise InterruptedError("user interrupted")
    with pytest.raises(InterruptedError, match="user interrupted"):
        runtime.dispatch_window_jobs(plan_relay_windows(31), 2, lambda w,j: j.index,
                                     lambda refs: [], lambda ref: None, torch.empty(31,2,2,3),
                                     interrupt_check=interrupted)


@pytest.mark.parametrize("owned,cluster_owned", [(True, True), (True, False), (False, False)])
def test_cleanup_releases_only_owned_resources(owned, cluster_owned):
    runtime = load_runtime()
    workers = [SimpleNamespace(alive=True, cached=True) for _ in range(2)]
    unrelated = SimpleNamespace(alive=True, cached=True)
    cluster = SimpleNamespace(alive=True)
    runtime.close_worker_resources(
        workers, owned=owned, owns_cluster=cluster_owned,
        kill_worker=lambda w: setattr(w, "alive", False),
        release_decoder=lambda w: setattr(w, "cached", False),
        shutdown_cluster=lambda: setattr(cluster, "alive", False),
    )
    assert unrelated.alive and unrelated.cached
    assert all(w.alive is not owned for w in workers)
    assert cluster.alive is not cluster_owned
    if not owned:
        assert not any(w.cached for w in workers)


def test_cleanup_attempts_other_worker_after_one_release_fails():
    runtime = load_runtime()
    first, second = object(), SimpleNamespace(alive=True)
    def kill(worker):
        if worker is first:
            raise RuntimeError("already dead")
        worker.alive = False
    with pytest.raises(RuntimeError):
        runtime.close_worker_resources([first, second], owned=True, owns_cluster=False,
                                      kill_worker=kill, release_decoder=lambda w: None,
                                      shutdown_cluster=lambda: None)
    assert not second.alive


@pytest.mark.parametrize("mutation", ["latent_count", "latent_size", "batch", "channels", "detail"])
def test_invalid_relay_input_fails_before_gpu_loading(mutation):
    worker = load_worker()
    latents = torch.zeros(1, 48, 3, 8, 8)
    images = torch.zeros(17, 64, 64, 3)
    detail = 1.0
    if mutation == "latent_count":
        latents = latents[:, :, :2]
    elif mutation == "latent_size":
        latents = latents[:, :, :, :7]
    elif mutation == "batch":
        latents = latents.expand(2, -1, -1, -1, -1)
    elif mutation == "channels":
        images = torch.zeros(17, 64, 64, 4)
    else:
        detail = float("nan")
    with pytest.raises((ValueError, TypeError)):
        worker.validate_relay_inputs(latents, images, scale=2, detail=detail)


def test_valid_input_preserves_short_last_window():
    worker = load_worker()
    jobs = worker.validate_relay_inputs(torch.zeros(1,48,3,8,8),
                                        torch.zeros(17,64,64,3), scale=2, detail=0.5)
    assert len(jobs) == 2
    assert (jobs[-1].pixel_start, jobs[-1].pixel_stop, jobs[-1].discard_prefix) == (15,17,1)


def test_owned_pool_closes_cluster_even_if_actor_resources_release_later(monkeypatch):
    import ray
    runtime = load_runtime()
    events = []
    actor = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "ours"))
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "kill", lambda *a, **k: events.append("kill"))
    monkeypatch.setattr(ray, "available_resources", lambda: {"GPU": 0})
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"GPU": 2})
    monkeypatch.setattr(ray._private.state, "actors", lambda: {
        "ours": {"State": "ALIVE"}})
    monkeypatch.setattr(ray, "shutdown", lambda: events.append("shutdown"))
    pool = runtime._FlashPool.__new__(runtime._FlashPool)
    pool.workers, pool.owns_cluster = [actor], True
    pool.close()
    assert events == ["kill", "shutdown"]


def test_owned_pool_never_shuts_down_cluster_with_another_actor(monkeypatch):
    import ray
    runtime = load_runtime()
    events = []
    actor = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "ours"))
    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "kill", lambda *a, **k: events.append("kill"))
    monkeypatch.setattr(ray, "available_resources", lambda: {"GPU": 2})
    monkeypatch.setattr(ray, "cluster_resources", lambda: {"GPU": 2})
    monkeypatch.setattr(ray._private.state, "actors", lambda: {
        "other": {"State": "ALIVE"}, "ours": {"State": "ALIVE"}})
    monkeypatch.setattr(ray, "shutdown", lambda: events.append("shutdown"))
    pool = runtime._FlashPool.__new__(runtime._FlashPool)
    pool.workers, pool.owns_cluster = [actor], True
    pool.close()
    assert events == ["kill"]


def test_actor_runtime_environment_accepts_local_shared_source(tmp_path):
    from ray._private.utils import parse_runtime_env_for_task_or_actor
    runtime = load_runtime()
    env = runtime._runtime_env(tmp_path)
    try:
        validated = parse_runtime_env_for_task_or_actor(env)
    except ValueError as error:
        pytest.fail(f"Actor environment illegally uploads a local directory: {error}")
    assert "env_vars" in validated
    assert "PYTHONPATH" in validated["env_vars"]

