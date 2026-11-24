"""Bounded Flash work distribution; ownership never extends to unrelated Ray actors."""
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
import uuid

import torch

from .planning import merge_window_result
from .worker import validate_relay_inputs

_LOG = logging.getLogger("raylight.relayvsr")
_POOL = None
_POOL_LOCK = threading.Lock()


def required_store_bytes(height, width, scale, worker_count=2, window_frames=16):
    if scale not in (2, 4) or worker_count not in (1, 2):
        raise ValueError("RelayVSR supports x2/x4 and one or two workers")
    if min(height, width, window_frames) < 1:
        raise ValueError("RelayVSR dimensions must be positive")
    stride = (16//scale)*2
    latent_h, latent_w = math.ceil(height/stride)*2, math.ceil(width/stride)*2
    inputs = window_frames*height*width*3*4 + min(2, window_frames)*48*latent_h*latent_w*4
    outputs = window_frames*height*width*scale**2*3*4
    budget = max(2**30, worker_count*(inputs+outputs)+2**28)
    return math.ceil(budget/2**28)*2**28


def validate_store_budget(required, available_ram, existing_capacity=None):
    if required > min(8*2**30, available_ram//4):
        raise MemoryError("RelayVSR object store exceeds safe host RAM budget")
    if existing_capacity is not None and existing_capacity < required:
        raise MemoryError("Ray object store capacity is insufficient for bounded RelayVSR windows")


def dispatch_window_jobs(jobs, worker_count, submit, poll, receive, output, interrupt_check=None):
    if worker_count not in (1, 2):
        raise ValueError("RelayVSR supports one or two workers")
    written = torch.zeros(output.shape[0], dtype=torch.bool)
    iterator = iter(jobs)
    pending = {}
    for worker in range(worker_count):
        job = next(iterator, None)
        if job is not None:
            pending[submit(worker, job)] = (worker, job)
    while pending:
        if interrupt_check is not None:
            interrupt_check()
        for ref in poll(tuple(pending)):
            if ref not in pending:
                raise RuntimeError("Unexpected RelayVSR window result")
            worker, job = pending.pop(ref)
            merge_window_result(output, written, job, receive(ref))
            next_job = next(iterator, None)
            if next_job is not None:
                pending[submit(worker, next_job)] = (worker, next_job)
    if not written.all():
        raise RuntimeError("RelayVSR output contains missing frames")
    return output


def close_worker_resources(workers, *, owned, owns_cluster, kill_worker, release_decoder, shutdown_cluster):
    errors = []
    for worker in workers:
        try:
            (kill_worker if owned else release_decoder)(worker)
        except Exception as error:
            errors.append(error)
    if owns_cluster:
        try:
            shutdown_cluster()
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(f"RelayVSR resource cleanup failed: {errors}") from errors[0]


def _checkpoint_scale(vae_path):
    from safetensors import safe_open
    with safe_open(str(vae_path), framework="pt", device="cpu") as f:
        config = json.loads((f.metadata() or {}).get("config", "{}"))
    scale = config.get("upscale_factor")
    if scale not in (2, 4):
        raise ValueError("RelayVSR Flash VAE needs official x2/x4 metadata")
    return scale


def _runtime_env(tmpdir):
    import raylight
    import folder_paths
    root = Path(folder_paths.__file__).resolve().parent
    source = Path(raylight.__file__).resolve().parent.parent
    paths = [str(root), str(source)]
    paths.extend(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    # All actors are local. Actor runtime_env cannot upload py_modules or a
    # working_dir from disk; use the same immutable deployed source via PATH.
    return {"env_vars": {
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(paths)),
        "COMFYUI_BASE_DIRECTORY": str(root),
        "RAY_TMPDIR": str(tmpdir), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }}


class _FlashPool:
    def __init__(self, count, store_bytes):
        import psutil
        import ray
        from raylight.nodes import _raylight_ray_tmpdir
        from .worker import RelayVSRFlashWorker
        self.key = (count, store_bytes)
        self.workers = []
        self.owns_cluster = False
        tmp = Path(os.environ.get("RAYLIGHT_RELAYVSR_TMPDIR", _raylight_ray_tmpdir()/"relayvsr")).resolve()
        tmp.mkdir(parents=True, exist_ok=True)
        validate_store_budget(store_bytes, psutil.virtual_memory().available)
        try:
            if not ray.is_initialized():
                if torch.cuda.device_count() < count:
                    raise RuntimeError(f"RelayVSR requires {count} visible CUDA devices")
                ray.init(num_cpus=2, num_gpus=min(2, torch.cuda.device_count()), include_dashboard=False,
                         object_store_memory=store_bytes, _temp_dir=str(tmp))
                self.owns_cluster = True
            validate_store_budget(store_bytes, psutil.virtual_memory().available,
                                  ray.cluster_resources().get("object_store_memory", 0))
            if ray.available_resources().get("GPU", 0) < count:
                raise RuntimeError("Ray GPUs occupied: pass ray_actors explicitly or release their owner first")
            actor = ray.remote(num_gpus=1, num_cpus=1)(RelayVSRFlashWorker)
            env = _runtime_env(tmp)
            for _ in range(count):
                self.workers.append(actor.options(runtime_env=env).remote())
            ray.get([worker.ready.remote() for worker in self.workers], timeout=120)
        except BaseException:
            self.close()
            raise

    def close(self):
        import ray
        if not ray.is_initialized():
            self.workers = []
            return
        # Never shut down an existing cluster. If other work has attached to our
        # cluster, leave the shared control plane alive after releasing our actors.
        shutdown = False
        if self.owns_cluster:
            try:
                owned_ids = {w._actor_id.hex() for w in self.workers}
                # Resource availability lags ray.kill and also misses CPU-only
                # actors. Inspect ownership before killing, fail closed on error.
                actors = ray._private.state.actors()
                shutdown = not any(key not in owned_ids and item.get("State") != "DEAD"
                                   for key, item in actors.items())
            except Exception:
                _LOG.warning("RelayVSR could not verify cluster ownership; retaining control plane")
        try:
            close_worker_resources(self.workers, owned=True, owns_cluster=shutdown,
                                   kill_worker=lambda w: ray.kill(w, no_restart=True),
                                   release_decoder=lambda w: None, shutdown_cluster=ray.shutdown)
        finally:
            self.workers = []
            self.owns_cluster = False


def release_relayvsr_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL = None


def run_parallel_flash(vae_path, samples, images, detail, worker_count=2,
                       keep_workers=False, borrowed_workers=(), interrupt_check=None):
    import psutil
    import ray
    global _POOL
    started = time.perf_counter()
    scale = _checkpoint_scale(vae_path)
    jobs = validate_relay_inputs(samples, images, scale, detail)
    if worker_count not in (1, 2):
        raise ValueError("RelayVSR supports one or two workers")
    count = min(worker_count, len(jobs))
    store = required_store_bytes(int(images.shape[1]), int(images.shape[2]), scale, count,
                                 max(j.pixel_stop-j.pixel_start for j in jobs))
    validate_store_budget(store, psutil.virtual_memory().available)
    if not _POOL_LOCK.acquire(blocking=False):
        raise RuntimeError("A RelayVSR decode is already using this process's Flash pool")
    pool = None
    workers = []
    borrowed = bool(borrowed_workers)
    succeeded = False
    trace = []
    operation = uuid.uuid4().hex
    try:
        if borrowed:
            if len(borrowed_workers) < count or not ray.is_initialized():
                raise RuntimeError("RelayVSR needs live explicitly supplied Ray workers")
            validate_store_budget(store, psutil.virtual_memory().available,
                                  ray.cluster_resources().get("object_store_memory", 0))
            workers = list(borrowed_workers[:count])
            ray.get([w.clear_sampling_vram.remote() for w in workers])
        else:
            key = (count, store)
            if _POOL is not None and _POOL.key != key:
                release_relayvsr_pool()
            if _POOL is None:
                _POOL = _FlashPool(count, store)
            pool = _POOL
            workers = pool.workers
        output = torch.empty((images.shape[0], images.shape[1]*scale, images.shape[2]*scale, 3),
                             dtype=torch.float32, device="cpu")
        samples = samples.detach().cpu()
        images = images.detach().cpu()

        def submit(worker, job):
            return workers[worker].ray_relayvsr_decode_window.remote(
                str(vae_path), samples[:, :, job.latent_start:job.latent_stop].contiguous(),
                images[job.pixel_start:job.pixel_stop].contiguous(), detail, asdict(job))

        def poll(refs):
            ready, _ = ray.wait(list(refs), num_returns=1, timeout=0.1)
            return ready

        def receive(ref):
            result = ray.get(ref)
            trace.append(result["metadata"])
            return result["pixels"]

        dispatch_window_jobs(jobs, count, submit, poll, receive, output, interrupt_check)
        succeeded = True
        return output
    finally:
        try:
            if borrowed and workers:
                close_worker_resources(workers, owned=False, owns_cluster=False,
                    kill_worker=lambda w: None,
                    release_decoder=lambda w: ray.get(w.free_relayvsr_vae.remote(), timeout=300),
                    shutdown_cluster=lambda: None)
            elif pool is not None and (not keep_workers or not succeeded):
                release_relayvsr_pool()
        finally:
            elapsed = time.perf_counter()-started
            _LOG.info("[Raylight] RelayVSR Flash %s windows=%d workers=%d seconds=%.3f",
                      "completed" if succeeded else "failed", len(trace), count, elapsed)
            try:
                directory = os.environ.get("RAYLIGHT_RELAYVSR_TRACE_DIR")
                if directory:
                    path = Path(directory)
                    path.mkdir(parents=True, exist_ok=True)
                    record = {"operation": operation, "completed": succeeded, "seconds": elapsed,
                              "workers": count, "store_bytes": store, "windows": trace}
                    (path/f"{operation}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            except OSError:
                _LOG.exception("RelayVSR diagnostic trace could not be saved")
            finally:
                _POOL_LOCK.release()

