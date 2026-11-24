"""Shared-process GPU Flash workers: no Ray startup or pixel serialization."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid

import torch

from .runtime import _checkpoint_scale, dispatch_window_jobs
from .worker import RelayFlashDecoder, validate_relay_inputs

_POOL = None
_LOCK = threading.Lock()
_LOG = logging.getLogger("raylight.relayvsr")


@torch.inference_mode()
def decode_flash_model(model, latents, images, detail, device):
    """Only official RelayFlash: its VAE process_output is identity."""
    from comfy.ldm.wan.model_vsr import RelayFlash
    if not isinstance(model, RelayFlash):
        raise ValueError("Direct decode is restricted to native RelayFlash")
    context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
    with context:
        shape = model.decode_output_shape(latents.shape, lq_images=images, detail=detail)
        output = torch.empty(shape, dtype=torch.float32, device="cpu")
        model.decode(latents.to(device=device, dtype=torch.float32), lq_images=images,
                     output_buffer=output, detail=detail)
        return output.movedim(1, -1).reshape(-1, *output.shape[-2:], 3)


class _GPUDecoder:
    def __init__(self, path, index):
        self.device = torch.device("cuda", index)
        self.decoder = RelayFlashDecoder(path, device=self.device)

    @torch.inference_mode()
    def decode(self, samples, images, detail, job):
        with torch.cuda.device(self.device):
            started = time.perf_counter()
            vae = self.decoder.load()
            vae.first_stage_model.to(device=self.device, dtype=torch.float32)
            pixels = decode_flash_model(vae.first_stage_model, samples, images, detail, self.device)
            return {"pixels": pixels, "metadata": {"window": asdict(job),
                "gpu_ids": [self.device.index], "seconds": time.perf_counter()-started,
                "pid": os.getpid(), "thread": threading.get_ident()}}

    def trim(self):
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def close(self):
        with torch.cuda.device(self.device):
            self.decoder.close()
            torch.cuda.empty_cache()


class _LocalPool:
    def __init__(self, path, count):
        import ray
        if torch.cuda.device_count() < count:
            raise RuntimeError(f"RelayVSR needs {count} visible CUDA devices")
        if ray.is_initialized() and ray.available_resources().get("GPU", 0) < count:
            raise RuntimeError("Ray GPUs occupied; use ray_actors mode with explicit borrowed workers")
        self.key = (str(Path(path).resolve()), count)
        self.decoders = [_GPUDecoder(path, i) for i in range(count)]
        self.executor = ThreadPoolExecutor(max_workers=count, thread_name_prefix="raylight-flash")

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        errors = []
        for decoder in self.decoders:
            try:
                decoder.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(f"Local Flash cleanup failed: {errors}") from errors[0]


def release_local_pool():
    global _POOL
    pool, _POOL = _POOL, None
    if pool is not None:
        pool.close()


def run_local_flash(vae_path, samples, images, detail, worker_count=2,
                    keep_workers=False, borrowed_workers=(), interrupt_check=None):
    global _POOL
    if borrowed_workers:
        raise ValueError("local_threads cannot borrow ray_actors; select ray_actors execution mode")
    if worker_count not in (1, 2):
        raise ValueError("RelayVSR supports one or two GPU workers")
    scale = _checkpoint_scale(vae_path)
    jobs = validate_relay_inputs(samples, images, scale, detail)
    count = min(worker_count, len(jobs))
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("A local RelayVSR decode is already active")
    started = time.perf_counter()
    succeeded = False
    trace = []
    try:
        key = (str(Path(vae_path).resolve()), count)
        if _POOL is not None and _POOL.key != key:
            release_local_pool()
        if _POOL is None:
            _POOL = _LocalPool(vae_path, count)
        pool = _POOL
        samples, images = samples.detach().cpu(), images.detach().cpu()
        output = torch.empty((images.shape[0], images.shape[1]*scale, images.shape[2]*scale, 3),
                             dtype=torch.float32, device="cpu")

        def submit(worker, job):
            return pool.executor.submit(pool.decoders[worker].decode,
                samples[:, :, job.latent_start:job.latent_stop], images[job.pixel_start:job.pixel_stop],
                detail, job)

        def poll(refs):
            return list(wait(refs, timeout=.1, return_when=FIRST_COMPLETED).done)

        def receive(ref):
            result = ref.result()
            trace.append(result["metadata"])
            return result["pixels"]

        dispatch_window_jobs(jobs, count, submit, poll, receive, output, interrupt_check)
        succeeded = True
        return output
    finally:
        try:
            if not succeeded or not keep_workers:
                release_local_pool()
            elif _POOL is not None:
                # Keep only useful weights, not scratch that could OOM the next sampler.
                for decoder in _POOL.decoders:
                    decoder.trim()
        finally:
            elapsed = time.perf_counter()-started
            _LOG.info("[Raylight] RelayVSR local Flash %s windows=%d workers=%d seconds=%.3f",
                      "completed" if succeeded else "failed", len(trace), count, elapsed)
            try:
                directory = os.environ.get("RAYLIGHT_RELAYVSR_TRACE_DIR")
                if directory:
                    path = Path(directory)
                    path.mkdir(parents=True, exist_ok=True)
                    record = {"operation": uuid.uuid4().hex, "mode": "local_threads",
                              "completed": succeeded, "seconds": elapsed, "workers": count,
                              "windows": trace}
                    (path/f"{record['operation']}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            except OSError:
                _LOG.exception("RelayVSR diagnostic trace could not be saved")
            finally:
                _LOCK.release()
