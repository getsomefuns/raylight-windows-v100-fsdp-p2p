"""Bounded window dispatch, independent of Ray cluster lifetime or transport."""
import json
import torch
from .planning import merge_window_result


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
        ready = list(poll(tuple(pending)))
        while ready:
            ref = ready.pop()
            if ref not in pending:
                raise RuntimeError("Unexpected RelayVSR window result")
            worker, job = pending.pop(ref)
            frames = receive(ref)
            merge_window_result(output, written, job, frames)
            del frames, ref
            next_job = next(iterator, None)
            if next_job is not None:
                pending[submit(worker, next_job)] = (worker, next_job)
        del ready
    if not written.all():
        raise RuntimeError("RelayVSR output contains missing frames")
    return output


def _checkpoint_scale(vae_path):
    from safetensors import safe_open
    with safe_open(str(vae_path), framework="pt", device="cpu") as f:
        config = json.loads((f.metadata() or {}).get("config", "{}"))
    scale = config.get("upscale_factor")
    if scale not in (2, 4):
        raise ValueError("RelayVSR Flash VAE needs official x2/x4 metadata")
    return scale
