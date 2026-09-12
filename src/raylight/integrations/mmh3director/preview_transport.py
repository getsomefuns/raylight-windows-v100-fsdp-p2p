from __future__ import annotations

from collections.abc import Callable, Sequence

import ray
import torch
from ray.util.queue import Empty, Full, Queue


DEFAULT_MAX_PREVIEW_FRAMES = 16


def _video_stream(x0):
    if isinstance(x0, torch.Tensor) and not bool(getattr(x0, "is_nested", False)):
        return x0
    if getattr(x0, "is_nested", False) and callable(getattr(x0, "unbind", None)):
        streams = list(x0.unbind())
        return streams[0] if streams else None
    if isinstance(x0, (tuple, list)) and x0:
        return x0[0]
    return None


def _should_emit(step: int, total_steps: int, preview_every: int) -> bool:
    last = max(0, int(total_steps) - 1)
    if int(preview_every) < 0:
        return int(step) >= last
    every = max(1, int(preview_every))
    return int(step) % every == 0 or int(step) >= last


def extract_preview_event(
    x0,
    *,
    step: int,
    total_steps: int,
    preview_every: int = 1,
    max_frames: int = DEFAULT_MAX_PREVIEW_FRAMES,
) -> dict | None:
    """Copy a bounded video-only x0 preview to CPU for the driver process."""

    if not _should_emit(step, total_steps, preview_every):
        return None
    video = _video_stream(x0)
    if not isinstance(video, torch.Tensor) or video.ndim != 5 or video.numel() == 0:
        return None

    frame_total = int(video.shape[2])
    cap = max(1, int(max_frames))
    if frame_total > cap:
        indices = torch.linspace(
            0,
            frame_total - 1,
            cap,
            device=video.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
        video = torch.index_select(video, 2, indices)
    video_cpu = video[:1].detach().to(device="cpu", copy=True).contiguous()
    return {
        "step": int(step),
        "total_steps": int(total_steps),
        "x0": video_cpu,
    }


def publish_preview_nonblocking(queue, event: dict | None) -> bool:
    if event is None:
        return False
    try:
        queue.put(event, block=False)
        return True
    except Full:
        return False


def create_preview_queue():
    return Queue(maxsize=1, actor_options={"num_cpus": 0})


def drain_preview_queue(queue, callback: Callable) -> int:
    drained = 0
    while True:
        try:
            event = queue.get(block=False)
        except Empty:
            break
        x0 = event["x0"]
        callback(int(event["step"]), x0, x0, int(event["total_steps"]))
        drained += 1
    return drained


def wait_for_worker_results(
    futures: Sequence,
    queue,
    callback: Callable,
    *,
    poll_seconds: float = 0.05,
) -> list:
    """Wait for Ray workers while forwarding any available preview events."""

    indexed = {future: index for index, future in enumerate(futures)}
    pending = list(futures)
    results = [None] * len(pending)
    while pending:
        ready, pending = ray.wait(
            pending,
            num_returns=1,
            timeout=max(0.001, float(poll_seconds)),
        )
        drain_preview_queue(queue, callback)
        for future in ready:
            results[indexed[future]] = ray.get(future)
    drain_preview_queue(queue, callback)
    return results


def close_preview_queue(queue) -> None:
    if queue is None:
        return
    try:
        queue.shutdown(force=True)
    except Exception:
        pass

