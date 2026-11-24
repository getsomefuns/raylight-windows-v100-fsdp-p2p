"""Exact official RelayVSR endpoint windows and strict CPU output assembly."""
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch


@dataclass(frozen=True)
class RelayWindow:
    index: int
    pixel_start: int
    pixel_stop: int
    latent_start: int
    latent_stop: int
    discard_prefix: int

    @property
    def output_start(self):
        return self.pixel_start + self.discard_prefix


def plan_relay_windows(frame_count: int, interval: int = 15) -> tuple[RelayWindow, ...]:
    for value, name in ((frame_count, "frame_count"), (interval, "interval")):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be positive")
    keys = list(range(0, frame_count-1, interval)) + [frame_count-1]
    jobs = []
    for index in range(max(1, len(keys)-1)):
        jobs.append(RelayWindow(
            index=index, pixel_start=keys[index],
            pixel_stop=keys[min(index+1, len(keys)-1)]+1,
            latent_start=index, latent_stop=min(index+2, len(keys)),
            discard_prefix=int(index > 0),
        ))
    return tuple(jobs)


def merge_window_result(output, written, job, frames) -> None:
    """Keep each duplicate endpoint as decoder context, not as a second output."""
    if not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[-1] != 3:
        raise ValueError("Relay output must be F,H,W,3")
    if output.device.type != "cpu" or output.dtype != torch.float32:
        raise ValueError("Relay output must be CPU float32")
    if (not isinstance(written, torch.Tensor) or written.shape != (output.shape[0],)
            or written.dtype != torch.bool or written.device.type != "cpu"):
        raise ValueError("Relay coverage mask must match the CPU frame count")
    if not isinstance(job, RelayWindow) or job.discard_prefix not in (0, 1):
        raise ValueError("Invalid Relay window")
    start, stop = job.output_start, job.pixel_stop
    if not 0 <= job.pixel_start <= start < stop <= output.shape[0]:
        raise ValueError("Relay window output range is invalid")
    expected = (job.pixel_stop-job.pixel_start, *output.shape[1:])
    if (not isinstance(frames, torch.Tensor) or tuple(frames.shape) != expected
            or frames.device.type != "cpu" or frames.dtype != output.dtype):
        raise ValueError(f"Relay window result shape/dtype mismatch; expected {expected}")
    if written[start:stop].any():
        raise ValueError("Relay window contains duplicate/already written frames")
    # Bound validation scratch to one frame rather than an entire high-resolution window.
    for frame in frames:
        if not np.isfinite(frame.detach().numpy()).all():
            raise ValueError("Relay window contains nonfinite / NaN pixels")
    output[start:stop].copy_(frames[job.discard_prefix:])
    written[start:stop] = True

