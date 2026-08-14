from __future__ import annotations

from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist


class A2ATracer:
    def __init__(self, rank: int, trace_dir: str | None):
        self.rank = rank
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.enabled = self.trace_dir is not None
        self._active = False
        self._closed = False
        self._capture_index = 0
        self._groups = {}
        self._original = None
        self._wrapper = None
        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self._install()

    def _install(self):
        self._original = dist.all_to_all_single

        def traced_all_to_all_single(output, input_tensor, *args, **kwargs):
            if not self._active:
                return self._original(output, input_tensor, *args, **kwargs)

            caller = sys._getframe(1)
            caller_name = f"{Path(caller.f_code.co_filename).name}:{caller.f_lineno}:{caller.f_code.co_name}"
            start = time.perf_counter()
            result = self._original(output, input_tensor, *args, **kwargs)
            elapsed = time.perf_counter() - start
            input_bytes = input_tensor.numel() * input_tensor.element_size()
            output_bytes = output.numel() * output.element_size()
            key = (
                tuple(input_tensor.shape),
                tuple(output.shape),
                str(input_tensor.dtype),
                input_bytes,
                output_bytes,
                caller_name,
            )
            group = self._groups.get(key)
            if group is None:
                group = {
                    "caller": caller_name,
                    "count": 0,
                    "dtype": str(input_tensor.dtype),
                    "input_bytes": input_bytes,
                    "input_shape": list(input_tensor.shape),
                    "max_elapsed_seconds": elapsed,
                    "min_elapsed_seconds": elapsed,
                    "output_bytes": output_bytes,
                    "output_shape": list(output.shape),
                    "total_elapsed_seconds": 0.0,
                }
                self._groups[key] = group
            group["count"] += 1
            group["total_elapsed_seconds"] += elapsed
            group["min_elapsed_seconds"] = min(group["min_elapsed_seconds"], elapsed)
            group["max_elapsed_seconds"] = max(group["max_elapsed_seconds"], elapsed)
            return result

        self._wrapper = traced_all_to_all_single
        dist.all_to_all_single = traced_all_to_all_single

    @contextmanager
    def capture(self, label: str):
        if not self.enabled:
            yield
            return

        self._groups = {}
        self._active = True
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        status = "ok"
        error_type = None
        try:
            yield
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            self._active = False
            self._write_summary(label, time.perf_counter() - start, status, error_type)

    def _write_summary(self, label: str, capture_seconds: float, status: str, error_type: str | None):
        groups = sorted(self._groups.values(), key=lambda group: (group["caller"], group["input_shape"], group["dtype"]))
        summary = {
            "capture_index": self._capture_index,
            "capture_seconds": capture_seconds,
            "error_type": error_type,
            "groups": groups,
            "label": label,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            "pid": os.getpid(),
            "rank": self.rank,
            "status": status,
            "total_input_bytes": sum(group["input_bytes"] * group["count"] for group in groups),
            "total_output_bytes": sum(group["output_bytes"] * group["count"] for group in groups),
            "call_count": sum(group["count"] for group in groups),
        }
        path = self.trace_dir / f"a2a-rank{self.rank}-pid{os.getpid()}.jsonl"
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
        self._capture_index += 1

    def close(self):
        if self._closed:
            return
        self._active = False
        if self.enabled and dist.all_to_all_single is self._wrapper:
            dist.all_to_all_single = self._original
        self._closed = True


def create_a2a_tracer(rank: int) -> A2ATracer:
    return A2ATracer(rank, os.environ.get("RAYLIGHT_A2A_TRACE_DIR"))


def trace_a2a_capture(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._a2a_tracer.capture(fn.__name__):
            return fn(self, *args, **kwargs)

    return wrapper
