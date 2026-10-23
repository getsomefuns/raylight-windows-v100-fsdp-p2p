from __future__ import annotations

import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time

import psutil
import torch


def _windows_commit_memory():
    if os.name != "nt":
        return {}

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", wintypes.DWORD),
            ("ProcessCount", wintypes.DWORD),
            ("ThreadCount", wintypes.DWORD),
        ]

    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        return {}
    page = int(info.PageSize)
    return {
        "system_commit_bytes": int(info.CommitTotal) * page,
        "system_commit_limit_bytes": int(info.CommitLimit) * page,
        "system_commit_peak_bytes": int(info.CommitPeak) * page,
    }


def capture_memory_snapshot():
    process_memory = psutil.Process().memory_full_info()
    system = psutil.virtual_memory()
    payload = {
        "system_total_bytes": int(system.total),
        "system_available_bytes": int(system.available),
        "system_used_bytes": int(system.used),
        "process_rss_bytes": int(process_memory.rss),
        "process_vms_bytes": int(process_memory.vms),
    }
    private = getattr(process_memory, "private", None)
    if private is not None:
        payload["process_private_bytes"] = int(private)
    pagefile = getattr(process_memory, "pagefile", None)
    if pagefile is not None:
        payload["process_pagefile_bytes"] = int(pagefile)
    payload.update(_windows_commit_memory())
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        payload.update(
            cuda_device=int(device),
            cuda_allocated_bytes=int(torch.cuda.memory_allocated(device)),
            cuda_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            cuda_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            cuda_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
        )
    return payload


def _default_output_path(rank):
    root = os.environ.get("RAYLIGHT_MEMORY_LOG_DIR")
    directory = Path(root) if root else Path.cwd() / "user" / "raylight_memory"
    directory /= datetime.now().strftime("%Y-%m-%d")
    label = "main" if rank is None else f"rank{rank}"
    return directory / f"raylight-memory-{label}-pid{os.getpid()}.jsonl"


class MemoryRunRecorder:
    """Record low-frequency host and CUDA peaks for one sampling operation."""

    def __init__(
        self,
        stage,
        *,
        operation_id=None,
        rank=None,
        fields=None,
        output_path=None,
        snapshot=capture_memory_snapshot,
        interval_seconds=1.0,
        start_thread=True,
    ):
        self.enabled = os.environ.get("RAYLIGHT_MEMORY_RECORD", "1") != "0"
        self.stage = str(stage)
        self.operation_id = operation_id
        self.rank = rank
        self.fields = dict(fields or {})
        self.output_path = Path(output_path) if output_path is not None else _default_output_path(rank)
        self.snapshot = snapshot
        self.interval_seconds = float(interval_seconds)
        self.start_thread = bool(start_thread)
        self.started_ns = None
        self.start_snapshot = {}
        self.end_snapshot = {}
        self.peaks = {}
        self._stop = threading.Event()
        self._thread = None
        self._finished = False

    def _merge(self, sample):
        for key, value in sample.items():
            if not isinstance(value, (int, float)):
                continue
            if key == "system_available_bytes":
                peak_key = "min_system_available_bytes"
                current = self.peaks.get(peak_key)
                self.peaks[peak_key] = value if current is None else min(current, value)
            elif key.endswith("_bytes"):
                peak_key = f"peak_{key}" if not key.startswith("cuda_peak_") else key
                self.peaks[peak_key] = max(self.peaks.get(peak_key, value), value)

    def poll_once(self):
        if not self.enabled:
            return {}
        try:
            sample = dict(self.snapshot())
        except (OSError, RuntimeError, StopIteration):
            return {}
        self._merge(sample)
        return sample

    def _poll(self):
        while not self._stop.wait(self.interval_seconds):
            self.poll_once()

    def __enter__(self):
        if not self.enabled:
            return self
        self.started_ns = time.time_ns()
        self.start_snapshot = self.poll_once()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if self.start_thread:
            self._thread = threading.Thread(target=self._poll, name="raylight-memory", daemon=True)
            self._thread.start()
        return self

    def finish(self, error=None):
        if not self.enabled or self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self.end_snapshot = self.poll_once()
        payload = {
            "event": "sampling_memory_summary",
            "stage": self.stage,
            "operation_id": self.operation_id,
            "rank": self.rank,
            "pid": os.getpid(),
            "started_ns": self.started_ns,
            "finished_ns": time.time_ns(),
            "status": "failed" if error is not None else "completed",
            "start": self.start_snapshot,
            "end": self.end_snapshot,
            **self.peaks,
            **self.fields,
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)[-4096:]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    def __exit__(self, exc_type, exc, _traceback):
        self.finish(exc)
        return False
