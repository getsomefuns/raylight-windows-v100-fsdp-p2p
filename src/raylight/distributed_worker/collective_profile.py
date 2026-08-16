"""Low-overhead aggregate profiling for the Windows CUDA P2P data plane."""

from __future__ import annotations

import os
import threading


_KINDS = ("all_gather", "all_to_all")
_SUM_FIELDS = (
    "calls",
    "payload_bytes",
    "remote_bytes",
    "chunks",
    "control_wait_ns",
    "submit_ns",
)


def _empty_stats() -> dict[str, int]:
    return {
        "calls": 0,
        "payload_bytes": 0,
        "remote_bytes": 0,
        "chunks": 0,
        "control_wait_ns": 0,
        "submit_ns": 0,
        "max_payload_bytes": 0,
    }


class CollectiveProfiler:
    """Keep aggregate counters without retaining tensors or per-call events."""

    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self._lock = threading.Lock() if self.enabled else None
        self._stats = {kind: _empty_stats() for kind in _KINDS} if self.enabled else {}

    def record(
        self,
        kind: str,
        payload_bytes: int,
        remote_bytes: int,
        chunks: int,
        control_wait_ns: int,
        submit_ns: int,
    ) -> None:
        if not self.enabled:
            return
        if kind not in _KINDS:
            raise ValueError(f"unsupported collective kind: {kind!r}")
        values = (payload_bytes, remote_bytes, chunks, control_wait_ns, submit_ns)
        if any(int(value) < 0 for value in values):
            raise ValueError("collective profile values must be non-negative")

        with self._lock:
            stats = self._stats[kind]
            stats["calls"] += 1
            stats["payload_bytes"] += int(payload_bytes)
            stats["remote_bytes"] += int(remote_bytes)
            stats["chunks"] += int(chunks)
            stats["control_wait_ns"] += int(control_wait_ns)
            stats["submit_ns"] += int(submit_ns)
            stats["max_payload_bytes"] = max(stats["max_payload_bytes"], int(payload_bytes))

    def snapshot(self, reset: bool = False) -> dict:
        if not self.enabled:
            return {
                "enabled": False,
                "collectives": {},
                "totals": _empty_stats(),
            }

        with self._lock:
            collectives = {
                kind: dict(values)
                for kind, values in self._stats.items()
            }
            totals = {
                field: sum(values[field] for values in collectives.values())
                for field in _SUM_FIELDS
            }
            totals["max_payload_bytes"] = max(
                values["max_payload_bytes"] for values in collectives.values()
            )
            if reset:
                self._stats = {kind: _empty_stats() for kind in _KINDS}
        return {
            "enabled": True,
            "collectives": collectives,
            "totals": totals,
        }


def create_collective_profiler() -> CollectiveProfiler:
    return CollectiveProfiler(os.environ.get("RAYLIGHT_P2P_PROFILE", "0") == "1")
