from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark_diag_wait", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _events(rank: int, pid: int) -> list[dict]:
    return [
        {
            "rank": rank,
            "pid": pid,
            "invocation": 1,
            "event": name,
            "time_ns": 11_000_000_000 + delta,
            "perf_ns": 1_000_000_000 + delta,
        }
        for name, delta in (
            ("sampler_entry", 0),
            ("load_models_gpu_begin", 1_000_000_000),
            ("load_models_gpu_end", 2_000_000_000),
            ("sample_begin", 3_000_000_000),
            ("sample_returned", 4_000_000_000),
            ("sampler_return", 5_000_000_000),
        )
    ]


def _text(events: list[dict]) -> str:
    return "".join(f"[RAYLIGHT_RANK_DIAG] {json.dumps(event)}\n" for event in events)


def test_wait_for_phase_diagnostics_polls_until_delayed_rank_is_complete():
    rank0 = _events(0, 100)
    rank1 = _events(1, 101)
    segments = iter(
        [
            (_text(rank0 + rank1[:-2]), 1000),
            (_text(rank0 + rank1), 1200),
        ]
    )
    calls = []

    phases, text, end_offset = benchmark.wait_for_phase_diagnostics(
        lambda: next(segments),
        execution_start_ms=10_000,
        execution_end_ms=20_000,
        timeout=5,
        sleep=lambda seconds: calls.append(seconds),
    )

    assert phases["worker_pids"] == [100, 101]
    assert end_offset == 1200
    assert "sampler_return" in text
    assert calls
