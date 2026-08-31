from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_director_workflow.py"
SPEC = importlib.util.spec_from_file_location("mmh3director_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _event(rank: int, invocation: int, name: str, perf_ns: int, time_ns: int) -> dict:
    return {
        "rank": rank,
        "pid": 1000 + rank,
        "invocation": invocation,
        "event": name,
        "perf_ns": perf_ns,
        "time_ns": time_ns,
    }


def test_summarize_director_phases_keeps_every_complete_invocation():
    events = []
    required = (
        "sampler_entry",
        "load_models_gpu_begin",
        "load_models_gpu_end",
        "sample_begin",
        "sample_returned",
        "sampler_return",
    )
    for invocation in (7, 8):
        for rank in (0, 1):
            base = invocation * 1_000_000_000 + rank * 10_000_000
            for offset, name in enumerate(required):
                events.append(_event(rank, invocation, name, base + offset * 100_000_000, base + offset * 100_000_000))

    summary = benchmark.summarize_director_phases(
        events,
        execution_start_ms=0,
        execution_end_ms=20_000,
    )

    assert summary["invocation_count"] == 2
    assert [row["invocation"] for row in summary["invocations"]] == [7, 8]
    assert summary["worker_pids"] == [1000, 1001]
    assert summary["sampling_seconds"] == pytest.approx(0.2)
    assert summary["worker_pids_by_rank"] == {"0": 1000, "1": 1001}


def test_summarize_director_phases_rejects_missing_peer_rank():
    names = (
        "sampler_entry",
        "load_models_gpu_begin",
        "load_models_gpu_end",
        "sample_begin",
        "sample_returned",
        "sampler_return",
    )
    events = [_event(0, 1, name, index, index) for index, name in enumerate(names)]

    with pytest.raises(RuntimeError, match="ranks 0 and 1"):
        benchmark.summarize_director_phases(events, execution_start_ms=0, execution_end_ms=1)


def test_validate_director_prompt_requires_adapter_dual_gpu_and_fsdp():
    prompt = {
        "5": {"class_type": "MiniMaxH3Director", "inputs": {}},
        "13": {
            "class_type": "RayInitializer",
            "inputs": {"GPU": 2, "FSDP": True},
        },
        "16": {"class_type": "RayMMH3DirectorModelAdapter", "inputs": {}},
        "7": {"class_type": "SaveVideo", "inputs": {}},
    }
    benchmark.validate_director_prompt(prompt)

    prompt["13"]["inputs"]["GPU"] = 1
    with pytest.raises(ValueError, match="GPU=2"):
        benchmark.validate_director_prompt(prompt)


def test_parse_args_defaults_to_cold_and_warm_runs():
    args = benchmark.parse_args(["workflow.json"])

    assert args.runs == 2


def test_parse_args_rejects_non_positive_runs():
    with pytest.raises(ValueError, match="runs must be positive"):
        benchmark.parse_args(["workflow.json", "--runs", "0"])
