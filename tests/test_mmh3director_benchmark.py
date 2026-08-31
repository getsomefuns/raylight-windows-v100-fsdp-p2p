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


def _profile(rank: int, elapsed_ns: int, calls: int, time_ns: int | None = None) -> dict:
    return {
        "rank": rank,
        "invocation": 0,
        "elapsed_ns": elapsed_ns,
        "time_ns": elapsed_ns if time_ns is None else time_ns,
        "status": "success",
        "profile": {"collectives": {"all_to_all": {"calls": calls, "chunks": calls}}},
    }


def test_summarize_director_phases_keeps_repeated_zero_invocations():
    events = []
    required = (
        "prepare_sampling_begin",
        "load_models_gpu_begin",
        "load_models_gpu_end",
        "prepare_sampling_end",
    )
    for operation_index in (0, 1):
        for rank in (0, 1):
            base = (operation_index + 1) * 1_000_000_000 + rank * 10_000_000
            for offset, name in enumerate(required):
                events.append(_event(rank, 0, name, base + offset * 100_000_000, base + offset * 100_000_000))
    profiles = [
        _profile(0, 1_000_000_000, 800, 1_400_000_000),
        _profile(1, 1_100_000_000, 800, 1_500_000_000),
        _profile(1, 2_100_000_000, 400, 3_500_000_000),
    ]

    summary = benchmark.summarize_director_phases(
        events,
        profiles,
        execution_start_ms=0,
        execution_end_ms=20_000,
    )

    assert summary["invocation_count"] == 2
    assert [row["operation_index"] for row in summary["invocations"]] == [0, 1]
    assert [row["all_to_all_calls"] for row in summary["invocations"]] == [800, 400]
    assert summary["worker_pids"] == [1000, 1001]
    assert summary["sampling_seconds"] == pytest.approx(3.2)
    assert summary["worker_pids_by_rank"] == {"0": 1000, "1": 1001}


def test_summarize_director_phases_rejects_missing_peer_rank():
    names = (
        "prepare_sampling_begin",
        "load_models_gpu_begin",
        "load_models_gpu_end",
        "prepare_sampling_end",
    )
    events = [_event(0, 1, name, index, index) for index, name in enumerate(names)]

    with pytest.raises(RuntimeError, match="ranks 0 and 1"):
        benchmark.summarize_director_phases(
            events,
            [_profile(0, 1, 1)],
            execution_start_ms=0,
            execution_end_ms=1,
        )


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
    assert args.reserve_vram_gib == 4.0


def test_parse_args_accepts_explicit_vram_reserve():
    args = benchmark.parse_args(
        ["workflow.json", "--reserve-vram-gib", "5.5"]
    )

    assert args.reserve_vram_gib == 5.5


def test_parse_args_rejects_non_positive_runs():
    with pytest.raises(ValueError, match="runs must be positive"):
        benchmark.parse_args(["workflow.json", "--runs", "0"])


def test_parse_args_rejects_non_positive_vram_reserve():
    with pytest.raises(ValueError, match="reserve-vram-gib must be positive"):
        benchmark.parse_args(["workflow.json", "--reserve-vram-gib", "0"])


def test_prepare_director_prompt_changes_seed_and_output_prefix_per_run():
    template = {
        "5": {
            "class_type": "MiniMaxH3Director",
            "inputs": {"seed": 42, "global_prompt": "same prompt"},
        },
        "7": {
            "class_type": "SaveVideo",
            "inputs": {"filename_prefix": "video/MMH3Director_O9Adaptive128"},
        },
    }

    cold = benchmark.prepare_director_prompt(template, 0)
    warm = benchmark.prepare_director_prompt(template, 1)

    assert cold["5"]["inputs"]["seed"] == 42
    assert warm["5"]["inputs"]["seed"] == 43
    assert cold["7"]["inputs"]["filename_prefix"].endswith("_run0")
    assert warm["7"]["inputs"]["filename_prefix"].endswith("_run1")
    assert template["5"]["inputs"]["seed"] == 42
