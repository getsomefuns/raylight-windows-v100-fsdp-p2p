from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark_hardening", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _complete_events(rank: int, pid: int, invocation: int, base_ns: int) -> list[dict]:
    return [
        {
            "rank": rank,
            "pid": pid,
            "invocation": invocation,
            "event": name,
            "time_ns": base_ns + delta,
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


def test_phase_summary_filters_stale_events_outside_history_window():
    events = _complete_events(0, 10, 1, 1_000_000_000)
    events += _complete_events(1, 11, 1, 1_000_000_000)
    events += _complete_events(0, 20, 2, 11_000_000_000)
    events += _complete_events(1, 21, 2, 11_000_000_000)

    phases = benchmark.summarize_phases(events, execution_start_ms=10_000, execution_end_ms=20_000)

    assert phases["worker_pids"] == [20, 21]


def test_phase_summary_rejects_two_complete_invocations_for_one_rank():
    events = _complete_events(0, 10, 1, 11_000_000_000)
    events += _complete_events(0, 20, 2, 12_000_000_000)
    events += _complete_events(1, 21, 2, 12_000_000_000)

    with pytest.raises(RuntimeError, match="one complete invocation"):
        benchmark.summarize_phases(events, execution_start_ms=10_000, execution_end_ms=20_000)


def test_terminate_closes_job_even_when_parent_already_exited():
    process = mock.Mock()
    process.poll.return_value = 1
    job = mock.Mock()

    benchmark.terminate_process_tree(process, job)

    job.close.assert_called_once_with()


def test_monitor_validation_rejects_missing_samples_or_gpu():
    with pytest.raises(RuntimeError, match="too few"):
        benchmark.validate_monitor_samples([], [], minimum_samples=5)

    bad_samples = [
        {
            "gpus": [{"index": 0}],
        }
        for _ in range(5)
    ]
    with pytest.raises(RuntimeError, match="GPU 0 and 1"):
        benchmark.validate_monitor_samples(bad_samples, [], minimum_samples=5)


def test_runtime_identity_requires_deployed_source_commit_and_clean_runtime():
    valid = {
        "source": {"head": "abc", "runtime_dirty": False, "status": " M README.md"},
        "installed": {"head": "deploy", "dirty": False, "status": ""},
        "deployed_source_commit": "abc",
    }
    benchmark.validate_runtime_identity(valid)

    mismatch = {**valid, "deployed_source_commit": "def"}
    with pytest.raises(RuntimeError, match="does not match"):
        benchmark.validate_runtime_identity(mismatch)

    dirty = {
        **valid,
        "source": {**valid["source"], "runtime_dirty": True},
    }
    with pytest.raises(RuntimeError, match="runtime source files are dirty"):
        benchmark.validate_runtime_identity(dirty)


def test_build_api_prompt_forwards_safe_fp16_compute_dtype():
    seen = {}

    def build_workflow(source, **kwargs):
        seen.update(kwargs)
        return {"source": source}

    builder = SimpleNamespace(
        SOURCE_WORKFLOWS={"i2v": Path("source.json")},
        build_workflow=build_workflow,
    )
    converter = SimpleNamespace(
        workflow_to_prompt=lambda workflow, object_info: {
            "init": {
                "class_type": "RayInitializer",
                "inputs": {},
            }
        }
    )

    with (
        mock.patch.object(benchmark, "_load_script", side_effect=[builder, converter]),
        mock.patch.object(benchmark, "request_json", return_value={}),
        mock.patch.object(benchmark, "validate_prompt"),
    ):
        benchmark.build_api_prompt(
            "http://127.0.0.1:8188",
            "i2v",
            "full",
            True,
            compute_dtype="fp16_h3_safe",
            turbo_variant="fl2v-turbo-8step",
        )

    assert seen["compute_dtype"] == "fp16_h3_safe"
    assert seen["turbo_variant"] == "fl2v-turbo-8step"


def test_parse_args_accepts_safe_fp16_compute_dtype():
    args = benchmark.parse_args(
        [
            "--mode",
            "i2v",
            "--profile",
            "full",
            "--compute-dtype",
            "fp16_h3_safe",
            "--turbo-variant",
            "fl2v-turbo-8step",
        ]
    )

    assert args.compute_dtype == "fp16_h3_safe"


def test_parse_args_rejects_safe_fp16_without_turbo_variant():
    with pytest.raises(ValueError, match="Turbo"):
        benchmark.parse_args(
            [
                "--mode",
                "i2v",
                "--profile",
                "full",
                "--compute-dtype",
                "fp16_h3_safe",
            ]
        )
