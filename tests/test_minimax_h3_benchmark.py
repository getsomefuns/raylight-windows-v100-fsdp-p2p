from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _prompt():
    return {
        "92": {
            "class_type": "SaveVideo",
            "inputs": {"filename_prefix": "video/original"},
        },
        "141": {
            "class_type": "RayInitializer",
            "inputs": {
                "GPU": 2,
                "FSDP": True,
                "FSDP_CPU_OFFLOAD": True,
            },
        },
        "142": {
            "class_type": "RayUNETLoader",
            "inputs": {"unet_name": "model.safetensors"},
        },
        "144": {
            "class_type": "XFuserSamplerCustomAdvanced",
            "inputs": {"noise_seed": 100},
        },
    }


def test_prepare_prompt_changes_seed_and_output_without_changing_topology():
    prompt = benchmark.prepare_prompt(_prompt(), "i2v", 2)

    assert prompt["144"]["inputs"]["noise_seed"] == 100 + 2 * benchmark.SEED_STRIDE
    assert prompt["92"]["inputs"]["filename_prefix"].endswith("i2v_run2")
    assert prompt["141"]["inputs"]["GPU"] == 2
    assert prompt["141"]["inputs"]["FSDP"] is True
    assert prompt["142"]["inputs"]["unet_name"] == "model.safetensors"


def test_validate_prompt_requires_dual_gpu_fsdp_and_one_sampler():
    benchmark.validate_prompt(_prompt())

    invalid = _prompt()
    invalid["141"]["inputs"]["GPU"] = 1
    try:
        benchmark.validate_prompt(invalid)
    except ValueError as exc:
        assert "GPU=2" in str(exc)
    else:
        raise AssertionError("single-GPU prompt was accepted")


def test_phase_summary_uses_slowest_rank_and_reports_tail():
    events = []
    for rank, offset in ((0, 0), (1, 1_000_000_000)):
        base = 10_000_000_000 + offset
        for name, delta in (
            ("sampler_entry", 0),
            ("load_models_gpu_begin", 2_000_000_000),
            ("load_models_gpu_end", 5_000_000_000),
            ("sample_begin", 6_000_000_000),
            ("sample_returned", 16_000_000_000),
            ("sampler_return", 17_000_000_000),
        ):
            events.append(
                {
                    "rank": rank,
                    "event": name,
                    "time_ns": base + delta,
                    "perf_ns": 1_000_000_000 + delta,
                    "pid": 100 + rank,
                    "invocation": 1,
                }
            )

    phases = benchmark.summarize_phases(
        events,
        execution_start_ms=9_000,
        execution_end_ms=30_000,
    )

    assert phases["rank_count"] == 2
    assert phases["worker_pids"] == [100, 101]
    assert phases["pre_sampler_seconds"] == 2.0
    assert phases["model_to_gpu_seconds"] == 3.0
    assert phases["sampling_seconds"] == 10.0
    assert phases["sampler_total_seconds"] == 17.0
    assert phases["decode_and_write_tail_seconds"] == 2.0


def test_reject_cached_sampler_detects_invalid_warm_measurement():
    history = {
        "status": {
            "messages": [
                ["execution_cached", {"nodes": ["92", "144"]}],
            ]
        }
    }

    try:
        benchmark.reject_cached_sampler(history, "144")
    except RuntimeError as exc:
        assert "sampler" in str(exc).lower()
    else:
        raise AssertionError("cached sampler measurement was accepted")


def test_server_environment_enables_native_windows_p2p():
    environment = benchmark.server_environment({})

    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert environment["RAYLIGHT_WINDOWS_P2P"] == "1"
    assert environment["USE_LIBUV"] == "0"
    assert environment["RAYLIGHT_GLOO_HOST"] == "127.0.0.1"
