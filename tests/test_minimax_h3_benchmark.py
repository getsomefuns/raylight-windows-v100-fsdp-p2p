from __future__ import annotations

import importlib.util
from pathlib import Path
import threading

import pytest


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

def test_configure_prompt_geometry_uses_exact_dimensions_duration_and_fps():
    prompt = {
        "10": {
            "class_type": "ResolutionSelector",
            "inputs": {"aspect_ratio": "16:9 (Widescreen)", "megapixels": 0.4, "multiple": 32},
        },
        "11": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "width": ["10", 0],
                "height": ["10", 1],
                "length": ["9", 1],
            },
        },
        "9": {
            "class_type": "ComfyMathExpression",
            "inputs": {"values.a": ["12", 0], "expression": "round(a * 24)"},
        },
        "12": {"class_type": "PrimitiveFloat", "inputs": {"value": 2.0}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["11", 0], "fps": 12}},
    }

    geometry = benchmark.configure_prompt_geometry(
        prompt,
        width=1120,
        height=768,
        duration=5.0,
        fps=24,
        expected_frames=124,
    )

    assert "10" not in prompt
    assert prompt["11"]["inputs"]["width"] == 1120
    assert prompt["11"]["inputs"]["height"] == 768
    assert prompt["11"]["inputs"]["length"] == 124
    assert prompt["12"]["inputs"]["value"] == 5.0
    assert prompt["13"]["inputs"]["fps"] == 24
    assert geometry == {
        "width": 1120,
        "height": 768,
        "duration_seconds": 5.0,
        "fps": 24,
        "expected_frames": 124,
        "playback_duration_seconds": 124 / 24,
    }


def test_summarize_node_timings_separates_load_preprocess_sample_decode_and_save():
    prompt = {
        "0": {"class_type": "RayInitializer", "inputs": {}},
        "1": {"class_type": "LoadImage", "inputs": {}},
        "2": {"class_type": "RayUNETLoader", "inputs": {}},
        "3": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {}},
        "4": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {}},
        "5": {"class_type": "VAEDecode", "inputs": {}},
        "6": {"class_type": "VAEDecodeAudio", "inputs": {}},
        "7": {"class_type": "CreateVideo", "inputs": {}},
        "8": {"class_type": "SaveVideo", "inputs": {}},
    }
    events = [
        {"prompt_id": "other", "node": "1", "time_ns": 0},
        {"prompt_id": "p", "node": "1", "time_ns": 0},
        {"prompt_id": "p", "node": "0", "time_ns": 1_000_000_000},
        {"prompt_id": "p", "node": "2", "time_ns": 3_000_000_000},
        {"prompt_id": "p", "node": "3", "time_ns": 6_000_000_000},
        {"prompt_id": "p", "node": "4", "time_ns": 8_000_000_000},
        {"prompt_id": "p", "node": "5", "time_ns": 18_000_000_000},
        {"prompt_id": "p", "node": "6", "time_ns": 20_000_000_000},
        {"prompt_id": "p", "node": "7", "time_ns": 21_000_000_000},
        {"prompt_id": "p", "node": "8", "time_ns": 22_000_000_000},
        {"prompt_id": "p", "node": None, "time_ns": 24_000_000_000},
    ]

    summary = benchmark.summarize_node_timings(events, prompt, prompt_id="p")

    assert summary["model_load_seconds"] == 3.0
    assert summary["preprocessing_seconds"] == 3.0
    assert summary["ray_initialization_seconds"] == 2.0
    assert summary["sampler_node_seconds"] == 10.0
    assert summary["vae_decode_seconds"] == 3.0
    assert summary["video_create_seconds"] == 1.0
    assert summary["video_save_seconds"] == 2.0
    assert summary["nodes"]["8"]["seconds"] == 2.0


def test_extract_sampler_progress_reports_final_rank_seconds_per_iteration():
    text = (
        "\x1b[36m(pid=101)\x1b[0m: 100%|##########| 8.00/8.00 [01:20<00:00, 10.0s/it]\n"
        "\x1b[36m(pid=102)\x1b[0m: 100%|##########| 8.00/8.00 [01:24<00:00, 10.5s/it]\n"
    )

    result = benchmark.extract_sampler_progress(text, {"0": 101, "1": 102})

    assert result == {
        "0": {"pid": 101, "seconds_per_iteration": 10.0},
        "1": {"pid": 102, "seconds_per_iteration": 10.5},
    }


def test_extract_sampler_progress_rejects_missing_known_worker_progress():
    with pytest.raises(RuntimeError, match="s/it"):
        benchmark.extract_sampler_progress("unrelated stderr", {"0": 101, "1": 102})


def test_extract_sampler_progress_accepts_rank_zero_progress_with_rank_diagnostics():
    text = "(pid=101): 100%|##########| 8.00/8.00 [01:20<00:00, 10.0s/it]"

    assert benchmark.extract_sampler_progress(text, {"0": 101, "1": 102}) == {
        "0": {"pid": 101, "seconds_per_iteration": 10.0}
    }

def test_decode_node_event_keeps_only_executing_messages():
    message = '{"type":"executing","data":{"prompt_id":"p","node":"42"}}'

    assert benchmark.decode_node_event(message, time_ns=123) == {
        "prompt_id": "p",
        "node": "42",
        "time_ns": 123,
    }
    assert benchmark.decode_node_event('{"type":"progress","data":{}}', time_ns=456) is None
    assert benchmark.decode_node_event(b"binary-preview", time_ns=789) is None


def test_decode_node_event_accepts_execution_success_as_terminal_boundary():
    message = '{"type":"execution_success","data":{"prompt_id":"p"}}'

    assert benchmark.decode_node_event(message, time_ns=321) == {
        "prompt_id": "p",
        "node": None,
        "time_ns": 321,
    }


def test_wait_for_terminal_node_event_handles_history_websocket_race():
    events = [{"prompt_id": "p", "node": "8", "time_ns": 1}]
    timer = threading.Timer(
        0.01,
        lambda: events.append({"prompt_id": "p", "node": None, "time_ns": 2}),
    )
    timer.start()
    try:
        terminal = benchmark.wait_for_terminal_node_event(
            events,
            prompt_id="p",
            timeout=0.5,
            poll_interval=0.001,
        )
    finally:
        timer.join()

    assert terminal["time_ns"] == 2


def test_parse_args_accepts_exact_baseline_geometry():
    args = benchmark.parse_args(
        [
            "--mode",
            "i2v",
            "--profile",
            "full",
            "--runs",
            "1",
            "--turbo-variant",
            "fl2v-turbo-8step",
            "--width",
            "1120",
            "--height",
            "768",
            "--duration",
            "5",
            "--fps",
            "24",
            "--expected-frames",
            "124",
        ]
    )

    assert args.width == 1120
    assert args.height == 768
    assert args.duration == 5.0
    assert args.fps == 24
    assert args.expected_frames == 124
