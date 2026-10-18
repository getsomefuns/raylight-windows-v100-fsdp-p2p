from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "minimax-h3"
    / "benchmark_dual_vae_decode.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("benchmark_dual_vae_decode", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_exposes_reproducible_quality_and_speed_gates(tmp_path: Path) -> None:
    module = _load_script()
    args = module.parse_args(
        [
            "--comfy-root",
            str(tmp_path / "ComfyUI"),
            "--vae-path",
            str(tmp_path / "video.safetensors"),
            "--latent-path",
            str(tmp_path / "seg.pt"),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )

    assert args.minimum_psnr_db == 58.0
    assert args.maximum_abs_error == 0.003
    assert args.maximum_warm_seconds == 42.0
    assert args.minimum_improvement_percent == 35.0
    assert args.maximum_gpu_peak_mib == 7 * 1024
    assert args.dual_runs == 5


def test_extract_video_latent_accepts_nested_av_cache() -> None:
    module = _load_script()
    video = torch.zeros(1, 24, 32, 4, 6)
    audio = torch.zeros(1, 32, 2, 12)
    nested = SimpleNamespace(is_nested=True, unbind=lambda: (video, audio))

    extracted = module.extract_video_latent({"samples": nested})

    assert extracted.shape == video.shape
    assert torch.equal(extracted, video)


def test_quality_metrics_are_streamed_and_exact() -> None:
    module = _load_script()
    baseline = torch.zeros(1, 3, 2, 2, 2)
    candidate = baseline.clone()
    candidate[:, :, 1, 0, 0] = 0.001

    metrics = module.quality_metrics(baseline, candidate)

    assert metrics["max_abs_error"] == pytest.approx(0.001)
    assert metrics["mse"] > 0
    assert metrics["psnr_db"] > 58
    assert metrics["finite"] is True


def test_normalize_native_output_restores_bcthw_layout() -> None:
    module = _load_script()
    flattened = torch.zeros(107, 8, 12, 3)

    normalized = module.normalize_native_output(flattened, batch=1, frames=107)

    assert normalized.shape == (1, 3, 107, 8, 12)


def test_measured_call_matches_comfyui_inference_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script()

    def sample_once(_stop, samples, _interval):
        samples.append(
            {
                "physical_used_mib": 1.0,
                "committed_mib": 2.0,
                "pagefile_used_mib": 3.0,
                "gpus": [
                    {"index": 0, "memory_mib": 4.0, "util_percent": 5.0, "power_w": 6.0},
                    {"index": 1, "memory_mib": 4.0, "util_percent": 5.0, "power_w": 6.0},
                ],
            }
        )

    monkeypatch.setattr(module, "_sample_resources", sample_once)
    monkeypatch.setattr(module.torch.cuda, "synchronize", lambda: None)

    enabled, _record = module._run_measured(torch.is_inference_mode_enabled, 0.01)

    assert enabled is True


def test_build_report_applies_all_acceptance_gates() -> None:
    module = _load_script()
    native = {"seconds": 64.0, "resources": {"gpus": {"0": {"peak_memory_mib": 6000}}}}
    dual_runs = [
        {"seconds": 40.0, "resources": {"gpus": {"0": {"peak_memory_mib": 6900}, "1": {"peak_memory_mib": 6800}}}},
        {"seconds": 39.0, "resources": {"gpus": {"0": {"peak_memory_mib": 7000}, "1": {"peak_memory_mib": 6950}}}},
    ]
    quality = {"psnr_db": 59.0, "max_abs_error": 0.002, "mse": 1e-6, "finite": True}

    report = module.build_report(
        native=native,
        dual_runs=dual_runs,
        quality=quality,
        thresholds={
            "minimum_psnr_db": 58.0,
            "maximum_abs_error": 0.003,
            "maximum_warm_seconds": 42.0,
            "minimum_improvement_percent": 35.0,
            "maximum_gpu_peak_mib": 7168.0,
            "maximum_dual_ram_growth_mib": 4096.0,
        },
    )

    assert report["comparison"]["warm_speedup"] == pytest.approx(64 / 39)
    assert report["comparison"]["warm_improvement_percent"] == pytest.approx(39.0625)
    assert all(report["gates"].values())
    assert report["passed"] is True


def test_build_report_fails_when_any_gpu_exceeds_memory_gate() -> None:
    module = _load_script()
    report = module.build_report(
        native={"seconds": 64.0, "resources": {"gpus": {}}},
        dual_runs=[
            {"seconds": 39.0, "resources": {"gpus": {"0": {"peak_memory_mib": 7200}, "1": {"peak_memory_mib": 6900}}}}
        ],
        quality={"psnr_db": 60.0, "max_abs_error": 0.001, "mse": 1e-7, "finite": True},
        thresholds={
            "minimum_psnr_db": 58.0,
            "maximum_abs_error": 0.003,
            "maximum_warm_seconds": 42.0,
            "minimum_improvement_percent": 35.0,
            "maximum_gpu_peak_mib": 7168.0,
            "maximum_dual_ram_growth_mib": 4096.0,
        },
    )

    assert report["gates"]["gpu_peak"] is False
    assert report["passed"] is False


def test_build_report_rejects_any_bad_dual_run_quality() -> None:
    module = _load_script()
    good = {"psnr_db": 60.0, "max_abs_error": 0.001, "mse": 1e-7, "finite": True}
    bad = {"psnr_db": 40.0, "max_abs_error": 0.1, "mse": 1e-2, "finite": True}
    report = module.build_report(
        native={"seconds": 64.0, "resources": {"gpus": {}}},
        dual_runs=[
            {
                "seconds": 39.0,
                "quality": bad,
                "resources": {
                    "end_physical_used_mib": 1000,
                    "gpus": {"0": {"peak_memory_mib": 6900}, "1": {"peak_memory_mib": 6800}},
                },
            },
            {
                "seconds": 39.0,
                "quality": good,
                "resources": {
                    "end_physical_used_mib": 1000,
                    "gpus": {"0": {"peak_memory_mib": 6900}, "1": {"peak_memory_mib": 6800}},
                },
            },
        ],
        quality=good,
        thresholds={
            "minimum_psnr_db": 58.0,
            "maximum_abs_error": 0.003,
            "maximum_warm_seconds": 42.0,
            "minimum_improvement_percent": 35.0,
            "maximum_gpu_peak_mib": 7168.0,
            "maximum_dual_ram_growth_mib": 4096.0,
        },
    )

    assert report["gates"]["psnr"] is False
    assert report["gates"]["max_abs_error"] is False
    assert report["passed"] is False
