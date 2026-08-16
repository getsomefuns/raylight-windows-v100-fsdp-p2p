import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSPECT_SCRIPT = REPO_ROOT / "scripts" / "minimax-h3" / "inspect-environment.ps1"


def test_inventory_reports_the_complete_fp8_baseline_contract(tmp_path):
    """A wrong/missing asset entry must make the pre-download inventory fail this contract."""
    model_root = tmp_path / "models"
    comfy_root = tmp_path / "ComfyUI"
    model_root.mkdir()
    comfy_root.mkdir()

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSPECT_SCRIPT),
            "-ModelRoot",
            str(model_root),
            "-ComfyRoot",
            str(comfy_root),
            "-PythonPath",
            sys.executable,
            "-SkipRuntimeProbes",
            "-Json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assets = {item["name"]: item for item in report["models"]}

    assert assets == {
        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors": {
            "name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
            "subdirectory": "diffusion_models",
            "expected_bytes": 20_958_205_608,
            "exists": False,
            "size_bytes": None,
            "complete": False,
        },
        "minimax_h3_ref2va_pruned_fp8_scaled.safetensors": {
            "name": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
            "subdirectory": "diffusion_models",
            "expected_bytes": 20_958_205_608,
            "exists": False,
            "size_bytes": None,
            "complete": False,
        },
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": {
            "name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "subdirectory": "text_encoders",
            "expected_bytes": 15_687_142_551,
            "exists": False,
            "size_bytes": None,
            "complete": False,
        },
        "minimax_h3_video_vae_fp16.safetensors": {
            "name": "minimax_h3_video_vae_fp16.safetensors",
            "subdirectory": "vae",
            "expected_bytes": 5_207_808_496,
            "exists": False,
            "size_bytes": None,
            "complete": False,
        },
        "minimax_h3_audio_vae_fp32.safetensors": {
            "name": "minimax_h3_audio_vae_fp32.safetensors",
            "subdirectory": "vae",
            "expected_bytes": 605_254_808,
            "exists": False,
            "size_bytes": None,
            "complete": False,
        },
    }
    assert report["summary"] == {
        "model_count": 5,
        "complete_count": 0,
        "missing_or_incomplete_count": 5,
        "expected_bytes": 63_416_617_071,
    }

def test_runtime_probe_returns_machine_readable_python_versions(tmp_path):
    """Broken Python probe quoting must fail instead of producing a partial inventory."""
    model_root = tmp_path / "models"
    comfy_root = tmp_path / "ComfyUI"
    model_root.mkdir()
    comfy_root.mkdir()

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSPECT_SCRIPT),
            "-ModelRoot",
            str(model_root),
            "-ComfyRoot",
            str(comfy_root),
            "-PythonPath",
            sys.executable,
            "-Json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    report = json.loads(result.stdout)
    assert report["runtime"]["python"]["python"] == "3.10.11"
    assert report["runtime"]["python"]["torch"] == "2.7.0+cu126"
    assert report["runtime"]["python"]["torch_cuda"] == "12.6"
    assert "sm_70" in report["runtime"]["python"]["sm_arches"]
