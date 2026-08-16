import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "scripts" / "minimax-h3" / "models.json"
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "minimax-h3" / "download-models.ps1"


def test_i2v_fp8_plan_uses_only_official_files_and_exact_remote_sizes(tmp_path):
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DOWNLOAD_SCRIPT),
            "-Group",
            "i2v-fp8",
            "-ModelRoot",
            str(tmp_path),
            "-PlanOnly",
            "-Json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["group"] == "i2v-fp8"
    assert plan["total_bytes"] == 42_458_411_463
    assert [item["relative_path"] for item in plan["files"]] == [
        "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae/minimax_h3_video_vae_fp16.safetensors",
        "vae/minimax_h3_audio_vae_fp32.safetensors",
    ]
    assert [item["expected_bytes"] for item in plan["files"]] == [
        20_958_205_608,
        15_687_142_551,
        5_207_808_496,
        605_254_808,
    ]
    assert all(
        item["url"]
        == "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/" + item["relative_path"]
        for item in plan["files"]
    )
    assert all(item["status"] == "missing" for item in plan["files"])
    assert not list(tmp_path.rglob("*"))


def test_manifest_keeps_precision_comparisons_in_separate_download_groups():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    groups = {
        group: [item["relative_path"] for item in manifest["models"] if group in item["groups"]]
        for group in ("i2v-fp8", "ref2va-fp8", "i2v-int8", "ref2va-int8", "turbo")
    }

    assert groups == {
        "i2v-fp8": [
            "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "vae/minimax_h3_video_vae_fp16.safetensors",
            "vae/minimax_h3_audio_vae_fp32.safetensors",
        ],
        "ref2va-fp8": [
            "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        ],
        "i2v-int8": [
            "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        ],
        "ref2va-int8": [
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ],
        "turbo": [
            "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
            "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
            "loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        ],
    }
