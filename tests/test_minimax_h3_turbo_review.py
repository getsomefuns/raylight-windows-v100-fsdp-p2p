from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "minimax-h3" / "download-models.ps1"
MANIFEST_PATH = REPO_ROOT / "scripts" / "minimax-h3" / "models.json"
SPEC = importlib.util.spec_from_file_location("minimax_h3_turbo_review", BENCHMARK_SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_plan_rejects_same_size_file_with_wrong_sha256(tmp_path):
    model_root = tmp_path / "models"
    target = model_root / "loras" / "turbo.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"nope")
    manifest = tmp_path / "models.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository": "https://example.invalid/models",
                "revision": "a" * 40,
                "models": [
                    {
                        "id": "fl2v-turbo-8step",
                        "relative_path": "loras/turbo.safetensors",
                        "expected_bytes": 4,
                        "sha256": "0" * 64,
                        "groups": ["turbo"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DOWNLOAD_SCRIPT),
            "-Group",
            "turbo",
            "-ModelId",
            "fl2v-turbo-8step",
            "-ModelRoot",
            str(model_root),
            "-ManifestPath",
            str(manifest),
            "-PlanOnly",
            "-Json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    file = json.loads(result.stdout)["files"][0]
    assert file["status"] == "invalid"
    assert file["expected_sha256"] == "0" * 64
    assert file["actual_sha256"] != file["expected_sha256"]


def test_turbo_variant_contract_rejects_mode_and_step_mismatch():
    spec = benchmark.resolve_turbo_variant("fl2v-turbo-8step", "i2v", None)
    assert spec["lora_name"] == "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    assert spec["steps"] == 8

    with pytest.raises(ValueError, match="mode"):
        benchmark.resolve_turbo_variant("ref2v-turbo-4step", "i2v", None)
    with pytest.raises(ValueError, match="steps"):
        benchmark.resolve_turbo_variant("fl2v-turbo-8step", "i2v", 4)


def test_turbo_manifest_declares_compatibility_contracts():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    turbo = {item["id"]: item for item in manifest["models"] if "turbo" in item["groups"]}
    assert (turbo["fl2v-turbo-8step"]["mode"], turbo["fl2v-turbo-8step"]["steps"]) == ("i2v", 8)
    assert (turbo["ref2v-turbo-4step"]["mode"], turbo["ref2v-turbo-4step"]["steps"]) == ("ref2va", 4)
