from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from comfy.weight_adapter.bypass import BypassForwardHook
from raylight.comfy_dist.sd import defer_adapter_device_move
from raylight.comfy_dist.weight_adapter.lora import LoRAAdapter


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
MANIFEST = REPO_ROOT / "scripts" / "minimax-h3" / "models.json"
SPEC = importlib.util.spec_from_file_location("minimax_h3_turbo_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def _prompt():
    return {
        "92": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "video/original"}},
        "141": {"class_type": "RayInitializer", "inputs": {"GPU": 2, "FSDP": True}},
        "142": {
            "class_type": "RayUNETLoader",
            "inputs": {"unet_name": "model.safetensors", "ray_actors_init": ["141", 0]},
        },
        "143": {
            "class_type": "RayBasicScheduler",
            "inputs": {"ray_actors": ["142", 0], "steps": 20},
        },
        "144": {"class_type": "XFuserSamplerCustomAdvanced", "inputs": {"noise_seed": 100}},
    }


def test_prepare_prompt_inserts_turbo_lora_and_official_step_count():
    prompt = benchmark.prepare_prompt(
        _prompt(),
        "i2v",
        0,
        lora_name="minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        steps=8,
        output_tag="turbo8",
    )

    lora_nodes = benchmark._nodes(prompt, "RayLoraLoader")
    assert len(lora_nodes) == 1
    lora_id, lora = lora_nodes[0]
    assert lora["inputs"] == {
        "lora_name": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "strength_model": 1.0,
    }
    assert prompt["142"]["inputs"]["lora"] == [lora_id, 0]
    assert prompt["143"]["inputs"]["steps"] == 8
    assert prompt["92"]["inputs"]["filename_prefix"] == (
        "video/raylight_o3/minimax_h3_i2v_turbo8_run0"
    )


def test_prepare_prompt_rejects_steps_without_exactly_one_scheduler():
    prompt = _prompt()
    del prompt["143"]

    try:
        benchmark.prepare_prompt(prompt, "i2v", 0, steps=8)
    except ValueError as exc:
        assert "RayBasicScheduler" in str(exc)
    else:
        raise AssertionError("Turbo prompt without a scheduler was accepted")


def test_turbo_manifest_is_revision_and_sha256_pinned():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["revision"] == "cec22ac7545ee166df6af79fda42bd41558f8558"
    turbo = {item["id"]: item for item in manifest["models"] if "turbo" in item["groups"]}
    assert turbo["fl2v-turbo-8step"]["sha256"] == (
        "2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e"
    )
    assert turbo["ref2v-turbo-4step"]["sha256"] == (
        "5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c"
    )


def test_lora_asset_identity_streams_hash_without_path_read_bytes(tmp_path):
    comfy_root = tmp_path / "ComfyUI"
    lora_root = comfy_root / "models" / "loras"
    lora_root.mkdir(parents=True)
    path = lora_root / "turbo.safetensors"
    path.write_bytes(b"test-lora-payload")

    with mock.patch.object(benchmark, "COMFY_ROOT", comfy_root):
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("not streaming")):
            identity = benchmark.lora_asset_identity(path.name)

    assert identity["size_bytes"] == len(b"test-lora-payload")
    assert identity["sha256"] == hashlib.sha256(b"test-lora-payload").hexdigest()


def test_bf16_lora_sidecar_follows_fp32_v100_compute_dtype():
    up = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    down = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.bfloat16)
    adapter = defer_adapter_device_move(LoRAAdapter({"up", "down"}, (up, down, 2.0, None, None, None)))
    layer = torch.nn.Linear(2, 2, bias=False, dtype=torch.float32)
    with torch.no_grad():
        layer.weight.zero_()
    hook = BypassForwardHook(layer, adapter, multiplier=1.0)
    hook.inject()
    seen_weight_dtypes = []
    original_linear = F.linear

    def tracking_linear(value, weight, bias=None):
        seen_weight_dtypes.append(weight.dtype)
        return original_linear(value, weight, bias)

    x = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    with mock.patch(
        "raylight.comfy_dist.weight_adapter.lora.F.linear",
        side_effect=tracking_linear,
    ):
        actual = layer(x)

    expected = original_linear(original_linear(x, down.float()), up.float())
    torch.testing.assert_close(actual, expected)
    assert len(seen_weight_dtypes) >= 2
    assert set(seen_weight_dtypes) == {torch.float32}
    hook.eject()


def test_bf16_lora_sidecar_follows_safe_fp16_branch_dtype():
    up = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    down = torch.tensor([[0.5, 1.0], [1.5, 2.0]], dtype=torch.bfloat16)
    adapter = defer_adapter_device_move(LoRAAdapter({"up", "down"}, (up, down, 2.0, None, None, None)))
    layer = torch.nn.Linear(2, 2, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.zero_()
    hook = BypassForwardHook(layer, adapter, multiplier=1.0)
    hook.inject()
    seen_weight_dtypes = []
    original_linear = F.linear

    def tracking_linear(value, weight, bias=None):
        seen_weight_dtypes.append(weight.dtype)
        return original_linear(value, weight, bias)

    x = torch.tensor([[1.0, -1.0]], dtype=torch.float16)
    with mock.patch(
        "raylight.comfy_dist.weight_adapter.lora.F.linear",
        side_effect=tracking_linear,
    ):
        actual = layer(x)

    expected = original_linear(original_linear(x, down.half()), up.half())
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert len(seen_weight_dtypes) >= 2
    assert set(seen_weight_dtypes) == {torch.float16}
    hook.eject()


def test_safe_attention_projection_scales_post_injection_lora_sidecar():
    from raylight.comfy_dist.minimax_h3_fp16 import (
        activate_minimax_h3_safe_fp16_model,
        safe_attention_output_projection,
    )

    layer = torch.nn.Linear(1, 1, bias=False, dtype=torch.float16)
    with torch.no_grad():
        layer.weight.fill_(1.0)
    model = SimpleNamespace(
        dtype=torch.float16,
        condition_proj=torch.nn.Identity(),
        blocks=[SimpleNamespace(attn=SimpleNamespace(out_proj=layer), mlp=SimpleNamespace())],
    )
    assert activate_minimax_h3_safe_fp16_model(model) is True

    up = torch.tensor([[2.0]], dtype=torch.bfloat16)
    down = torch.tensor([[1.0]], dtype=torch.bfloat16)
    adapter = defer_adapter_device_move(
        LoRAAdapter({"up", "down"}, (up, down, 1.0, None, None, None))
    )
    hook = BypassForwardHook(layer, adapter, multiplier=1.0)
    hook.inject()
    try:
        actual = safe_attention_output_projection(
            layer,
            torch.tensor([[60_000.0]], dtype=torch.float16),
        )
    finally:
        hook.eject()

    assert actual.dtype is torch.float32
    assert torch.isfinite(actual).all()
    assert actual.item() == pytest.approx(180_000.0, rel=3e-3)
