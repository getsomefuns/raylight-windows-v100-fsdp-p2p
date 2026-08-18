import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "scripts" / "minimax-h3" / "build_workflows.py"
WORKFLOW_ROOT = REPO_ROOT / "example_workflows"


def _load_builder():
    spec = importlib.util.spec_from_file_location("minimax_h3_workflow_builder", BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _node(workflow, node_type):
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    assert len(matches) == 1
    return matches[0]


def _nodes(workflow, node_type):
    return [node for node in workflow["nodes"] if node["type"] == node_type]


def _assert_conditioning_link(workflow, source_type):
    source = _node(workflow, source_type)
    guider = _node(workflow, "RayBasicGuider")
    source_slot = next(i for i, output in enumerate(source["outputs"]) if output["name"] == "positive")
    target_slot = next(i for i, input_ in enumerate(guider["inputs"]) if input_["name"] == "conditioning")
    link_id = guider["inputs"][target_slot]["link"]
    assert link_id is not None
    assert link_id in source["outputs"][source_slot]["links"]
    assert [link_id, source["id"], source_slot, guider["id"], target_slot, "CONDITIONING"] in workflow["links"]


def _assert_initializer_wait_link(workflow, source_type):
    source = _node(workflow, source_type)
    initializer = _node(workflow, "RayInitializer")
    source_slot = next(i for i, output in enumerate(source["outputs"]) if output["name"] == "positive")
    wait_slot = next(i for i, input_ in enumerate(initializer["inputs"]) if input_["name"] == "wait_for")
    link_id = initializer["inputs"][wait_slot]["link"]
    assert link_id is not None
    assert link_id in source["outputs"][source_slot]["links"]
    assert [link_id, source["id"], source_slot, initializer["id"], wait_slot, "CONDITIONING"] in workflow["links"]


def _assert_lora_link(workflow, filename):
    lora = _node(workflow, "RayLoraLoader")
    unet = _node(workflow, "RayUNETLoader")
    lora_slot = next(i for i, output in enumerate(lora["outputs"]) if output["name"] == "ray_lora")
    unet_slot = next(i for i, input_ in enumerate(unet["inputs"]) if input_["name"] == "lora")
    link_id = unet["inputs"][unet_slot]["link"]

    assert lora["widgets_values"] == [filename, 1.0]
    assert link_id is not None
    assert link_id in lora["outputs"][lora_slot]["links"]
    assert [link_id, lora["id"], lora_slot, unet["id"], unet_slot, "RAY_LORA"] in workflow["links"]


def _workflow_bytes(path):
    return path.read_bytes()



def test_i2v_builder_enables_the_supported_windows_v100_hybrid_topology():
    builder = _load_builder()
    source_path = WORKFLOW_ROOT / "Minimax_H3_I2V_Raylight.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))

    built = builder.build_workflow(
        source_path,
        mode="i2v",
        profile="full",
    )

    assert _node(built, "RayInitializer")["widgets_values"] == [
        "local",
        "default",
        2,
        2,
        1,
        1,
        1,
        True,
        False,
        True,
        True,
        "TORCH_EFFICIENT",
        True,
        True,
    ]
    assert _node(built, "RayUNETLoader")["widgets_values"][0] == (
        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    )
    assert _node(built, "LoadImage")["widgets_values"][0] == "minimax_h3_i2v_spear_portals.jpg"
    assert _node(built, "SaveVideo")["widgets_values"][0] == (
        "video/MiniMax_H3_I2V_Windows_V100_FSDP_FP8"
    )
    assert _node(built, "ResolutionSelector")["widgets_values"] == ["1:1 (Square)", 0.4, 32]
    assert _node(built, "PrimitiveFloat")["widgets_values"] == [2]
    assert _node(built, "RayBasicScheduler")["widgets_values"] == ["simple", 20, 1]
    assert _node(built, "MiniMaxH3ImageToVideo")["widgets_values"][0] == (
        _node(source, "MiniMaxH3ImageToVideo")["widgets_values"][0]
    )
    _assert_conditioning_link(built, "MiniMaxH3ImageToVideo")
    _assert_initializer_wait_link(built, "MiniMaxH3ImageToVideo")
    assert len(built["links"]) == len(source["links"]) + 2
    assert json.loads(source_path.read_text(encoding="utf-8")) == source


def test_ref2va_builder_reuses_the_validation_image_without_changing_the_prompt():
    builder = _load_builder()
    source_path = WORKFLOW_ROOT / "Minimax_H3_REF2VA_Raylight.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))

    built = builder.build_workflow(
        source_path,
        mode="ref2va",
        profile="full",
    )

    assert _node(built, "RayInitializer")["widgets_values"] == [
        "local",
        "default",
        2,
        2,
        1,
        1,
        1,
        True,
        False,
        True,
        True,
        "TORCH_EFFICIENT",
        True,
        True,
    ]
    assert _node(built, "RayUNETLoader")["widgets_values"][0] == (
        "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
    )
    assert [node["widgets_values"][0] for node in _nodes(built, "LoadImage")] == [
        "minimax_h3_ref2va_green_robots.jpg",
        "minimax_h3_ref2va_green_robots.jpg",
    ]
    assert _node(built, "SaveVideo")["widgets_values"][0] == (
        "video/MiniMax_H3_REF2VA_Windows_V100_FSDP_FP8"
    )
    assert _node(built, "ResolutionSelector")["widgets_values"] == ["16:9 (Widescreen)", 0.4, 32]
    assert _node(built, "PrimitiveFloat")["widgets_values"] == [5]
    assert _node(built, "RayBasicScheduler")["widgets_values"] == ["simple", 20, 1]
    assert _node(built, "PrimitiveStringMultiline")["widgets_values"][0] == (
        _node(source, "PrimitiveStringMultiline")["widgets_values"][0]
    )
    _assert_conditioning_link(built, "MiniMaxH3ReferenceToVideo")
    _assert_initializer_wait_link(built, "MiniMaxH3ReferenceToVideo")
    assert len(built["links"]) == len(source["links"]) + 1
    assert json.loads(source_path.read_text(encoding="utf-8")) == source


def test_smoke_profile_keeps_fsdp_cpu_offload_disabled():
    builder = _load_builder()
    built = builder.build_workflow(
        WORKFLOW_ROOT / "Minimax_H3_I2V_Raylight.json",
        mode="i2v",
        profile="smoke",
    )

    initializer_widgets = _node(built, "RayInitializer")["widgets_values"]
    assert initializer_widgets[9] is True


def test_i2v_turbo_workflow_pins_official_lora_steps_and_output_without_mutating_base():
    builder = _load_builder()
    source_path = WORKFLOW_ROOT / "Minimax_H3_I2V_Raylight.json"
    base_path = WORKFLOW_ROOT / "Minimax_H3_I2V_Windows_V100_FSDP.json"
    source_before = _workflow_bytes(source_path)
    base_before = _workflow_bytes(base_path)

    built = builder.build_workflow(
        source_path,
        mode="i2v",
        profile="full",
        turbo_variant="fl2v-turbo-8step",
    )

    _assert_lora_link(
        built,
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    )
    assert _node(built, "RayBasicScheduler")["widgets_values"] == ["simple", 8, 1]
    assert _node(built, "SaveVideo")["widgets_values"][0] == (
        "video/MiniMax_H3_I2V_Windows_V100_FSDP_FP8_Turbo8"
    )
    assert _workflow_bytes(source_path) == source_before
    assert _workflow_bytes(base_path) == base_before


def test_ref2va_turbo_workflow_pins_official_lora_steps_and_output_without_mutating_base():
    builder = _load_builder()
    source_path = WORKFLOW_ROOT / "Minimax_H3_REF2VA_Raylight.json"
    base_path = WORKFLOW_ROOT / "Minimax_H3_REF2VA_Windows_V100_FSDP.json"
    source_before = _workflow_bytes(source_path)
    base_before = _workflow_bytes(base_path)

    built = builder.build_workflow(
        source_path,
        mode="ref2va",
        profile="full",
        turbo_variant="ref2v-turbo-4step",
    )

    _assert_lora_link(
        built,
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
    )
    assert _node(built, "RayBasicScheduler")["widgets_values"] == ["simple", 4, 1]
    assert _node(built, "SaveVideo")["widgets_values"][0] == (
        "video/MiniMax_H3_REF2VA_Windows_V100_FSDP_FP8_Turbo4"
    )
    assert _workflow_bytes(source_path) == source_before
    assert _workflow_bytes(base_path) == base_before


def test_turbo_workflow_rejects_a_variant_for_the_wrong_mode():
    builder = _load_builder()

    try:
        builder.build_workflow(
            WORKFLOW_ROOT / "Minimax_H3_I2V_Raylight.json",
            mode="i2v",
            profile="full",
            turbo_variant="ref2v-turbo-4step",
        )
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError("REF2VA Turbo variant was accepted for I2V")


def test_i2v_four_step_variant_uses_a_four_step_output_label():
    builder = _load_builder()

    built = builder.build_workflow(
        WORKFLOW_ROOT / "Minimax_H3_I2V_Raylight.json",
        mode="i2v",
        profile="full",
        turbo_variant="fl2v-turbo-4step",
    )

    assert _node(built, "RayBasicScheduler")["widgets_values"] == ["simple", 4, 1]
    assert _node(built, "SaveVideo")["widgets_values"][0].endswith("_Turbo4")


def test_cli_rejects_turbo_smoke_before_writing_workflows(monkeypatch, tmp_path):
    builder = _load_builder()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(BUILDER_PATH),
            "--mode",
            "all",
            "--profile",
            "smoke",
            "--turbo",
            "--output-dir",
            str(tmp_path),
        ],
    )

    with pytest.raises(ValueError, match="profile full"):
        builder.main()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("mode", "variant"),
    [("i2v", "fl2v-turbo-8step"), ("ref2va", "ref2v-turbo-4step")],
)
def test_safe_fp16_workflow_changes_only_the_loader_mode_and_output_label(mode, variant):
    builder = _load_builder()
    default = builder.build_workflow(
        builder.SOURCE_WORKFLOWS[mode],
        mode=mode,
        profile="full",
        turbo_variant=variant,
    )
    safe = builder.build_workflow(
        builder.SOURCE_WORKFLOWS[mode],
        mode=mode,
        profile="full",
        turbo_variant=variant,
        compute_dtype="fp16_h3_safe",
    )

    assert _node(default, "RayUNETLoader")["widgets_values"][1] == "default"
    assert _node(safe, "RayUNETLoader")["widgets_values"][1] == "fp16_h3_safe"
    assert _node(safe, "SaveVideo")["widgets_values"][0].endswith("_SafeFP16")

    _node(safe, "RayUNETLoader")["widgets_values"][1] = "default"
    _node(safe, "SaveVideo")["widgets_values"][0] = _node(default, "SaveVideo")["widgets_values"][0]
    assert safe == default


def test_safe_fp16_workflow_rejects_smoke_profile():
    builder = _load_builder()
    with pytest.raises(ValueError, match="full"):
        builder.build_workflow(
            builder.SOURCE_WORKFLOWS["i2v"],
            mode="i2v",
            profile="smoke",
            compute_dtype="fp16_h3_safe",
        )


def test_safe_fp16_workflow_requires_a_pinned_turbo_variant():
    builder = _load_builder()
    with pytest.raises(ValueError, match="Turbo"):
        builder.build_workflow(
            builder.SOURCE_WORKFLOWS["i2v"],
            mode="i2v",
            profile="full",
            compute_dtype="fp16_h3_safe",
        )


@pytest.mark.parametrize(
    ("mode", "variant", "filename"),
    [
        ("i2v", "fl2v-turbo-8step", "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json"),
        ("ref2va", "ref2v-turbo-4step", "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json"),
    ],
)
def test_committed_turbo_workflow_is_the_exact_generated_artifact(tmp_path, mode, variant, filename):
    builder = _load_builder()
    expected = builder.build_workflow(
        builder.SOURCE_WORKFLOWS[mode],
        mode=mode,
        profile="full",
        turbo_variant=variant,
    )
    generated = builder.write_workflow(expected, tmp_path / filename)

    assert generated.read_bytes() == (WORKFLOW_ROOT / filename).read_bytes()


@pytest.mark.parametrize(
    ("mode", "variant", "filename"),
    [
        (
            "i2v",
            "fl2v-turbo-8step",
            "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json",
        ),
        (
            "ref2va",
            "ref2v-turbo-4step",
            "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json",
        ),
    ],
)
def test_committed_safe_fp16_workflow_is_exact_generated_artifact(tmp_path, mode, variant, filename):
    builder = _load_builder()
    expected = builder.build_workflow(
        builder.SOURCE_WORKFLOWS[mode],
        mode=mode,
        profile="full",
        turbo_variant=variant,
        compute_dtype="fp16_h3_safe",
    )
    generated = builder.write_workflow(expected, tmp_path / filename)

    assert generated.read_bytes() == (WORKFLOW_ROOT / filename).read_bytes()
