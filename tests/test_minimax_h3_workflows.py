import importlib.util
import json
from pathlib import Path


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
