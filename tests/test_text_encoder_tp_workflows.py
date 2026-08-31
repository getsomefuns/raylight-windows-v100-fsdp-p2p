from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPO_ROOT / "example_workflows"

CASES = {
    "Minimax_H3_I2V_DualGPU_Text_Encoder_TP2_Raylight.json": (
        "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json",
        "MiniMaxH3ImageToVideo",
    ),
    "Minimax_H3_REF2VA_DualGPU_Text_Encoder_TP2_Raylight.json": (
        "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json",
        "MiniMaxH3ReferenceToVideo",
    ),
    "Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json": (
        "MMH3Director_Windows_V100_FSDP.json",
        "MiniMaxH3Director",
    ),
}


def _load(name: str) -> dict:
    return json.loads((WORKFLOW_ROOT / name).read_text(encoding="utf-8"))


def _load_builder():
    path = REPO_ROOT / "scripts" / "minimax-h3" / "build_tp2_workflows.py"
    spec = importlib.util.spec_from_file_location("raylight_tp2_workflow_builder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _node(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    assert len(matches) == 1, f"expected one {node_type}, got {len(matches)}"
    return matches[0]


def _link(workflow: dict, link_id: int) -> list:
    matches = [link for link in workflow["links"] if link[0] == link_id]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("generated_name,case", CASES.items())
def test_tp2_workflow_has_one_explicit_loader_and_no_stock_cliploader(generated_name, case) -> None:
    workflow = _load(generated_name)
    loader = _node(workflow, "RayMiniMaxH3DualGPUCLIPLoader")

    assert not any(node["type"] == "CLIPLoader" for node in workflow["nodes"])
    assert loader["widgets_values"] == [
        "MMH3\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
        if "Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json" == generated_name
        else "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "tp2",
    ]
    assert loader["properties"]["Node name for S&R"] == "RayMiniMaxH3DualGPUCLIPLoader"


@pytest.mark.parametrize("generated_name,case", CASES.items())
def test_tp2_loader_serializes_actor_initialization_before_unet_and_clip_consumer(generated_name, case) -> None:
    _, clip_consumer_type = case
    workflow = _load(generated_name)
    initializer = _node(workflow, "RayInitializer")
    loader = _node(workflow, "RayMiniMaxH3DualGPUCLIPLoader")
    unet = _node(workflow, "RayUNETLoader")
    consumer = _node(workflow, clip_consumer_type)

    init_input = next(item for item in loader["inputs"] if item["name"] == "ray_actors_init")
    unet_input = next(item for item in unet["inputs"] if item["name"] == "ray_actors_init")
    clip_input = next(item for item in consumer["inputs"] if item["name"] == "clip")
    clip_slot = consumer["inputs"].index(clip_input)

    assert _link(workflow, init_input["link"])[1:5] == [initializer["id"], 0, loader["id"], 0]
    assert _link(workflow, unet_input["link"])[1:5] == [loader["id"], 1, unet["id"], 0]
    assert _link(workflow, clip_input["link"])[1:5] == [loader["id"], 0, consumer["id"], clip_slot]
    assert sum(1 for link in workflow["links"] if link[1] == loader["id"] and link[2] == 0) == 1


@pytest.mark.parametrize(
    "generated_name", [
        "Minimax_H3_I2V_DualGPU_Text_Encoder_TP2_Raylight.json",
        "Minimax_H3_REF2VA_DualGPU_Text_Encoder_TP2_Raylight.json",
    ]
)
def test_plain_workflows_remove_the_old_conditioning_to_initializer_cycle(generated_name) -> None:
    workflow = _load(generated_name)
    initializer = _node(workflow, "RayInitializer")
    wait_for = next(item for item in initializer["inputs"] if item["name"] == "wait_for")
    assert wait_for["link"] is None


def test_director_controls_and_sampling_path_match_the_validated_workflow() -> None:
    source = _load("MMH3Director_Windows_V100_FSDP.json")
    built = _load("Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json")
    source_director = _node(source, "MiniMaxH3Director")
    built_director = _node(built, "MiniMaxH3Director")

    assert built_director["widgets_values"] == source_director["widgets_values"]
    assert [item["name"] for item in built_director["inputs"]] == [
        item["name"] for item in source_director["inputs"]
    ]
    for node_type in (
        "RayLoraLoader",
        "RayUNETLoader",
        "RayMMH3DirectorModelAdapter",
        "CreateVideo",
        "SaveVideo",
    ):
        assert _node(built, node_type)["widgets_values"] == _node(source, node_type)["widgets_values"]


@pytest.mark.parametrize("generated_name,case", CASES.items())
def test_committed_tp2_workflows_are_reproducible_from_frozen_sources(generated_name, case) -> None:
    source_name, _ = case
    builder = _load_builder()
    source = _load(source_name)
    expected = builder.build_tp2_workflow(
        source,
        remove_initializer_wait=generated_name != "Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json",
    )

    assert _load(generated_name) == expected
