from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "build_o12_director_workflow.py"
SPEC = importlib.util.spec_from_file_location("o12_director_workflow", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def _combined_workflow():
    return {
        "nodes": [
            {"id": 5, "type": "MiniMaxH3Director", "inputs": [{"name": "model", "link": 20}], "widgets_values": [None] * 21},
            {"id": 7, "type": "SaveVideo", "widgets_values": ["old", "auto", "auto"]},
            {"id": 12, "type": "RayLoraLoader", "widgets_values": ["turbo4.safetensors", 1]},
            {"id": 14, "type": "RayMMH3DirectorModelAdapter", "outputs": [{"links": [18]}]},
            {"id": 15, "type": "TESpeedMiniMaxH3", "inputs": [{"name": "model", "link": 18}], "outputs": [{"links": [19]}], "widgets_values": [0.08, 0.1, 0.9, 2, "auto", "4-step LoRA"]},
            {"id": 16, "type": "SolAttnV100", "inputs": [{"name": "model", "link": 19}], "outputs": [{"links": [20]}]},
        ],
        "links": [
            [18, 14, 0, 15, 0, "MODEL"],
            [19, 15, 0, 16, 0, "MODEL"],
            [20, 16, 0, 5, 0, "MODEL"],
        ],
        "last_node_id": 16,
        "last_link_id": 20,
    }


@pytest.mark.parametrize(
    ("variant", "types", "route"),
    [
        ("none", {"MiniMaxH3Director", "SaveVideo", "RayLoraLoader", "RayMMH3DirectorModelAdapter"}, [(14, 5)]),
        ("te", {"MiniMaxH3Director", "SaveVideo", "RayLoraLoader", "RayMMH3DirectorModelAdapter", "TESpeedMiniMaxH3"}, [(14, 15), (15, 5)]),
        ("sol", {"MiniMaxH3Director", "SaveVideo", "RayLoraLoader", "RayMMH3DirectorModelAdapter", "SolAttnV100"}, [(14, 16), (16, 5)]),
        ("combined", {"MiniMaxH3Director", "SaveVideo", "RayLoraLoader", "RayMMH3DirectorModelAdapter", "TESpeedMiniMaxH3", "SolAttnV100"}, [(14, 15), (15, 16), (16, 5)]),
    ],
)
def test_derive_variant_builds_exact_model_route(variant, types, route):
    workflow = builder.derive_variant(_combined_workflow(), variant=variant, turbo_steps=4)

    assert {node["type"] for node in workflow["nodes"]} == types
    assert [(link[1], link[3]) for link in workflow["links"]] == route


def test_derive_variant_switches_ref2va_to_turbo8():
    workflow = builder.derive_variant(_combined_workflow(), variant="combined", turbo_steps=8)
    by_type = {node["type"]: node for node in workflow["nodes"]}

    assert by_type["MiniMaxH3Director"]["widgets_values"][13] == 8
    assert by_type["TESpeedMiniMaxH3"]["widgets_values"][5] == "8-step LoRA"
    assert by_type["RayLoraLoader"]["widgets_values"][0] == (
        "MMH3\\Ref2va\\lightx2v\\minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
    )


def test_derive_variant_rejects_unknown_variant_or_steps():
    with pytest.raises(ValueError, match="variant"):
        builder.derive_variant(_combined_workflow(), variant="mystery", turbo_steps=4)
    with pytest.raises(ValueError, match="turbo_steps"):
        builder.derive_variant(_combined_workflow(), variant="none", turbo_steps=6)


def test_all_committed_o12_workflows_are_valid_json():
    workflow_root = Path(__file__).parents[1] / "example_workflows" / "o12"

    parsed = [json.loads(path.read_text(encoding="utf-8")) for path in workflow_root.glob("*.json")]

    assert len(parsed) == 8
