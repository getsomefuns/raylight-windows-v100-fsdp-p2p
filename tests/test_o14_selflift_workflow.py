from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "minimax-h3"
    / "build_o14_selflift_workflow.py"
)
SPEC = importlib.util.spec_from_file_location("o14_selflift_workflow", SCRIPT)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(builder)


def _base_workflow() -> dict:
    return {
        "nodes": [
            {
                "id": 5,
                "type": "MiniMaxH3Director",
                "inputs": [
                    {"name": "model", "link": 19},
                    {"name": "refine", "link": 24},
                ],
                "widgets_values": [None] * 21,
            },
            {
                "id": 7,
                "type": "SaveVideo",
                "widgets_values": ["old/path", "auto", "auto"],
            },
            {
                "id": 18,
                "type": "MiniMaxH3DirectorRefine",
                "outputs": [{"links": [24]}],
            },
        ],
        "links": [
            [19, 16, 0, 5, 0, "MODEL"],
            [24, 18, 0, 5, 6, "MMX_DIR_REFINE"],
        ],
        "last_node_id": 18,
        "last_link_id": 24,
    }


def test_build_adds_selflift_before_refine_and_uses_v100_fp16_lift():
    workflow = builder.build_workflow(_base_workflow())
    by_type = {node["type"]: node for node in workflow["nodes"]}
    director = by_type["MiniMaxH3Director"]
    selflift = by_type["MiniMaxH3DirectorSelfLift"]

    assert next(item for item in director["inputs"] if item["name"] == "selflift")[
        "link"
    ] == 25
    assert next(item for item in director["inputs"] if item["name"] == "refine")[
        "link"
    ] == 24
    assert next(link for link in workflow["links"] if link[0] == 25)[3:6] == [
        5,
        7,
        "MMX_DIR_SELFLIFT",
    ]
    assert next(link for link in workflow["links"] if link[0] == 24)[4] == 8
    assert selflift["widgets_values"][1:6] == [
        "highres_steps",
        2,
        6,
        0.5,
        "euler",
    ]
    assert (
        selflift["widgets_values"][8]
        == "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    )
    assert by_type["SaveVideo"]["widgets_values"][0].startswith(
        "video/raylight_o14/"
    )


def test_without_sol_routes_adapter_to_director_and_refine_scheduler():
    base = _base_workflow()
    base["nodes"].extend(
        [
            {
                "id": 14,
                "type": "RayMMH3DirectorModelAdapter",
                "outputs": [{"links": [18]}],
            },
            {
                "id": 16,
                "type": "SolAttnV100",
                "inputs": [{"name": "model", "link": 18}],
                "outputs": [{"links": [19, 21]}],
            },
            {
                "id": 17,
                "type": "BasicScheduler",
                "inputs": [{"name": "model", "link": 21}],
            },
        ]
    )
    base["links"].extend(
        [
            [18, 14, 0, 16, 0, "MODEL"],
            [21, 16, 0, 17, 0, "MODEL"],
        ]
    )

    workflow = builder.build_workflow(base, include_sol=False)
    by_type = {node["type"]: node for node in workflow["nodes"]}

    assert "SolAttnV100" not in by_type
    routes = {
        (link[1], link[3])
        for link in workflow["links"]
        if link[5] == "MODEL"
    }
    assert (14, 5) in routes
    assert (14, 17) in routes
    assert by_type["SaveVideo"]["widgets_values"][0].endswith("_NoSol")


def test_without_refine_disconnects_pack_and_keeps_first_pass_output():
    workflow = builder.build_workflow(
        _base_workflow(),
        include_refine=False,
    )
    by_type = {node["type"]: node for node in workflow["nodes"]}
    director = by_type["MiniMaxH3Director"]

    assert "MiniMaxH3DirectorRefine" not in by_type
    assert next(item for item in director["inputs"] if item["name"] == "refine")[
        "link"
    ] is None
    assert not any(link[0] == 24 for link in workflow["links"])
    assert by_type["SaveVideo"]["widgets_values"][0].endswith("_NoRefine")


def test_compute_dtype_override_changes_only_ray_loader():
    base = _base_workflow()
    base["nodes"].append(
        {
            "id": 3,
            "type": "RayUNETLoader",
            "widgets_values": ["model.safetensors", "fp16_h3_safe"],
        }
    )

    workflow = builder.build_workflow(base, compute_dtype="default")

    loader = next(node for node in workflow["nodes"] if node["type"] == "RayUNETLoader")
    assert loader["widgets_values"] == ["model.safetensors", "default"]
    assert next(
        node for node in workflow["nodes"] if node["type"] == "SaveVideo"
    )["widgets_values"][0].endswith("_DefaultDType")


def test_selflift_scale_override_builds_noop_control():
    workflow = builder.build_workflow(_base_workflow(), selflift_scale=1.0)

    selflift = next(
        node for node in workflow["nodes"] if node["type"] == "MiniMaxH3DirectorSelfLift"
    )
    save = next(node for node in workflow["nodes"] if node["type"] == "SaveVideo")
    assert selflift["widgets_values"][4] == 1.0
    assert save["widgets_values"][0].endswith("_Scale1Control")


def test_committed_o14_workflow_is_valid_and_has_one_selflift():
    path = (
        Path(__file__).parents[1]
        / "example_workflows"
        / "o14"
        / "MMH3Director_O14_REF2VA_Turbo4_Sol_SelfLift_Refine2.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))

    assert sum(
        node["type"] == "MiniMaxH3DirectorSelfLift"
        for node in workflow["nodes"]
    ) == 1


def test_committed_no_sol_control_is_valid_and_has_no_sol_node():
    path = (
        Path(__file__).parents[1]
        / "example_workflows"
        / "o14"
        / "MMH3Director_O14_REF2VA_Turbo4_SelfLift_Refine2_NoSol.json"
    )
    workflow = json.loads(path.read_text(encoding="utf-8"))

    assert not any(node["type"] == "SolAttnV100" for node in workflow["nodes"])
