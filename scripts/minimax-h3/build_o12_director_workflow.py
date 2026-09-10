"""Build the O12 MiniMax H3 Director acceleration validation workflow.

The builder converts an existing single-GPU Director workflow to Raylight and
inserts the two optional third-party MODEL patches in this order:

    Ray adapter -> TE-Speed -> Sol-Attn -> MiniMax H3 Director

It deliberately reads node shapes from a known-good Raylight workflow instead
of duplicating Raylight's frontend schema in this script.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _node(workflow: dict[str, Any], node_type: str) -> dict[str, Any]:
    return next(node for node in workflow["nodes"] if node["type"] == node_type)


def _set_id_and_position(
    node: dict[str, Any], node_id: int, position: list[float]
) -> dict[str, Any]:
    node = copy.deepcopy(node)
    node["id"] = node_id
    node["pos"] = position
    return node


def _input(node: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in node["inputs"] if item["name"] == name)


def _sol_node(node_id: int, position: list[float]) -> dict[str, Any]:
    widget_inputs: list[tuple[str, str]] = [
        ("fp16_safe", "BOOLEAN"),
        ("tau", "FLOAT"),
        ("start_percent", "FLOAT"),
        ("end_percent", "FLOAT"),
        ("min_tokens", "INT"),
        ("dense_blocks", "STRING"),
        ("h3_prefix_tokens", "INT"),
        ("topk_blocks", "INT"),
        ("debug_nan", "BOOLEAN"),
        ("profile", "BOOLEAN"),
    ]
    inputs: list[dict[str, Any]] = [
        {
            "localized_name": "model",
            "name": "model",
            "type": "MODEL",
            "link": 19,
        }
    ]
    inputs.extend(
        {
            "localized_name": name,
            "name": name,
            "type": input_type,
            "widget": {"name": name},
            "link": None,
        }
        for name, input_type in widget_inputs
    )
    return {
        "id": node_id,
        "type": "SolAttnV100",
        "pos": position,
        "size": [330, 334],
        "flags": {},
        "order": 7,
        "mode": 0,
        "inputs": inputs,
        "outputs": [
            {
                "localized_name": "model",
                "name": "model",
                "type": "MODEL",
                "links": [20],
            }
        ],
        "properties": {"Node name for S&R": "SolAttnV100"},
        "widgets_values": [
            True,
            0.75,
            0.2,
            0.9,
            1024,
            "0-1,-1",
            1024,
            32,
            False,
            False,
        ],
        "color": "#233",
        "bgcolor": "#355",
    }


def build_workflow(
    base: dict[str, Any],
    ray_template: dict[str, Any],
    te_template: dict[str, Any],
) -> dict[str, Any]:
    workflow = copy.deepcopy(base)
    original_unet = _node(workflow, "UNETLoader")
    original_clip = _node(workflow, "CLIPLoader")
    original_lora = _node(workflow, "LoraLoaderModelOnly")
    director = _node(workflow, "MiniMaxH3Director")
    save_video = _node(workflow, "SaveVideo")

    unet_name = original_unet["widgets_values"][0]
    if unet_name == "minimax_h3\\minimax_h3_ref2va_pruned_fp8_scaled.safetensors":
        unet_name = "minimax_h3\\ref2va\\minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
    lora_name = original_lora["widgets_values"][0]
    if lora_name == "MMH3\\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors":
        lora_name = (
            "MMH3\\Ref2va\\lightx2v\\"
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
        )

    ray_unet = _set_id_and_position(
        _node(ray_template, "RayUNETLoader"),
        original_unet["id"],
        original_unet["pos"],
    )
    ray_unet["widgets_values"] = [unet_name, "fp16_h3_safe"]
    _input(ray_unet, "ray_actors_init")["link"] = 16
    _input(ray_unet, "lora")["link"] = 15
    ray_unet["outputs"][0]["links"] = [17]

    ray_clip = _set_id_and_position(
        _node(ray_template, "RayMiniMaxH3DualGPUCLIPLoader"),
        original_clip["id"],
        original_clip["pos"],
    )
    ray_clip["widgets_values"] = [original_clip["widgets_values"][0], "tp2", "AUTO"]
    _input(ray_clip, "ray_actors_init")["link"] = 14
    ray_clip["outputs"][0]["links"] = [2]
    ray_clip["outputs"][1]["links"] = [16]

    ray_lora = _set_id_and_position(
        _node(ray_template, "RayLoraLoader"),
        original_lora["id"],
        original_lora["pos"],
    )
    ray_lora["widgets_values"] = [lora_name, original_lora["widgets_values"][1]]
    ray_lora["outputs"][0]["links"] = [15]

    initializer = _set_id_and_position(
        _node(ray_template, "RayInitializer"), 13, [-1740.0, 100.0]
    )
    initializer["widgets_values"] = [
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
        "XFORMERS_MEMORY_EFFICIENT",
        True,
        True,
    ]
    initializer["outputs"][0]["links"] = [14]

    adapter = _set_id_and_position(
        _node(ray_template, "RayMMH3DirectorModelAdapter"), 14, [-650.0, 100.0]
    )
    _input(adapter, "ray_actors")["link"] = 17
    adapter["outputs"][0]["links"] = [18]

    te_speed = _set_id_and_position(
        _node(te_template, "TESpeedMiniMaxH3"), 15, [-300.0, 80.0]
    )
    _input(te_speed, "model")["link"] = 18
    te_speed["outputs"][0]["links"] = [19]
    te_speed["widgets_values"] = [0.08, 0.1, 0.9, 2, "auto", "4-step LoRA"]

    sol = _sol_node(16, [80.0, 80.0])
    _input(director, "model")["link"] = 20
    save_video["widgets_values"][0] = "video/raylight_o12/MMH3Director_REF2VA_Turbo4_TE_Sol"

    replacements = {
        original_unet["id"]: ray_unet,
        original_clip["id"]: ray_clip,
        original_lora["id"]: ray_lora,
    }
    workflow["nodes"] = [replacements.get(node["id"], node) for node in workflow["nodes"]]
    workflow["nodes"].extend([initializer, adapter, te_speed, sol])

    workflow["links"] = [link for link in workflow["links"] if link[0] not in {12, 13}]
    workflow["links"].extend(
        [
            [14, 13, 0, 2, 0, "RAY_ACTORS_INIT"],
            [15, 12, 0, 1, 1, "RAY_LORA"],
            [16, 2, 1, 1, 0, "RAY_ACTORS_INIT"],
            [17, 1, 0, 14, 0, "RAY_ACTORS"],
            [18, 14, 0, 15, 0, "MODEL"],
            [19, 15, 0, 16, 0, "MODEL"],
            [20, 16, 0, director["id"], 0, "MODEL"],
        ]
    )
    workflow["last_node_id"] = 16
    workflow["last_link_id"] = 20
    return workflow


def derive_variant(
    combined: dict[str, Any], *, variant: str, turbo_steps: int
) -> dict[str, Any]:
    """Derive a directly comparable accelerator/step workflow from O12 combined."""
    if variant not in {"none", "te", "sol", "combined"}:
        raise ValueError("variant must be one of: none, te, sol, combined")
    if turbo_steps not in {4, 8}:
        raise ValueError("turbo_steps must be 4 or 8")

    workflow = copy.deepcopy(combined)
    by_type = {node["type"]: node for node in workflow["nodes"]}
    director = by_type["MiniMaxH3Director"]
    adapter = by_type["RayMMH3DirectorModelAdapter"]
    te_speed = by_type["TESpeedMiniMaxH3"]
    sol = by_type["SolAttnV100"]
    lora = by_type["RayLoraLoader"]
    save_video = by_type["SaveVideo"]

    include_te = variant in {"te", "combined"}
    include_sol = variant in {"sol", "combined"}
    removed_ids = set()
    if not include_te:
        removed_ids.add(te_speed["id"])
    if not include_sol:
        removed_ids.add(sol["id"])
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in removed_ids]

    route_ids = {adapter["id"], te_speed["id"], sol["id"]}
    workflow["links"] = [
        link
        for link in workflow["links"]
        if not (link[5] == "MODEL" and link[1] in route_ids)
    ]

    route_nodes = [adapter]
    if include_te:
        route_nodes.append(te_speed)
    if include_sol:
        route_nodes.append(sol)
    route_nodes.append(director)

    first_link_id = max([link[0] for link in workflow["links"]], default=0) + 1
    new_links = []
    for offset, (source, target) in enumerate(zip(route_nodes, route_nodes[1:])):
        link_id = first_link_id + offset
        new_links.append([link_id, source["id"], 0, target["id"], 0, "MODEL"])
        source["outputs"][0]["links"] = [link_id]
        _input(target, "model")["link"] = link_id
    workflow["links"].extend(new_links)

    director["widgets_values"][13] = turbo_steps
    if turbo_steps == 8:
        lora["widgets_values"][0] = (
            "MMH3\\Ref2va\\lightx2v\\"
            "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
        )
        te_speed["widgets_values"][5] = "8-step LoRA"
    else:
        te_speed["widgets_values"][5] = "4-step LoRA"

    suffix = {"none": "Base", "te": "TE", "sol": "Sol", "combined": "TE_Sol"}[variant]
    save_video["widgets_values"][0] = (
        f"video/raylight_o12/MMH3Director_REF2VA_Turbo{turbo_steps}_{suffix}"
    )
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--ray-template", required=True, type=Path)
    parser.add_argument("--te-template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--variant", choices=("none", "te", "sol", "combined"), default="combined"
    )
    parser.add_argument("--turbo-steps", choices=(4, 8), type=int, default=4)
    args = parser.parse_args()

    workflow = derive_variant(
        build_workflow(
            _load(args.base), _load(args.ray_template), _load(args.te_template)
        ),
        variant=args.variant,
        turbo_steps=args.turbo_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
