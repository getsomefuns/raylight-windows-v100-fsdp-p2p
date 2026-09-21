"""Build the O14 Director SelfLift hardware-validation workflow.

The input is the accepted O13 workflow. The output keeps every model, seed,
prompt, frame, SOL, and Director Refine setting unchanged, then inserts the
official Director SelfLift pack before Refine. This makes the O13 and O14
hardware measurements directly comparable.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


OUTPUT_PREFIX = (
    "video/raylight_o14/"
    "MMH3Director_O14_REF2VA_Turbo4_Sol_SelfLift_Refine2"
)
LIFT_MODEL = "minimax_h3_latent_upscaler_3d_fp16.safetensors"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _node(workflow: dict[str, Any], node_type: str) -> dict[str, Any]:
    return next(node for node in workflow["nodes"] if node["type"] == node_type)


def _widget_input(name: str, input_type: str) -> dict[str, Any]:
    return {
        "localized_name": name,
        "name": name,
        "type": input_type,
        "widget": {"name": name},
        "link": None,
    }


def _selflift_node(
    node_id: int, link_id: int, position: list[float]
) -> dict[str, Any]:
    widget_inputs = [
        ("bd_grp_selflift_sample", "BDGROUP"),
        ("split_mode", "COMBO"),
        ("highres_steps", "INT"),
        ("transition_step", "INT"),
        ("lowres_scale", "FLOAT"),
        ("sampler_mode", "COMBO"),
        ("native_low_carry", "BOOLEAN"),
        ("bd_grp_selflift_lift", "BDGROUP"),
        ("latent_upscale_model", "COMBO"),
        ("latent_upsample", "COMBO"),
        ("rho", "FLOAT"),
        ("w_min", "FLOAT"),
        ("w_max", "FLOAT"),
        ("enable_latent_chunking", "BOOLEAN"),
        ("bd_grp_selflift_tile", "BDGROUP"),
        ("enable_tiling", "BOOLEAN"),
        ("tile_count", "INT"),
        ("tile_overlap", "INT"),
    ]
    inputs = [_widget_input(name, input_type) for name, input_type in widget_inputs]
    inputs.append(
        {
            "localized_name": "model_hires",
            "name": "model_hires",
            "type": "MODEL",
            "shape": 7,
            "link": None,
        }
    )
    return {
        "id": node_id,
        "type": "MiniMaxH3DirectorSelfLift",
        "pos": position,
        "size": [420, 600],
        "flags": {},
        "order": 19,
        "mode": 0,
        "inputs": inputs,
        "outputs": [
            {
                "localized_name": "selflift",
                "name": "selflift",
                "type": "MMX_DIR_SELFLIFT",
                "links": [link_id],
            }
        ],
        "properties": {
            "Node name for S&R": "MiniMaxH3DirectorSelfLift",
        },
        "widgets_values": [
            "Progressive sample",
            "highres_steps",
            2,
            6,
            0.5,
            "euler",
            True,
            "Lift / 3D",
            LIFT_MODEL,
            "bilinear",
            0.0,
            0.25,
            0.75,
            False,
            "High-res tiles",
            False,
            2,
            128,
        ],
        "color": "#243447",
        "bgcolor": "#35536b",
    }


def _input(node: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in node["inputs"] if item["name"] == name)


def _remove_sol(workflow: dict[str, Any]) -> None:
    sol = _node(workflow, "SolAttnV100")
    adapter = _node(workflow, "RayMMH3DirectorModelAdapter")
    director = _node(workflow, "MiniMaxH3Director")
    scheduler = _node(workflow, "BasicScheduler")

    workflow["nodes"] = [
        node for node in workflow["nodes"] if node["id"] != sol["id"]
    ]
    workflow["links"] = [
        link
        for link in workflow["links"]
        if not (
            link[5] == "MODEL"
            and (link[1] == sol["id"] or link[3] == sol["id"])
        )
    ]

    first_link_id = max(link[0] for link in workflow["links"]) + 1
    director_link_id = first_link_id
    scheduler_link_id = first_link_id + 1
    workflow["links"].extend(
        [
            [
                director_link_id,
                adapter["id"],
                0,
                director["id"],
                0,
                "MODEL",
            ],
            [
                scheduler_link_id,
                adapter["id"],
                0,
                scheduler["id"],
                0,
                "MODEL",
            ],
        ]
    )
    adapter["outputs"][0]["links"] = [director_link_id, scheduler_link_id]
    _input(director, "model")["link"] = director_link_id
    _input(scheduler, "model")["link"] = scheduler_link_id


def build_workflow(
    base: dict[str, Any],
    *,
    include_sol: bool = True,
    include_refine: bool = True,
    compute_dtype: str | None = None,
    selflift_scale: float = 0.5,
) -> dict[str, Any]:
    workflow = copy.deepcopy(base)
    director = _node(workflow, "MiniMaxH3Director")
    save_video = _node(workflow, "SaveVideo")

    if any(
        node["type"] == "MiniMaxH3DirectorSelfLift"
        for node in workflow["nodes"]
    ):
        raise ValueError("base workflow already contains SelfLift")

    node_id = max(node["id"] for node in workflow["nodes"]) + 1
    link_id = max(link[0] for link in workflow["links"]) + 1
    director_pos = director.get("pos", [0.0, 0.0])
    selflift = _selflift_node(
        node_id,
        link_id,
        [director_pos[0], director_pos[1] + 1080],
    )
    if not 0.1 <= float(selflift_scale) <= 1.0:
        raise ValueError("SelfLift scale must be between 0.1 and 1.0")
    selflift["widgets_values"][4] = float(selflift_scale)

    refine_input = next(
        item for item in director["inputs"] if item["name"] == "refine"
    )
    refine_index = director["inputs"].index(refine_input)
    director["inputs"].insert(
        refine_index,
        {
            "localized_name": "selflift",
            "name": "selflift",
            "type": "MMX_DIR_SELFLIFT",
            "shape": 7,
            "link": link_id,
        },
    )

    refine_link_id = refine_input.get("link")
    if refine_link_id is not None:
        refine_link = next(
            link for link in workflow["links"] if link[0] == refine_link_id
        )
        refine_link[4] = 8

    workflow["nodes"].append(selflift)
    workflow["links"].append(
        [
            link_id,
            node_id,
            0,
            director["id"],
            7,
            "MMX_DIR_SELFLIFT",
        ]
    )
    output_prefix = OUTPUT_PREFIX if include_sol else OUTPUT_PREFIX + "_NoSol"
    if not include_sol:
        _remove_sol(workflow)
    if not include_refine:
        refine_link_id = refine_input.get("link")
        refine_node_ids = {
            link[1]
            for link in workflow["links"]
            if refine_link_id is not None and link[0] == refine_link_id
        }
        workflow["nodes"] = [
            node
            for node in workflow["nodes"]
            if node["id"] not in refine_node_ids
        ]
        workflow["links"] = [
            link
            for link in workflow["links"]
            if link[0] != refine_link_id
            and link[1] not in refine_node_ids
            and link[3] not in refine_node_ids
        ]
        refine_input["link"] = None
        output_prefix += "_NoRefine"
    if compute_dtype is not None:
        if compute_dtype not in {"default", "fp16_h3_safe"}:
            raise ValueError(f"unsupported RayUNETLoader compute dtype: {compute_dtype}")
        ray_unet = _node(workflow, "RayUNETLoader")
        ray_unet["widgets_values"][1] = compute_dtype
        if compute_dtype == "default":
            output_prefix += "_DefaultDType"
    if float(selflift_scale) == 1.0:
        output_prefix += "_Scale1Control"
    save_video["widgets_values"][0] = output_prefix
    workflow["last_node_id"] = node_id
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    return workflow


def build_single_gpu_control(
    base: dict[str, Any],
    standard_template: dict[str, Any],
    lora_template: dict[str, Any],
) -> dict[str, Any]:
    """Replace Ray nodes with ordinary ComfyUI loaders for attribution testing."""

    workflow = copy.deepcopy(base)
    ray_unet = _node(workflow, "RayUNETLoader")
    ray_clip = _node(workflow, "RayMiniMaxH3DualGPUCLIPLoader")
    ray_lora = _node(workflow, "RayLoraLoader")
    initializer = _node(workflow, "RayInitializer")
    adapter = _node(workflow, "RayMMH3DirectorModelAdapter")
    director = _node(workflow, "MiniMaxH3Director")
    scheduler = _node(workflow, "BasicScheduler")
    save_video = _node(workflow, "SaveVideo")

    unet = copy.deepcopy(_node(standard_template, "UNETLoader"))
    unet["id"] = ray_unet["id"]
    unet["pos"] = ray_unet["pos"]
    unet["widgets_values"] = [ray_unet["widgets_values"][0], "default"]

    clip = copy.deepcopy(_node(standard_template, "CLIPLoader"))
    clip["id"] = ray_clip["id"]
    clip["pos"] = ray_clip["pos"]
    clip["widgets_values"] = [
        ray_clip["widgets_values"][0],
        "minimax",
        "default",
    ]
    clip["outputs"][0]["links"] = [
        link[0]
        for link in workflow["links"]
        if link[1] == ray_clip["id"] and link[5] == "CLIP"
    ]

    lora = copy.deepcopy(_node(lora_template, "LoraLoaderModelOnly"))
    lora["id"] = ray_lora["id"]
    lora["pos"] = ray_lora["pos"]
    lora["widgets_values"] = list(ray_lora["widgets_values"])
    lora["widgets_values_named"] = {
        "lora_name": ray_lora["widgets_values"][0],
        "strength_model": ray_lora["widgets_values"][1],
    }

    removed_ids = {initializer["id"], adapter["id"]}
    replacements = {
        ray_unet["id"]: unet,
        ray_clip["id"]: clip,
        ray_lora["id"]: lora,
    }
    workflow["nodes"] = [
        replacements.get(node["id"], node)
        for node in workflow["nodes"]
        if node["id"] not in removed_ids
    ]
    rewired_ids = removed_ids | {ray_unet["id"], ray_lora["id"]}
    workflow["links"] = [
        link
        for link in workflow["links"]
        if link[1] not in rewired_ids and link[3] not in rewired_ids
    ]

    first_link_id = max(link[0] for link in workflow["links"]) + 1
    unet_lora = first_link_id
    lora_director = first_link_id + 1
    lora_scheduler = first_link_id + 2
    workflow["links"].extend(
        [
            [unet_lora, unet["id"], 0, lora["id"], 0, "MODEL"],
            [
                lora_director,
                lora["id"],
                0,
                director["id"],
                0,
                "MODEL",
            ],
            [
                lora_scheduler,
                lora["id"],
                0,
                scheduler["id"],
                0,
                "MODEL",
            ],
        ]
    )
    unet["outputs"][0]["links"] = [unet_lora]
    _input(lora, "model")["link"] = unet_lora
    lora["outputs"][0]["links"] = [lora_director, lora_scheduler]
    _input(director, "model")["link"] = lora_director
    _input(scheduler, "model")["link"] = lora_scheduler
    save_video["widgets_values"][0] = OUTPUT_PREFIX + "_SingleGPU_Control"
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--without-sol", action="store_true")
    parser.add_argument("--without-refine", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=("default", "fp16_h3_safe"),
    )
    parser.add_argument("--selflift-scale", type=float, default=0.5)
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--standard-template", type=Path)
    parser.add_argument("--lora-template", type=Path)
    args = parser.parse_args()

    workflow = build_workflow(
        _load(args.base),
        include_sol=not args.without_sol,
        include_refine=not args.without_refine,
        compute_dtype=args.compute_dtype,
        selflift_scale=args.selflift_scale,
    )
    if args.single_gpu:
        if args.standard_template is None or args.lora_template is None:
            parser.error(
                "--single-gpu requires --standard-template and --lora-template"
            )
        workflow = build_single_gpu_control(
            workflow,
            _load(args.standard_template),
            _load(args.lora_template),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
