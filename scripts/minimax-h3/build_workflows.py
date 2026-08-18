from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO_ROOT / "example_workflows"
MODEL_MANIFEST = Path(__file__).with_name("models.json")

SOURCE_WORKFLOWS = {
    "i2v": WORKFLOW_ROOT / "Minimax_H3_I2V_Raylight.json",
    "ref2va": WORKFLOW_ROOT / "Minimax_H3_REF2VA_Raylight.json",
}
OUTPUT_WORKFLOWS = {
    "i2v": WORKFLOW_ROOT / "Minimax_H3_I2V_Windows_V100_FSDP.json",
    "ref2va": WORKFLOW_ROOT / "Minimax_H3_REF2VA_Windows_V100_FSDP.json",
}
TURBO_OUTPUT_WORKFLOWS = {
    "i2v": WORKFLOW_ROOT / "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json",
    "ref2va": WORKFLOW_ROOT / "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json",
}
SAFE_FP16_OUTPUT_WORKFLOWS = {
    "i2v": WORKFLOW_ROOT / "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json",
    "ref2va": WORKFLOW_ROOT / "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json",
}
DEFAULT_TURBO_VARIANTS = {
    "i2v": "fl2v-turbo-8step",
    "ref2va": "ref2v-turbo-4step",
}
DIFFUSION_MODELS = {
    "i2v": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "ref2va": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
}
OUTPUT_PREFIXES = {
    "i2v": "video/MiniMax_H3_I2V_Windows_V100_FSDP_FP8",
    "ref2va": "video/MiniMax_H3_REF2VA_Windows_V100_FSDP_FP8",
}
INPUT_FILENAMES = {
    "i2v": "minimax_h3_i2v_spear_portals.jpg",
    "ref2va": "minimax_h3_ref2va_green_robots.jpg",
}
RAY_INITIALIZER_WIDGETS = [
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
    False,
    "TORCH_EFFICIENT",
    True,
    True,
]
RAY_INITIALIZER_FSDP_CPU_OFFLOAD_INDEX = 10
SMOKE_DEFAULTS = {
    "i2v": {"megapixels": 0.2, "duration": 1.0, "steps": 12},
    "ref2va": {"megapixels": 0.2, "duration": 2.0, "steps": 12},
}


def _nodes(workflow: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    return [node for node in workflow.get("nodes", []) if node.get("type") == node_type]


def _one_node(workflow: dict[str, Any], node_type: str) -> dict[str, Any]:
    matches = _nodes(workflow, node_type)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {node_type} node, found {len(matches)}")
    return matches[0]


def _set_first_widget(node: dict[str, Any], value: Any) -> None:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or not widgets:
        raise ValueError(f"Node {node.get('type')} has no editable widgets_values")
    widgets[0] = value


def _set_unet_compute_dtype(node: dict[str, Any], value: str) -> None:
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) < 2:
        raise ValueError("RayUNETLoader has no editable weight_dtype widget")
    widgets[1] = value


def _ensure_conditioning_link(workflow: dict[str, Any], mode: str) -> None:
    source_type = "MiniMaxH3ImageToVideo" if mode == "i2v" else "MiniMaxH3ReferenceToVideo"
    source = _one_node(workflow, source_type)
    guider = _one_node(workflow, "RayBasicGuider")
    source_slot = next(i for i, output in enumerate(source["outputs"]) if output["name"] == "positive")
    target_slot = next(i for i, input_ in enumerate(guider["inputs"]) if input_["name"] == "conditioning")
    source_output = source["outputs"][source_slot]
    target_input = guider["inputs"][target_slot]
    current_link_id = target_input.get("link")

    if current_link_id is not None:
        expected = [current_link_id, source["id"], source_slot, guider["id"], target_slot, "CONDITIONING"]
        if expected not in workflow["links"]:
            raise ValueError("RayBasicGuider conditioning link does not come from the MiniMax positive output")
        return

    existing_link_ids = [link[0] for link in workflow.get("links", [])]
    link_id = max([int(workflow.get("last_link_id", 0)), *existing_link_ids], default=0) + 1
    workflow["last_link_id"] = link_id
    workflow.setdefault("links", []).append(
        [link_id, source["id"], source_slot, guider["id"], target_slot, "CONDITIONING"]
    )
    target_input["link"] = link_id
    source_output.setdefault("links", []).append(link_id)


def _ensure_initializer_wait_link(workflow: dict[str, Any], mode: str) -> None:
    source_type = "MiniMaxH3ImageToVideo" if mode == "i2v" else "MiniMaxH3ReferenceToVideo"
    source = _one_node(workflow, source_type)
    initializer = _one_node(workflow, "RayInitializer")
    source_slot = next(i for i, output in enumerate(source["outputs"]) if output["name"] == "positive")
    source_output = source["outputs"][source_slot]

    inputs = initializer.setdefault("inputs", [])
    wait_slots = [i for i, input_ in enumerate(inputs) if input_["name"] == "wait_for"]
    if len(wait_slots) > 1:
        raise ValueError("RayInitializer has more than one wait_for input")
    if wait_slots:
        target_slot = wait_slots[0]
        target_input = inputs[target_slot]
    else:
        target_slot = len(inputs)
        target_input = {"name": "wait_for", "type": "*", "link": None}
        inputs.append(target_input)

    current_link_id = target_input.get("link")
    if current_link_id is not None:
        expected = [current_link_id, source["id"], source_slot, initializer["id"], target_slot, "CONDITIONING"]
        if expected not in workflow["links"]:
            raise ValueError("RayInitializer wait_for link does not come from the MiniMax positive output")
        return

    existing_link_ids = [link[0] for link in workflow.get("links", [])]
    link_id = max([int(workflow.get("last_link_id", 0)), *existing_link_ids], default=0) + 1
    workflow["last_link_id"] = link_id
    workflow.setdefault("links", []).append(
        [link_id, source["id"], source_slot, initializer["id"], target_slot, "CONDITIONING"]
    )
    target_input["link"] = link_id
    source_output.setdefault("links", []).append(link_id)


def _resolve_turbo_variant(variant_id: str, mode: str) -> dict[str, Any]:
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    matches = [
        item
        for item in manifest.get("models", [])
        if item.get("id") == variant_id and "turbo" in item.get("groups", [])
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown Turbo variant: {variant_id}")
    item = matches[0]
    if item.get("mode") != mode:
        raise ValueError(f"Turbo variant {variant_id} mode is {item.get('mode')}, not {mode}")
    relative_path = Path(item["relative_path"])
    if relative_path.parent.as_posix() != "loras" or relative_path.name != relative_path.as_posix().split("/")[-1]:
        raise ValueError(f"Turbo variant {variant_id} has an invalid LoRA path")
    steps = int(item["steps"])
    if steps < 1:
        raise ValueError(f"Turbo variant {variant_id} has invalid steps")
    return {"lora_name": relative_path.name, "steps": steps}


def _ensure_lora_link(workflow: dict[str, Any], lora_name: str) -> None:
    unet = _one_node(workflow, "RayUNETLoader")
    lora_inputs = [i for i, input_ in enumerate(unet.get("inputs", [])) if input_["name"] == "lora"]
    if len(lora_inputs) != 1:
        raise ValueError(f"Expected exactly one RayUNETLoader lora input, found {len(lora_inputs)}")
    target_slot = lora_inputs[0]
    if unet["inputs"][target_slot].get("link") is not None:
        raise ValueError("RayUNETLoader already has a LoRA link")
    if _nodes(workflow, "RayLoraLoader"):
        raise ValueError("Workflow already contains a RayLoraLoader")

    node_ids = [int(node["id"]) for node in workflow.get("nodes", [])]
    node_id = max([int(workflow.get("last_node_id", 0)), *node_ids], default=0) + 1
    workflow["last_node_id"] = node_id
    link_ids = [int(link[0]) for link in workflow.get("links", [])]
    link_id = max([int(workflow.get("last_link_id", 0)), *link_ids], default=0) + 1
    workflow["last_link_id"] = link_id

    unet_position = unet.get("pos", [0, 0])
    lora_node = {
        "id": node_id,
        "type": "RayLoraLoader",
        "pos": [float(unet_position[0]), float(unet_position[1]) + 150.0],
        "size": [270, 82],
        "flags": {},
        "order": max((int(node.get("order", -1)) for node in workflow.get("nodes", [])), default=-1) + 1,
        "mode": 0,
        "inputs": [
            {
                "name": "prev_ray_lora",
                "shape": 7,
                "type": "RAY_LORA",
                "link": None,
            }
        ],
        "outputs": [
            {
                "name": "ray_lora",
                "type": "RAY_LORA",
                "links": [link_id],
            }
        ],
        "properties": {"Node name for S&R": "RayLoraLoader"},
        "widgets_values": [lora_name, 1.0],
    }
    workflow.setdefault("nodes", []).append(lora_node)
    workflow.setdefault("links", []).append([link_id, node_id, 0, unet["id"], target_slot, "RAY_LORA"])
    unet["inputs"][target_slot]["link"] = link_id


def build_workflow(
    source_path: str | Path,
    *,
    mode: str,
    input_filename: str | None = None,
    profile: str = "full",
    megapixels: float | None = None,
    duration: float | None = None,
    steps: int | None = None,
    turbo_variant: str | None = None,
    compute_dtype: str = "default",
) -> dict[str, Any]:
    if mode not in SOURCE_WORKFLOWS:
        raise ValueError(f"Unsupported MiniMax workflow mode: {mode}")
    if input_filename is None:
        input_filename = INPUT_FILENAMES[mode]
    if profile not in {"full", "smoke"}:
        raise ValueError(f"Unsupported workflow profile: {profile}")
    if compute_dtype not in {"default", "fp16_h3_safe"}:
        raise ValueError(f"Unsupported MiniMax compute dtype: {compute_dtype}")
    if compute_dtype == "fp16_h3_safe" and profile != "full":
        raise ValueError("fp16_h3_safe requires the full MiniMax H3 profile")
    if compute_dtype == "fp16_h3_safe" and turbo_variant is None:
        raise ValueError("fp16_h3_safe requires a pinned MiniMax H3 Turbo variant")
    if not input_filename:
        raise ValueError("input_filename must not be empty")

    turbo = _resolve_turbo_variant(turbo_variant, mode) if turbo_variant else None
    if turbo is not None and steps is not None and int(steps) != turbo["steps"]:
        raise ValueError(f"Turbo variant {turbo_variant} requires {turbo['steps']} steps, not {steps}")

    source_path = Path(source_path)
    workflow = json.loads(source_path.read_text(encoding="utf-8"))
    _ensure_conditioning_link(workflow, mode)
    _ensure_initializer_wait_link(workflow, mode)

    initializer = _one_node(workflow, "RayInitializer")
    if len(initializer.get("widgets_values", [])) != len(RAY_INITIALIZER_WIDGETS):
        raise ValueError(
            "Unexpected RayInitializer schema: "
            f"expected {len(RAY_INITIALIZER_WIDGETS)} widgets, "
            f"found {len(initializer.get('widgets_values', []))}"
        )
    initializer_widgets = list(RAY_INITIALIZER_WIDGETS)
    initializer_widgets[RAY_INITIALIZER_FSDP_CPU_OFFLOAD_INDEX] = profile == "full"
    initializer["widgets_values"] = initializer_widgets

    unet = _one_node(workflow, "RayUNETLoader")
    _set_first_widget(unet, DIFFUSION_MODELS[mode])
    _set_unet_compute_dtype(unet, compute_dtype)
    _set_first_widget(_one_node(workflow, "SaveVideo"), OUTPUT_PREFIXES[mode])

    load_images = _nodes(workflow, "LoadImage")
    expected_images = 1 if mode == "i2v" else 2
    if len(load_images) != expected_images:
        raise ValueError(f"Expected {expected_images} LoadImage nodes for {mode}, found {len(load_images)}")
    for node in load_images:
        _set_first_widget(node, input_filename)

    if profile == "smoke":
        defaults = SMOKE_DEFAULTS[mode]
        megapixels = defaults["megapixels"] if megapixels is None else megapixels
        duration = defaults["duration"] if duration is None else duration
        steps = defaults["steps"] if steps is None else steps

    if turbo is not None:
        steps = turbo["steps"]
        _ensure_lora_link(workflow, turbo["lora_name"])
        _set_first_widget(
            _one_node(workflow, "SaveVideo"),
            f"{OUTPUT_PREFIXES[mode]}_Turbo{turbo['steps']}",
        )

    if compute_dtype == "fp16_h3_safe":
        save_video = _one_node(workflow, "SaveVideo")
        _set_first_widget(save_video, f"{save_video['widgets_values'][0]}_SafeFP16")

    if megapixels is not None:
        if megapixels <= 0:
            raise ValueError("megapixels must be positive")
        resolution = _one_node(workflow, "ResolutionSelector")
        resolution["widgets_values"][1] = float(megapixels)

    if duration is not None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        _one_node(workflow, "PrimitiveFloat")["widgets_values"] = [float(duration)]

    if steps is not None:
        if steps < 1:
            raise ValueError("steps must be at least 1")
        scheduler = _one_node(workflow, "RayBasicScheduler")
        scheduler["widgets_values"][1] = int(steps)

    return workflow


def write_workflow(workflow: dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(workflow, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MiniMax H3 Windows V100 FSDP workflows")
    parser.add_argument("--mode", choices=("i2v", "ref2va", "all"), default="all")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--input-filename")
    parser.add_argument("--megapixels", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-dir", type=Path, default=WORKFLOW_ROOT)
    parser.add_argument("--turbo", action="store_true")
    parser.add_argument(
        "--compute-dtype",
        choices=("default", "fp16_h3_safe"),
        default="default",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.turbo and args.profile != "full":
        raise ValueError("--turbo requires --profile full")
    if args.compute_dtype == "fp16_h3_safe" and not args.turbo:
        raise ValueError("--compute-dtype fp16_h3_safe requires --turbo")
    modes = ("i2v", "ref2va") if args.mode == "all" else (args.mode,)
    for mode in modes:
        turbo_variant = DEFAULT_TURBO_VARIANTS[mode] if args.turbo else None
        workflow = build_workflow(
            SOURCE_WORKFLOWS[mode],
            mode=mode,
            input_filename=args.input_filename,
            profile=args.profile,
            megapixels=args.megapixels,
            duration=args.duration,
            steps=args.steps,
            turbo_variant=turbo_variant,
            compute_dtype=args.compute_dtype,
        )
        if args.compute_dtype == "fp16_h3_safe":
            output_name = SAFE_FP16_OUTPUT_WORKFLOWS[mode].name
        elif args.turbo:
            output_name = TURBO_OUTPUT_WORKFLOWS[mode].name
        else:
            output_name = OUTPUT_WORKFLOWS[mode].name
        if args.profile == "smoke" and not args.turbo:
            output_name = output_name.replace(".json", "_Smoke.json")
        output_path = write_workflow(workflow, args.output_dir / output_name)
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
