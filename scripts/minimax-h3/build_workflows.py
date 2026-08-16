from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO_ROOT / "example_workflows"

SOURCE_WORKFLOWS = {
    "i2v": WORKFLOW_ROOT / "Minimax_H3_I2V_Raylight.json",
    "ref2va": WORKFLOW_ROOT / "Minimax_H3_REF2VA_Raylight.json",
}
OUTPUT_WORKFLOWS = {
    "i2v": WORKFLOW_ROOT / "Minimax_H3_I2V_Windows_V100_FSDP.json",
    "ref2va": WORKFLOW_ROOT / "Minimax_H3_REF2VA_Windows_V100_FSDP.json",
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
    False,
    True,
]
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


def build_workflow(
    source_path: str | Path,
    *,
    mode: str,
    input_filename: str | None = None,
    profile: str = "full",
    megapixels: float | None = None,
    duration: float | None = None,
    steps: int | None = None,
) -> dict[str, Any]:
    if mode not in SOURCE_WORKFLOWS:
        raise ValueError(f"Unsupported MiniMax workflow mode: {mode}")
    if input_filename is None:
        input_filename = INPUT_FILENAMES[mode]
    if profile not in {"full", "smoke"}:
        raise ValueError(f"Unsupported workflow profile: {profile}")
    if not input_filename:
        raise ValueError("input_filename must not be empty")

    source_path = Path(source_path)
    workflow = json.loads(source_path.read_text(encoding="utf-8"))

    initializer = _one_node(workflow, "RayInitializer")
    if len(initializer.get("widgets_values", [])) != len(RAY_INITIALIZER_WIDGETS):
        raise ValueError(
            "Unexpected RayInitializer schema: "
            f"expected {len(RAY_INITIALIZER_WIDGETS)} widgets, "
            f"found {len(initializer.get('widgets_values', []))}"
        )
    initializer["widgets_values"] = list(RAY_INITIALIZER_WIDGETS)

    _set_first_widget(_one_node(workflow, "RayUNETLoader"), DIFFUSION_MODELS[mode])
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    modes = ("i2v", "ref2va") if args.mode == "all" else (args.mode,)
    for mode in modes:
        workflow = build_workflow(
            SOURCE_WORKFLOWS[mode],
            mode=mode,
            input_filename=args.input_filename,
            profile=args.profile,
            megapixels=args.megapixels,
            duration=args.duration,
            steps=args.steps,
        )
        output_name = OUTPUT_WORKFLOWS[mode].name
        if args.profile == "smoke":
            output_name = output_name.replace(".json", "_Smoke.json")
        output_path = write_workflow(workflow, args.output_dir / output_name)
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
