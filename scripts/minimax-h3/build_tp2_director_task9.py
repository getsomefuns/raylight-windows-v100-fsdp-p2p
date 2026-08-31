"""Build Task 9 Director UI workflows on the FSDP + TP2 graph."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def _one_node(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow.get("nodes", []) if node.get("type") == node_type]
    if len(matches) != 1:
        raise ValueError(f"expected one {node_type}, found {len(matches)}")
    return matches[0]


def _timeline(director: dict) -> dict:
    widgets = director.get("widgets_values", [])
    if len(widgets) <= 11 or not isinstance(widgets[11], str):
        raise ValueError("MiniMaxH3Director timeline widget is missing")
    payload = json.loads(widgets[11])
    if not isinstance(payload.get("segments"), list) or not payload["segments"]:
        raise ValueError("MiniMaxH3Director timeline must contain segments")
    return payload


def build_variant(
    base_tp2: dict,
    stock_director: dict,
    *,
    width: int | None = None,
    height: int | None = None,
    segment_count: int | None = None,
    output_name: str,
) -> dict:
    """Transfer Director controls/assets onto the validated FSDP + TP2 graph."""

    if (width is None) != (height is None):
        raise ValueError("width and height must be provided together")
    if width is not None and (width <= 0 or height is None or height <= 0):
        raise ValueError("width and height must be positive")
    if segment_count is not None and segment_count <= 0:
        raise ValueError("segment_count must be positive")

    result = deepcopy(base_tp2)
    source_director = _one_node(stock_director, "MiniMaxH3Director")
    target_director = _one_node(result, "MiniMaxH3Director")
    target_director["widgets_values"] = deepcopy(source_director["widgets_values"])

    stock_unet = _one_node(stock_director, "UNETLoader")
    stock_lora = _one_node(stock_director, "LoraLoaderModelOnly")
    ray_unet = _one_node(result, "RayUNETLoader")
    ray_lora = _one_node(result, "RayLoraLoader")
    ray_unet["widgets_values"][0] = stock_unet["widgets_values"][0]
    ray_lora["widgets_values"][0] = stock_lora["widgets_values"][0]
    ray_lora["widgets_values"][1] = stock_lora["widgets_values"][1]

    timeline = _timeline(target_director)
    segments = timeline["segments"]
    if segment_count is not None:
        if len(segments) < segment_count:
            raise ValueError(
                f"source Director has {len(segments)} segments, requested {segment_count}"
            )
        segments = deepcopy(segments[:segment_count])
        cursor = 0
        for segment in segments:
            segment["start"] = cursor
            cursor += int(segment.get("frameCount") or segment.get("length"))
        timeline["segments"] = segments
    total_frames = sum(int(item.get("frameCount") or item.get("length")) for item in segments)
    timeline["totalFrames"] = total_frames
    target_director["widgets_values"][10] = total_frames

    if width is not None and height is not None:
        target_director["widgets_values"][7] = int(width)
        target_director["widgets_values"][8] = int(height)
        target_director["widgets_values"][9] = int(max(width, height))
        timeline["width"] = int(width)
        timeline["height"] = int(height)
        timeline["refMaxSize"] = int(max(width, height))
        output = timeline.setdefault("output", {})
        output["width"] = int(width)
        output["height"] = int(height)
        output["longEdge"] = int(max(width, height))
    target_director["widgets_values"][11] = json.dumps(
        timeline, ensure_ascii=False, separators=(",", ":")
    )

    save = _one_node(result, "SaveVideo")
    save["widgets_values"][0] = f"video/raylight_o10/director/{output_name}"
    result["revision"] = int(result.get("revision", 0)) + 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--segments", type=int)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8-sig"))
    source = json.loads(args.source.read_text(encoding="utf-8-sig"))
    result = build_variant(
        base,
        source,
        width=args.width,
        height=args.height,
        segment_count=args.segments,
        output_name=args.output_name,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
