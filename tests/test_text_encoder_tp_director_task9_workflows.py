from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "minimax-h3"
    / "build_tp2_director_task9.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("build_tp2_director_task9", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base():
    return {
        "revision": 1,
        "nodes": [
            {"type": "RayMiniMaxH3DualGPUCLIPLoader", "widgets_values": ["qwen", "tp2"]},
            {"type": "RayUNETLoader", "widgets_values": ["old-unet"]},
            {"type": "RayLoraLoader", "widgets_values": ["old-lora", 1.0]},
            {
                "type": "MiniMaxH3Director",
                "widgets_values": ["t2v"] * 11 + [json.dumps({"segments": [{"length": 1}]})],
            },
            {"type": "SaveVideo", "widgets_values": ["old", "auto", "auto"]},
        ],
    }


def _stock():
    timeline = {
        "totalFrames": 219,
        "width": 864,
        "height": 480,
        "refMaxSize": 864,
        "output": {"width": 864, "height": 480, "longEdge": 864},
        "segments": [
            {"start": 0, "length": 73, "frameCount": 73},
            {"start": 73, "length": 73, "frameCount": 73},
            {"start": 146, "length": 73, "frameCount": 73},
        ],
    }
    widgets = ["r2v", "prompt", "sampling", 1, 42, "fixed", 24, 864, 480, 864, 219, json.dumps(timeline)]
    return {
        "nodes": [
            {"type": "UNETLoader", "widgets_values": ["ref2va.safetensors", "default"]},
            {"type": "LoraLoaderModelOnly", "widgets_values": ["turbo4.safetensors", 0.8]},
            {"type": "MiniMaxH3Director", "widgets_values": widgets},
        ]
    }


def test_builds_single_segment_960x544_on_existing_tp2_graph():
    module = _module()

    result = module.build_variant(
        _base(),
        _stock(),
        width=960,
        height=544,
        segment_count=1,
        output_name="single",
    )

    by_type = {node["type"]: node for node in result["nodes"]}
    assert "CLIPLoader" not in by_type
    assert by_type["RayUNETLoader"]["widgets_values"][0] == "ref2va.safetensors"
    assert by_type["RayLoraLoader"]["widgets_values"][:2] == ["turbo4.safetensors", 0.8]
    director = by_type["MiniMaxH3Director"]
    assert director["widgets_values"][7:11] == [960, 544, 960, 73]
    timeline = json.loads(director["widgets_values"][11])
    assert len(timeline["segments"]) == 1
    assert timeline["totalFrames"] == 73
    assert timeline["output"]["width"] == 960
    assert by_type["SaveVideo"]["widgets_values"][0].startswith(
        "video/raylight_o10/director/"
    )
