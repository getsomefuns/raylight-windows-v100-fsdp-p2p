"""Prepare the local O11 MiniMax H3 Director maximum-input validation workflow."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


REFERENCE_IMAGES = (
    "红模参考.png",
    "白模参考1.png",
    "白模参考2.png",
    "白模参考3.png",
    "白模参考4.png",
    "白模参考5.png",
    "白模参考6.png",
)
REFERENCE_VIDEO = "宗主舞白模有背景15s_去水印_无音轨.mp4"
REFERENCE_AUDIO = "宗主舞白模有背景15s.WAV"
OUTPUT_PREFIX = "video/raylight_o11/MMH3Director_O11_15s_7img_1video_1audio_0.6MP"


def _node(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node.get("type") == node_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {node_type} node, found {len(matches)}")
    return matches[0]


def _media_entry(index: int, field: str, relative_path: str) -> dict:
    return {
        "index": index,
        field: relative_path,
        "fileName": "",
        "type": "input",
        "subfolder": "",
    }


def build_workflow(base_workflow: dict, prompt: str, media_subdir: str) -> dict:
    workflow = copy.deepcopy(base_workflow)
    media_subdir = media_subdir.replace("\\", "/").strip("/")
    rel = lambda name: f"{media_subdir}/{name}"

    loader = _node(workflow, "RayMiniMaxH3DualGPUCLIPLoader")
    loader["widgets_values"] = [
        "MMH3\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "tp2",
        "AUTO",
    ]
    _node(workflow, "RayLoraLoader")["widgets_values"] = [
        "MMH3\\minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
        1.0,
    ]
    _node(workflow, "RayUNETLoader")["widgets_values"] = [
        "minimax_h3\\minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        "fp16_h3_safe",
    ]
    _node(workflow, "SaveVideo")["widgets_values"] = [OUTPUT_PREFIX, "auto", "auto"]

    task_type = "r2v — 参考主体生视频(Reference to Video)"
    timeline = {
        "version": 5,
        "editMode": "global",
        "timelineMode": "gen_blank",
        "totalFrames": 365,
        "frameRate": 24,
        "width": 1056,
        "height": 608,
        "refMaxSize": 1056,
        "output": {
            "mode": "fixed",
            "aspectRatio": "16:9 (宽屏)",
            "megapixels": 0.6,
            "multiple": 32,
            "longEdge": 1056,
            "width": 1056,
            "height": 608,
            "maxExportFrames": 0,
            "exportMode": "all",
            "audioMode": "generate",
            "continuityEnabled": False,
            "continuityOverlapFrames": 22,
        },
        "videoClips": [],
        "video": {
            "fileName": "",
            "videoFile": "",
            "subfolder": "",
            "type": "input",
            "frames": [],
            "frameMap": [],
        },
        "global": {
            "taskType": task_type,
            "prompt": prompt,
            "refs": [
                _media_entry(index, "imageFile", rel(name))
                for index, name in enumerate(REFERENCE_IMAGES)
            ],
            "refVideos": [_media_entry(0, "videoFile", rel(REFERENCE_VIDEO))],
            "refAudios": [_media_entry(0, "audioFile", rel(REFERENCE_AUDIO))],
            "referenceVideo": {},
            "continuousReference": False,
            "genImage": {"imageFile": ""},
            "commonEnabled": True,
            "commonCollapsed": False,
        },
        "segments": [
            {
                "id": "o11-max-input",
                "start": 0,
                "length": 365,
                "frameCount": 365,
                "durationSec": 15,
                "prompt": "",
                "negativePrompt": "",
                "taskType": "",
                "refs": [],
                "refVideos": [],
                "refAudios": [],
                "referenceVideo": {},
                "genImage": {"imageFile": "", "fileName": ""},
                "continuityFromPrev": False,
            }
        ],
        "gen": {"defaultFrameCount": 365},
        "runSelectEnabled": False,
        "runSelection": [],
        "liveTaePreview": False,
    }

    director = _node(workflow, "MiniMaxH3Director")
    values = list(director["widgets_values"])
    values[0] = task_type
    values[1] = prompt
    values[3] = 1.0
    values[4] = 42
    values[5] = "fixed"
    values[6:11] = [24, 1056, 608, 1056, 365]
    values[11] = json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))
    values[13] = 4
    values[14] = "res_multistep"
    values[15] = "simple"
    values[16] = 12
    values[17] = 3
    values[19] = True
    values[20] = False
    director["widgets_values"] = values
    workflow.setdefault("extra", {})["note"] = (
        "O11 validation: Director REF2VA Turbo4, single pass, 15s/24fps, "
        "1056x608, 7 images + 1 silent video + 1 audio."
    )
    return workflow


def validate_media(media_dir: Path) -> None:
    missing = [
        name
        for name in REFERENCE_IMAGES + (REFERENCE_VIDEO, REFERENCE_AUDIO)
        if not (media_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing O11 reference media: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--media-subdir", default="动作迁移/宗主舞蹈")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    validate_media(args.media_dir)
    base = json.loads(args.base.read_text(encoding="utf-8"))
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    workflow = build_workflow(base, prompt, args.media_subdir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(workflow, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
