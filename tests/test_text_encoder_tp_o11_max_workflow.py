from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "minimax-h3"
    / "build_o11_max_context_workflow.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_o11_max_context_workflow", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_o11_prompt_is_exactly_7000_meaningful_characters():
    module = _load_module()
    prompt = module.build_acceptance_prompt()

    assert module.normalize_prompt(prompt) == prompt
    assert len(prompt) == 7000
    assert prompt.count("<Picture ") >= 7
    assert "<Video 1>" in prompt
    assert "14.5至15.0秒" in prompt
    assert "15.0至15.5秒" not in prompt
    assert len(set(prompt)) >= 150
    assert not any(token * 20 in prompt for token in (" ", "。", "测", "a", "0"))


def test_o11_api_prompt_has_exact_boundary_and_stops_at_conditioning():
    module = _load_module()
    prompt = module.build_api_prompt(
        checkpoint="text_encoder.safetensors",
        video_vae="video_vae.safetensors",
        audio_vae="audio_vae.safetensors",
        image_files=[f"image-{index}.png" for index in range(7)],
        video_file="reference-15s.mp4",
        attention_backend="AUTO",
    )
    by_type: dict[str, list[dict]] = {}
    for node in prompt.values():
        by_type.setdefault(node["class_type"], []).append(node)

    assert len(by_type["LoadImage"]) == 7
    assert len(by_type["LoadVideo"]) == 1
    assert len(by_type["GetVideoComponents"]) == 1
    assert len(by_type["MiniMaxH3ReferenceToVideo"]) == 1
    assert len(by_type["PreviewAny"]) == 1
    assert not ({"KSampler", "XFuserSamplerCustomAdvanced", "VAEDecode"} & set(by_type))

    source = by_type["MiniMaxH3ReferenceToVideo"][0]["inputs"]
    assert source["width"] == 1056
    assert source["height"] == 608
    assert source["length"] == 362
    assert source["ref_image_size"] == "match"
    assert len(source["prompt"]) == 7000
    assert {
        key for key in source if key.startswith("ref_images.ref_image_")
    } == {f"ref_images.ref_image_{index}" for index in range(7)}
    assert "ref_videos.ref_video_0" in source

    loader = by_type["RayMiniMaxH3DualGPUCLIPLoader"][0]["inputs"]
    assert loader["execution_mode"] == "tp2"
    assert loader["text_encoder_attention"] == "AUTO"


def test_o11_fixture_validation_rejects_duplicate_or_short_media(tmp_path, monkeypatch):
    module = _load_module()
    images = []
    for index in range(7):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(bytes([index]) * 8)
        images.append(path)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")

    monkeypatch.setattr(
        module,
        "probe_image",
        lambda path: {"width": 854, "height": 480},
    )
    monkeypatch.setattr(
        module,
        "probe_video",
        lambda path: {
            "width": 1920,
            "height": 1080,
            "fps": 24.0,
            "frames": 360,
            "duration_seconds": 15.0,
        },
    )
    relative_names = [f"refs/image-{index}.png" for index in range(7)]
    validated = module.validate_media(
        images,
        video,
        image_names=relative_names,
        video_name="refs/video.mp4",
    )
    assert [row["file"] for row in validated["images"]] == relative_names
    assert validated["video"]["file"] == "refs/video.mp4"
    assert len({row["sha256"] for row in validated["images"]}) == 7

    images[-1].write_bytes(images[0].read_bytes())
    with pytest.raises(ValueError, match="unique SHA-256"):
        module.validate_media(images, video, image_names=relative_names)

    images[-1].write_bytes(b"unique-again")
    monkeypatch.setattr(
        module,
        "probe_image",
        lambda path: {"width": 853, "height": 479},
    )
    with pytest.raises(ValueError, match="short edge"):
        module.validate_media(images, video, image_names=relative_names)


def test_o11_ui_workflow_matches_api_boundary(tmp_path):
    module = _load_module()
    output = tmp_path / "workflow.json"
    workflow = module.build_ui_workflow(
        checkpoint="text_encoder.safetensors",
        video_vae="video_vae.safetensors",
        audio_vae="audio_vae.safetensors",
        image_files=[f"image-{index}.png" for index in range(7)],
        video_file="reference-15s.mp4",
        attention_backend="AUTO",
    )
    output.write_text(json.dumps(workflow), encoding="utf-8")

    assert workflow["version"] == pytest.approx(0.4)
    nodes = workflow["nodes"]
    assert len([node for node in nodes if node["type"] == "LoadImage"]) == 7
    assert len([node for node in nodes if node["type"] == "LoadVideo"]) == 1
    assert len([node for node in nodes if node["type"] == "GetVideoComponents"]) == 1
    source = next(node for node in nodes if node["type"] == "MiniMaxH3ReferenceToVideo")
    assert source["widgets_values"][-5:] == [
        module.build_acceptance_prompt(),
        1056,
        608,
        362,
        "match",
    ]
    assert max(link[0] for link in workflow["links"]) == workflow["last_link_id"]
    assert max(node["id"] for node in nodes) == workflow["last_node_id"]
