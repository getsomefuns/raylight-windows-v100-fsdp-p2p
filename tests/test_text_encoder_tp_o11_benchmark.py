from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "minimax-h3"
    / "benchmark_o11_max_context.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_o11_max_context", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    images = []
    for index in range(7):
        relative = f"refs/image-{index}.png"
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{index}".encode())
        images.append((relative, path))
    video_relative = "refs/video.mp4"
    video_path = input_root / video_relative
    video_path.write_bytes(b"video")
    return input_root, images, (video_relative, video_path)


def test_validate_frozen_fixture_checks_prompt_and_all_media_hashes(tmp_path):
    module = _load_module()
    input_root, images, video = _fixture(tmp_path)
    prompt = "p" * 7000
    api = {
        "1": {
            "class_type": "RayMiniMaxH3DualGPUCLIPLoader",
            "inputs": {"text_encoder_attention": "AUTO"},
        },
        "30": {
            "class_type": "MiniMaxH3ReferenceToVideo",
                "inputs": {
                    "prompt": prompt,
                    "width": 1056,
                    "height": 608,
                    "length": 362,
                    **{
                    f"ref_images.ref_image_{index}": [str(10 + index), 0]
                    for index in range(7)
                },
                "ref_videos.ref_video_0": ["21", 0],
            },
        },
    }
    for index, (relative, _) in enumerate(images):
        api[str(10 + index)] = {"class_type": "LoadImage", "inputs": {"image": relative}}
    api["20"] = {"class_type": "LoadVideo", "inputs": {"file": video[0]}}
    manifest = {
        "prompt": {"normalized_characters": 7000, "sha256": module.sha256_text(prompt)},
        "media": {
            "images": [
                {"file": relative, "sha256": module.sha256_file(path)}
                for relative, path in images
            ],
            "video": {"file": video[0], "sha256": module.sha256_file(video[1])},
        },
    }

    summary = module.validate_frozen_fixture(api, manifest, input_root)
    assert summary["prompt_characters"] == 7000
    assert summary["image_count"] == 7
    assert summary["backend"] == "AUTO"

    images[3][1].write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        module.validate_frozen_fixture(api, manifest, input_root)


def test_select_backend_changes_only_o11_loader(tmp_path):
    module = _load_module()
    api = {
        "1": {
            "class_type": "RayMiniMaxH3DualGPUCLIPLoader",
            "inputs": {"text_encoder_attention": "AUTO"},
        },
        "2": {"class_type": "RayInitializer", "inputs": {"XFuser_attention": "TORCH_EFFICIENT"}},
    }

    selected = module.select_backend(api, "XFORMERS_GQA")
    assert selected["1"]["inputs"]["text_encoder_attention"] == "XFORMERS_GQA"
    assert selected["2"]["inputs"]["XFuser_attention"] == "TORCH_EFFICIENT"
    assert api["1"]["inputs"]["text_encoder_attention"] == "AUTO"

    with pytest.raises(ValueError, match="exactly one"):
        module.select_backend({}, "AUTO")


def test_failure_report_is_written_before_reraising(tmp_path):
    module = _load_module()
    report_path = tmp_path / "report.json"
    report = {"backend": "AUTO", "records": []}

    with pytest.raises(RuntimeError, match="boom"):
        with module.failure_report(report_path, report):
            raise RuntimeError("boom")

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert "boom" in saved["error"]


def test_prompt_history_state_fails_immediately_on_comfy_execution_error():
    module = _load_module()
    entry = {
        "status": {
            "status_str": "error",
            "completed": False,
            "messages": [["execution_error", {"exception_message": "CUDA out of memory"}]],
        }
    }

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        module.prompt_history_state(entry)

    assert module.prompt_history_state(
        {"status": {"status_str": "success", "completed": True}}
    )
    assert not module.prompt_history_state(
        {"status": {"status_str": "running", "completed": False}}
    )
