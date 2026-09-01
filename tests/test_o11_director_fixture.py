import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "minimax-h3" / "prepare_o11_director_fixture.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_o11_director_fixture", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _node(workflow, node_type):
    return next(node for node in workflow["nodes"] if node["type"] == node_type)


def test_builds_single_pass_o11_director_contract():
    module = _load_script()
    base = json.loads(
        (REPO_ROOT / "example_workflows" / "Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json")
        .read_text(encoding="utf-8")
    )

    workflow = module.build_workflow(base, "test prompt", "动作迁移/宗主舞蹈")

    assert all(node["type"] != "MiniMaxH3DirectorRefine" for node in workflow["nodes"])
    assert _node(workflow, "RayMiniMaxH3DualGPUCLIPLoader")["widgets_values"] == [
        "MMH3\\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "tp2",
        "AUTO",
    ]
    assert _node(workflow, "RayLoraLoader")["widgets_values"][0].endswith(
        "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
    )
    assert _node(workflow, "RayUNETLoader")["widgets_values"] == [
        "minimax_h3\\minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
        "fp16_h3_safe",
    ]
    assert _node(workflow, "SaveVideo")["widgets_values"][0] == (
        "video/raylight_o11/MMH3Director_O11_15s_7img_1video_1audio_0.6MP"
    )

    director = _node(workflow, "MiniMaxH3Director")
    assert director["widgets_values"][6:11] == [24, 1056, 608, 1056, 365]
    assert director["widgets_values"][13] == 4
    timeline = json.loads(director["widgets_values"][11])
    assert timeline["output"]["megapixels"] == 0.6
    assert timeline["output"]["width"] == 1056
    assert timeline["output"]["height"] == 608
    assert timeline["totalFrames"] == 365
    assert timeline["global"]["prompt"] == "test prompt"
    assert [item["imageFile"] for item in timeline["global"]["refs"]] == [
        "动作迁移/宗主舞蹈/红模参考.png",
        "动作迁移/宗主舞蹈/白模参考1.png",
        "动作迁移/宗主舞蹈/白模参考2.png",
        "动作迁移/宗主舞蹈/白模参考3.png",
        "动作迁移/宗主舞蹈/白模参考4.png",
        "动作迁移/宗主舞蹈/白模参考5.png",
        "动作迁移/宗主舞蹈/白模参考6.png",
    ]
    assert timeline["global"]["refVideos"][0]["videoFile"].endswith(
        "宗主舞白模有背景15s_去水印_无音轨.mp4"
    )
    assert timeline["global"]["refAudios"][0]["audioFile"].endswith(
        "宗主舞白模有背景15s.WAV"
    )


def test_cli_validates_all_local_media(tmp_path):
    module = _load_script()
    media = tmp_path / "media"
    media.mkdir()
    for name in module.REFERENCE_IMAGES + (module.REFERENCE_VIDEO, module.REFERENCE_AUDIO):
        (media / name).write_bytes(b"fixture")

    module.validate_media(media)

