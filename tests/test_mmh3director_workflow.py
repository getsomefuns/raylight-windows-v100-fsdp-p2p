import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / "example_workflows" / "MMH3Director_Windows_V100_FSDP.json"


def _load_workflow():
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _node_by_type(workflow, node_type):
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    assert len(matches) == 1, f"expected one {node_type}, got {len(matches)}"
    return matches[0]


def _link_by_id(workflow):
    return {link[0]: link for link in workflow["links"]}


def test_director_adapter_example_has_one_unambiguous_model_path():
    workflow = _load_workflow()
    initializer = _node_by_type(workflow, "RayInitializer")
    ray_lora = _node_by_type(workflow, "RayLoraLoader")
    ray_unet = _node_by_type(workflow, "RayUNETLoader")
    adapter = _node_by_type(workflow, "RayMMH3DirectorModelAdapter")
    director = _node_by_type(workflow, "MiniMaxH3Director")
    links = _link_by_id(workflow)

    unet_init_link = links[next(item["link"] for item in ray_unet["inputs"] if item["name"] == "ray_actors_init")]
    unet_lora_link = links[next(item["link"] for item in ray_unet["inputs"] if item["name"] == "lora")]
    adapter_link = links[next(item["link"] for item in adapter["inputs"] if item["name"] == "ray_actors")]
    director_model_link = links[next(item["link"] for item in director["inputs"] if item["name"] == "model")]

    assert unet_init_link[1:5] == [initializer["id"], 0, ray_unet["id"], 0]
    assert unet_lora_link[1:5] == [ray_lora["id"], 0, ray_unet["id"], 1]
    assert adapter_link[1:5] == [ray_unet["id"], 0, adapter["id"], 0]
    assert director_model_link[1:5] == [adapter["id"], 0, director["id"], 0]


def test_director_remains_the_only_sampling_control_surface():
    workflow = _load_workflow()
    director = _node_by_type(workflow, "MiniMaxH3Director")
    node_types = {node["type"] for node in workflow["nodes"]}

    assert "UNETLoader" not in node_types
    assert "LoraLoaderModelOnly" not in node_types
    assert not any("Sampler" in node_type and node_type.startswith("Ray") for node_type in node_types)
    assert not any("Scheduler" in node_type and node_type.startswith("Ray") for node_type in node_types)
    assert not any("Guider" in node_type and node_type.startswith("Ray") for node_type in node_types)

    input_names = [item["name"] for item in director["inputs"]]
    for retained in (
        "model",
        "video_vae",
        "audio_vae",
        "clip",
        "task_type",
        "global_prompt",
        "cfg",
        "seed",
        "frame_rate",
        "width",
        "height",
        "total_frames",
        "timeline_data",
        "steps",
        "sampler",
        "scheduler",
        "shift_video",
        "shift_audio",
        "clear_vram_between_segments",
        "export_source_images",
    ):
        assert retained in input_names


def test_example_retains_the_validated_turbo8_720p_five_second_baseline():
    workflow = _load_workflow()
    director = _node_by_type(workflow, "MiniMaxH3Director")
    ray_lora = _node_by_type(workflow, "RayLoraLoader")
    ray_unet = _node_by_type(workflow, "RayUNETLoader")
    widgets = director["widgets_values"]

    assert widgets[0] == "t2v — 文生视频(Text to Video)"
    assert widgets[3:6] == [1, 42, "fixed"]
    assert widgets[6:11] == [24, 1280, 704, 1280, 124]
    assert widgets[13:18] == [8, "res_multistep", "simple", 12, 3]
    assert widgets[19:21] == [True, False]
    assert json.loads(widgets[11])["segments"][0]["durationSec"] == 5
    assert ray_unet["widgets_values"] == [
        "minimax_h3\\minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        "fp16_h3_safe",
    ]
    assert ray_lora["widgets_values"] == [
        "MMH3\\minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        1.0,
    ]


def test_workflow_keeps_existing_output_pipeline():
    workflow = _load_workflow()
    for node_type in (
        "CLIPLoader",
        "VAELoader",
        "MiniMaxH3Director",
        "CreateVideo",
        "SaveVideo",
    ):
        assert any(node["type"] == node_type for node in workflow["nodes"])
