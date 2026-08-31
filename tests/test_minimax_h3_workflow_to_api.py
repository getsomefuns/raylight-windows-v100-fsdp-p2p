import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERTER_PATH = REPO_ROOT / "scripts" / "minimax-h3" / "workflow_to_api.py"


def _load_converter():
    spec = importlib.util.spec_from_file_location("minimax_h3_workflow_to_api", CONVERTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_converter_preserves_links_and_consumes_connected_widget_slots():
    converter = _load_converter()
    workflow = {
        "nodes": [
            {"id": 1, "type": "Source", "mode": 0, "inputs": [], "widgets_values": ["model.safetensors"]},
            {
                "id": 2,
                "type": "Target",
                "mode": 0,
                "title": "Target title",
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": 10},
                    {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 11},
                ],
                "widgets_values": ["prompt text", 1344, 768, "ui-only-control"],
            },
            {"id": 3, "type": "Primitive", "mode": 0, "inputs": [], "widgets_values": [608]},
            {"id": 4, "type": "MarkdownNote", "mode": 0, "inputs": [], "widgets_values": ["note"]},
        ],
        "links": [
            [10, 1, 0, 2, 0, "MODEL"],
            [11, 3, 0, 2, 1, "INT"],
        ],
    }
    object_info = {
        "Source": {
            "input": {"required": {"model_name": [["model.safetensors"], {}]}},
            "input_order": {"required": ["model_name"]},
        },
        "Primitive": {
            "input": {"required": {"value": ["INT", {}]}},
            "input_order": {"required": ["value"]},
        },
        "Target": {
            "input": {
                "required": {
                    "model": ["MODEL", {}],
                    "prompt": ["STRING", {}],
                    "width": ["INT", {}],
                    "height": ["INT", {}],
                }
            },
            "input_order": {"required": ["model", "prompt", "width", "height"]},
        },
    }

    prompt = converter.workflow_to_prompt(workflow, object_info)

    assert prompt == {
        "1": {"class_type": "Source", "inputs": {"model_name": "model.safetensors"}},
        "2": {
            "class_type": "Target",
            "inputs": {
                "model": ["1", 0],
                "prompt": "prompt text",
                "width": ["3", 0],
                "height": 768,
            },
            "_meta": {"title": "Target title"},
        },
        "3": {"class_type": "Primitive", "inputs": {"value": 608}},
    }


def test_converter_rejects_unknown_executable_nodes():
    converter = _load_converter()
    workflow = {
        "nodes": [{"id": 9, "type": "MissingNode", "mode": 0, "inputs": [], "widgets_values": []}],
        "links": [],
    }

    try:
        converter.workflow_to_prompt(workflow, {})
    except ValueError as exc:
        assert "MissingNode" in str(exc)
    else:
        raise AssertionError("unknown executable node was silently discarded")


def test_converter_serializes_combo_and_dynamic_combo_widgets():
    converter = _load_converter()
    workflow = {
        "nodes": [
            {
                "id": 7,
                "type": "SaveLikeNode",
                "mode": 0,
                "inputs": [],
                "widgets_values": ["auto", "h264"],
            }
        ],
        "links": [],
    }
    object_info = {
        "SaveLikeNode": {
            "input": {
                "required": {
                    "format": ["COMBO", {"options": ["auto", "mp4"]}],
                    "codec": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "auto"}, {"key": "h264"}]}],
                }
            },
            "input_order": {"required": ["format", "codec"]},
        }
    }

    prompt = converter.workflow_to_prompt(workflow, object_info)

    assert prompt["7"]["inputs"] == {"format": "auto", "codec": "h264"}


def test_converter_skips_mmh3director_frontend_only_widget_slots():
    converter = _load_converter()
    timeline = '{"totalFrames":124,"frameRate":24}'
    workflow = {
        "nodes": [
            {
                "id": 5,
                "type": "MiniMaxH3Director",
                "mode": 0,
                "inputs": [],
                "widgets_values": [
                    "t2v",
                    "prompt",
                    "sample group",
                    1.0,
                    42,
                    "fixed",
                    24,
                    1280,
                    704,
                    1280,
                    124,
                    timeline,
                    "advanced group",
                    8,
                    "res_multistep",
                    "simple",
                    12.0,
                    3.0,
                    "performance group",
                    True,
                    False,
                ],
            }
        ],
        "links": [],
    }
    input_names = [
        "task_type",
        "global_prompt",
        "bd_grp_sample",
        "cfg",
        "seed",
        "frame_rate",
        "width",
        "height",
        "ref_max_size",
        "total_frames",
        "timeline_data",
        "bd_grp_advanced",
        "steps",
        "sampler",
        "scheduler",
        "shift_video",
        "shift_audio",
        "bd_grp_perf",
        "clear_vram_between_segments",
        "export_source_images",
    ]
    object_info = {
        "MiniMaxH3Director": {
            "input": {
                "required": {
                    name: [
                        "STRING"
                        if name in {"global_prompt", "timeline_data"}
                        else "BOOLEAN"
                        if name in {"clear_vram_between_segments", "export_source_images"}
                        else "FLOAT"
                        if name in {"cfg", "frame_rate", "shift_video", "shift_audio"}
                        else "INT"
                        if name in {"seed", "width", "height", "ref_max_size", "total_frames", "steps"}
                        else "BDGROUP"
                        if name in {"bd_grp_sample", "bd_grp_advanced", "bd_grp_perf"}
                        else ["t2v", "res_multistep", "simple"],
                        {},
                    ]
                    for name in input_names
                }
            },
            "input_order": {"required": input_names},
        }
    }

    prompt = converter.workflow_to_prompt(workflow, object_info)

    assert prompt["5"]["inputs"] == {
        "task_type": "t2v",
        "global_prompt": "prompt",
        "bd_grp_sample": "sample group",
        "cfg": 1.0,
        "seed": 42,
        "frame_rate": 24,
        "width": 1280,
        "height": 704,
        "ref_max_size": 1280,
        "total_frames": 124,
        "timeline_data": timeline,
        "bd_grp_advanced": "advanced group",
        "steps": 8,
        "sampler": "res_multistep",
        "scheduler": "simple",
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "bd_grp_perf": "performance group",
        "clear_vram_between_segments": True,
        "export_source_images": False,
    }


def test_converter_omits_bypassed_source_and_its_optional_downstream_link():
    converter = _load_converter()
    workflow = {
        "nodes": [
            {
                "id": 19,
                "type": "UpscaleModelLoader",
                "mode": 4,
                "inputs": [],
                "widgets_values": ["4x-UltraSharp.pth"],
            },
            {
                "id": 20,
                "type": "MiniMaxH3DirectorRefine",
                "mode": 0,
                "inputs": [
                    {"name": "upscale_model", "type": "UPSCALE_MODEL", "link": 18},
                ],
                "widgets_values": ["refine"],
            },
        ],
        "links": [[18, 19, 0, 20, 0, "UPSCALE_MODEL"]],
    }
    object_info = {
        "UpscaleModelLoader": {
            "input": {"required": {"model_name": [["4x-UltraSharp.pth"], {}]}},
            "input_order": {"required": ["model_name"]},
        },
        "MiniMaxH3DirectorRefine": {
            "input": {
                "required": {"mode": [["refine"], {}]},
                "optional": {"upscale_model": ["UPSCALE_MODEL", {}]},
            },
            "input_order": {"required": ["mode"], "optional": ["upscale_model"]},
        },
    }

    prompt = converter.workflow_to_prompt(workflow, object_info)

    assert "19" not in prompt
    assert prompt["20"]["inputs"] == {"mode": "refine"}


def test_converter_rejects_bypassed_source_feeding_required_input():
    converter = _load_converter()
    workflow = {
        "nodes": [
            {"id": 1, "type": "Source", "mode": 4, "inputs": [], "widgets_values": []},
            {
                "id": 2,
                "type": "Target",
                "mode": 0,
                "inputs": [{"name": "model", "type": "MODEL", "link": 9}],
                "widgets_values": [],
            },
        ],
        "links": [[9, 1, 0, 2, 0, "MODEL"]],
    }
    object_info = {
        "Source": {"input": {"required": {}}},
        "Target": {"input": {"required": {"model": ["MODEL", {}]}}},
    }

    try:
        converter.workflow_to_prompt(workflow, object_info)
    except ValueError as exc:
        assert "required input Target.model" in str(exc)
    else:
        raise AssertionError("bypassed source silently removed a required input")
