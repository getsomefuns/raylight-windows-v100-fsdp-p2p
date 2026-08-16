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
