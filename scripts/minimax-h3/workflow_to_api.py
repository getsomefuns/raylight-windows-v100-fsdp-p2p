#!/usr/bin/env python3
"""Convert a ComfyUI UI workflow into an API prompt using live node metadata."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


UI_ONLY_NODE_TYPES = {"Note", "MarkdownNote"}
WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}


def _as_list(value: Any) -> list[Any]:
    if value is None or value == {}:
        return []
    return value if isinstance(value, list) else [value]


def _is_widget(input_spec: Any) -> bool:
    if not isinstance(input_spec, list) or not input_spec:
        return False
    input_type = input_spec[0]
    options = input_spec[1] if len(input_spec) > 1 and isinstance(input_spec[1], dict) else {}
    if options.get("forceInput"):
        return False
    return isinstance(input_type, list) or input_type in WIDGET_TYPES or "options" in options


def workflow_to_prompt(workflow: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    links = {link[0]: link for link in workflow.get("links", [])}
    prompt: dict[str, Any] = {}

    for node in workflow.get("nodes", []):
        node_type = node["type"]
        if node_type in UI_ONLY_NODE_TYPES:
            continue
        if node.get("mode", 0) != 0:
            raise ValueError(f"unsupported node mode {node.get('mode')} for {node_type} ({node['id']})")
        if node_type not in object_info:
            raise ValueError(f"node type is not installed: {node_type} ({node['id']})")

        node_inputs: dict[str, Any] = {}
        connected_names: set[str] = set()
        for ui_input in _as_list(node.get("inputs")):
            if not isinstance(ui_input, dict) or ui_input.get("link") is None:
                continue
            link_id = ui_input["link"]
            if link_id not in links:
                raise ValueError(f"missing link {link_id} for {node_type} ({node['id']})")
            link = links[link_id]
            name = ui_input["name"]
            node_inputs[name] = [str(link[1]), link[2]]
            connected_names.add(name)

        definition = object_info[node_type]
        definitions = definition.get("input", {})
        input_order = definition.get("input_order", {})
        widgets = _as_list(node.get("widgets_values"))
        widget_index = 0

        for section in ("required", "optional"):
            section_definitions = definitions.get(section, {}) or {}
            ordered_names = input_order.get(section) or list(section_definitions)
            for name in ordered_names:
                spec = section_definitions.get(name)
                if not _is_widget(spec):
                    continue
                if widget_index >= len(widgets):
                    if name not in connected_names:
                        raise ValueError(f"missing widget value for {node_type}.{name} ({node['id']})")
                    continue
                value = widgets[widget_index]
                widget_index += 1
                if name not in connected_names:
                    node_inputs[name] = value

        api_node: dict[str, Any] = {"class_type": node_type, "inputs": node_inputs}
        if node.get("title"):
            api_node["_meta"] = {"title": node["title"]}
        prompt[str(node["id"])] = api_node

    return prompt


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_object_info(source: str) -> dict[str, Any]:
    path = Path(source)
    if path.exists():
        return _read_json(path)
    url = source.rstrip("/")
    if not url.endswith("/object_info"):
        url += "/object_info"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--object-info", default="http://127.0.0.1:8188")
    args = parser.parse_args()

    prompt = workflow_to_prompt(_read_json(args.workflow), _load_object_info(args.object_info))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(prompt)} executable nodes to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
