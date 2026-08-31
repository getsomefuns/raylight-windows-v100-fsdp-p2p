"""Build MiniMax H3 workflows that explicitly select the dual-V100 TP2 CLIP loader."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPO_ROOT / "example_workflows"
WORKFLOWS = {
    "Minimax_H3_I2V_DualGPU_Text_Encoder_TP2_Raylight.json": (
        "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json",
        True,
    ),
    "Minimax_H3_REF2VA_DualGPU_Text_Encoder_TP2_Raylight.json": (
        "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json",
        True,
    ),
    "Minimax_H3_DualGPU_Text_Encoder_TP2_Raylight.json": (
        "MMH3Director_Windows_V100_FSDP.json",
        False,
    ),
}


def _one_node(workflow: dict, node_type: str) -> dict:
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {node_type}, found {len(matches)}")
    return matches[0]


def _remove_link(workflow: dict, link_id: int) -> None:
    links = [link for link in workflow["links"] if int(link[0]) == int(link_id)]
    if len(links) != 1:
        raise ValueError(f"expected exactly one workflow link {link_id}, found {len(links)}")
    link = links[0]
    source = next(node for node in workflow["nodes"] if node["id"] == link[1])
    for output in source.get("outputs", []):
        if output.get("links") and link_id in output["links"]:
            output["links"].remove(link_id)
    workflow["links"].remove(link)


def build_tp2_workflow(source: dict, *, remove_initializer_wait: bool) -> dict:
    workflow = deepcopy(source)
    clip = _one_node(workflow, "CLIPLoader")
    initializer = _one_node(workflow, "RayInitializer")
    unet = _one_node(workflow, "RayUNETLoader")

    init_input = next(item for item in unet["inputs"] if item["name"] == "ray_actors_init")
    init_link_id = init_input.get("link")
    if init_link_id is None:
        raise ValueError("RayUNETLoader ray_actors_init must be linked in the source workflow")
    init_link = next(link for link in workflow["links"] if link[0] == init_link_id)
    if init_link[1:5] != [initializer["id"], 0, unet["id"], 0]:
        raise ValueError(f"unexpected RayInitializer to RayUNETLoader link: {init_link}")

    if remove_initializer_wait:
        wait_input = next(item for item in initializer["inputs"] if item["name"] == "wait_for")
        if wait_input.get("link") is not None:
            _remove_link(workflow, wait_input["link"])
            wait_input["link"] = None

    clip_name = clip["widgets_values"][0]
    previous_clip_output = clip["outputs"][0]
    clip.update(
        {
            "type": "RayMiniMaxH3DualGPUCLIPLoader",
            "title": "MiniMax H3 Text Encoder (Dual V100 TP2)",
            "inputs": [
                {
                    "name": "ray_actors_init",
                    "type": "RAY_ACTORS_INIT",
                    "link": init_link_id,
                }
            ],
            "outputs": [
                {
                    "name": "clip",
                    "type": "CLIP",
                    "links": list(previous_clip_output.get("links") or []),
                },
                {
                    "name": "ray_actors_init",
                    "type": "RAY_ACTORS_INIT",
                    "links": [],
                },
            ],
            "widgets_values": [clip_name, "tp2"],
        }
    )
    clip.setdefault("properties", {})["Node name for S&R"] = "RayMiniMaxH3DualGPUCLIPLoader"

    # Reuse the old link ID for Initializer -> TP2 loader, then add one link
    # for TP2 loader -> RayUNETLoader. This serializes worker construction
    # before either the stock fallback or TP2 proxy can be used.
    init_link[3] = clip["id"]
    init_link[4] = 0
    new_link_id = int(workflow["last_link_id"]) + 1
    new_link = [new_link_id, clip["id"], 1, unet["id"], 0, "RAY_ACTORS_INIT"]
    workflow["links"].append(new_link)
    clip["outputs"][1]["links"].append(new_link_id)
    init_input["link"] = new_link_id
    workflow["last_link_id"] = new_link_id
    workflow["revision"] = int(workflow.get("revision", 0)) + 1
    return workflow


def write_all() -> list[Path]:
    written = []
    for output_name, (source_name, remove_wait) in WORKFLOWS.items():
        source_path = WORKFLOW_ROOT / source_name
        output_path = WORKFLOW_ROOT / output_name
        source = json.loads(source_path.read_text(encoding="utf-8"))
        built = build_tp2_workflow(source, remove_initializer_wait=remove_wait)
        output_path.write_text(
            json.dumps(built, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        written.append(output_path)
    return written


if __name__ == "__main__":
    for path in write_all():
        print(path)
