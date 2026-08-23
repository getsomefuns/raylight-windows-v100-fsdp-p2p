import hashlib
import json
from pathlib import Path

import pytest

from raylight import nodes
from raylight.comfy_extra_dist import nodes_custom_sampler


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = REPO_ROOT / "example_workflows"

ACCEPTED_WORKFLOWS = {
    "LTX2_3_i2v_Raylight_Windows_FSDP_5s.json": ("2b0b6e3bab1a7adb7f5e3a8b627b0bfe5535b3d8b3e949ecb7644225f71eef84"),
    "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8.json": ("2f261d43e0477e21165793edfade592e4d4eef948b0434c81db8180a8fc926a5"),
    "Minimax_H3_I2V_Windows_V100_FSDP_Turbo8_FP16_Experimental.json": ("faa485b80b87126bb45223f06c935f8372611f0d98efcd4106ee7111d3e41b02"),
    "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json": ("af038101e582efd163d93f1fd22be7abe5d0c3a93b00b90c332f52d9ff2377ee"),
    "Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json": (
        "fccd1b7ed6519cc81fa66f26a956061587f8cebce567f3ddef07ba0752a32f35"
    ),
}

CORE_NODE_CONTRACTS = {
    "RayInitializer": {
        "class": nodes.RayInitializer,
        "required": (
            "ray_cluster_address",
            "ray_cluster_namespace",
            "GPU",
            "ulysses_degree",
            "ring_degree",
            "cfg_degree",
            "dp_degree",
            "sync_ulysses",
            "clear_vram_after_sampling",
            "FSDP",
            "FSDP_CPU_OFFLOAD",
            "XFuser_attention",
            "skip_comm_test",
            "use_mmap",
        ),
        "optional": ("wait_for",),
        "return_types": ("RAY_ACTORS_INIT",),
        "return_names": ("ray_actors_init",),
        "function": "spawn_actor",
        "display": "Ray Init Actor",
    },
    "RayInitializerAdvanced": {
        "class": nodes.RayInitializerAdvanced,
        "required": (
            "ray_cluster_address",
            "ray_cluster_namespace",
            "GPU",
            "GPU_SELECT",
            "ulysses_degree",
            "ring_degree",
            "clear_vram_after_sampling",
            "cfg_degree",
            "dp_degree",
            "sync_ulysses",
            "FSDP",
            "FSDP_CPU_OFFLOAD",
            "XFuser_attention",
            "skip_comm_test",
            "use_mmap",
        ),
        "optional": (
            "wait_for",
            "ray_object_store_gb",
            "ray_dashboard_address",
            "torch_dist_address",
        ),
        "return_types": ("RAY_ACTORS_INIT",),
        "return_names": ("ray_actors_init",),
        "function": "spawn_actor",
        "display": "Ray Init Actor (Advanced)",
    },
    "RayUNETLoader": {
        "class": nodes.RayUNETLoader,
        "required": ("unet_name", "weight_dtype", "ray_actors_init"),
        "optional": ("lora",),
        "return_types": ("RAY_ACTORS",),
        "return_names": ("ray_actors",),
        "function": "load_ray_unet",
        "display": "Load Diffusion Model (Ray)",
    },
    "RayKill": {
        "class": nodes.RayKill,
        "required": ("ray_actors", "kill_mode"),
        "optional": (),
        "return_types": (),
        "return_names": (),
        "function": "kill_ray",
        "display": "Kill Ray",
    },
}

SAMPLER_CONTRACTS = {
    "XFuserKSamplerAdvanced": (
        ("LATENT", "RAY_ACTORS"),
        ("latent", "ray_actors"),
        "XFuser KSampler (Advanced)",
    ),
    "UnifiedParallelSampler": (
        ("LATENT", "RAY_ACTORS"),
        ("latent", "ray_actors"),
        "Unified Parallel Sampler (Advance)",
    ),
    "DPKSamplerAdvanced": (
        ("LATENT", "RAY_ACTORS"),
        ("latent", "ray_actors"),
        "Data Parallel KSampler (Advanced)",
    ),
    "XFuserSamplerCustomAdvanced": (
        ("LATENT", "LATENT", "RAY_ACTORS"),
        ("output", "denoised_output", "ray_actors"),
        "XFuser SamplerCustom Advanced",
    ),
    "XFuserSamplerCustom": (
        ("LATENT", "RAY_ACTORS"),
        ("latent", "ray_actors"),
        "XFuser SamplerCustom",
    ),
    "UnifiedParallelSamplerCustomAdvanced": (
        ("LATENT", "LATENT", "RAY_ACTORS"),
        ("output", "denoised_output", "ray_actors"),
        "Unified Parallel SamplerCustom Advanced",
    ),
    "UnifiedParallelSamplerCustom": (
        ("LATENT", "RAY_ACTORS"),
        ("latent", "ray_actors"),
        "Unified Parallel SamplerCustom (Advance)",
    ),
    "DPSamplerCustomAdvanced": (
        ("LATENT", "LATENT", "RAY_ACTORS"),
        ("output", "denoised_output", "ray_actors"),
        "Data Parallel SamplerCustom Advanced",
    ),
    "DPSamplerCustom": (
        ("LATENT", "RAY_ACTORS"),
        ("latent", "ray_actors"),
        "Data Parallel SamplerCustom",
    ),
}


def _all_node_mappings():
    return {**nodes.NODE_CLASS_MAPPINGS, **nodes_custom_sampler.NODE_CLASS_MAPPINGS}


def _all_display_mappings():
    return {
        **nodes.NODE_DISPLAY_NAME_MAPPINGS,
        **nodes_custom_sampler.NODE_DISPLAY_NAME_MAPPINGS,
    }


@pytest.mark.parametrize("node_id", CORE_NODE_CONTRACTS)
def test_existing_core_node_schema_is_unchanged(node_id):
    contract = CORE_NODE_CONTRACTS[node_id]
    node_class = contract["class"]
    input_types = node_class.INPUT_TYPES()

    assert tuple(input_types.get("required", {})) == contract["required"]
    assert tuple(input_types.get("optional", {})) == contract["optional"]
    assert node_class.RETURN_TYPES == contract["return_types"]
    assert getattr(node_class, "RETURN_NAMES", ()) == contract["return_names"]
    assert node_class.FUNCTION == contract["function"]
    assert _all_display_mappings()[node_id] == contract["display"]


def test_ray_unet_loader_still_returns_only_ray_actors():
    assert nodes.RayUNETLoader.RETURN_TYPES == ("RAY_ACTORS",)


@pytest.mark.parametrize("node_id", SAMPLER_CONTRACTS)
def test_existing_sampler_public_contract_is_unchanged(node_id):
    return_types, return_names, display_name = SAMPLER_CONTRACTS[node_id]
    node_class = _all_node_mappings()[node_id]

    assert node_class.RETURN_TYPES == return_types
    assert getattr(node_class, "RETURN_NAMES", ()) == return_names
    assert node_class.FUNCTION == "ray_sample"
    assert _all_display_mappings()[node_id] == display_name


@pytest.mark.parametrize("filename, expected_sha256", ACCEPTED_WORKFLOWS.items())
def test_accepted_workflow_bytes_are_unchanged(filename, expected_sha256):
    workflow_path = WORKFLOW_ROOT / filename
    assert hashlib.sha256(workflow_path.read_bytes()).hexdigest() == expected_sha256


@pytest.mark.parametrize("filename", ACCEPTED_WORKFLOWS)
def test_accepted_workflow_links_and_raylight_nodes_still_resolve(filename):
    workflow = json.loads((WORKFLOW_ROOT / filename).read_text(encoding="utf-8"))
    nodes_by_id = {node["id"]: node for node in workflow["nodes"]}
    assert len(nodes_by_id) == len(workflow["nodes"])

    raylight_node_ids = {
        "RayBasicGuider",
        "RayBasicScheduler",
        "RayCFGGuider",
        "RayInitializer",
        "RayLoraLoader",
        "RayUNETLoader",
        "XFuserSamplerCustomAdvanced",
    }
    mappings = _all_node_mappings()
    referenced_raylight_nodes = {node["type"] for node in workflow["nodes"] if node["type"] in raylight_node_ids}
    assert referenced_raylight_nodes
    assert referenced_raylight_nodes <= mappings.keys()

    for link_id, source_id, source_slot, target_id, target_slot, _link_type in workflow["links"]:
        source = nodes_by_id[source_id]
        target = nodes_by_id[target_id]
        assert source_slot < len(source.get("outputs", ())), (filename, link_id, source_id, source_slot)
        assert target_slot < len(target.get("inputs", ())), (filename, link_id, target_id, target_slot)
        assert link_id in (source["outputs"][source_slot].get("links") or ())
        assert target["inputs"][target_slot].get("link") == link_id
