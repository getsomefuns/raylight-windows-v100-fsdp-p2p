from pathlib import Path


def test_fsdp_quantized_model_reuse_is_not_excluded():
    source = (
        Path(__file__).parents[1] / "src" / "raylight" / "nodes.py"
    ).read_text(encoding="utf-8")

    assert (
        "transition = _fsdp_actor_model_transition(gpu_actors, unet_path, model_options)"
        in source
    )
    assert (
        'if parallel_dict["is_quant"] is False:\n'
        "                already_loaded = ray.get("
    ) not in source
