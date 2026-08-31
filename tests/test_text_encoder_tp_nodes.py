from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest

from raylight.text_encoder_parallel import nodes as tp_nodes
from raylight.text_encoder_parallel.clip_proxy import MiniMaxH3TP2ClipProxy


def _payload(worker_count: int = 2):
    workers = [mock.Mock(name=f"worker-{rank}") for rank in range(worker_count)]
    ray_actors = {"workers": workers, "world_size": worker_count}
    actor_fn = mock.Mock(name="actor-factory")
    return [ray_actors, actor_fn], ray_actors, workers


def test_node_schema_registration_and_display_name_are_explicit() -> None:
    node = tp_nodes.RayMiniMaxH3DualGPUCLIPLoader
    schema = node.INPUT_TYPES()

    assert list(schema) == ["required"]
    assert list(schema["required"]) == ["clip_name", "execution_mode", "ray_actors_init"]
    assert schema["required"]["execution_mode"][0] == ["tp2", "single_gpu_fallback"]
    assert schema["required"]["ray_actors_init"] == ("RAY_ACTORS_INIT",)
    assert node.RETURN_TYPES == ("CLIP", "RAY_ACTORS_INIT")
    assert node.RETURN_NAMES == ("clip", "ray_actors_init")
    assert node.FUNCTION == "load_clip"
    assert node.CATEGORY == "Raylight/text_encoder"
    assert tp_nodes.NODE_CLASS_MAPPINGS == {"RayMiniMaxH3DualGPUCLIPLoader": node}
    assert tp_nodes.NODE_DISPLAY_NAME_MAPPINGS == {
        "RayMiniMaxH3DualGPUCLIPLoader": "Load MiniMax H3 Text Encoder (Dual V100 TP2)"
    }


def test_tp2_mode_returns_proxy_and_preserves_original_actor_payload() -> None:
    payload, ray_actors, workers = _payload()
    checkpoint = "E:/models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    tokenizer = object()

    with (
        mock.patch.object(
            tp_nodes, "ensure_fresh_actors", return_value=(ray_actors, workers, {"global_world_size": 2})
        ) as ensure,
        mock.patch.object(tp_nodes.folder_paths, "get_full_path_or_raise", return_value=checkpoint),
        mock.patch.object(tp_nodes.folder_paths, "get_folder_paths", return_value=["E:/embeddings"]),
        mock.patch.object(tp_nodes, "MiniMaxH3Tokenizer", return_value=tokenizer),
    ):
        clip, returned = tp_nodes.RayMiniMaxH3DualGPUCLIPLoader().load_clip(
            "MMH3/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "tp2", payload
        )

    assert isinstance(clip, MiniMaxH3TP2ClipProxy)
    assert clip.tokenizer is tokenizer
    assert returned is payload
    ensure.assert_called_once_with(payload)


def test_tp2_proxy_lease_context_uses_same_workers_and_unique_operation_ids() -> None:
    payload, ray_actors, workers = _payload()
    observed = []

    @contextmanager
    def lease(actors, operation_id, owner):
        observed.append((actors, operation_id, owner))
        yield ("rank0", "rank1")

    with (
        mock.patch.object(
            tp_nodes, "ensure_fresh_actors", return_value=(ray_actors, workers, {"global_world_size": 2})
        ),
        mock.patch.object(
            tp_nodes.folder_paths,
            "get_full_path_or_raise",
            return_value="E:/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        ),
        mock.patch.object(tp_nodes.folder_paths, "get_folder_paths", return_value=[]),
        mock.patch.object(tp_nodes, "MiniMaxH3Tokenizer", return_value=object()),
        mock.patch.object(tp_nodes, "external_text_encoder_gpu_lease", side_effect=lease),
    ):
        clip, _ = tp_nodes.RayMiniMaxH3DualGPUCLIPLoader().load_clip(
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "tp2", payload
        )
        with (
            mock.patch.object(tp_nodes.comfy.model_management, "unload_all_models") as unload,
            mock.patch.object(tp_nodes.comfy.model_management, "soft_empty_cache") as empty,
        ):
            with clip._execution_context_factory():
                pass
            with clip._execution_context_factory():
                pass

    assert [row[0] for row in observed] == [ray_actors, ray_actors]
    assert [row[2] for row in observed] == ["MiniMaxH3TP2", "MiniMaxH3TP2"]
    assert observed[0][1] != observed[1][1]
    assert all(value.startswith("minimax-h3-tp2-") for _, value, _ in observed)
    assert unload.call_count == 2
    assert empty.call_count == 2


@pytest.mark.parametrize("worker_count", [0, 1, 3])
def test_tp2_mode_fails_closed_unless_exactly_two_workers(worker_count: int) -> None:
    payload, ray_actors, workers = _payload(worker_count)
    with mock.patch.object(
        tp_nodes,
        "ensure_fresh_actors",
        return_value=(ray_actors, workers, {"global_world_size": worker_count}),
    ):
        with pytest.raises(ValueError, match="exactly two Ray workers"):
            tp_nodes.RayMiniMaxH3DualGPUCLIPLoader().load_clip("model.safetensors", "tp2", payload)


def test_single_gpu_fallback_is_explicit_and_uses_stock_minimax_loader() -> None:
    payload, ray_actors, workers = _payload()
    stock_clip = object()
    with (
        mock.patch.object(
            tp_nodes, "ensure_fresh_actors", return_value=(ray_actors, workers, {"global_world_size": 2})
        ),
        mock.patch.object(
            tp_nodes.folder_paths,
            "get_full_path_or_raise",
            return_value="E:/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        ),
        mock.patch.object(tp_nodes.folder_paths, "get_folder_paths", return_value=["E:/embeddings"]),
        mock.patch.object(tp_nodes.comfy.sd, "load_clip", return_value=stock_clip) as load,
    ):
        clip, returned = tp_nodes.RayMiniMaxH3DualGPUCLIPLoader().load_clip(
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "single_gpu_fallback", payload
        )

    assert clip is stock_clip
    assert returned is payload
    load.assert_called_once_with(
        ckpt_paths=[str(Path("E:/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"))],
        embedding_directory=["E:/embeddings"],
        clip_type=tp_nodes.comfy.sd.CLIPType.MINIMAX,
    )


def test_invalid_mode_never_silently_falls_back() -> None:
    payload, ray_actors, workers = _payload()
    with mock.patch.object(
        tp_nodes, "ensure_fresh_actors", return_value=(ray_actors, workers, {"global_world_size": 2})
    ):
        with pytest.raises(ValueError, match="unsupported MiniMax H3 text encoder mode"):
            tp_nodes.RayMiniMaxH3DualGPUCLIPLoader().load_clip("model.safetensors", "auto", payload)


def test_loader_rejects_unsupported_checkpoint_before_constructing_proxy() -> None:
    payload, ray_actors, workers = _payload()
    with (
        mock.patch.object(
            tp_nodes, "ensure_fresh_actors", return_value=(ray_actors, workers, {"global_world_size": 2})
        ),
        mock.patch.object(
            tp_nodes.folder_paths, "get_full_path_or_raise", return_value="E:/other_model.safetensors"
        ),
        pytest.raises(ValueError, match="qwen3vl_32b_minimax_h3_nvfp4_awq"),
    ):
        tp_nodes.RayMiniMaxH3DualGPUCLIPLoader().load_clip("other_model.safetensors", "tp2", payload)
