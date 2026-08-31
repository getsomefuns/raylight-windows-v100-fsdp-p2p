"""ComfyUI node exposing the MiniMax H3 dual-V100 TP2 text encoder."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import uuid

import comfy.sd
from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
import folder_paths

from raylight.distributed_worker.ray_worker import ensure_fresh_actors
from raylight.distributed_worker.sampling_lifecycle import external_text_encoder_gpu_lease

from .clip_proxy import (
    MiniMaxH3CheckpointPreprocessor,
    MiniMaxH3TP2ClipProxy,
    load_minimax_h3_tp2_backbone,
)


SUPPORTED_CHECKPOINT = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
EXECUTION_MODES = ["tp2", "single_gpu_fallback"]


def _validate_checkpoint(path: str) -> Path:
    checkpoint = Path(path)
    if checkpoint.name.casefold() != SUPPORTED_CHECKPOINT.casefold():
        raise ValueError(
            "MiniMax H3 dual-V100 TP2 currently supports only "
            f"{SUPPORTED_CHECKPOINT}; selected {checkpoint.name}"
        )
    return checkpoint


class RayMiniMaxH3DualGPUCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("text_encoders"),),
                "execution_mode": (EXECUTION_MODES,),
                "ray_actors_init": ("RAY_ACTORS_INIT",),
            }
        }

    RETURN_TYPES = ("CLIP", "RAY_ACTORS_INIT")
    RETURN_NAMES = ("clip", "ray_actors_init")
    FUNCTION = "load_clip"
    CATEGORY = "Raylight/text_encoder"
    DESCRIPTION = (
        "Loads the MiniMax H3 Qwen3-VL-32B NVFP4 text encoder as native-Windows "
        "tensor parallelism across exactly two V100 GPUs. TP2 is explicit and never "
        "silently falls back to the stock single-GPU CLIP loader."
    )

    def load_clip(self, clip_name, execution_mode, ray_actors_init):
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f"unsupported MiniMax H3 text encoder mode: {execution_mode!r}")

        if execution_mode == "tp2":
            ray_actors, workers, parallel_dict = ensure_fresh_actors(ray_actors_init)
            configured_world_size = int(parallel_dict.get("global_world_size", len(workers)))
            if len(workers) != 2 or configured_world_size != 2:
                raise ValueError(
                    "MiniMax H3 TP2 requires exactly two Ray workers; "
                    f"found workers={len(workers)}, configured_world_size={configured_world_size}"
                )
        else:
            ray_actors = None

        checkpoint = _validate_checkpoint(
            folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        )
        embedding_directory = folder_paths.get_folder_paths("embeddings")

        if execution_mode == "single_gpu_fallback":
            clip = comfy.sd.load_clip(
                ckpt_paths=[str(checkpoint)],
                embedding_directory=embedding_directory,
                clip_type=comfy.sd.CLIPType.MINIMAX,
            )
            return clip, ray_actors_init

        @contextmanager
        def execution_context():
            operation_id = f"minimax-h3-tp2-{uuid.uuid4().hex}"
            with external_text_encoder_gpu_lease(
                ray_actors, operation_id, "MiniMaxH3TP2"
            ):
                yield

        clip = MiniMaxH3TP2ClipProxy(
            tokenizer=MiniMaxH3Tokenizer(embedding_directory=embedding_directory),
            preprocessor_factory=lambda: MiniMaxH3CheckpointPreprocessor(checkpoint),
            backbone_factory=lambda: load_minimax_h3_tp2_backbone(checkpoint),
            execution_context_factory=execution_context,
        )
        return clip, ray_actors_init


NODE_CLASS_MAPPINGS = {
    "RayMiniMaxH3DualGPUCLIPLoader": RayMiniMaxH3DualGPUCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayMiniMaxH3DualGPUCLIPLoader": "Load MiniMax H3 Text Encoder (Dual V100 TP2)",
}


__all__ = [
    "EXECUTION_MODES",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "RayMiniMaxH3DualGPUCLIPLoader",
    "SUPPORTED_CHECKPOINT",
]
