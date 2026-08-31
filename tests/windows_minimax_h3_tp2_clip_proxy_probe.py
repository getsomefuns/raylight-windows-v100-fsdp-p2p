"""Gate D probe for the real MiniMax H3 CLIP proxy and 50-layer TP2 backbone."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
import torch.nn.functional as F


DEFAULT_PROMPT = (
    "A cinematic tracking shot follows three emerald robots through a crystalline city, "
    "purple lightning reflecting across polished armor, coherent motion, detailed metal surfaces."
)


def _install_comfy(comfy_root: Path) -> None:
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)
    # Match the accepted single-V100 baseline.  Standalone probes do not pass
    # through ComfyUI's main.py, so argument parsing must be enabled explicitly
    # before model_management is imported.
    sys.argv = [sys.argv[0], "--disable-cuda-malloc", "--reserve-vram", "4"]
    from comfy.options import enable_args_parsing

    enable_args_parsing()


def _save_tensor(path: Path, *, cond: torch.Tensor, tags: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cond": cond.to("cpu"), "tags": tags.to("cpu")}, path)


def _load_image(path: Path | None, size: int) -> torch.Tensor | None:
    if path is None:
        return None
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    if size > 0:
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(image).copy()).to(torch.float32).div(255).unsqueeze(0)


def _tokenize_kwargs(case: str, image: torch.Tensor | None) -> dict:
    if case == "text":
        return {}
    if image is None:
        raise ValueError(f"--image is required for {case}")
    if case == "i2v":
        return {"images": [image]}
    if case == "ref2va":
        return {
            "minimax_ref_items": [
                {"type": "image", "data": image},
                {"type": "image", "data": torch.flip(image, dims=(2,))},
            ]
        }
    if case == "video":
        frames = torch.cat((image, torch.flip(image, dims=(2,))), dim=0)
        return {
            "minimax_ref_items": [
                {"type": "video", "data": frames, "timestamps": [0.0, 0.5]}
            ]
        }
    raise ValueError(f"unsupported probe case: {case}")


def _summary(cond: torch.Tensor, tags: torch.Tensor, *, elapsed_s: float) -> dict:
    return {
        "shape": list(cond.shape),
        "dtype": str(cond.dtype).removeprefix("torch."),
        "finite": bool(torch.isfinite(cond).all().item()),
        "token_count": int(cond.shape[-2]),
        "text_tags": int((tags == 1).sum().item()),
        "vision_tags": int((tags == 0).sum().item()),
        "elapsed_s": elapsed_s,
        "gpu_peak_mib": [torch.cuda.max_memory_allocated(rank) / 2**20 for rank in (0, 1)],
    }


def run_single_gpu(
    checkpoint: Path, prompt: str, tensor_output: Path, tokenize_kwargs: dict
) -> dict:
    import comfy.sd

    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(checkpoint)],
        embedding_directory=[],
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    tokens = clip.tokenize(prompt, **tokenize_kwargs)
    output = clip.encode_from_tokens(tokens, return_dict=True)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    cond = output["cond"].detach().to("cpu")
    tags = output["minimax_token_tags"].detach().to("cpu")
    result = _summary(cond, tags, elapsed_s=elapsed)
    _save_tensor(tensor_output, cond=cond, tags=tags)
    del output, cond, tags, tokens, clip
    try:
        import comfy.model_management

        comfy.model_management.unload_all_models()
    finally:
        gc.collect()
        torch.cuda.empty_cache()
    result["gpu_allocated_after_close_mib"] = [
        torch.cuda.memory_allocated(rank) / 2**20 for rank in (0, 1)
    ]
    return result


def run_tp2(
    checkpoint: Path,
    prompt: str,
    tensor_output: Path,
    reference: Path | None,
    tokenize_kwargs: dict,
) -> dict:
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    from raylight.text_encoder_parallel.clip_proxy import (
        MiniMaxH3CheckpointPreprocessor,
        MiniMaxH3TP2ClipProxy,
        load_minimax_h3_tp2_backbone,
    )

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.reset_peak_memory_stats(1)
    proxy = MiniMaxH3TP2ClipProxy(
        tokenizer=MiniMaxH3Tokenizer(embedding_directory=[]),
        preprocessor_factory=lambda: MiniMaxH3CheckpointPreprocessor(checkpoint),
        backbone_factory=lambda: load_minimax_h3_tp2_backbone(checkpoint),
    )
    started = time.perf_counter()
    tokens = proxy.tokenize(prompt, **tokenize_kwargs)
    output = proxy.encode_from_tokens(tokens, return_dict=True)
    elapsed = time.perf_counter() - started
    cond = output["cond"].detach().to("cpu")
    tags = output["minimax_token_tags"].detach().to("cpu")
    result = _summary(cond, tags, elapsed_s=elapsed)
    _save_tensor(tensor_output, cond=cond, tags=tags)
    if reference is not None:
        expected = torch.load(reference, map_location="cpu", weights_only=True)
        expected_cond = expected["cond"].float()
        if tuple(expected_cond.shape) != tuple(cond.shape):
            raise AssertionError(
                f"conditioning shape mismatch: TP2={tuple(cond.shape)} single={tuple(expected_cond.shape)}"
            )
        cosine = float(
            F.cosine_similarity(cond.double().flatten(), expected_cond.double().flatten(), dim=0).item()
        )
        max_abs = float((cond.float() - expected_cond).abs().max().item())
        tags_equal = torch.equal(tags, expected["tags"])
        result["reference"] = {
            "cosine": cosine,
            "max_abs_error": max_abs,
            "tags_equal": tags_equal,
        }
        if cosine < 0.999 or not tags_equal:
            raise AssertionError(
                f"Gate D numeric failure: cosine={cosine}, max_abs={max_abs}, tags_equal={tags_equal}"
            )
    del output, cond, tags, tokens, proxy
    gc.collect()
    for rank in (0, 1):
        torch.cuda.empty_cache()
    result["gpu_allocated_after_close_mib"] = [
        torch.cuda.memory_allocated(rank) / 2**20 for rank in (0, 1)
    ]
    if any(value > 64 for value in result["gpu_allocated_after_close_mib"]):
        raise AssertionError(f"TP2 GPU allocations were not released: {result['gpu_allocated_after_close_mib']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "tp2"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--case", choices=("text", "i2v", "ref2va", "video"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--tensor-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _install_comfy(args.comfy_root.resolve())
    checkpoint = args.checkpoint.resolve()
    image = _load_image(args.image.resolve() if args.image else None, args.image_size)
    case = args.case or ("i2v" if image is not None else "text")
    tokenize_kwargs = _tokenize_kwargs(case, image)
    if args.mode == "single":
        result = run_single_gpu(
            checkpoint, args.prompt, args.tensor_output.resolve(), tokenize_kwargs
        )
    else:
        result = run_tp2(
            checkpoint,
            args.prompt,
            args.tensor_output.resolve(),
            args.reference.resolve() if args.reference else None,
            tokenize_kwargs,
        )
    payload = {
        "schema_version": 1,
        "mode": args.mode,
        "case": case,
        "checkpoint": checkpoint.name,
        "torch_version": torch.__version__,
        "result": result,
        "gate_d_pass": bool(result["finite"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
