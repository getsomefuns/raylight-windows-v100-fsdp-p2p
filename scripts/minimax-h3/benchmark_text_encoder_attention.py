"""Measure MiniMax H3 TP2-local attention capacity on one CUDA device."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raylight.text_encoder_parallel.attention_backends import (  # noqa: E402
    TextEncoderAttentionBackend,
    resolve_attention_backend,
    run_text_encoder_attention,
)


Q_HEADS = 32
KV_HEADS = 4
HEAD_DIM = 128
MIB = 1024**2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        required=True,
        choices=[backend.value for backend in TextEncoderAttentionBackend],
    )
    parser.add_argument("--sequence", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--json-output", type=Path)
    return parser


def benchmark_attention(
    backend: str,
    sequence: int,
    *,
    device: str = "cuda:0",
    dtype: str = "float32",
) -> dict[str, object]:
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise ValueError("sequence must be a positive integer")
    requested = TextEncoderAttentionBackend(str(backend).strip().upper())
    if requested is TextEncoderAttentionBackend.MATH_DEBUG and sequence > 2048:
        raise ValueError("MATH_DEBUG is limited to 2048 tokens")
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise ValueError("attention capacity probe requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    selected = resolve_attention_backend(requested, probe_device=cuda_device)
    try:
        torch_dtype = {"float32": torch.float32, "float16": torch.float16}[dtype]
    except KeyError as exc:
        raise ValueError("dtype must be float32 or float16") from exc
    torch.cuda.empty_cache()
    query = key = value = output = None
    try:
        query = torch.randn(1, Q_HEADS, sequence, HEAD_DIM, device=cuda_device, dtype=torch_dtype)
        key = torch.randn(1, KV_HEADS, sequence, HEAD_DIM, device=cuda_device, dtype=torch_dtype)
        value = torch.randn(1, KV_HEADS, sequence, HEAD_DIM, device=cuda_device, dtype=torch_dtype)
        input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in (query, key, value))
        torch.cuda.synchronize(cuda_device)
        # Fixture construction can transiently allocate random-number workspaces.
        # Gate B measures the attention call with its live Q/K/V inputs, not that setup noise.
        torch.cuda.reset_peak_memory_stats(cuda_device)
        started = time.perf_counter()
        output = run_text_encoder_attention(query, key, value, backend=selected, is_causal=True)
        torch.cuda.synchronize(cuda_device)
        elapsed = time.perf_counter() - started
        finite = bool(torch.isfinite(output).all().item())
        peak_mib = torch.cuda.max_memory_allocated(cuda_device) / MIB
        return {
            "backend": selected.value,
            "requested_backend": requested.value,
            "sequence": sequence,
            "q_heads": Q_HEADS,
            "kv_heads": KV_HEADS,
            "head_dim": HEAD_DIM,
            "dtype": str(torch_dtype).removeprefix("torch."),
            "input_mib": input_bytes / MIB,
            "peak_mib": peak_mib,
            "elapsed_seconds": elapsed,
            "finite": finite,
            "math_fallback_detected": (
                selected is TextEncoderAttentionBackend.MATH_DEBUG
                and requested is not TextEncoderAttentionBackend.MATH_DEBUG
            ),
        }
    finally:
        del output, value, key, query
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = benchmark_attention(args.backend, args.sequence, device=args.device, dtype=args.dtype)
    encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
