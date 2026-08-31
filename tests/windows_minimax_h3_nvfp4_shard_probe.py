"""Gate B probe for real MiniMax H3 Qwen3-VL NVFP4 linear weights."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout
from safetensors import safe_open

from raylight.text_encoder_parallel.nvfp4_sharding import shard_nvfp4_linear

LAYERS = (
    ("q_proj", "model.layers.0.self_attn.q_proj", (8192, 5120), "row", 0),
    ("o_proj", "model.layers.0.self_attn.o_proj", (5120, 8192), "column", 1),
    ("up_proj", "model.layers.0.mlp.up_proj", (25600, 5120), "row", 0),
    ("down_proj", "model.layers.0.mlp.down_proj", (5120, 25600), "column", 1),
)


def _load_weight(handle, prefix: str, logical_shape: tuple[int, int]) -> QuantizedTensor:
    qdata = handle.get_tensor(f"{prefix}.weight")
    block_scale = handle.get_tensor(f"{prefix}.weight_scale")
    scale = handle.get_tensor(f"{prefix}.weight_scale_2")
    params = TensorCoreNVFP4Layout.Params(
        scale=scale,
        block_scale=block_scale,
        orig_dtype=torch.float16,
        orig_shape=logical_shape,
        transposed=False,
    )
    return QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params)


def run_probe(checkpoint: Path) -> dict:
    results = []
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for label, prefix, logical_shape, axis, cat_dim in LAYERS:
            weight = _load_weight(handle, prefix, logical_shape)

            shard_started = time.perf_counter()
            shards = [shard_nvfp4_linear(weight, axis=axis, rank=rank) for rank in (0, 1)]
            shard_seconds = time.perf_counter() - shard_started

            full_started = time.perf_counter()
            expected = weight.dequantize()
            full_dequant_seconds = time.perf_counter() - full_started
            rank_results = []
            for rank, shard in enumerate(shards):
                actual = shard.qweight.dequantize()
                expected_slice = expected.chunk(2, dim=cat_dim)[rank]
                equal = torch.equal(actual, expected_slice)
                finite = bool(torch.isfinite(actual).all().item())
                rank_results.append(
                    {
                        "rank": rank,
                        "logical_shape": list(shard.logical_shape),
                        "storage_shape": list(shard.qweight._qdata.shape),
                        "element_exact": equal,
                        "finite": finite,
                    }
                )
                del actual, expected_slice

            result = {
                "label": label,
                "key": prefix,
                "axis": axis,
                "logical_shape": list(logical_shape),
                "source_qdata_bytes": weight._qdata.nbytes,
                "source_block_scale_bytes": weight._params.block_scale.nbytes,
                "shard_qdata_bytes_total": sum(shard.qweight._qdata.nbytes for shard in shards),
                "shard_block_scale_bytes_total": sum(shard.qweight._params.block_scale.nbytes for shard in shards),
                "shard_seconds": shard_seconds,
                "full_dequant_seconds": full_dequant_seconds,
                "ranks": rank_results,
            }
            if not all(rank["element_exact"] and rank["finite"] for rank in rank_results):
                raise AssertionError(f"{label} failed NVFP4 shard reconstruction: {rank_results}")
            if result["shard_qdata_bytes_total"] != result["source_qdata_bytes"]:
                raise AssertionError(f"{label} duplicated or lost packed qdata")
            if result["shard_block_scale_bytes_total"] != result["source_block_scale_bytes"]:
                raise AssertionError(f"{label} duplicated or lost block scales")
            results.append(result)
            del expected, shards, weight
            gc.collect()

    return {
        "schema_version": 1,
        "checkpoint": checkpoint.name,
        "torch_version": torch.__version__,
        "layers": results,
        "gate_b_pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
