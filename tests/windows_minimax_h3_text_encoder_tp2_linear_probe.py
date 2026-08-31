"""Dual-V100 probe for O10 Task 4 tensor-parallel Linear primitives."""

from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import comfy_kitchen as ck
import torch
import torch.nn.functional as F
from comfy_kitchen.tensor import QuantizedTensor, TensorCoreNVFP4Layout
from safetensors import safe_open

from raylight.text_encoder_parallel.linear import ColumnParallelLinear, RowParallelLinear
from raylight.text_encoder_parallel.nvfp4_sharding import QuantizedLinearShard, shard_nvfp4_linear
from raylight.text_encoder_parallel.topology import validate_windows_v100_topology


def _cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item())


def _cuda_shards(shards):
    return tuple(replace(shard, qweight=shard.qweight.to(f"cuda:{rank}")) for rank, shard in enumerate(shards))


def _stats_dict(stats) -> dict:
    return {
        "rank0_ms": stats.rank0_ms,
        "rank1_ms": stats.rank1_ms,
        "wall_ms": stats.wall_ms,
        "overlap_ms": stats.overlap_ms,
        "overlap_ratio": stats.overlap_ratio,
    }


def _run_dense_overlap_probe() -> dict:
    torch.manual_seed(101)
    local_out, in_features, tokens = 4096, 4096, 2048
    weights = (
        torch.randn((local_out, in_features), device="cuda:0", dtype=torch.float32),
        torch.randn((local_out, in_features), device="cuda:1", dtype=torch.float32),
    )
    shards = tuple(
        QuantizedLinearShard("row", tuple(weight.shape), rank, 2, weight)
        for rank, weight in enumerate(weights)
    )
    cpu_input = torch.randn((tokens, in_features), dtype=torch.float32)
    inputs = (cpu_input.to("cuda:0"), cpu_input.to("cuda:1"))
    layer = ColumnParallelLinear(shards)
    layer.forward(inputs)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    layer.forward(inputs)
    stats = layer.last_execution
    if stats is None or stats.overlap_ms <= 0.0 or stats.overlap_ratio <= 0.0:
        raise AssertionError(f"dense local Linear kernels did not overlap: {stats}")
    return {
        "dtype": "float32",
        "tokens": tokens,
        "local_weight_shape": [local_out, in_features],
        **_stats_dict(stats),
    }


def _run_nvfp4_column_probe() -> dict:
    torch.manual_seed(202)
    source = torch.randn((256, 128), dtype=torch.float16)
    weight = QuantizedTensor.from_float(source, "TensorCoreNVFP4Layout")
    shards = _cuda_shards(tuple(shard_nvfp4_linear(weight, axis="row", rank=rank) for rank in (0, 1)))
    x = torch.randn((17, 128), dtype=torch.float16)
    expected = F.linear(x, weight.dequantize())
    layer = ColumnParallelLinear(shards)
    layer.forward((x.to("cuda:0"), x.to("cuda:1")))
    outputs = layer.forward((x.to("cuda:0"), x.to("cuda:1")))
    actual = torch.cat(tuple(output.cpu() for output in outputs), dim=-1)
    cosine = _cosine(actual, expected)
    finite = bool(torch.isfinite(actual).all().item())
    if cosine < 0.9995 or not finite:
        raise AssertionError(f"NVFP4 column Linear failed: cosine={cosine}, finite={finite}")
    return {
        "axis": "row/output-channel",
        "logical_shape": [256, 128],
        "cosine": cosine,
        "max_abs_error": float(torch.max(torch.abs(actual.float() - expected.float())).item()),
        "finite": finite,
        "execution": _stats_dict(layer.last_execution),
    }


def _run_nvfp4_row_probe() -> dict:
    torch.manual_seed(303)
    source = torch.randn((128, 256), dtype=torch.float16)
    weight = QuantizedTensor.from_float(source, "TensorCoreNVFP4Layout")
    shards = _cuda_shards(tuple(shard_nvfp4_linear(weight, axis="column", rank=rank) for rank in (0, 1)))
    x = torch.randn((19, 256), dtype=torch.float16)
    expected = F.linear(x, weight.dequantize())
    input_parts = x.chunk(2, dim=-1)
    layer = RowParallelLinear(shards)
    cuda_inputs = (input_parts[0].contiguous().to("cuda:0"), input_parts[1].contiguous().to("cuda:1"))
    layer.forward(cuda_inputs, operation_id=1)
    outputs = layer.forward(
        cuda_inputs,
        operation_id=2,
    )
    actual0, actual1 = outputs[0].cpu(), outputs[1].cpu()
    cosine = _cosine(actual0, expected)
    finite = bool(torch.isfinite(actual0).all().item() and torch.isfinite(actual1).all().item())
    ranks_equal = torch.equal(actual0, actual1)
    if cosine < 0.9995 or not finite or not ranks_equal:
        raise AssertionError(
            f"NVFP4 row Linear failed: cosine={cosine}, finite={finite}, ranks_equal={ranks_equal}"
        )
    return {
        "axis": "column/input-channel",
        "logical_shape": [128, 256],
        "cosine": cosine,
        "max_abs_error": float(torch.max(torch.abs(actual0.float() - expected.float())).item()),
        "finite": finite,
        "ranks_equal": ranks_equal,
        "execution": _stats_dict(layer.last_execution),
        "collective": {
            "bytes_sent": layer.last_collective.bytes_sent,
            "elapsed_ms": layer.last_collective.elapsed_ms,
            "operation_id": layer.last_collective.operation_id,
        },
    }


def _run_real_q_proj_overlap_probe(checkpoint: Path) -> dict:
    prefix = "model.layers.0.self_attn.q_proj"
    logical_shape = (8192, 5120)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        params = TensorCoreNVFP4Layout.Params(
            scale=handle.get_tensor(f"{prefix}.weight_scale_2"),
            block_scale=handle.get_tensor(f"{prefix}.weight_scale"),
            orig_dtype=torch.float16,
            orig_shape=logical_shape,
            transposed=False,
        )
        weight = QuantizedTensor(handle.get_tensor(f"{prefix}.weight"), "TensorCoreNVFP4Layout", params)
    shards = _cuda_shards(tuple(shard_nvfp4_linear(weight, axis="row", rank=rank) for rank in (0, 1)))
    x = torch.randn((221, logical_shape[1]), dtype=torch.float16)
    inputs = (x.to("cuda:0"), x.to("cuda:1"))
    layer = ColumnParallelLinear(shards)
    layer.forward(inputs)
    outputs = layer.forward(inputs)
    stats = layer.last_execution
    finite = bool(torch.isfinite(outputs[0]).all().item() and torch.isfinite(outputs[1]).all().item())
    if stats is None or stats.overlap_ms <= 0.0 or stats.overlap_ratio <= 0.0 or not finite:
        raise AssertionError(f"real q_proj local Linear kernels did not overlap or produced non-finite output: {stats}")
    return {
        "key": prefix,
        "tokens": 221,
        "logical_shape": list(logical_shape),
        "local_shape": list(shards[0].logical_shape),
        "finite": finite,
        **_stats_dict(stats),
    }


def run_probe(checkpoint: Path) -> dict:
    ck.registry.disable("cuda")
    ck.registry.disable("triton")
    validate_windows_v100_topology(torch_module=torch, devices=(0, 1))
    result = {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "devices": [torch.cuda.get_device_name(0), torch.cuda.get_device_name(1)],
        "dense_overlap": _run_dense_overlap_probe(),
        "real_nvfp4_q_proj_overlap": _run_real_q_proj_overlap_probe(checkpoint),
        "nvfp4_column": _run_nvfp4_column_probe(),
        "nvfp4_row": _run_nvfp4_row_probe(),
        "task4_pass": True,
    }
    torch.cuda.empty_cache()
    return result


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
