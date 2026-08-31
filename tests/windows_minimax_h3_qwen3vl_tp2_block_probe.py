"""Gate C probe: real MiniMax H3 Qwen3-VL layer 0 on dual V100 TP2."""

from __future__ import annotations

from dataclasses import replace
import argparse
import gc
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

from raylight.text_encoder_parallel.collectives import CudaPeerGroup2
from raylight.text_encoder_parallel.linear import ColumnParallelLinear, RowParallelLinear, TorchDualCudaLinearExecutor
from raylight.text_encoder_parallel.nvfp4_sharding import shard_nvfp4_linear
from raylight.text_encoder_parallel.qwen3vl_tp import (
    MINIMAX_H3_QWEN3VL_CONFIG,
    TP2Qwen3VLBlock,
    apply_qwen3vl_rope,
    build_qwen3vl_rope,
)
from raylight.text_encoder_parallel.topology import validate_windows_v100_topology

PREFIXES = {
    "q": "model.layers.0.self_attn.q_proj",
    "k": "model.layers.0.self_attn.k_proj",
    "v": "model.layers.0.self_attn.v_proj",
    "o": "model.layers.0.self_attn.o_proj",
    "gate": "model.layers.0.mlp.gate_proj",
    "up": "model.layers.0.mlp.up_proj",
    "down": "model.layers.0.mlp.down_proj",
}
SHAPES = {
    "q": (8192, 5120),
    "k": (1024, 5120),
    "v": (1024, 5120),
    "o": (5120, 8192),
    "gate": (25600, 5120),
    "up": (25600, 5120),
    "down": (5120, 25600),
}
AXES = {"q": "row", "k": "row", "v": "row", "o": "column", "gate": "row", "up": "row", "down": "column"}


def _load_qweight(handle, name: str) -> QuantizedTensor:
    prefix = PREFIXES[name]
    params = TensorCoreNVFP4Layout.Params(
        scale=handle.get_tensor(f"{prefix}.weight_scale_2"),
        block_scale=handle.get_tensor(f"{prefix}.weight_scale"),
        orig_dtype=torch.float16,
        orig_shape=SHAPES[name],
        transposed=False,
    )
    return QuantizedTensor(handle.get_tensor(f"{prefix}.weight"), "TensorCoreNVFP4Layout", params)


def _rms(x, weight):
    return F.rms_norm(x, (x.shape[-1],), weight.to(device=x.device, dtype=x.dtype), MINIMAX_H3_QWEN3VL_CONFIG.rms_norm_eps)


def _single_gpu_baseline(handle, weights, hidden, position_ids, mask, norms, scales):
    config = MINIMAX_H3_QWEN3VL_CONFIG

    def linear(name, x, pre_scale=None):
        if pre_scale is not None:
            x = x * pre_scale.to(device=x.device, dtype=x.dtype)
        cuda_weight = weights[name].to("cuda:0")
        output = F.linear(x, cuda_weight)
        torch.cuda.synchronize(0)
        del cuda_weight
        torch.cuda.empty_cache()
        return output

    residual = hidden
    x = _rms(hidden, norms["input"])
    batch, sequence = x.shape[:2]
    q = linear("q", x).view(batch, sequence, config.num_attention_heads, config.head_dim).transpose(1, 2)
    k = linear("k", x).view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    v = linear("v", x).view(batch, sequence, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    q = _rms(q, norms["q"])
    k = _rms(k, norms["k"])
    rope = build_qwen3vl_rope(config, position_ids, device=hidden.device)
    q, k = apply_qwen3vl_rope(q, k, rope)
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
    x = residual + linear("o", attended, scales["o"])
    residual = x
    x = _rms(x, norms["post"])
    x = F.silu(linear("gate", x)) * linear("up", x)
    return residual + linear("down", x, scales["down"])


def _cuda_shards(weight, axis):
    return tuple(
        replace(shard, qweight=shard.qweight.to(f"cuda:{rank}"))
        for rank in (0, 1)
        for shard in (shard_nvfp4_linear(weight, axis=axis, rank=rank),)
    )


def _stats(stats):
    return {
        "rank0_ms": stats.rank0_ms,
        "rank1_ms": stats.rank1_ms,
        "wall_ms": stats.wall_ms,
        "overlap_ms": stats.overlap_ms,
        "overlap_ratio": stats.overlap_ratio,
    }


def run_probe(checkpoint: Path, *, tokens: int) -> dict:
    ck.registry.disable("cuda")
    ck.registry.disable("triton")
    validate_windows_v100_topology(torch_module=torch, devices=(0, 1))
    torch.manual_seed(601)
    config = MINIMAX_H3_QWEN3VL_CONFIG

    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        weights = {name: _load_qweight(handle, name) for name in PREFIXES}
        norms = {
            "input": handle.get_tensor("model.layers.0.input_layernorm.weight").to(torch.float16),
            "post": handle.get_tensor("model.layers.0.post_attention_layernorm.weight").to(torch.float16),
            "q": handle.get_tensor("model.layers.0.self_attn.q_norm.weight").to(torch.float16),
            "k": handle.get_tensor("model.layers.0.self_attn.k_norm.weight").to(torch.float16),
        }
        scales = {
            "o": handle.get_tensor("model.layers.0.self_attn.o_proj.pre_quant_scale").to(torch.float16),
            "down": handle.get_tensor("model.layers.0.mlp.down_proj.pre_quant_scale").to(torch.float16),
        }

        hidden_cpu = torch.randn((1, tokens, config.hidden_size), dtype=torch.float16) * 0.05
        hidden0 = hidden_cpu.to("cuda:0")
        position_ids = torch.arange(tokens).unsqueeze(0)
        mask0 = torch.full((tokens, tokens), torch.finfo(torch.float16).min / 4, device="cuda:0", dtype=torch.float16).triu_(1)
        expected = _single_gpu_baseline(handle, weights, hidden0, position_ids, mask0, norms, scales).cpu()
        del hidden0, mask0
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.reset_peak_memory_stats(1)
        shards = {name: _cuda_shards(weight, AXES[name]) for name, weight in weights.items()}
        executor = TorchDualCudaLinearExecutor()
        collective = CudaPeerGroup2()
        norm_pairs = {
            name: (value.to("cuda:0"), value.to("cuda:1")) for name, value in norms.items()
        }
        scale_pairs = {
            name: tuple(part.contiguous().to(f"cuda:{rank}") for rank, part in enumerate(value.chunk(2)))
            for name, value in scales.items()
        }
        block = TP2Qwen3VLBlock(
            config,
            q_proj=ColumnParallelLinear(shards["q"], executor=executor),
            k_proj=ColumnParallelLinear(shards["k"], executor=executor),
            v_proj=ColumnParallelLinear(shards["v"], executor=executor),
            o_proj=RowParallelLinear(
                shards["o"], pre_quant_scale=scale_pairs["o"], executor=executor, collective=collective
            ),
            gate_proj=ColumnParallelLinear(shards["gate"], executor=executor),
            up_proj=ColumnParallelLinear(shards["up"], executor=executor),
            down_proj=RowParallelLinear(
                shards["down"], pre_quant_scale=scale_pairs["down"], executor=executor, collective=collective
            ),
            input_layernorm=norm_pairs["input"],
            post_attention_layernorm=norm_pairs["post"],
            q_norm=norm_pairs["q"],
            k_norm=norm_pairs["k"],
        )
        hidden = (hidden_cpu.to("cuda:0"), hidden_cpu.to("cuda:1"))
        mask = torch.full((tokens, tokens), torch.finfo(torch.float16).min / 4, dtype=torch.float16).triu_(1)
        block.forward(hidden[0], hidden[1], position_ids, attention_mask=mask, operation_id=1)
        calls_before = collective.profile_snapshot()["calls"]
        actual = block.forward(hidden[0], hidden[1], position_ids, attention_mask=mask, operation_id=3)
        calls_after = collective.profile_snapshot()["calls"]
        actual0, actual1 = actual[0].cpu(), actual[1].cpu()

        cosine = float(F.cosine_similarity(actual0.float().flatten(), expected.float().flatten(), dim=0).item())
        max_abs = float(torch.max(torch.abs(actual0.float() - expected.float())).item())
        finite = bool(torch.isfinite(actual0).all().item() and torch.isfinite(actual1).all().item())
        ranks_equal = torch.equal(actual0, actual1)
        if cosine < 0.9995 or not finite or not ranks_equal:
            raise AssertionError(
                f"Gate C numeric failure: cosine={cosine}, max_abs={max_abs}, finite={finite}, ranks_equal={ranks_equal}"
            )
        if calls_after - calls_before != 2:
            raise AssertionError(f"Gate C expected exactly two P2P SUM calls, got {calls_after - calls_before}")

        linear_stats = {
            name: _stats(getattr(block, name).last_execution)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        }
        result = {
            "schema_version": 1,
            "checkpoint": checkpoint.name,
            "tokens": tokens,
            "torch_version": torch.__version__,
            "numeric": {
                "cosine": cosine,
                "max_abs_error": max_abs,
                "finite": finite,
                "ranks_equal": ranks_equal,
            },
            "collective_calls_measured_block": calls_after - calls_before,
            "o_proj_operation_id": block.o_proj.last_collective.operation_id,
            "down_proj_operation_id": block.down_proj.last_collective.operation_id,
            "linear_stats": linear_stats,
            "gpu_peak_mib": [
                torch.cuda.max_memory_allocated(0) / (1024**2),
                torch.cuda.max_memory_allocated(1) / (1024**2),
            ],
            "gate_c_pass": True,
        }
        executor.close()
        collective.close()
        del block, shards, hidden, actual
        gc.collect()
        torch.cuda.empty_cache()
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_probe(args.checkpoint.resolve(), tokens=args.tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
