from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from comfy_kitchen.tensor import QuantizedTensor

from raylight.text_encoder_parallel.linear import ColumnParallelLinear, LinearExecutionStats, RowParallelLinear
from raylight.text_encoder_parallel.nvfp4_sharding import QuantizedLinearShard, shard_nvfp4_linear
from raylight.text_encoder_parallel.qwen3vl_tp import (
    MINIMAX_H3_QWEN3VL_CONFIG,
    TP2Qwen3VLBlock,
    TP2Qwen3VLConfig,
    apply_qwen3vl_rope,
    build_qwen3vl_rope,
)


class CpuPairExecutor:
    def run(self, inputs, weights, biases):
        outputs = tuple(F.linear(x, weight, bias) for x, weight, bias in zip(inputs, weights, biases))
        return outputs, self._stats()

    def run_functions(self, functions):
        return tuple(function() for function in functions), self._stats()

    @staticmethod
    def _stats():
        return LinearExecutionStats(1.0, 1.0, 1.5, 0.5, 0.5)


class CpuPeerGroup:
    def __init__(self) -> None:
        self.operation_ids = []

    def all_reduce_sum(self, rank0, rank1, operation_id):
        self.operation_ids.append(operation_id)
        total = rank0 + rank1
        return total, total.clone(), operation_id


class RecordingMathAttention:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, query, key, value, *, is_causal):
        self.calls.append(
            {
                "query_shape": tuple(query.shape),
                "key_shape": tuple(key.shape),
                "value_shape": tuple(value.shape),
                "is_causal": is_causal,
            }
        )
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            enable_gqa=query.shape[1] != key.shape[1],
        )


def _dense_shards(weight: torch.Tensor, axis: str):
    dim = 0 if axis == "row" else 1
    return tuple(
        QuantizedLinearShard(axis, tuple(piece.shape), rank, 2, piece.contiguous())
        for rank, piece in enumerate(weight.chunk(2, dim=dim))
    )


def _quant_shards(weight: torch.Tensor, axis: str):
    quantized = QuantizedTensor.from_float(weight, "TensorCoreNVFP4Layout")
    return quantized, tuple(shard_nvfp4_linear(quantized, axis=axis, rank=rank) for rank in (0, 1))


def _config(*, hidden=16, intermediate=32, heads=4, kv_heads=2, head_dim=4):
    return TP2Qwen3VLConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        rms_norm_eps=1e-6,
        rope_theta=5000000.0,
        rope_dims=None,
        interleaved_mrope=False,
    )


def _make_weights(config, *, quantized=False):
    torch.manual_seed(501)
    shapes = {
        "q": (config.num_attention_heads * config.head_dim, config.hidden_size),
        "k": (config.num_key_value_heads * config.head_dim, config.hidden_size),
        "v": (config.num_key_value_heads * config.head_dim, config.hidden_size),
        "o": (config.hidden_size, config.num_attention_heads * config.head_dim),
        "gate": (config.intermediate_size, config.hidden_size),
        "up": (config.intermediate_size, config.hidden_size),
        "down": (config.hidden_size, config.intermediate_size),
    }
    dtype = torch.float16 if quantized else torch.float32
    full = {name: torch.randn(shape, dtype=dtype) / shape[1] ** 0.5 for name, shape in shapes.items()}
    shard_axes = {"q": "row", "k": "row", "v": "row", "o": "column", "gate": "row", "up": "row", "down": "column"}
    if quantized:
        pairs = {name: _quant_shards(weight, shard_axes[name]) for name, weight in full.items()}
        full = {name: pair[0].dequantize() for name, pair in pairs.items()}
        shards = {name: pair[1] for name, pair in pairs.items()}
    else:
        shards = {name: _dense_shards(weight, shard_axes[name]) for name, weight in full.items()}
    return full, shards


def _make_block(
    config,
    shards,
    executor,
    peer_group,
    norms,
    scales,
    attention_forward,
    *,
    mlp_chunk_tokens=None,
):
    return TP2Qwen3VLBlock(
        config,
        q_proj=ColumnParallelLinear(shards["q"], executor=executor),
        k_proj=ColumnParallelLinear(shards["k"], executor=executor),
        v_proj=ColumnParallelLinear(shards["v"], executor=executor),
        o_proj=RowParallelLinear(
            shards["o"], pre_quant_scale=scales["o"], executor=executor, collective=peer_group
        ),
        gate_proj=ColumnParallelLinear(shards["gate"], executor=executor),
        up_proj=ColumnParallelLinear(shards["up"], executor=executor),
        down_proj=RowParallelLinear(
            shards["down"], pre_quant_scale=scales["down"], executor=executor, collective=peer_group
        ),
        input_layernorm=norms["input"],
        post_attention_layernorm=norms["post"],
        q_norm=norms["q"],
        k_norm=norms["k"],
        attention_forward=attention_forward,
        mlp_chunk_tokens=mlp_chunk_tokens,
    )


def _baseline_block(x, config, weights, norms, scales, position_ids):
    residual = x
    hidden = F.rms_norm(x, (config.hidden_size,), norms["input"][0], config.rms_norm_eps)
    batch, seq, _ = hidden.shape
    q = F.linear(hidden, weights["q"]).view(batch, seq, config.num_attention_heads, config.head_dim).transpose(1, 2)
    k = F.linear(hidden, weights["k"]).view(batch, seq, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    v = F.linear(hidden, weights["v"]).view(batch, seq, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    q = F.rms_norm(q, (config.head_dim,), norms["q"][0], config.rms_norm_eps)
    k = F.rms_norm(k, (config.head_dim,), norms["k"][0], config.rms_norm_eps)
    rope = build_qwen3vl_rope(config, position_ids, device=x.device)
    q, k = apply_qwen3vl_rope(q, k, rope)
    attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    attended = attended.transpose(1, 2).reshape(batch, seq, -1)
    hidden = residual + F.linear(attended * scales["o_full"], weights["o"])
    residual = hidden
    hidden = F.rms_norm(hidden, (config.hidden_size,), norms["post"][0], config.rms_norm_eps)
    mlp = F.silu(F.linear(hidden, weights["gate"])) * F.linear(hidden, weights["up"])
    return residual + F.linear(mlp * scales["down_full"], weights["down"])


def _case(*, quantized=False):
    config = _config(hidden=128, intermediate=256, heads=16, kv_heads=16, head_dim=16) if quantized else _config()
    weights, shards = _make_weights(config, quantized=quantized)
    dtype = torch.float16 if quantized else torch.float32
    torch.manual_seed(502)
    norm_values = {
        "input": torch.randn(config.hidden_size, dtype=dtype).abs() + 0.5,
        "post": torch.randn(config.hidden_size, dtype=dtype).abs() + 0.5,
        "q": torch.randn(config.head_dim, dtype=dtype).abs() + 0.5,
        "k": torch.randn(config.head_dim, dtype=dtype).abs() + 0.5,
    }
    norms = {name: (value, value.clone()) for name, value in norm_values.items()}
    o_full = torch.linspace(0.6, 1.4, config.num_attention_heads * config.head_dim, dtype=dtype)
    down_full = torch.linspace(0.7, 1.3, config.intermediate_size, dtype=dtype)
    scales = {
        "o_full": o_full,
        "down_full": down_full,
        "o": tuple(part.contiguous() for part in o_full.chunk(2)),
        "down": tuple(part.contiguous() for part in down_full.chunk(2)),
    }
    executor, peer_group = CpuPairExecutor(), CpuPeerGroup()
    attention_forward = RecordingMathAttention()
    block = _make_block(config, shards, executor, peer_group, norms, scales, attention_forward)
    return config, weights, norms, scales, block, peer_group, attention_forward


def test_freezes_minimax_h3_qwen3vl_32b_contract() -> None:
    assert MINIMAX_H3_QWEN3VL_CONFIG.hidden_size == 5120
    assert MINIMAX_H3_QWEN3VL_CONFIG.intermediate_size == 25600
    assert MINIMAX_H3_QWEN3VL_CONFIG.num_attention_heads == 64
    assert MINIMAX_H3_QWEN3VL_CONFIG.num_key_value_heads == 8
    assert MINIMAX_H3_QWEN3VL_CONFIG.head_dim == 128
    assert MINIMAX_H3_QWEN3VL_CONFIG.local_attention_heads == 32
    assert MINIMAX_H3_QWEN3VL_CONFIG.local_key_value_heads == 4


def test_fp32_tp2_block_matches_full_block_and_uses_exactly_two_sums() -> None:
    config, weights, norms, scales, block, peer_group, attention_forward = _case()
    x = torch.randn((1, 7, config.hidden_size), dtype=torch.float32)
    position_ids = torch.arange(x.shape[1]).unsqueeze(0)

    actual = block.forward(x, x.clone(), position_ids, operation_id=11)
    expected = _baseline_block(x, config, weights, norms, scales, position_ids)

    assert torch.max(torch.abs(actual[0] - expected)).item() <= 1e-4
    assert torch.equal(actual[0], actual[1])
    assert peer_group.operation_ids == [11, 12]
    assert len(attention_forward.calls) == 2
    assert all(call["is_causal"] is True for call in attention_forward.calls)
    assert all(len(call["query_shape"]) == 4 for call in attention_forward.calls)


def test_chunked_mlp_matches_full_block_and_assigns_one_collective_per_chunk() -> None:
    config, weights, norms, scales, _block, peer_group, attention_forward = _case()
    _full, shards = _make_weights(config)
    block = _make_block(
        config,
        shards,
        CpuPairExecutor(),
        peer_group,
        norms,
        scales,
        attention_forward,
        mlp_chunk_tokens=4,
    )
    x = torch.randn((1, 10, config.hidden_size), dtype=torch.float32)
    position_ids = torch.arange(x.shape[1]).unsqueeze(0)

    actual = block.forward(x, x.clone(), position_ids, operation_id=11)
    expected = _baseline_block(x, config, weights, norms, scales, position_ids)

    assert torch.max(torch.abs(actual[0] - expected)).item() <= 1e-4
    assert torch.equal(actual[0], actual[1])
    assert peer_group.operation_ids == [11, 12, 13, 14]
    assert block.collective_count(10) == 4


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.5])
def test_mlp_chunk_tokens_must_be_a_positive_integer(invalid) -> None:
    config, _weights, norms, scales, _block, peer_group, attention_forward = _case()
    _full, shards = _make_weights(config)

    with pytest.raises(ValueError, match="mlp_chunk_tokens"):
        _make_block(
            config,
            shards,
            CpuPairExecutor(),
            peer_group,
            norms,
            scales,
            attention_forward,
            mlp_chunk_tokens=invalid,
        )


def test_nvfp4_tp2_block_meets_cosine_and_finite_gate() -> None:
    config, weights, norms, scales, block, peer_group, attention_forward = _case(quantized=True)
    x = torch.randn((1, 5, config.hidden_size), dtype=torch.float16)
    position_ids = torch.arange(x.shape[1]).unsqueeze(0)

    actual = block.forward(x, x.clone(), position_ids, operation_id=21)
    expected = _baseline_block(x, config, weights, norms, scales, position_ids)
    cosine = F.cosine_similarity(actual[0].float().flatten(), expected.float().flatten(), dim=0)

    assert torch.isfinite(actual[0]).all()
    assert torch.equal(actual[0], actual[1])
    assert cosine >= 0.9995
    assert peer_group.operation_ids == [21, 22]
    assert len(attention_forward.calls) == 2


@pytest.mark.parametrize("depth", [2, 8, 50])
def test_fp32_synthetic_stack_keeps_ranks_order_and_error_bounded(depth: int) -> None:
    config, weights, norms, scales, block, peer_group, attention_forward = _case()
    actual = torch.randn((1, 3, config.hidden_size), dtype=torch.float32) * 0.1
    expected = actual.clone()
    position_ids = torch.arange(actual.shape[1]).unsqueeze(0)

    ranks = (actual, actual.clone())
    for layer_index in range(depth):
        ranks = block.forward(ranks[0], ranks[1], position_ids, operation_id=layer_index * 2 + 1)
        expected = _baseline_block(expected, config, weights, norms, scales, position_ids)

    max_error = torch.max(torch.abs(ranks[0] - expected)).item()
    assert torch.isfinite(ranks[0]).all()
    assert torch.equal(ranks[0], ranks[1])
    assert max_error <= 1e-3
    assert peer_group.operation_ids == list(range(1, depth * 2 + 1))
    assert len(attention_forward.calls) == depth * 2
