from __future__ import annotations

import torch
import torch.nn.functional as F
from comfy_kitchen.tensor import QuantizedTensor

from raylight.text_encoder_parallel.linear import (
    ColumnParallelLinear,
    LinearExecutionStats,
    RowParallelLinear,
)
from raylight.text_encoder_parallel.nvfp4_sharding import QuantizedLinearShard, shard_nvfp4_linear


class CpuPairExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, inputs, weights, biases):
        self.calls += 1
        outputs = tuple(F.linear(x, weight, bias) for x, weight, bias in zip(inputs, weights, biases))
        return outputs, LinearExecutionStats(
            rank0_ms=2.0,
            rank1_ms=3.0,
            wall_ms=3.5,
            overlap_ms=1.5,
            overlap_ratio=0.75,
        )


class CpuPeerGroup:
    def __init__(self) -> None:
        self.operation_ids = []

    def all_reduce_sum(self, rank0, rank1, operation_id):
        self.operation_ids.append(operation_id)
        result = rank0 + rank1
        return result, result.clone(), {"operation_id": operation_id}


def _dense_shards(weight: torch.Tensor, axis: str) -> tuple[QuantizedLinearShard, QuantizedLinearShard]:
    dim = 0 if axis == "row" else 1
    pieces = weight.chunk(2, dim=dim)
    return tuple(
        QuantizedLinearShard(
            axis=axis,
            logical_shape=tuple(piece.shape),
            rank=rank,
            world_size=2,
            qweight=piece.contiguous(),
        )
        for rank, piece in enumerate(pieces)
    )


def test_dense_column_parallel_matches_full_linear() -> None:
    torch.manual_seed(10)
    x = torch.randn((7, 12), dtype=torch.float32)
    weight = torch.randn((16, 12), dtype=torch.float32)
    bias = torch.randn((16,), dtype=torch.float32)
    executor = CpuPairExecutor()
    layer = ColumnParallelLinear(
        _dense_shards(weight, "row"),
        bias=tuple(piece.contiguous() for piece in bias.chunk(2)),
        executor=executor,
    )

    outputs = layer.forward((x, x.clone()))

    actual = torch.cat(outputs, dim=-1)
    expected = F.linear(x, weight, bias)
    assert torch.max(torch.abs(actual - expected)).item() <= 1e-4
    assert executor.calls == 1
    assert layer.last_execution is not None
    assert layer.last_execution.overlap_ratio == 0.75


def test_dense_row_parallel_matches_full_linear_and_uses_p2p_sum() -> None:
    torch.manual_seed(20)
    x = torch.randn((5, 12), dtype=torch.float32)
    weight = torch.randn((8, 12), dtype=torch.float32)
    bias = torch.randn((8,), dtype=torch.float32)
    peer_group = CpuPeerGroup()
    layer = RowParallelLinear(
        _dense_shards(weight, "column"),
        bias=(bias, bias.clone()),
        executor=CpuPairExecutor(),
        collective=peer_group,
    )

    outputs = layer.forward(tuple(part.contiguous() for part in x.chunk(2, dim=-1)), operation_id=17)

    expected = F.linear(x, weight, bias)
    assert torch.allclose(outputs[0], expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(outputs[0], outputs[1])
    assert peer_group.operation_ids == [17]
    assert layer.last_collective == {"operation_id": 17}


def test_nvfp4_column_parallel_meets_cosine_and_finite_gate() -> None:
    torch.manual_seed(30)
    source = torch.randn((256, 128), dtype=torch.float16)
    weight = QuantizedTensor.from_float(source, "TensorCoreNVFP4Layout")
    shards = tuple(shard_nvfp4_linear(weight, axis="row", rank=rank) for rank in (0, 1))
    x = torch.randn((11, 128), dtype=torch.float16)
    layer = ColumnParallelLinear(shards, executor=CpuPairExecutor())

    actual = torch.cat(layer.forward((x, x.clone())), dim=-1)
    expected = F.linear(x, weight.dequantize())

    assert torch.isfinite(actual).all()
    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) >= 0.9995


def test_nvfp4_row_parallel_meets_cosine_and_finite_gate() -> None:
    torch.manual_seed(40)
    source = torch.randn((128, 256), dtype=torch.float16)
    weight = QuantizedTensor.from_float(source, "TensorCoreNVFP4Layout")
    shards = tuple(shard_nvfp4_linear(weight, axis="column", rank=rank) for rank in (0, 1))
    x = torch.randn((13, 256), dtype=torch.float16)
    layer = RowParallelLinear(
        shards,
        executor=CpuPairExecutor(),
        collective=CpuPeerGroup(),
    )

    actual = layer.forward(tuple(part.contiguous() for part in x.chunk(2, dim=-1)), operation_id=1)[0]
    expected = F.linear(x, weight.dequantize())

    assert torch.isfinite(actual).all()
    assert F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) >= 0.9995


def test_rejects_mismatched_rank_axis_and_world_size() -> None:
    weight = torch.ones((8, 8), dtype=torch.float32)
    row_shards = list(_dense_shards(weight, "row"))
    row_shards[1] = QuantizedLinearShard("column", (4, 8), 1, 2, row_shards[1].qweight)

    try:
        ColumnParallelLinear(tuple(row_shards), executor=CpuPairExecutor())
    except ValueError as error:
        assert "axis" in str(error)
    else:
        raise AssertionError("mismatched shard axes must be rejected")
