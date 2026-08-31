from __future__ import annotations

import pytest
import torch
from comfy_kitchen.tensor import QuantizedTensor

from raylight.text_encoder_parallel.nvfp4_sharding import (
    Nvfp4ShardError,
    nvfp4_shard_plan,
    shard_nvfp4_linear,
)


@pytest.mark.parametrize(
    ("logical_shape", "axis", "expected_shape"),
    [
        ((8192, 5120), "row", (4096, 5120)),  # q_proj
        ((5120, 8192), "column", (5120, 4096)),  # o_proj
        ((25600, 5120), "row", (12800, 5120)),  # up_proj
        ((5120, 25600), "column", (5120, 12800)),  # down_proj
    ],
)
def test_plans_representative_minimax_h3_linear_shards(
    logical_shape: tuple[int, int],
    axis: str,
    expected_shape: tuple[int, int],
) -> None:
    plans = [nvfp4_shard_plan(logical_shape, axis=axis, rank=rank) for rank in (0, 1)]

    assert plans[0].logical_shape == expected_shape
    assert plans[1].logical_shape == expected_shape
    assert plans[0].qdata_elements + plans[1].qdata_elements == logical_shape[0] * logical_shape[1] // 2


@pytest.mark.parametrize(
    ("shape", "axis", "cat_dim"),
    [
        ((256, 128), "row", 0),
        ((128, 256), "column", 1),
    ],
)
def test_shards_reconstruct_full_nvfp4_dequantization_exactly(
    shape: tuple[int, int],
    axis: str,
    cat_dim: int,
) -> None:
    source = torch.linspace(-3.0, 3.0, shape[0] * shape[1], dtype=torch.float16).reshape(shape)
    weight = QuantizedTensor.from_float(source, "TensorCoreNVFP4Layout")
    expected = weight.dequantize()

    shards = [shard_nvfp4_linear(weight, axis=axis, rank=rank) for rank in (0, 1)]
    reconstructed = torch.cat([shard.qweight.dequantize() for shard in shards], dim=cat_dim)

    assert shards[0].logical_shape == tuple(size // 2 if dim == cat_dim else size for dim, size in enumerate(shape))
    assert tuple(shards[0].qweight.shape) == shards[0].logical_shape
    assert torch.equal(reconstructed, expected)
    assert all(torch.isfinite(shard.qweight.dequantize()).all() for shard in shards)


def test_sharding_never_dequantizes_the_full_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.arange(256 * 128, dtype=torch.float16).reshape(256, 128)
    weight = QuantizedTensor.from_float(source, "TensorCoreNVFP4Layout")

    def forbidden_dequantize(self):
        raise AssertionError("full dequantization is forbidden while creating a shard")

    monkeypatch.setattr(QuantizedTensor, "dequantize", forbidden_dequantize)
    shard = shard_nvfp4_linear(weight, axis="row", rank=0)

    assert shard.qweight._qdata.numel() == weight._qdata.numel() // 2
    assert shard.qweight._params.scale.data_ptr() == weight._params.scale.data_ptr()


@pytest.mark.parametrize(
    ("logical_shape", "axis", "message"),
    [
        ((192, 128), "row", "128-row"),
        ((128, 192), "column", "64-column"),
        ((128, 129), "column", "16-column"),
        ((128, 128), "depth", "axis"),
    ],
)
def test_rejects_shapes_that_cannot_be_split_without_requantization(
    logical_shape: tuple[int, int], axis: str, message: str
) -> None:
    with pytest.raises((Nvfp4ShardError, ValueError), match=message):
        nvfp4_shard_plan(logical_shape, axis=axis, rank=0)


@pytest.mark.parametrize("rank", [-1, 2, 3, True])
def test_rejects_invalid_rank(rank) -> None:
    with pytest.raises((Nvfp4ShardError, ValueError), match="rank"):
        nvfp4_shard_plan((256, 128), axis="row", rank=rank)


def test_rejects_non_nvfp4_and_logically_transposed_weights() -> None:
    with pytest.raises(Nvfp4ShardError, match="QuantizedTensor"):
        shard_nvfp4_linear(torch.ones((256, 128), dtype=torch.float16), axis="row", rank=0)

    source = torch.ones((256, 128), dtype=torch.float16)
    transposed = QuantizedTensor.from_float(source, "TensorCoreNVFP4Layout").t()
    with pytest.raises(Nvfp4ShardError, match="transposed"):
        shard_nvfp4_linear(transposed, axis="row", rank=0)
