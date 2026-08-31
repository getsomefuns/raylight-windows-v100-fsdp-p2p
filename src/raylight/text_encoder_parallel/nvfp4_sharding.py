"""Read-only NVFP4 weight sharding for MiniMax H3 text-encoder TP2.

The packed four-bit payload can be sliced directly.  NVFP4 block scales cannot:
cuBLAS stores them in a tiled/swizzled layout, so this module only unswizzles the
small FP8 scale matrix, slices it, and swizzles the shard again.  It never
materializes the logical weight as FP16/FP32 while constructing a shard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from comfy_kitchen.float_utils import from_blocked, to_blocked
from comfy_kitchen.tensor import QuantizedTensor

ShardAxis = Literal["row", "column"]


class Nvfp4ShardError(ValueError):
    """The requested NVFP4 tensor cannot be split without requantization."""


@dataclass(frozen=True)
class Nvfp4ShardPlan:
    """Storage and logical bounds for one rank of a two-way NVFP4 shard."""

    axis: ShardAxis
    rank: int
    world_size: int
    source_shape: tuple[int, int]
    logical_shape: tuple[int, int]
    logical_row_slice: slice
    logical_column_slice: slice
    qdata_row_slice: slice
    qdata_column_slice: slice
    block_scale_row_slice: slice
    block_scale_column_slice: slice

    @property
    def qdata_elements(self) -> int:
        rows = self.qdata_row_slice.stop - self.qdata_row_slice.start
        columns = self.qdata_column_slice.stop - self.qdata_column_slice.start
        return rows * columns


@dataclass(frozen=True)
class QuantizedLinearShard:
    """One rank's quantized linear weight and its tensor-parallel metadata."""

    axis: ShardAxis
    logical_shape: tuple[int, int]
    rank: int
    world_size: int
    qweight: QuantizedTensor


def _validate_shape(logical_shape: tuple[int, int]) -> tuple[int, int]:
    if not isinstance(logical_shape, tuple) or len(logical_shape) != 2:
        raise Nvfp4ShardError("logical_shape must be a two-dimensional tuple")
    rows, columns = logical_shape
    if isinstance(rows, bool) or isinstance(columns, bool) or not isinstance(rows, int) or not isinstance(columns, int):
        raise Nvfp4ShardError("logical_shape dimensions must be integers")
    if rows <= 0 or columns <= 0:
        raise Nvfp4ShardError("logical_shape dimensions must be positive")
    if rows % 16:
        raise Nvfp4ShardError("NVFP4 source weight must have 16-row alignment")
    if columns % 16:
        raise Nvfp4ShardError("NVFP4 source weight must have 16-column alignment")
    return rows, columns


def nvfp4_shard_plan(
    logical_shape: tuple[int, int],
    *,
    axis: ShardAxis,
    rank: int,
    world_size: int = 2,
) -> Nvfp4ShardPlan:
    """Plan a strict TP2 shard that needs neither padding nor requantization."""

    if axis not in ("row", "column"):
        raise Nvfp4ShardError("axis must be 'row' or 'column'")
    if isinstance(world_size, bool) or world_size != 2:
        raise Nvfp4ShardError("world_size must be exactly 2 for MiniMax H3 TP2")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world_size:
        raise Nvfp4ShardError("rank must be integer 0 or 1")

    rows, columns = _validate_shape(logical_shape)
    if axis == "row":
        shard_rows = rows // world_size
        if rows % world_size or shard_rows % 128:
            raise Nvfp4ShardError("row shards require each rank to keep 128-row scale-tile alignment")
        row_start = rank * shard_rows
        row_stop = row_start + shard_rows
        column_start, column_stop = 0, columns
    else:
        shard_columns = columns // world_size
        if columns % world_size or shard_columns % 64:
            raise Nvfp4ShardError("column shards require each rank to keep 64-column scale-tile alignment")
        row_start, row_stop = 0, rows
        column_start = rank * shard_columns
        column_stop = column_start + shard_columns

    logical_rows = slice(row_start, row_stop)
    logical_columns = slice(column_start, column_stop)
    return Nvfp4ShardPlan(
        axis=axis,
        rank=rank,
        world_size=world_size,
        source_shape=(rows, columns),
        logical_shape=(row_stop - row_start, column_stop - column_start),
        logical_row_slice=logical_rows,
        logical_column_slice=logical_columns,
        qdata_row_slice=logical_rows,
        qdata_column_slice=slice(column_start // 2, column_stop // 2),
        block_scale_row_slice=logical_rows,
        block_scale_column_slice=slice(column_start // 16, column_stop // 16),
    )


def _validate_weight(weight: QuantizedTensor) -> tuple[int, int]:
    if not isinstance(weight, QuantizedTensor):
        raise Nvfp4ShardError("weight must be a comfy_kitchen QuantizedTensor")
    if getattr(weight, "_layout_cls", None) != "TensorCoreNVFP4Layout":
        raise Nvfp4ShardError("weight must use TensorCoreNVFP4Layout")
    params = weight._params
    if getattr(params, "transposed", False):
        raise Nvfp4ShardError("logically transposed NVFP4 weights are not supported")
    logical_shape = tuple(params.orig_shape)
    rows, columns = _validate_shape(logical_shape)
    expected_storage_shape = (rows, columns // 2)
    if tuple(weight._qdata.shape) != expected_storage_shape:
        raise Nvfp4ShardError(
            f"padded NVFP4 storage is not supported: expected {expected_storage_shape}, got {tuple(weight._qdata.shape)}"
        )
    expected_scale_elements = rows * (columns // 16)
    if weight._params.block_scale.numel() != expected_scale_elements:
        raise Nvfp4ShardError(
            f"unexpected NVFP4 block_scale size: expected {expected_scale_elements}, got {weight._params.block_scale.numel()}"
        )
    return rows, columns


def shard_nvfp4_linear(
    weight: QuantizedTensor,
    *,
    axis: ShardAxis,
    rank: int,
    world_size: int = 2,
) -> QuantizedLinearShard:
    """Create one row/column shard without dequantizing the logical weight."""

    rows, columns = _validate_weight(weight)
    plan = nvfp4_shard_plan((rows, columns), axis=axis, rank=rank, world_size=world_size)

    qdata = weight._qdata[plan.qdata_row_slice, plan.qdata_column_slice].contiguous()
    unblocked_scale = from_blocked(weight._params.block_scale, num_rows=rows, num_cols=columns // 16)
    scale_shard = unblocked_scale[plan.block_scale_row_slice, plan.block_scale_column_slice]
    blocked_scale = to_blocked(scale_shard, flatten=False)

    params_type = type(weight._params)
    params = params_type(
        scale=weight._params.scale,
        orig_dtype=weight._params.orig_dtype,
        orig_shape=plan.logical_shape,
        block_scale=blocked_scale,
        transposed=False,
    )
    qweight = type(weight)(qdata, weight._layout_cls, params)
    return QuantizedLinearShard(
        axis=axis,
        logical_shape=plan.logical_shape,
        rank=rank,
        world_size=world_size,
        qweight=qweight,
    )


__all__ = [
    "Nvfp4ShardError",
    "Nvfp4ShardPlan",
    "QuantizedLinearShard",
    "nvfp4_shard_plan",
    "shard_nvfp4_linear",
]
