"""MiniMax H3 text-encoder parallelism experiments."""

from .baseline import BaselineRecord
from .collectives import CudaPeerGroup2, PeerCollectiveStats
from .linear import ColumnParallelLinear, RowParallelLinear
from .nvfp4_sharding import QuantizedLinearShard, shard_nvfp4_linear
from .qwen3vl_tp import MINIMAX_H3_QWEN3VL_CONFIG, TP2Qwen3VLBlock, TP2Qwen3VLConfig

__all__ = [
    "BaselineRecord",
    "CudaPeerGroup2",
    "ColumnParallelLinear",
    "PeerCollectiveStats",
    "MINIMAX_H3_QWEN3VL_CONFIG",
    "QuantizedLinearShard",
    "RowParallelLinear",
    "TP2Qwen3VLBlock",
    "TP2Qwen3VLConfig",
    "shard_nvfp4_linear",
]
