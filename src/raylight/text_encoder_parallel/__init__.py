"""MiniMax H3 text-encoder parallelism experiments."""

from .baseline import BaselineRecord
from .clip_proxy import (
    MiniMaxH3CheckpointPreprocessor,
    MiniMaxH3TP2ClipProxy,
    PreparedMiniMaxConditioning,
    TP2Qwen3VLBackbone,
    load_minimax_h3_tp2_backbone,
    prepare_minimax_token_sections,
)
from .collectives import CudaPeerGroup2, PeerCollectiveStats
from .linear import ColumnParallelLinear, RowParallelLinear
from .nvfp4_sharding import QuantizedLinearShard, shard_nvfp4_linear
from .qwen3vl_tp import MINIMAX_H3_QWEN3VL_CONFIG, TP2Qwen3VLBlock, TP2Qwen3VLConfig

__all__ = [
    "BaselineRecord",
    "CudaPeerGroup2",
    "ColumnParallelLinear",
    "MiniMaxH3CheckpointPreprocessor",
    "MiniMaxH3TP2ClipProxy",
    "PeerCollectiveStats",
    "MINIMAX_H3_QWEN3VL_CONFIG",
    "QuantizedLinearShard",
    "PreparedMiniMaxConditioning",
    "RowParallelLinear",
    "TP2Qwen3VLBlock",
    "TP2Qwen3VLBackbone",
    "TP2Qwen3VLConfig",
    "shard_nvfp4_linear",
    "load_minimax_h3_tp2_backbone",
    "prepare_minimax_token_sections",
]
