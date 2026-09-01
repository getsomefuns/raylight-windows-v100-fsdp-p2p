"""MiniMax H3 text-encoder parallelism experiments."""

from .baseline import BaselineRecord
from .attention_backends import (
    AttentionBackendUnavailable,
    TextEncoderAttentionBackend,
    resolve_attention_backend,
    run_text_encoder_attention,
)
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
from .lifecycle import DualGpuLeaseCoordinator, ExternalGpuLease, WorkerGpuLifecycle
from .nvfp4_sharding import QuantizedLinearShard, shard_nvfp4_linear
from .qwen3vl_tp import MINIMAX_H3_QWEN3VL_CONFIG, TP2Qwen3VLBlock, TP2Qwen3VLConfig

__all__ = [
    "AttentionBackendUnavailable",
    "BaselineRecord",
    "CudaPeerGroup2",
    "ColumnParallelLinear",
    "DualGpuLeaseCoordinator",
    "ExternalGpuLease",
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
    "TextEncoderAttentionBackend",
    "WorkerGpuLifecycle",
    "resolve_attention_backend",
    "run_text_encoder_attention",
    "shard_nvfp4_linear",
    "load_minimax_h3_tp2_backbone",
    "prepare_minimax_token_sections",
]
