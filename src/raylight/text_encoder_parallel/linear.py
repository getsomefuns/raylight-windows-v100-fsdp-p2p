"""Two-device tensor-parallel Linear primitives for MiniMax H3 Text Encoder."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence
import time

from .collectives import CudaPeerGroup2
from .nvfp4_sharding import QuantizedLinearShard


@dataclass(frozen=True)
class LinearExecutionStats:
    """CUDA timing evidence for one pair of concurrently submitted linears."""

    rank0_ms: float
    rank1_ms: float
    wall_ms: float
    overlap_ms: float
    overlap_ratio: float


def _pair(values: Sequence[Any], *, name: str) -> tuple[Any, Any]:
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly rank 0 and rank 1")
    return values[0], values[1]


def _validate_shards(
    shards: Sequence[QuantizedLinearShard], *, expected_axis: str
) -> tuple[QuantizedLinearShard, QuantizedLinearShard]:
    rank0, rank1 = _pair(shards, name="shards")
    for expected_rank, shard in enumerate((rank0, rank1)):
        if not isinstance(shard, QuantizedLinearShard):
            raise ValueError(f"rank {expected_rank} weight must be a QuantizedLinearShard")
        if shard.axis != expected_axis:
            raise ValueError(f"rank {expected_rank} shard axis must be {expected_axis!r}, got {shard.axis!r}")
        if shard.rank != expected_rank:
            raise ValueError(f"shard rank order must be 0, 1; position {expected_rank} contains rank {shard.rank}")
        if shard.world_size != 2:
            raise ValueError("Text Encoder tensor parallel Linear requires world_size=2")
        if tuple(shard.qweight.shape) != tuple(shard.logical_shape):
            raise ValueError(f"rank {expected_rank} qweight shape does not match logical_shape")
    return rank0, rank1


def _normalize_bias(bias: Sequence[Any] | None) -> tuple[Any | None, Any | None]:
    if bias is None:
        return None, None
    return _pair(bias, name="bias")


class TorchDualCudaLinearExecutor:
    """Submit one local Linear to each V100 without serial synchronization."""

    def __init__(self, *, torch_module=None) -> None:
        if torch_module is None:
            import torch as torch_module

        self.torch = torch_module
        self.streams = (
            torch_module.cuda.Stream(device=0),
            torch_module.cuda.Stream(device=1),
        )
        self._submit_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="raylight-te-tp2")
        self._closed = False

    def _validate(self, inputs, weights, biases) -> None:
        for rank, (input_tensor, weight, bias) in enumerate(zip(inputs, weights, biases)):
            for label, tensor in (("input", input_tensor), ("weight", weight)):
                device = getattr(tensor, "device", None)
                if getattr(device, "type", None) != "cuda" or device.index != rank:
                    raise ValueError(f"rank {rank} {label} must be on cuda:{rank}")
            if bias is not None:
                device = getattr(bias, "device", None)
                if getattr(device, "type", None) != "cuda" or device.index != rank:
                    raise ValueError(f"rank {rank} bias must be on cuda:{rank}")

    def run(self, inputs, weights, biases):
        if self._closed:
            raise RuntimeError("TorchDualCudaLinearExecutor is closed")
        torch = self.torch
        inputs = _pair(inputs, name="inputs")
        weights = _pair(weights, name="weights")
        biases = _pair(biases, name="biases")
        self._validate(inputs, weights, biases)

        ready_events = (torch.cuda.Event(blocking=False), torch.cuda.Event(blocking=False))
        start_events = (
            torch.cuda.Event(enable_timing=True, blocking=False),
            torch.cuda.Event(enable_timing=True, blocking=False),
        )
        end_events = (
            torch.cuda.Event(enable_timing=True, blocking=False),
            torch.cuda.Event(enable_timing=True, blocking=False),
        )
        for rank in (0, 1):
            with torch.cuda.device(rank):
                ready_events[rank].record(torch.cuda.current_stream(rank))

        def submit_rank(rank: int):
            with torch.cuda.device(rank), torch.cuda.stream(self.streams[rank]):
                self.streams[rank].wait_event(ready_events[rank])
                start_events[rank].record(self.streams[rank])
                output = torch.nn.functional.linear(inputs[rank], weights[rank], biases[rank])
                end_events[rank].record(self.streams[rank])
                return output

        wall_started = time.perf_counter()
        futures = tuple(self._submit_pool.submit(submit_rank, rank) for rank in (0, 1))
        outputs = tuple(future.result() for future in futures)
        end_events[0].synchronize()
        end_events[1].synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        rank_ms = (
            float(start_events[0].elapsed_time(end_events[0])),
            float(start_events[1].elapsed_time(end_events[1])),
        )
        shorter_ms = min(rank_ms)
        overlap_ms = min(shorter_ms, max(0.0, rank_ms[0] + rank_ms[1] - wall_ms))
        overlap_ratio = overlap_ms / shorter_ms if shorter_ms > 0.0 else 0.0
        stats = LinearExecutionStats(
            rank0_ms=rank_ms[0],
            rank1_ms=rank_ms[1],
            wall_ms=wall_ms,
            overlap_ms=overlap_ms,
            overlap_ratio=overlap_ratio,
        )
        return outputs, stats

    def close(self) -> None:
        if self._closed:
            return
        self._submit_pool.shutdown(wait=True, cancel_futures=True)
        self.streams = (None, None)
        self._closed = True


class ColumnParallelLinear:
    """Output-channel-sharded Linear; each rank receives replicated hidden state."""

    def __init__(self, shards, *, bias=None, executor=None) -> None:
        self.shards = _validate_shards(shards, expected_axis="row")
        self.bias = _normalize_bias(bias)
        self.executor = executor if executor is not None else TorchDualCudaLinearExecutor()
        self.last_execution: LinearExecutionStats | None = None
        in_features = self.shards[0].logical_shape[1]
        if self.shards[1].logical_shape[1] != in_features:
            raise ValueError("column-parallel rank weights must have the same input width")
        for rank, rank_bias in enumerate(self.bias):
            if rank_bias is not None and tuple(rank_bias.shape) != (self.shards[rank].logical_shape[0],):
                raise ValueError(f"rank {rank} bias shape does not match local output width")

    def forward(self, inputs):
        inputs = _pair(inputs, name="inputs")
        if tuple(inputs[0].shape[:-1]) != tuple(inputs[1].shape[:-1]):
            raise ValueError("column-parallel rank inputs must have matching leading dimensions")
        for rank, input_tensor in enumerate(inputs):
            expected = self.shards[rank].logical_shape[1]
            if input_tensor.shape[-1] != expected:
                raise ValueError(f"rank {rank} input width must be {expected}, got {input_tensor.shape[-1]}")
        outputs, stats = self.executor.run(
            inputs,
            tuple(shard.qweight for shard in self.shards),
            self.bias,
        )
        self.last_execution = stats
        return outputs

    __call__ = forward


class RowParallelLinear:
    """Input-channel-sharded Linear followed by the strict CUDA P2P SUM."""

    def __init__(self, shards, *, bias=None, executor=None, collective=None) -> None:
        self.shards = _validate_shards(shards, expected_axis="column")
        self.bias = _normalize_bias(bias)
        self.executor = executor if executor is not None else TorchDualCudaLinearExecutor()
        self.collective = collective if collective is not None else CudaPeerGroup2()
        self.last_execution: LinearExecutionStats | None = None
        self.last_collective = None
        out_features = self.shards[0].logical_shape[0]
        if self.shards[1].logical_shape[0] != out_features:
            raise ValueError("row-parallel rank weights must have the same output width")
        for rank, rank_bias in enumerate(self.bias):
            if rank_bias is not None and tuple(rank_bias.shape) != (out_features,):
                raise ValueError(f"rank {rank} bias shape does not match full output width")

    def forward(self, inputs, *, operation_id: int):
        inputs = _pair(inputs, name="inputs")
        if tuple(inputs[0].shape[:-1]) != tuple(inputs[1].shape[:-1]):
            raise ValueError("row-parallel rank inputs must have matching leading dimensions")
        for rank, input_tensor in enumerate(inputs):
            expected = self.shards[rank].logical_shape[1]
            if input_tensor.shape[-1] != expected:
                raise ValueError(f"rank {rank} input width must be {expected}, got {input_tensor.shape[-1]}")

        partials, stats = self.executor.run(
            inputs,
            tuple(shard.qweight for shard in self.shards),
            (None, None),
        )
        partials = tuple(partial.contiguous() for partial in partials)
        rank0, rank1, collective_stats = self.collective.all_reduce_sum(
            partials[0], partials[1], operation_id=operation_id
        )
        if self.bias[0] is not None:
            rank0 = rank0 + self.bias[0]
        if self.bias[1] is not None:
            rank1 = rank1 + self.bias[1]
        self.last_execution = stats
        self.last_collective = collective_stats
        return rank0, rank1

    __call__ = forward


__all__ = [
    "ColumnParallelLinear",
    "LinearExecutionStats",
    "RowParallelLinear",
    "TorchDualCudaLinearExecutor",
]
