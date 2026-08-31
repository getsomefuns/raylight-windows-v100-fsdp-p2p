"""Single-process, two-GPU CUDA P2P sum collective for Text Encoder TP2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from .topology import validate_windows_v100_topology


@dataclass(frozen=True)
class PeerCollectiveStats:
    bytes_sent: int
    elapsed_ms: float
    operation_id: int


@dataclass(frozen=True)
class _BufferKey:
    shape: tuple[int, ...]
    dtype: str
    rank0_device: int
    rank1_device: int


@dataclass
class _TorchBuffers:
    recv0: Any
    recv1: Any
    out0: Any
    out1: Any
    copy_done0: Any
    copy_done1: Any
    add_done0: Any
    add_done1: Any


def _dtype_name(dtype: Any) -> str:
    return str(dtype).lower()


def _validate_tensor_pair(rank0: Any, rank1: Any, operation_id: int) -> _BufferKey:
    if isinstance(operation_id, bool) or not isinstance(operation_id, int) or operation_id <= 0:
        raise ValueError("operation_id must be a positive integer")
    for rank, tensor in enumerate((rank0, rank1)):
        if not getattr(tensor, "is_cuda", False) or getattr(tensor.device, "type", None) != "cuda":
            raise ValueError(f"rank {rank} partial must be a CUDA tensor")
        if tensor.device.index != rank:
            raise ValueError(f"rank {rank} partial must be on device {rank}")
        if not tensor.is_contiguous():
            raise ValueError(f"rank {rank} partial must be contiguous")
    if tuple(rank0.shape) != tuple(rank1.shape):
        raise ValueError(f"partial shape mismatch: {tuple(rank0.shape)} != {tuple(rank1.shape)}")
    if rank0.dtype != rank1.dtype:
        raise ValueError(f"partial dtype mismatch: {rank0.dtype} != {rank1.dtype}")
    if _dtype_name(rank0.dtype) not in {"float16", "torch.float16", "float32", "torch.float32"}:
        raise ValueError(f"TP2 P2P SUM supports only FP16/FP32, got {rank0.dtype}")
    if rank0.numel() != rank1.numel():
        raise ValueError("partial tensors must contain the same number of elements")
    return _BufferKey(
        shape=tuple(rank0.shape),
        dtype=_dtype_name(rank0.dtype),
        rank0_device=rank0.device.index,
        rank1_device=rank1.device.index,
    )


class TorchCudaPeerBackend:
    """PyTorch CUDA implementation using two peer copies and local adds."""

    def __init__(self, *, torch_module=None, topology_validator=validate_windows_v100_topology):
        if torch_module is None:
            import torch as torch_module

        self.torch = torch_module
        self.topology_validator = topology_validator
        self.stream0 = None
        self.stream1 = None
        self._closed = False

    def validate(self, devices: tuple[int, int]) -> None:
        if self._closed:
            raise RuntimeError("CUDA peer backend is closed")
        self.topology_validator(torch_module=self.torch, devices=devices)
        self.stream0 = self.torch.cuda.Stream(device=devices[0])
        self.stream1 = self.torch.cuda.Stream(device=devices[1])

    def allocate_buffers(self, rank0: Any, rank1: Any) -> _TorchBuffers:
        torch = self.torch
        return _TorchBuffers(
            recv0=torch.empty_like(rank0),
            recv1=torch.empty_like(rank1),
            out0=torch.empty_like(rank0),
            out1=torch.empty_like(rank1),
            copy_done0=torch.cuda.Event(blocking=False),
            copy_done1=torch.cuda.Event(blocking=False),
            add_done0=torch.cuda.Event(blocking=False),
            add_done1=torch.cuda.Event(blocking=False),
        )

    def all_reduce_sum(self, rank0: Any, rank1: Any, buffers: _TorchBuffers):
        if self._closed or self.stream0 is None or self.stream1 is None:
            raise RuntimeError("CUDA peer backend is not validated or is closed")
        torch = self.torch

        # The local linears may run on arbitrary streams. Complete both partials
        # before issuing the two peer copies; the peer copies themselves overlap.
        torch.cuda.current_stream(0).synchronize()
        torch.cuda.current_stream(1).synchronize()
        started = time.perf_counter()
        with torch.cuda.device(0), torch.cuda.stream(self.stream0):
            buffers.recv0.copy_(rank1, non_blocking=True)
            buffers.copy_done0.record(self.stream0)
        with torch.cuda.device(1), torch.cuda.stream(self.stream1):
            buffers.recv1.copy_(rank0, non_blocking=True)
            buffers.copy_done1.record(self.stream1)
        buffers.copy_done0.synchronize()
        buffers.copy_done1.synchronize()

        # Do not overlap add with the opposite copy: a caller may feed a reused
        # output buffer back as the next partial, and early in-place overwrite
        # would otherwise race the second peer read.
        with torch.cuda.device(0), torch.cuda.stream(self.stream0):
            torch.add(rank0, buffers.recv0, out=buffers.out0)
            buffers.add_done0.record(self.stream0)
        with torch.cuda.device(1), torch.cuda.stream(self.stream1):
            torch.add(rank1, buffers.recv1, out=buffers.out1)
            buffers.add_done1.record(self.stream1)
        buffers.add_done0.synchronize()
        buffers.add_done1.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return buffers.out0, buffers.out1, elapsed_ms

    def close(self) -> None:
        self.stream0 = None
        self.stream1 = None
        self._closed = True


class CudaPeerGroup2:
    """Strict operation-ordered sum collective for CUDA device 0 and 1."""

    def __init__(self, *, devices: tuple[int, int] = (0, 1), backend=None):
        self.devices = devices
        self.backend = backend if backend is not None else TorchCudaPeerBackend()
        self._validated = False
        self._closed = False
        self._last_operation_id = 0
        self._buffer_key = None
        self._buffers = None
        self._calls = 0
        self._total_bytes_sent = 0
        self._total_elapsed_ms = 0.0
        self._slowest_elapsed_ms = 0.0
        self._slowest_operation_id = None

    def validate(self) -> None:
        if self._closed:
            raise RuntimeError("CudaPeerGroup2 is closed")
        if not self._validated:
            self.backend.validate(self.devices)
            self._validated = True

    def all_reduce_sum(self, rank0_partial: Any, rank1_partial: Any, operation_id: int):
        if self._closed:
            raise RuntimeError("CudaPeerGroup2 is closed")
        key = _validate_tensor_pair(rank0_partial, rank1_partial, operation_id)
        if operation_id <= self._last_operation_id:
            raise ValueError(
                "operation_id must strictly increase: "
                f"last={self._last_operation_id}, current={operation_id}"
            )
        self.validate()
        if key != self._buffer_key:
            self._buffers = self.backend.allocate_buffers(rank0_partial, rank1_partial)
            self._buffer_key = key
        out0, out1, elapsed_ms = self.backend.all_reduce_sum(
            rank0_partial, rank1_partial, self._buffers
        )
        bytes_sent = rank0_partial.numel() * rank0_partial.element_size() * 2
        stats = PeerCollectiveStats(
            bytes_sent=bytes_sent,
            elapsed_ms=float(elapsed_ms),
            operation_id=operation_id,
        )
        self._last_operation_id = operation_id
        self._calls += 1
        self._total_bytes_sent += bytes_sent
        self._total_elapsed_ms += stats.elapsed_ms
        if stats.elapsed_ms > self._slowest_elapsed_ms:
            self._slowest_elapsed_ms = stats.elapsed_ms
            self._slowest_operation_id = operation_id
        return out0, out1, stats

    def profile_snapshot(self) -> dict[str, int | float | None]:
        return {
            "calls": self._calls,
            "total_bytes_sent": self._total_bytes_sent,
            "total_elapsed_ms": self._total_elapsed_ms,
            "slowest_elapsed_ms": self._slowest_elapsed_ms,
            "slowest_operation_id": self._slowest_operation_id,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._buffers = None
        self._buffer_key = None
        self.backend.close()
        self._closed = True
