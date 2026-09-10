"""Two-rank, single-machine CUDA IPC/P2P all-to-all for Windows TCC."""

from __future__ import annotations

import ctypes
import json
from ctypes import wintypes
import mmap
import os
import struct
import sys
import time

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
from raylight.distributed_worker.collective_profile import create_collective_profiler



DEFAULT_WINDOWS_P2P_CAPACITY_BYTES = 128 * 1024 * 1024


def should_use_safetensors_mmap(parallel_dict, unet_path):
    """Honor explicit mmap for safetensors, including FP8/quantized checkpoints."""
    return bool(parallel_dict.get("use_mmap", True)) and unet_path.lower().endswith(".safetensors")


def synchronized_model_load(local_budget, load_model, reduce_min, barrier):
    """Use one VRAM budget on every rank and release them only after all loads finish."""
    synchronized_budget = reduce_min(local_budget)
    result = load_model(synchronized_budget)
    barrier()
    return result


class P2PGroupError(RuntimeError):
    """The endpoint is no longer safe to use after a collective failure."""


class CudaEventWork(dist.Work):
    """Torch Work handle backed by a CUDA completion event."""

    def __init__(self, event, device):
        super().__init__()
        self._event = event
        self._device = device

    def wait(self, timeout=None):
        torch.cuda.current_stream(self._device).wait_event(self._event)
        return True

    def is_completed(self):
        return self._event.query()

    def synchronize(self):
        return self.wait()


def export_cuda_tensor_ipc(tensor):
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("CUDA IPC export requires a contiguous CUDA tensor")
    rebuild, args = reduce_tensor(tensor)
    if rebuild is not rebuild_cuda_tensor:
        raise RuntimeError("PyTorch did not select the CUDA IPC tensor reducer")
    return {
        "size": tuple(args[1]),
        "stride": tuple(args[2]),
        "tensor_offset": args[3],
        "dtype": str(args[5]).removeprefix("torch."),
        "storage_device": args[6],
        "storage_handle": args[7],
        "storage_size_bytes": args[8],
        "storage_offset_bytes": args[9],
        "requires_grad": args[10],
        "ref_counter_handle": args[11],
        "ref_counter_offset": args[12],
        "event_handle": args[13],
        "event_sync_required": args[14],
    }


def import_cuda_tensor_ipc(metadata):
    dtype = getattr(torch, metadata["dtype"])
    return rebuild_cuda_tensor(
        torch.Tensor,
        torch.Size(metadata["size"]),
        tuple(metadata["stride"]),
        metadata["tensor_offset"],
        torch.storage.TypedStorage,
        dtype,
        metadata["storage_device"],
        metadata["storage_handle"],
        metadata["storage_size_bytes"],
        metadata["storage_offset_bytes"],
        metadata["requires_grad"],
        metadata["ref_counter_handle"],
        metadata["ref_counter_offset"],
        metadata["event_handle"],
        metadata["event_sync_required"],
    )


def make_all_to_all_router(endpoint, fallback):
    """Route only the supported two-rank synchronous CUDA collective to P2P."""

    def all_to_all_single(
        output,
        input_tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
        async_op=False,
    ):
        supported = (
            output.is_cuda
            and input_tensor.is_cuda
            and not async_op
            and output_split_sizes is None
            and input_split_sizes is None
            and dist.get_world_size(group) == 2
        )
        if supported:
            endpoint.all_to_all_single(output, input_tensor)
            return None
        return fallback(
            output,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

    return all_to_all_single


def make_all_gather_into_tensor_router(endpoint, fallback):
    """Route supported two-rank CUDA all-gather operations to P2P."""

    def all_gather_into_tensor(
        output_tensor,
        input_tensor,
        group=None,
        async_op=False,
    ):
        if (
            output_tensor.is_cuda
            and input_tensor.is_cuda
            and dist.get_world_size(group) == 2
        ):
            return endpoint.all_gather_into_tensor(
                output_tensor,
                input_tensor,
                async_op=async_op,
            )
        if output_tensor.is_cuda or input_tensor.is_cuda:
            raise RuntimeError(
                "Windows P2P all-gather refuses CUDA fallback to Gloo"
            )
        return fallback(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )

    return all_gather_into_tensor


def install_collective_routers(endpoint, dist_module=dist):
    """Install the Windows P2P data-plane routers and return originals."""
    originals = {
        "all_to_all_single": dist_module.all_to_all_single,
        "all_gather_into_tensor": dist_module.all_gather_into_tensor,
    }
    dist_module.all_to_all_single = make_all_to_all_router(
        endpoint,
        originals["all_to_all_single"],
    )
    dist_module.all_gather_into_tensor = make_all_gather_into_tensor_router(
        endpoint,
        originals["all_gather_into_tensor"],
    )
    return originals


def restore_collective_routers(dist_module, originals):
    """Restore torch.distributed collectives previously replaced by P2P routers."""
    dist_module.all_to_all_single = originals["all_to_all_single"]
    dist_module.all_gather_into_tensor = originals["all_gather_into_tensor"]


def copy_plan(rank: int, total_bytes: int):
    """Return destination/source byte slices for a two-rank equal all-to-all."""
    if rank not in (0, 1):
        raise ValueError(f"rank must be 0 or 1, got {rank}")
    if total_bytes < 0 or total_bytes % 2:
        raise ValueError(f"total_bytes must be non-negative and divisible by 2, got {total_bytes}")
    half = total_bytes // 2
    if rank == 0:
        return ((0, half, "input", 0, half), (half, total_bytes, "peer", 0, half))
    return ((0, half, "peer", 0, half), (half, total_bytes, "input", half, total_bytes))


def _validate_all_to_all_chunk_geometry(
    rank: int,
    total_bytes: int,
    chunk_start: int,
    chunk_end: int,
) -> int:
    if rank not in (0, 1):
        raise ValueError(f"rank must be 0 or 1, got {rank}")
    if total_bytes < 0 or total_bytes % 2:
        raise ValueError(
            f"total_bytes must be non-negative and divisible by 2, got {total_bytes}"
        )
    remote_bytes = total_bytes // 2
    if not 0 <= chunk_start < chunk_end <= remote_bytes:
        raise ValueError(
            "chunk range must satisfy "
            f"0 <= start < end <= {remote_bytes}, got [{chunk_start}, {chunk_end})"
        )
    return remote_bytes


def all_to_all_chunk_send_slice(
    rank: int,
    total_bytes: int,
    chunk_start: int,
    chunk_end: int,
) -> tuple[int, int]:
    """Return the absolute input byte slice sent for one equal all-to-all chunk."""
    remote_bytes = _validate_all_to_all_chunk_geometry(
        rank,
        total_bytes,
        chunk_start,
        chunk_end,
    )
    offset = remote_bytes if rank == 0 else 0
    return offset + chunk_start, offset + chunk_end


def all_to_all_chunk_copy_plan(
    rank: int,
    total_bytes: int,
    chunk_start: int,
    chunk_end: int,
):
    """Return output/source byte slices for one equal all-to-all chunk."""
    remote_bytes = _validate_all_to_all_chunk_geometry(
        rank,
        total_bytes,
        chunk_start,
        chunk_end,
    )
    chunk_bytes = chunk_end - chunk_start
    local_offset = 0 if rank == 0 else remote_bytes
    peer_offset = remote_bytes if rank == 0 else 0
    local = (
        local_offset + chunk_start,
        local_offset + chunk_end,
        "input",
        local_offset + chunk_start,
        local_offset + chunk_end,
    )
    peer = (
        peer_offset + chunk_start,
        peer_offset + chunk_end,
        "peer",
        0,
        chunk_bytes,
    )
    return (local, peer) if rank == 0 else (peer, local)


def chunk_ranges(total_bytes: int, capacity_bytes: int) -> tuple[tuple[int, int], ...]:
    """Split a byte payload into ordered half-open ranges."""
    if total_bytes < 0:
        raise ValueError("total_bytes must be non-negative")
    if capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive")
    return tuple(
        (start, min(start + capacity_bytes, total_bytes))
        for start in range(0, total_bytes, capacity_bytes)
    )


def all_gather_copy_plan(rank: int, shard_bytes: int):
    """Return destination/source byte slices for a two-rank all-gather."""
    if rank == 0:
        return ((0, shard_bytes, "input", 0, shard_bytes), (shard_bytes, shard_bytes * 2, "peer", 0, shard_bytes))
    return ((0, shard_bytes, "peer", 0, shard_bytes), (shard_bytes, shard_bytes * 2, "input", 0, shard_bytes))

def all_gather_chunk_copy_plan(
    rank: int,
    shard_bytes: int,
    chunk_start: int,
    chunk_end: int,
):
    """Return byte copies for one chunk of a two-rank all-gather."""
    chunk_bytes = chunk_end - chunk_start
    local_start = rank * shard_bytes + chunk_start
    peer_start = (1 - rank) * shard_bytes + chunk_start
    local = (
        local_start,
        local_start + chunk_bytes,
        "input",
        chunk_start,
        chunk_end,
    )
    peer = (
        peer_start,
        peer_start + chunk_bytes,
        "peer",
        0,
        chunk_bytes,
    )
    return (local, peer) if rank == 0 else (peer, local)

def send_slice(rank: int, total_bytes: int):
    copy_plan(rank, total_bytes)
    half = total_bytes // 2
    return (half, total_bytes) if rank == 0 else (0, half)


def validate_collective(output, input_tensor, capacity_bytes: int) -> int:
    if not input_tensor.is_cuda or not output.is_cuda:
        raise ValueError("Windows CUDA P2P all-to-all requires CUDA tensors")
    if not input_tensor.is_contiguous() or not output.is_contiguous():
        raise ValueError("Windows CUDA P2P all-to-all requires contiguous tensors")
    if input_tensor.dtype != output.dtype:
        raise ValueError("input and output must use the same dtype")
    if input_tensor.numel() != output.numel():
        raise ValueError("input and output must contain the same number of elements")
    total_bytes = input_tensor.numel() * input_tensor.element_size()
    if total_bytes % 2:
        raise ValueError(f"payload bytes must be divisible by 2, got {total_bytes}")
    return total_bytes


def validate_all_gather(output, input_tensor, capacity_bytes: int) -> int:
    """Return the local shard size for a two-rank all-gather."""
    if not input_tensor.is_cuda or not output.is_cuda:
        raise ValueError("Windows CUDA P2P all-gather requires CUDA tensors")
    if not input_tensor.is_contiguous() or not output.is_contiguous():
        raise ValueError("Windows CUDA P2P all-gather requires contiguous tensors")
    if input_tensor.dtype != output.dtype:
        raise ValueError("all-gather input and output must use the same dtype")
    if output.numel() != input_tensor.numel() * 2:
        raise ValueError("all-gather output must contain exactly twice the input elements")
    shard_bytes = input_tensor.numel() * input_tensor.element_size()
    return shard_bytes


class CudaP2PAllToAll:
    """Reusable two-rank P2P endpoint with one shared send buffer per rank.

    Control-plane synchronization is intentionally injected. Phase 1 uses a
    multiprocessing test transport; Phase 2 will provide a Ray/TCPStore one.
    """

    def __init__(
        self,
        rank: int,
        capacity_bytes: int,
        control,
        timeout_seconds: float = 10.0,
    ):
        if rank not in (0, 1):
            raise ValueError(f"rank must be 0 or 1, got {rank}")
        if capacity_bytes <= 0 or capacity_bytes % 2:
            raise ValueError("capacity_bytes must be positive and divisible by 2")
        self.rank = rank
        self.capacity_bytes = capacity_bytes
        self.control = control
        if control is None:
            raise ValueError("strict P2P mode requires a control plane")
        self.timeout_seconds = timeout_seconds
        self._poisoned_reason = None
        # Zero is reserved for an uninitialized shared-memory control slot.
        self._operation_id = 1
        self._last_call_perf = None
        self._profiler = create_collective_profiler()
        self._profile_control_wait_ns = 0
        self._diag_enabled = os.environ.get("RAYLIGHT_RANK_DIAG", "0") == "1"
        self._send_buffer = torch.empty(capacity_bytes, dtype=torch.uint8, device="cuda:0")
        self._ready_event = torch.cuda.Event(interprocess=True)
        self._consumed_event = torch.cuda.Event(interprocess=True)
        self._stream = torch.cuda.Stream(device=0)
        self._peer_buffer = None
        self._peer_ready_event = None
        self._peer_consumed_event = None
        self._buffer_operation_id = None

    def local_handles(self):
        # Keep the cross-process metadata pickle free of dynamically imported
        # Raylight types. CUDA tensor reduction supplies the storage IPC handle.
        return (
            self._send_buffer,
            self._ready_event.ipc_handle(),
            self._consumed_event.ipc_handle(),
        )

    def connect(self, peer_handles):
        peer_buffer, ready_event_handle, consumed_event_handle = peer_handles
        self._peer_buffer = peer_buffer
        self._peer_ready_event = torch.cuda.Event.from_ipc_handle(0, ready_event_handle)
        self._peer_consumed_event = torch.cuda.Event.from_ipc_handle(0, consumed_event_handle)

    def local_ipc_metadata(self):
        return {
            "buffer": export_cuda_tensor_ipc(self._send_buffer),
            "ready_event": self._ready_event.ipc_handle(),
            "consumed_event": self._consumed_event.ipc_handle(),
        }

    def connect_ipc_metadata(self, peer_metadata):
        self._peer_buffer = import_cuda_tensor_ipc(peer_metadata["buffer"])
        self._peer_ready_event = torch.cuda.Event.from_ipc_handle(
            0,
            peer_metadata["ready_event"],
        )
        self._peer_consumed_event = torch.cuda.Event.from_ipc_handle(
            0,
            peer_metadata["consumed_event"],
        )

    def _ensure_healthy(self):
        if self._poisoned_reason is not None:
            raise P2PGroupError(f"P2P group is poisoned: {self._poisoned_reason}")

    def _fail(self, exc: BaseException):
        self._poisoned_reason = f"{type(exc).__name__}: {exc}"
        raise P2PGroupError(self._poisoned_reason) from exc

    def _wait_for_peer_buffer_consumption(self):
        """Wait until the peer has recorded consumption of our prior buffer.

        CUDA IPC events are reusable.  The CPU handshake is required before
        waiting on the event so a fast rank cannot observe the event's older
        generation and overwrite the shared send buffer while the peer still
        copies the previous chunk.
        """
        if self._buffer_operation_id is None:
            return
        self.control.wait_consumed(
            1 - self.rank,
            self._buffer_operation_id,
            timeout_seconds=self.timeout_seconds,
        )
        self._stream.wait_event(self._peer_consumed_event)

    def _exchange_collective_geometry(self, total_bytes, collective_name):
        operation_id = self._operation_id
        self._operation_id += 1
        self.control.publish_ready(self.rank, operation_id, total_bytes)
        wait_started_ns = time.perf_counter_ns()
        peer_total_bytes = self.control.wait_ready(
            1 - self.rank,
            operation_id,
            timeout_seconds=self.timeout_seconds,
        )
        wait_ns = time.perf_counter_ns() - wait_started_ns
        if peer_total_bytes != total_bytes:
            raise ValueError(
                f"{collective_name} size mismatch at operation {operation_id}: "
                f"local={total_bytes}, peer={peer_total_bytes}"
            )
        return operation_id, wait_ns

    def _all_to_all_chunk(
        self,
        input_bytes,
        output_bytes,
        total_bytes,
        chunk_start,
        chunk_end,
    ):
        operation_id = self._operation_id
        self._operation_id += 1
        chunk_bytes = chunk_end - chunk_start

        with torch.cuda.stream(self._stream):
            self._wait_for_peer_buffer_consumption()
            send_start, send_end = all_to_all_chunk_send_slice(
                self.rank,
                total_bytes,
                chunk_start,
                chunk_end,
            )
            self._send_buffer[:chunk_bytes].copy_(
                input_bytes[send_start:send_end],
                non_blocking=True,
            )
            self._ready_event.record(self._stream)

        self.control.publish_ready(self.rank, operation_id, chunk_bytes)
        wait_started_ns = time.perf_counter_ns()
        peer_chunk_bytes = self.control.wait_ready(
            1 - self.rank,
            operation_id,
            timeout_seconds=self.timeout_seconds,
        )
        wait_ns = time.perf_counter_ns() - wait_started_ns
        if peer_chunk_bytes != chunk_bytes:
            raise ValueError(
                f"all-to-all chunk size mismatch at operation {operation_id}: "
                f"local={chunk_bytes}, peer={peer_chunk_bytes}"
            )

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(self._peer_ready_event)
            for dst_start, dst_end, source_kind, src_start, src_end in (
                all_to_all_chunk_copy_plan(
                    self.rank,
                    total_bytes,
                    chunk_start,
                    chunk_end,
                )
            ):
                source = input_bytes if source_kind == "input" else self._peer_buffer
                output_bytes[dst_start:dst_end].copy_(
                    source[src_start:src_end],
                    non_blocking=True,
                )
            self._consumed_event.record(self._stream)
        self.control.publish_consumed(self.rank, operation_id)
        self._buffer_operation_id = operation_id
        return operation_id, wait_ns

    def all_to_all_single(self, output, input_tensor):
        self._ensure_healthy()
        if self._peer_buffer is None:
            raise P2PGroupError("P2P endpoint is not connected")
        total_bytes = validate_collective(output, input_tensor, self.capacity_bytes)
        operation_id = None
        call_perf = time.perf_counter()
        submit_started_ns = time.perf_counter_ns() if self._profiler.enabled else 0
        idle_seconds = None if self._last_call_perf is None else call_perf - self._last_call_perf
        self._last_call_perf = call_perf
        log_boundary = self._diag_enabled and (idle_seconds is None or idle_seconds >= 1.0)
        current_stream = torch.cuda.current_stream(0)
        remote_bytes = total_bytes // 2
        chunks = chunk_ranges(remote_bytes, self.capacity_bytes)
        chunk_count = len(chunks)
        control_wait_ns = 0

        try:
            if log_boundary:
                self._diag(
                    "collective_after_idle",
                    operation_id=self._operation_id,
                    idle_seconds=idle_seconds,
                    total_bytes=total_bytes,
                    remote_bytes=remote_bytes,
                    capacity_bytes=self.capacity_bytes,
                    chunks=chunk_count,
                )
            input_bytes = input_tensor.view(torch.uint8).reshape(-1)
            output_bytes = output.view(torch.uint8).reshape(-1)

            self._stream.wait_stream(current_stream)
            if chunk_count != 1:
                operation_id, geometry_wait_ns = self._exchange_collective_geometry(
                    total_bytes,
                    "all-to-all",
                )
                control_wait_ns += geometry_wait_ns

            for chunk_start, chunk_end in chunks:
                operation_id, chunk_wait_ns = self._all_to_all_chunk(
                    input_bytes,
                    output_bytes,
                    total_bytes,
                    chunk_start,
                    chunk_end,
                )
                control_wait_ns += chunk_wait_ns

            wait_seconds = control_wait_ns / 1_000_000_000
            if log_boundary or wait_seconds >= 0.05:
                self._diag(
                    "peer_ready_observed",
                    operation_id=operation_id,
                    wait_seconds=wait_seconds,
                    total_bytes=total_bytes,
                    remote_bytes=remote_bytes,
                    capacity_bytes=self.capacity_bytes,
                    chunks=chunk_count,
                )

            if chunks:
                current_stream.wait_event(self._consumed_event)
            if self._profiler.enabled:
                self._profiler.record(
                    "all_to_all",
                    payload_bytes=total_bytes,
                    remote_bytes=remote_bytes,
                    chunks=chunk_count,
                    control_wait_ns=control_wait_ns,
                    submit_ns=time.perf_counter_ns() - submit_started_ns,
                )
            return output
        except Exception as exc:
            peer_operation = None
            try:
                peer_operation = self.control.peek_ready(1 - self.rank, operation_id)
            except Exception:
                pass
            self._diag(
                "collective_error",
                operation_id=operation_id,
                error=f"{type(exc).__name__}: {exc}",
                peer_operation=peer_operation,
                total_bytes=total_bytes,
                remote_bytes=remote_bytes,
                capacity_bytes=self.capacity_bytes,
                chunks=chunk_count,
            )
            self._fail(exc)

    def _all_gather_chunk(
        self,
        input_bytes,
        output_bytes,
        shard_bytes,
        chunk_start,
        chunk_end,
    ):
        operation_id = self._operation_id
        self._operation_id += 1
        chunk_bytes = chunk_end - chunk_start

        with torch.cuda.stream(self._stream):
            self._wait_for_peer_buffer_consumption()
            self._send_buffer[:chunk_bytes].copy_(
                input_bytes[chunk_start:chunk_end],
                non_blocking=True,
            )
            self._ready_event.record(self._stream)

        self.control.publish_ready(self.rank, operation_id, chunk_bytes)
        wait_started_ns = time.perf_counter_ns() if self._profiler.enabled else 0
        peer_chunk_bytes = self.control.wait_ready(
            1 - self.rank,
            operation_id,
            timeout_seconds=self.timeout_seconds,
        )
        if self._profiler.enabled:
            self._profile_control_wait_ns += time.perf_counter_ns() - wait_started_ns
        if peer_chunk_bytes != chunk_bytes:
            raise ValueError(
                f"all-gather chunk size mismatch at operation {operation_id}: "
                f"local={chunk_bytes}, peer={peer_chunk_bytes}"
            )

        with torch.cuda.stream(self._stream):
            self._stream.wait_event(self._peer_ready_event)
            for dst_start, dst_end, source_kind, src_start, src_end in (
                all_gather_chunk_copy_plan(
                    self.rank, shard_bytes, chunk_start, chunk_end
                )
            ):
                source = input_bytes if source_kind == "input" else self._peer_buffer
                output_bytes[dst_start:dst_end].copy_(
                    source[src_start:src_end], non_blocking=True
                )
            self._consumed_event.record(self._stream)
        self.control.publish_consumed(self.rank, operation_id)
        self._buffer_operation_id = operation_id
        return operation_id

    def all_gather_into_tensor(self, output, input_tensor, async_op=False):
        self._ensure_healthy()
        if self._peer_buffer is None:
            raise P2PGroupError("P2P endpoint is not connected")

        shard_bytes = validate_all_gather(output, input_tensor, self.capacity_bytes)
        operation_id = None
        submit_started_ns = time.perf_counter_ns() if self._profiler.enabled else 0
        chunk_count = 0
        self._profile_control_wait_ns = 0
        current_stream = torch.cuda.current_stream(0)

        try:
            input_bytes = input_tensor.view(torch.uint8).reshape(-1)
            output_bytes = output.view(torch.uint8).reshape(-1)

            self._stream.wait_stream(current_stream)
            for chunk_start, chunk_end in chunk_ranges(shard_bytes, self.capacity_bytes):
                chunk_count += 1
                operation_id = self._all_gather_chunk(
                    input_bytes, output_bytes, shard_bytes, chunk_start, chunk_end
                )

            current_stream.wait_event(self._consumed_event)
            if self._profiler.enabled:
                self._profiler.record(
                    "all_gather",
                    payload_bytes=shard_bytes * 2,
                    remote_bytes=shard_bytes,
                    chunks=chunk_count,
                    control_wait_ns=self._profile_control_wait_ns,
                    submit_ns=time.perf_counter_ns() - submit_started_ns,
                )
            if async_op:
                return CudaEventWork(self._consumed_event, 0)
            return None
        except Exception as exc:
            peer_operation = None
            try:
                peer_operation = self.control.peek_ready(1 - self.rank, operation_id)
            except Exception:
                pass
            self._diag(
                "all_gather_error",
                operation_id=operation_id,
                error=f"{type(exc).__name__}: {exc}",
                peer_operation=peer_operation,
                total_bytes=shard_bytes * 2,
            )
            self._fail(exc)

    def profile_snapshot(self, reset=False):
        return self._profiler.snapshot(reset=reset)

    def _diag(self, event, **fields):
        if not self._diag_enabled:
            return
        payload = {
            "event": event,
            "perf_ns": time.perf_counter_ns(),
            "pid": os.getpid(),
            "rank": self.rank,
            "time_ns": time.time_ns(),
        }
        payload.update(fields)
        print(f"[RAYLIGHT_P2P_DIAG] {json.dumps(payload, sort_keys=True)}", flush=True)
    def close(self):
        self._peer_buffer = None
        self._peer_ready_event = None
        self._peer_consumed_event = None


class PollingDictControl:
    """Small Phase-1 control adapter over process-safe dicts."""

    def __init__(self, ready, consumed):
        self.ready = ready
        self.consumed = consumed

    @staticmethod
    def _wait(mapping, key, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if key in mapping:
                return mapping[key]
            time.sleep(0.0001)
        raise TimeoutError(f"timed out waiting for control key {key}")

    def publish_ready(self, rank, operation_id, total_bytes):
        self.ready[(rank, operation_id)] = total_bytes

    def wait_ready(self, rank, operation_id, timeout_seconds):
        return self._wait(self.ready, (rank, operation_id), timeout_seconds)

    def publish_consumed(self, rank, operation_id):
        self.consumed[(rank, operation_id)] = True

    def wait_consumed(self, rank, operation_id, timeout_seconds):
        return self._wait(self.consumed, (rank, operation_id), timeout_seconds)


class BarrierControl:
    """Low-overhead Phase-1 control adapter for spawned local processes."""

    def __init__(self, rank, ready_barrier, consumed_barrier, ready_operations, ready_sizes):
        self.rank = rank
        self.ready_barrier = ready_barrier
        self.consumed_barrier = consumed_barrier
        self.ready_operations = ready_operations
        self.ready_sizes = ready_sizes

    def publish_ready(self, rank, operation_id, total_bytes):
        if rank != self.rank:
            raise ValueError(f"control rank mismatch: endpoint={rank}, control={self.rank}")
        self.ready_operations[rank] = operation_id
        self.ready_sizes[rank] = total_bytes

    def wait_ready(self, rank, operation_id, timeout_seconds):
        self.ready_barrier.wait(timeout=timeout_seconds)
        peer_operation = self.ready_operations[rank]
        if peer_operation != operation_id:
            raise ValueError(
                f"collective operation mismatch: local={operation_id}, peer={peer_operation}"
            )
        return self.ready_sizes[rank]

    def publish_consumed(self, rank, operation_id):
        if rank != self.rank:
            raise ValueError(f"control rank mismatch: endpoint={rank}, control={self.rank}")

    def wait_consumed(self, rank, operation_id, timeout_seconds):
        self.consumed_barrier.wait(timeout=timeout_seconds)
        return True


class WindowsNamedControl:
    """Low-latency local control plane shared by independent Windows actors."""

    _SLOT = struct.Struct("<qq")
    _WAIT_OBJECT_0 = 0
    _WAIT_TIMEOUT = 258

    def __init__(self, group_name: str, rank: int):
        if sys.platform != "win32":
            raise RuntimeError("WindowsNamedControl is only available on Windows")
        if rank not in (0, 1):
            raise ValueError(f"rank must be 0 or 1, got {rank}")
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in group_name)
        if not safe_name:
            raise ValueError("group_name must contain at least one usable character")
        self.rank = rank
        self._mapping = mmap.mmap(
            -1,
            self._SLOT.size * 4,
            tagname=f"Local\\RaylightP2P_{safe_name}_metadata",
            access=mmap.ACCESS_WRITE,
        )
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateEventW.argtypes = (
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        )
        self._kernel32.CreateEventW.restype = wintypes.HANDLE
        self._kernel32.SetEvent.argtypes = (wintypes.HANDLE,)
        self._kernel32.SetEvent.restype = wintypes.BOOL
        self._kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        self._kernel32.WaitForSingleObject.restype = wintypes.DWORD
        self._kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._ready_events = tuple(
            self._create_event(f"Local\\RaylightP2P_{safe_name}_ready_{event_rank}")
            for event_rank in range(2)
        )
        self._consumed_events = tuple(
            self._create_event(
                f"Local\\RaylightP2P_{safe_name}_consumed_{event_rank}"
            )
            for event_rank in range(2)
        )
        self._closed = False

    def _create_event(self, name):
        handle = self._kernel32.CreateEventW(None, False, False, name)
        if not handle:
            raise OSError(ctypes.get_last_error(), f"CreateEventW failed for {name}")
        return handle

    def publish_ready(self, rank, operation_id, total_bytes):
        if rank != self.rank:
            raise ValueError(f"control rank mismatch: endpoint={rank}, control={self.rank}")
        self._SLOT.pack_into(self._mapping, rank * self._SLOT.size, operation_id, total_bytes)
        if not self._kernel32.SetEvent(self._ready_events[rank]):
            raise OSError(ctypes.get_last_error(), f"SetEvent failed for rank {rank}")

    def wait_ready(self, rank, operation_id, timeout_seconds):
        timeout_ms = max(0, min(round(timeout_seconds * 1000), 0xFFFFFFFE))
        result = self._kernel32.WaitForSingleObject(self._ready_events[rank], timeout_ms)
        if result == self._WAIT_TIMEOUT:
            raise TimeoutError(
                f"timed out waiting for rank {rank} operation {operation_id} after {timeout_seconds}s"
            )
        if result != self._WAIT_OBJECT_0:
            raise OSError(ctypes.get_last_error(), f"WaitForSingleObject failed with status {result}")
        peer_operation, total_bytes = self._SLOT.unpack_from(
            self._mapping,
            rank * self._SLOT.size,
        )
        if peer_operation != operation_id:
            raise ValueError(
                f"collective operation mismatch: local={operation_id}, peer={peer_operation}"
            )
        return total_bytes

    def publish_consumed(self, rank, operation_id):
        if rank != self.rank:
            raise ValueError(f"control rank mismatch: endpoint={rank}, control={self.rank}")
        offset = (2 + rank) * self._SLOT.size
        self._SLOT.pack_into(self._mapping, offset, operation_id, 1)
        if not self._kernel32.SetEvent(self._consumed_events[rank]):
            raise OSError(
                ctypes.get_last_error(),
                f"SetEvent failed for consumed rank {rank}",
            )

    def wait_consumed(self, rank, operation_id, timeout_seconds):
        timeout_ms = max(0, min(round(timeout_seconds * 1000), 0xFFFFFFFE))
        result = self._kernel32.WaitForSingleObject(
            self._consumed_events[rank], timeout_ms
        )
        if result == self._WAIT_TIMEOUT:
            raise TimeoutError(
                f"timed out waiting for rank {rank} to consume operation "
                f"{operation_id} after {timeout_seconds}s"
            )
        if result != self._WAIT_OBJECT_0:
            raise OSError(
                ctypes.get_last_error(),
                f"WaitForSingleObject consumed failed with status {result}",
            )
        peer_operation, _ = self._SLOT.unpack_from(
            self._mapping,
            (2 + rank) * self._SLOT.size,
        )
        if peer_operation != operation_id:
            raise ValueError(
                "collective consumed operation mismatch: "
                f"local={operation_id}, peer={peer_operation}"
            )
        return True

    def close(self):
        if self._closed:
            return
        self._closed = True
        for handle in self._ready_events + self._consumed_events:
            self._kernel32.CloseHandle(handle)
        self._mapping.close()


class WindowsSpinControl:
    """Append-only shared-memory handshake without a kernel transition per call."""

    _ENTRY = struct.Struct("<qq")
    _SLOT_COUNT = 131_072
    _READY_CHANNEL = 0
    _CONSUMED_CHANNEL = 1
    _CHANNEL_COUNT = 2

    def __init__(self, group_name: str, rank: int):
        if sys.platform != "win32":
            raise RuntimeError("WindowsSpinControl is only available on Windows")
        if rank not in (0, 1):
            raise ValueError(f"rank must be 0 or 1, got {rank}")
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in group_name)
        if not safe_name:
            raise ValueError("group_name must contain at least one usable character")
        self.rank = rank
        self._mapping = mmap.mmap(
            -1,
            self._ENTRY.size * 2 * self._SLOT_COUNT * self._CHANNEL_COUNT,
            tagname=f"Local\\RaylightP2P_{safe_name}_spin",
            access=mmap.ACCESS_WRITE,
        )
        self._closed = False

    @classmethod
    def _slot(cls, operation_id):
        if operation_id <= 0:
            raise ValueError(f"operation id must be positive, got {operation_id}")
        # Store the absolute operation id in a reusable ring slot. A rank cannot
        # advance to the next collective until its peer has published the current
        # one, so it cannot lap the peer by the entire ring capacity.
        return operation_id % cls._SLOT_COUNT

    @classmethod
    def _offset(cls, rank, operation_id, channel=_READY_CHANNEL):
        if channel not in (cls._READY_CHANNEL, cls._CONSUMED_CHANNEL):
            raise ValueError(f"unsupported control channel {channel}")
        slot = cls._slot(operation_id)
        channel_offset = channel * 2 * cls._SLOT_COUNT
        return (channel_offset + rank * cls._SLOT_COUNT + slot) * cls._ENTRY.size

    def publish_ready(self, rank, operation_id, total_bytes):
        if rank != self.rank:
            raise ValueError(f"control rank mismatch: endpoint={rank}, control={self.rank}")
        offset = self._offset(rank, operation_id)
        # Publish size before operation id. On the target x86_64 Windows system,
        # stores become visible in order to the peer process.
        struct.pack_into("<q", self._mapping, offset + 8, total_bytes)
        struct.pack_into("<q", self._mapping, offset, operation_id)

    def wait_ready(self, rank, operation_id, timeout_seconds):
        offset = self._offset(rank, operation_id)
        deadline = time.perf_counter() + timeout_seconds
        spins = 0
        while True:
            peer_operation = struct.unpack_from("<q", self._mapping, offset)[0]
            if peer_operation == operation_id:
                return struct.unpack_from("<q", self._mapping, offset + 8)[0]
            if peer_operation > operation_id:
                raise ValueError(
                    f"collective operation mismatch: local={operation_id}, peer={peer_operation}"
                )
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for rank {rank} operation {operation_id} after {timeout_seconds}s"
                )
            spins += 1
            if spins % 4096 == 0:
                time.sleep(0)

    def peek_ready(self, rank, operation_id):
        offset = self._offset(rank, operation_id)
        return struct.unpack_from("<q", self._mapping, offset)[0]

    def publish_consumed(self, rank, operation_id):
        if rank != self.rank:
            raise ValueError(f"control rank mismatch: endpoint={rank}, control={self.rank}")
        offset = self._offset(
            rank,
            operation_id,
            channel=self._CONSUMED_CHANNEL,
        )
        struct.pack_into("<q", self._mapping, offset, operation_id)

    def wait_consumed(self, rank, operation_id, timeout_seconds):
        offset = self._offset(
            rank,
            operation_id,
            channel=self._CONSUMED_CHANNEL,
        )
        deadline = time.perf_counter() + timeout_seconds
        spins = 0
        while True:
            peer_operation = struct.unpack_from("<q", self._mapping, offset)[0]
            if peer_operation == operation_id:
                return True
            if peer_operation > operation_id:
                raise ValueError(
                    "collective consumed operation mismatch: "
                    f"local={operation_id}, peer={peer_operation}"
                )
            if time.perf_counter() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for rank {rank} to consume operation "
                    f"{operation_id} after {timeout_seconds}s"
                )
            spins += 1
            if spins % 4096 == 0:
                time.sleep(0)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._mapping.close()
