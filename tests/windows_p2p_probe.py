"""Spawned two-V100 correctness and throughput probe for windows_p2p.py."""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src/raylight/distributed_worker/windows_p2p.py"
)
REAL_SIZES = (516_096, 4_194_304, 8_388_608, 14_417_920, 28_835_840, 57_671_680, 115_343_360)
DTYPES = {
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
    "int32": "int32",
    "uint8": "uint8",
}


def validation_positions(remote_bytes: int, capacity_bytes: int) -> tuple[int, ...]:
    positions = {0, remote_bytes - 1}
    for boundary in range(capacity_bytes, remote_bytes, capacity_bytes):
        positions.add(boundary - 1)
        positions.add(boundary)
    return tuple(sorted(position for position in positions if 0 <= position < remote_bytes))


def chunk_byte_value(rank: int, half_index: int, chunk_index: int) -> int:
    return rank * 96 + half_index * 32 + chunk_index + 1


def expected_output_byte(
    output_rank: int,
    output_half: int,
    relative_offset: int,
    capacity_bytes: int,
) -> int:
    source_rank, source_half = (
        ((0, 0), (1, 0))
        if output_rank == 0
        else ((0, 1), (1, 1))
    )[output_half]
    return chunk_byte_value(
        source_rank,
        source_half,
        relative_offset // capacity_bytes,
    )


def _load_module(rank):
    spec = importlib.util.spec_from_file_location(f"raylight_windows_p2p_rank_{rank}", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _worker(
    rank,
    group_name,
    handles_queues,
    benchmark_barrier,
    results,
    iterations,
    sizes,
    capacity_bytes,
    dtype_name,
    shape,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch

    module = _load_module(rank)
    torch.cuda.set_device(0)
    control = module.WindowsSpinControl(group_name, rank)
    endpoint = module.CudaP2PAllToAll(
        rank,
        capacity_bytes,
        control,
        timeout_seconds=10,
    )
    endpoint_allocated_bytes = torch.cuda.memory_allocated()
    handles_queues[1 - rank].put(endpoint.local_handles())
    peer_handles = handles_queues[rank].get(timeout=60)
    endpoint.connect(peer_handles)

    dtype = getattr(torch, DTYPES[dtype_name])
    rows = []
    for size_bytes in sizes:
        element_count = size_bytes // torch.empty((), dtype=dtype).element_size()
        tensor_shape = shape or (element_count,)
        if torch.tensor(tensor_shape).prod().item() != element_count:
            raise ValueError(f"shape {tensor_shape} does not match {element_count} elements")
        source = torch.empty(tensor_shape, dtype=dtype, device="cuda:0")
        source_bytes = source.view(torch.uint8).reshape(-1)
        remote_bytes = size_bytes // 2
        for half_index in (0, 1):
            half_offset = half_index * remote_bytes
            for chunk_index, chunk_start in enumerate(
                range(0, remote_bytes, capacity_bytes)
            ):
                chunk_end = min(chunk_start + capacity_bytes, remote_bytes)
                source_bytes[
                    half_offset + chunk_start : half_offset + chunk_end
                ].fill_(chunk_byte_value(rank, half_index, chunk_index))
        output = torch.empty_like(source)
        payload_allocated_bytes = torch.cuda.memory_allocated()

        for _ in range(3):
            endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        output_bytes = output.view(torch.uint8).reshape(-1)
        for output_half in (0, 1):
            half_offset = output_half * remote_bytes
            for position in validation_positions(remote_bytes, capacity_bytes):
                observed = int(output_bytes[half_offset + position].item())
                expected = expected_output_byte(
                    rank,
                    output_half,
                    position,
                    capacity_bytes,
                )
                if observed != expected:
                    raise RuntimeError(
                        f"rank={rank} size={size_bytes} half={output_half} "
                        f"position={position} value={observed} expected={expected}"
                    )

        benchmark_barrier.wait(timeout=10)
        start = time.perf_counter()
        for _ in range(iterations):
            endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        # Half of each all-to-all payload is remote traffic in each direction.
        remote_gib_s = (size_bytes / 2 / 2**30) * iterations / elapsed
        profile = endpoint.profile_snapshot(reset=False)
        all_to_all_profile = profile.get("collectives", {}).get("all_to_all", {})
        profile_calls = all_to_all_profile.get("calls", 0)
        profile_chunks = all_to_all_profile.get("chunks", 0)
        next_operation_id = endpoint._operation_id
        rows.append(
            (
                size_bytes,
                iterations,
                elapsed,
                remote_gib_s,
                endpoint_allocated_bytes,
                payload_allocated_bytes,
                next_operation_id,
                profile_calls,
                profile_chunks,
            )
        )
        benchmark_barrier.wait(timeout=10)

    results.put((rank, rows))
    benchmark_barrier.wait(timeout=10)
    endpoint.close()
    control.close()


def resolve_probe_geometry(capacity_bytes, size_bytes, stress_iterations):
    sizes = tuple(size_bytes) if size_bytes is not None else (
        (516_096,) if stress_iterations else REAL_SIZES
    )
    if any(size <= 0 for size in sizes):
        raise ValueError("size_bytes must be positive")
    if any(size % 2 for size in sizes):
        raise ValueError("size_bytes must be divisible by 2")
    resolved_capacity = max(sizes) // 2 if capacity_bytes is None else capacity_bytes
    if resolved_capacity <= 0:
        raise ValueError("capacity_bytes must be positive")
    return resolved_capacity, sizes


def main(
    iterations,
    stress_iterations,
    dtype_name,
    shape,
    capacity_bytes=None,
    size_bytes=None,
    profile=False,
):
    if profile:
        os.environ["RAYLIGHT_P2P_PROFILE"] = "1"
    ctx = mp.get_context("spawn")
    handles_queues = [ctx.Queue(), ctx.Queue()]
    benchmark_barrier = ctx.Barrier(2)
    results = ctx.Queue()
    group_name = f"probe_{os.getpid()}_{time.time_ns()}"
    capacity_bytes, sizes = resolve_probe_geometry(
        capacity_bytes,
        size_bytes,
        stress_iterations,
    )
    effective_iterations = stress_iterations or iterations
    processes = [
        ctx.Process(
            target=_worker,
            args=(
                rank,
                group_name,
                handles_queues,
                benchmark_barrier,
                results,
                effective_iterations,
                sizes,
                capacity_bytes,
                dtype_name,
                shape,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    rows = []
    try:
        for _ in processes:
            rows.append(results.get(timeout=180))
    except queue.Empty:
        pass
    for process in processes:
        process.join(timeout=10)
        if process.is_alive():
            process.kill()
        print(f"pid={process.pid} exitcode={process.exitcode}")
    for rank, rank_rows in sorted(rows):
        for (
            size_bytes,
            count,
            elapsed,
            gib_s,
            endpoint_allocated_bytes,
            payload_allocated_bytes,
            next_operation_id,
            profile_calls,
            profile_chunks,
        ) in rank_rows:
            print(
                f"rank={rank} bytes={size_bytes} iterations={count} "
                f"elapsed_s={elapsed:.6f} remote_GiB_s={gib_s:.3f} "
                f"endpoint_allocated_bytes={endpoint_allocated_bytes} "
                f"payload_allocated_bytes={payload_allocated_bytes} "
                f"next_operation_id={next_operation_id} "
                f"profile_calls={profile_calls} profile_chunks={profile_chunks}"
            )
    if len(rows) != 2 or any(process.exitcode != 0 for process in processes):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--stress-iterations", type=int, default=0)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--shape", type=int, nargs="+")
    parser.add_argument("--capacity-bytes", type=int)
    parser.add_argument("--size-bytes", type=int, nargs="+")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    main(
        args.iterations,
        args.stress_iterations,
        args.dtype,
        tuple(args.shape) if args.shape else None,
        args.capacity_bytes,
        tuple(args.size_bytes) if args.size_bytes else None,
        args.profile,
    )
