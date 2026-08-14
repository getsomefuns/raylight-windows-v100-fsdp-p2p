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


def _load_module(rank):
    spec = importlib.util.spec_from_file_location(f"raylight_windows_p2p_rank_{rank}", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _worker(rank, group_name, handles_queues, benchmark_barrier, results, iterations, sizes, dtype_name, shape):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch

    module = _load_module(rank)
    torch.cuda.set_device(0)
    capacity = max(sizes) // 2
    control = module.WindowsSpinControl(group_name, rank)
    endpoint = module.CudaP2PAllToAll(rank, capacity, control, timeout_seconds=10)
    handles_queues[1 - rank].put(endpoint.local_handles())
    peer_handles = handles_queues[rank].get(timeout=60)
    endpoint.connect(peer_handles)

    dtype = getattr(torch, DTYPES[dtype_name])
    rows = []
    for size_bytes in sizes:
        element_count = size_bytes // torch.empty((), dtype=dtype).element_size()
        half = element_count // 2
        tensor_shape = shape or (element_count,)
        if torch.tensor(tensor_shape).prod().item() != element_count:
            raise ValueError(f"shape {tensor_shape} does not match {element_count} elements")
        source = torch.empty(tensor_shape, dtype=dtype, device="cuda:0")
        source_flat = source.reshape(-1)
        source_flat[:half].fill_(rank * 100 + 1)
        source_flat[half:].fill_(rank * 100 + 2)
        output = torch.empty_like(source)

        for _ in range(3):
            endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        expected = (1, 101) if rank == 0 else (2, 102)
        output_flat = output.reshape(-1)
        observed = (int(output_flat[0].item()), int(output_flat[half].item()))
        if observed != expected:
            raise RuntimeError(
                f"rank={rank} size={size_bytes} values={observed} expected={expected}"
            )

        benchmark_barrier.wait(timeout=10)
        start = time.perf_counter()
        for _ in range(iterations):
            endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        # Half of each all-to-all payload is remote traffic in each direction.
        remote_gib_s = (size_bytes / 2 / 2**30) * iterations / elapsed
        rows.append((size_bytes, iterations, elapsed, remote_gib_s))
        benchmark_barrier.wait(timeout=10)

    results.put((rank, rows))
    benchmark_barrier.wait(timeout=10)
    endpoint.close()
    control.close()


def main(iterations, stress_iterations, dtype_name, shape):
    ctx = mp.get_context("spawn")
    handles_queues = [ctx.Queue(), ctx.Queue()]
    benchmark_barrier = ctx.Barrier(2)
    results = ctx.Queue()
    group_name = f"probe_{os.getpid()}_{time.time_ns()}"
    sizes = (516_096,) if stress_iterations else REAL_SIZES
    effective_iterations = stress_iterations or iterations
    processes = [
        ctx.Process(
            target=_worker,
            args=(rank, group_name, handles_queues, benchmark_barrier, results, effective_iterations, sizes, dtype_name, shape),
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
        for size_bytes, count, elapsed, gib_s in rank_rows:
            print(
                f"rank={rank} bytes={size_bytes} iterations={count} "
                f"elapsed_s={elapsed:.6f} remote_GiB_s={gib_s:.3f}"
            )
    if len(rows) != 2 or any(process.exitcode != 0 for process in processes):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--stress-iterations", type=int, default=0)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--shape", type=int, nargs="+")
    args = parser.parse_args()
    main(
        args.iterations,
        args.stress_iterations,
        args.dtype,
        tuple(args.shape) if args.shape else None,
    )
