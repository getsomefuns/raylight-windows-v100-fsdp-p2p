"""Full-element correctness probe for real Raylight all-to-all payloads."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time


MODULE_PATH = Path(__file__).parents[1] / "src/raylight/distributed_worker/windows_p2p.py"
REAL_SHAPES = (
    (2, 63, 1, 16, 64),
    (2, 512, 1, 16, 128),
    (2, 1760, 1, 16, 128),
    (2, 7040, 1, 16, 128),
)


def _load_module(rank):
    name = f"raylight_windows_p2p_full_rank_{rank}"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _expected(torch, rank, element_count):
    half = element_count // 2
    local = torch.arange(element_count, dtype=torch.int32, device="cuda:0")
    peer = local + 10_000_000
    if rank == 0:
        return torch.cat((local[:half], peer[:half]))
    return torch.cat((local[half:], peer[half:]))


def _worker(rank, group_name, handle_queues, barrier, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch

    module = _load_module(rank)
    torch.cuda.set_device(0)
    capacity = 64 * 1024 * 1024
    control = module.WindowsSpinControl(group_name, rank)
    endpoint = module.CudaP2PAllToAll(rank, capacity, control, timeout_seconds=10)
    handle_queues[1 - rank].put(endpoint.local_handles())
    endpoint.connect(handle_queues[rank].get(timeout=60))

    checks = []
    try:
        for shape in REAL_SHAPES:
            element_count = 1
            for dimension in shape:
                element_count *= dimension
            # int32 matches the four-byte float32 payloads observed in the LTX trace.
            source = torch.arange(element_count, dtype=torch.int32, device="cuda:0")
            source.add_(rank * 10_000_000)
            source = source.reshape(shape)
            output = torch.empty_like(source)
            barrier.wait(timeout=30)
            endpoint.all_to_all_single(output, source)
            torch.cuda.synchronize()
            expected = _expected(torch, rank, element_count)
            observed = output.reshape(-1)
            mismatches = torch.count_nonzero(observed != expected).item()
            maximum_error = torch.max(torch.abs(observed - expected)).item()
            checks.append(
                {
                    "shape": shape,
                    "bytes": source.numel() * source.element_size(),
                    "mismatches": mismatches,
                    "maximum_error": maximum_error,
                }
            )
            if mismatches:
                raise RuntimeError(f"rank={rank} shape={shape} mismatches={mismatches}")
        results.put((rank, checks, None))
    except Exception as exc:
        results.put((rank, checks, repr(exc)))
        raise
    finally:
        endpoint.close()
        control.close()


def main():
    ctx = mp.get_context("spawn")
    handle_queues = [ctx.Queue(), ctx.Queue()]
    barrier = ctx.Barrier(2)
    results = ctx.Queue()
    group_name = f"full_{os.getpid()}_{time.time_ns()}"
    processes = [
        ctx.Process(target=_worker, args=(rank, group_name, handle_queues, barrier, results))
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
    for rank, checks, error in sorted(rows):
        for check in checks:
            print(f"rank={rank} {check}")
        if error:
            print(f"rank={rank} error={error}")
    if len(rows) != 2 or any(process.exitcode != 0 for process in processes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
