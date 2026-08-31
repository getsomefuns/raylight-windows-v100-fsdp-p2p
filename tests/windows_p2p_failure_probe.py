"""Failure-injection probe: operation mismatch and missing peer must fail fast."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
import time


MODULE_PATH = Path(__file__).parents[1] / "src/raylight/distributed_worker/windows_p2p.py"


class SecondChunkMismatchControl:
    def __init__(self, delegate, rank, target_operation_id):
        self._delegate = delegate
        self._rank = rank
        self._target_operation_id = target_operation_id

    def publish_ready(self, rank, operation_id, total_bytes):
        if rank == self._rank == 1 and operation_id == self._target_operation_id:
            total_bytes -= 1
        return self._delegate.publish_ready(rank, operation_id, total_bytes)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def _load_module(rank):
    import sys
    spec = importlib.util.spec_from_file_location(f"raylight_windows_p2p_failure_{rank}", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _worker(rank, mode, group_name, queues, process_barrier, results):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    import torch

    module = _load_module(rank)
    torch.cuda.set_device(0)
    control = module.WindowsSpinControl(group_name, rank)
    endpoint_control = (
        SecondChunkMismatchControl(control, rank, target_operation_id=3)
        if mode == "second_chunk_mismatch"
        else control
    )
    endpoint = module.CudaP2PAllToAll(
        rank,
        2 * 1024 * 1024,
        endpoint_control,
        timeout_seconds=2,
    )
    queues[1 - rank].put(endpoint.local_handles())
    endpoint.connect(queues[rank].get(timeout=30))
    process_barrier.wait(timeout=10)

    if mode == "mismatch" and rank == 1:
        endpoint._operation_id += 1
    if mode == "missing_peer" and rank == 1:
        results.put((rank, "exited_before_collective", 0.0, False))
        return

    total_bytes = 8 * 1024 * 1024 if mode == "second_chunk_mismatch" else 4096
    source = torch.ones(total_bytes // 4, dtype=torch.float32, device="cuda:0")
    output = torch.empty_like(source)
    started = time.perf_counter()
    try:
        endpoint.all_to_all_single(output, source)
    except module.P2PGroupError as exc:
        elapsed = time.perf_counter() - started
        future_rejected = False
        try:
            endpoint.all_to_all_single(output, source)
        except module.P2PGroupError as poisoned:
            if "poisoned" not in str(poisoned):
                raise
            future_rejected = True
        results.put((rank, str(exc), elapsed, future_rejected))
    else:
        results.put(
            (
                rank,
                "unexpected_success",
                time.perf_counter() - started,
                False,
            )
        )


def run_mode(mode):
    ctx = mp.get_context("spawn")
    group_name = f"failure_{mode}_{os.getpid()}_{time.time_ns()}"
    queues = [ctx.Queue(), ctx.Queue()]
    process_barrier = ctx.Barrier(2)
    results = ctx.Queue()
    processes = [
        ctx.Process(target=_worker, args=(rank, mode, group_name, queues, process_barrier, results))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    rows = [results.get(timeout=15) for _ in processes]
    for process in processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
        if process.exitcode != 0:
            raise RuntimeError(f"{mode}: worker {process.pid} failed with {process.exitcode}")
    for row in sorted(rows):
        print(
            f"mode={mode} rank={row[0]} elapsed_s={row[2]:.3f} "
            f"future_rejected={row[3]} result={row[1]}"
        )
    rank0 = next(row for row in rows if row[0] == 0)
    if rank0[1] == "unexpected_success" or rank0[2] > 5 or not rank0[3]:
        raise RuntimeError(f"{mode}: rank 0 did not fail fast: {rank0}")


if __name__ == "__main__":
    run_mode("mismatch")
    run_mode("missing_peer")
    run_mode("second_chunk_mismatch")
