"""Measure native Windows Gloo CUDA all-to-all throughput on two local GPUs."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


SIZES_MIB = (1, 4, 16, 64, 128)


def _load_windows_gloo():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "raylight"
        / "distributed_worker"
        / "windows_gloo.py"
    )
    spec = importlib.util.spec_from_file_location("raylight_windows_gloo", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _iterations(size_mib: int) -> int:
    if size_mib <= 4:
        return 30
    if size_mib <= 16:
        return 15
    return 6


def _worker(rank: int, port: int, use_sync: bool) -> None:
    os.environ["USE_LIBUV"] = "0"

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    windows_gloo = _load_windows_gloo()
    selected_host = windows_gloo.init_windows_gloo_process_group(
        rank=rank,
        world_size=2,
        master_addr="127.0.0.1",
        port=port,
        timeout=timedelta(seconds=90),
    )
    if rank == 0:
        print(f"backend={dist.get_backend()} gloo_device={selected_host} use_sync={use_sync}", flush=True)

    try:
        for size_mib in SIZES_MIB:
            elements = size_mib * 1024 * 1024 // 2
            source = torch.full((elements,), rank + 1, dtype=torch.float16, device=f"cuda:{rank}")
            output = torch.empty_like(source)
            for _ in range(3):
                dist.all_to_all_single(output, source)
                if use_sync:
                    torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()

            iterations = _iterations(size_mib)
            start = time.perf_counter()
            for _ in range(iterations):
                dist.all_to_all_single(output, source)
                if use_sync:
                    torch.cuda.synchronize()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            elapsed_tensor = torch.tensor([elapsed], dtype=torch.float64)
            dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
            if rank == 0:
                seconds = elapsed_tensor.item()
                logical_gib_s = (size_mib / 1024) * iterations / seconds
                remote_gib_s = logical_gib_s / 2
                print(
                    f"size_mib={size_mib} iterations={iterations} elapsed_s={seconds:.6f} "
                    f"logical_GiB_s={logical_gib_s:.4f} remote_per_rank_GiB_s={remote_gib_s:.4f}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


def _parent(port: int, use_sync: bool) -> int:
    script = str(Path(__file__).resolve())
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                script,
                "--worker",
                str(rank),
                "--port",
                str(port),
                "--use-sync",
                str(int(use_sync)),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        for rank in range(2)
    ]
    failed = False
    for process in processes:
        output, _ = process.communicate(timeout=240)
        print(output, end="")
        failed |= process.returncode != 0
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--port", type=int, default=29641)
    parser.add_argument("--use-sync", type=int, default=1)
    args = parser.parse_args()
    if args.worker is None:
        raise SystemExit(_parent(args.port, bool(args.use_sync)))
    _worker(args.worker, args.port, bool(args.use_sync))
