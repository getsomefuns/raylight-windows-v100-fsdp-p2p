"""Minimal two-process Windows TCPStore diagnostic probe."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


def child(rank: int, port: int, master_addr: str) -> None:
    import torch.distributed as dist

    started = time.perf_counter()
    print(f"rank={rank} create addr={master_addr}:{port}", flush=True)
    store = dist.TCPStore(
        master_addr,
        port,
        2,
        rank == 0,
        timeout=timedelta(seconds=30),
        wait_for_workers=True,
        use_libuv=False,
    )
    print(f"rank={rank} created elapsed={time.perf_counter() - started:.3f}s", flush=True)
    store.set(f"ready-{rank}", "1")
    store.wait(["ready-0", "ready-1"])
    print(f"rank={rank} synchronized", flush=True)
    del store
    gc.collect()
    print(f"rank={rank} destroyed elapsed={time.perf_counter() - started:.3f}s", flush=True)


def parent(port: int, master_addr: str) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--port",
        str(port),
        "--master-addr",
        master_addr,
    ]
    env = os.environ.copy()
    env["USE_LIBUV"] = "0"
    processes = [
        subprocess.Popen(command + ["--rank", str(rank)], env=env)
        for rank in range(2)
    ]
    return max(process.wait(timeout=60) for process in processes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--master-addr", required=True)
    args = parser.parse_args()
    if args.child:
        child(args.rank, args.port, args.master_addr)
        return 0
    return parent(args.port, args.master_addr)


if __name__ == "__main__":
    raise SystemExit(main())
