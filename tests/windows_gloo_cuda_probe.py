"""Two-process probe for native Windows Gloo CUDA all-to-all support."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path


def _worker(rank: int, port: int) -> None:
    os.environ["USE_LIBUV"] = "0"
    os.environ.pop("GLOO_SOCKET_IFNAME", None)

    import torch
    import torch.distributed as dist

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "raylight"
        / "distributed_worker"
        / "windows_gloo.py"
    )
    spec = importlib.util.spec_from_file_location("raylight_windows_gloo", module_path)
    windows_gloo = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(windows_gloo)
    print(f"rank={rank} stage=init_process_group", flush=True)
    selected_host = windows_gloo.init_windows_gloo_process_group(
        rank=rank,
        world_size=2,
        master_addr="127.0.0.1",
        port=port,
        timeout=timedelta(seconds=30),
    )
    print(f"rank={rank} gloo_device={selected_host}", flush=True)
    try:
        print(f"rank={rank} stage=barrier", flush=True)
        dist.barrier()
        print(f"rank={rank} stage=cuda", flush=True)
        torch.cuda.set_device(rank)
        source = torch.arange(8, device=f"cuda:{rank}", dtype=torch.float32) + rank * 100
        output = torch.empty_like(source)
        dist.all_to_all_single(output, source)
        print(f"rank={rank} output={output.cpu().tolist()}", flush=True)
    finally:
        dist.destroy_process_group()


def _parent(port: int) -> int:
    script = str(Path(__file__).resolve())
    processes = [
        subprocess.Popen(
            [sys.executable, script, "--worker", str(rank), "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(2)
    ]
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and any(process.poll() is None for process in processes):
        time.sleep(0.1)
    failed = False
    for process in processes:
        if process.poll() is None:
            process.kill()
            failed = True
        output, _ = process.communicate()
        if process.returncode != 0 and "Traceback" not in output:
            output += f"probe stopped pid={process.pid} returncode={process.returncode}\n"
        print(output, end="")
        failed |= process.returncode != 0
    return int(failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--port", type=int, default=29631)
    args = parser.parse_args()
    if args.worker is None:
        raise SystemExit(_parent(args.port))
    _worker(args.worker, args.port)
