"""Measure aggregate profiler overhead on the real two-V100 P2P endpoint."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_memory_usage_threshold"] = "1"

import ray


REPO_ROOT = Path(__file__).parents[1]
MODULE_PATH = REPO_ROOT / "src/raylight/distributed_worker/windows_p2p.py"
CAPACITY_BYTES = 18 * 1024 * 1024
TARGET_REMOTE_BYTES = 100 * 1024**3


@ray.remote(num_gpus=1)
class ProfileProbeActor:
    def __init__(self, rank: int, group_name: str):
        import torch

        spec = importlib.util.spec_from_file_location(
            f"raylight_windows_p2p_profile_{rank}", MODULE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.cuda.set_device(0)
        self.rank = rank
        self.module = module
        self.control = module.WindowsSpinControl(group_name, rank)
        self.endpoint = module.CudaP2PAllToAll(
            rank, CAPACITY_BYTES, self.control, timeout_seconds=10
        )

        a2a_elements = (CAPACITY_BYTES * 2) // 4
        half = a2a_elements // 2
        self.a2a_source = torch.empty(a2a_elements, dtype=torch.float32, device="cuda:0")
        self.a2a_source[:half].fill_(rank * 100 + 1)
        self.a2a_source[half:].fill_(rank * 100 + 2)
        self.a2a_output = torch.empty_like(self.a2a_source)
        gather_elements = CAPACITY_BYTES // 4
        self.gather_source = torch.full(
            (gather_elements,), rank + 1, dtype=torch.float32, device="cuda:0"
        )
        self.gather_output = torch.empty(
            gather_elements * 2, dtype=torch.float32, device="cuda:0"
        )

    def metadata(self):
        return self.endpoint.local_ipc_metadata()

    def connect(self, peer_metadata):
        self.endpoint.connect_ipc_metadata(peer_metadata)
        return True

    def warmup(self, iterations: int):
        import torch

        for _ in range(iterations):
            self.endpoint.all_to_all_single(self.a2a_output, self.a2a_source)
            self.endpoint.all_gather_into_tensor(
                self.gather_output, self.gather_source, async_op=False
            )
        torch.cuda.synchronize()
        return True

    def run_trial(self, enabled: bool, iterations: int):
        import torch
        from raylight.distributed_worker.collective_profile import CollectiveProfiler

        self.endpoint._profiler = CollectiveProfiler(enabled=enabled)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            self.endpoint.all_to_all_single(self.a2a_output, self.a2a_source)
            self.endpoint.all_gather_into_tensor(
                self.gather_output, self.gather_source, async_op=False
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        half = self.a2a_output.numel() // 2
        expected_a2a = (1, 101) if self.rank == 0 else (2, 102)
        observed_a2a = (
            int(self.a2a_output[0].item()),
            int(self.a2a_output[half].item()),
        )
        gather_elements = self.gather_source.numel()
        observed_gather = (
            int(self.gather_output[0].item()),
            int(self.gather_output[gather_elements].item()),
        )
        profile = self.endpoint.profile_snapshot(reset=False)
        remote_bytes = iterations * CAPACITY_BYTES * 2
        return {
            "enabled": enabled,
            "elapsed_seconds": elapsed,
            "expected_a2a": expected_a2a,
            "expected_gather": (1, 2),
            "observed_a2a": observed_a2a,
            "observed_gather": observed_gather,
            "profile": profile,
            "rank": self.rank,
            "remote_gib_s": remote_bytes / 2**30 / elapsed,
        }

    def close(self):
        self.endpoint.close()
        self.control.close()


def main():
    iterations = math.ceil(TARGET_REMOTE_BYTES / (CAPACITY_BYTES * 2))
    ray.init(num_gpus=2, include_dashboard=False, ignore_reinit_error=True)
    actors = []
    report = {
        "capacity_bytes": CAPACITY_BYTES,
        "iterations": iterations,
        "target_remote_bytes": TARGET_REMOTE_BYTES,
        "trials": [],
    }
    try:
        group_name = f"profile_{os.getpid()}_{time.time_ns()}"
        actors = [ProfileProbeActor.remote(rank, group_name) for rank in range(2)]
        metadata = ray.get([actor.metadata.remote() for actor in actors])
        ray.get([actors[rank].connect.remote(metadata[1 - rank]) for rank in range(2)])
        ray.get([actor.warmup.remote(200) for actor in actors], timeout=60)

        orders = ((False, True), (True, False), (False, True), (True, False), (False, True))
        for pair_index, pair in enumerate(orders):
            for enabled in pair:
                ranks = ray.get(
                    [actor.run_trial.remote(enabled, iterations) for actor in actors],
                    timeout=120,
                )
                for rank in ranks:
                    if rank["observed_a2a"] != rank["expected_a2a"]:
                        raise RuntimeError(rank)
                    if rank["observed_gather"] != rank["expected_gather"]:
                        raise RuntimeError(rank)
                    if enabled:
                        collectives = rank["profile"]["collectives"]
                        if collectives["all_to_all"]["calls"] != iterations:
                            raise RuntimeError(rank)
                        if collectives["all_gather"]["calls"] != iterations:
                            raise RuntimeError(rank)
                trial = {
                    "enabled": enabled,
                    "pair_index": pair_index,
                    "elapsed_seconds": max(rank["elapsed_seconds"] for rank in ranks),
                    "minimum_remote_gib_s": min(rank["remote_gib_s"] for rank in ranks),
                    "ranks": ranks,
                }
                report["trials"].append(trial)
                print(json.dumps(trial, sort_keys=True), flush=True)

        off = [trial["elapsed_seconds"] for trial in report["trials"] if not trial["enabled"]]
        on = [trial["elapsed_seconds"] for trial in report["trials"] if trial["enabled"]]
        report["median_off_seconds"] = statistics.median(off)
        report["median_on_seconds"] = statistics.median(on)
        report["overhead_percent"] = (
            report["median_on_seconds"] / report["median_off_seconds"] - 1.0
        ) * 100.0
        report["accepted"] = report["overhead_percent"] < 2.0

        output_dir = REPO_ROOT.parents[2] / "logs/f4"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "f4-p2p-profiler-overhead.json"
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"summary": report, "output": str(output_path)}, sort_keys=True))
        if not report["accepted"]:
            raise RuntimeError(
                f"profiler overhead {report['overhead_percent']:.3f}% exceeds 2% gate"
            )
    finally:
        if actors:
            ray.get([actor.close.remote() for actor in actors], timeout=30)
        ray.shutdown()


if __name__ == "__main__":
    main()
