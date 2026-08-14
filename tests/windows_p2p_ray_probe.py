"""Ray actor probe for raw CUDA IPC metadata and NVLink data plane."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import time

os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_memory_usage_threshold"] = "1"

import ray


MODULE_PATH = Path(__file__).parents[1] / "src/raylight/distributed_worker/windows_p2p.py"


@ray.remote(num_gpus=1)
class P2PActor:
    def __init__(self, rank, group_name, capacity_bytes):
        import sys
        import torch

        spec = importlib.util.spec_from_file_location(f"raylight_windows_p2p_ray_{rank}", MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        torch.cuda.set_device(0)
        self.rank = rank
        self.module = module
        self.control = module.WindowsSpinControl(group_name, rank)
        self.endpoint = module.CudaP2PAllToAll(rank, capacity_bytes, self.control, timeout_seconds=10)

    def metadata(self):
        return self.endpoint.local_ipc_metadata()

    def connect(self, peer_metadata):
        self.endpoint.connect_ipc_metadata(peer_metadata)
        return True

    def run(self, iterations, size_bytes):
        import torch

        elements = size_bytes // 4
        half = elements // 2
        source = torch.empty(elements, dtype=torch.float32, device="cuda:0")
        source[:half].fill_(self.rank * 100 + 1)
        source[half:].fill_(self.rank * 100 + 2)
        output = torch.empty_like(source)
        for _ in range(3):
            self.endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            self.endpoint.all_to_all_single(output, source)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        expected = (1, 101) if self.rank == 0 else (2, 102)
        observed = (int(output[0].item()), int(output[half].item()))
        return {
            "rank": self.rank,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
            "values": observed,
            "expected": expected,
            "elapsed_seconds": elapsed,
            "remote_gib_s": (size_bytes / 2 / 2**30) * iterations / elapsed,
        }

    def close(self):
        self.endpoint.close()
        self.control.close()


if __name__ == "__main__":
    size_bytes = 115_343_360
    iterations = 100
    minimum_gib_s = float(os.environ.get("RAYLIGHT_WINDOWS_P2P_MIN_GIB_S", "50"))
    ray.init(num_gpus=2, include_dashboard=False, ignore_reinit_error=True)
    actors = []
    try:
        group_name = f"ray_{os.getpid()}_{time.time_ns()}"
        actors = [P2PActor.remote(rank, group_name, size_bytes // 2) for rank in range(2)]
        metadata = ray.get([actor.metadata.remote() for actor in actors])
        ray.get([actors[rank].connect.remote(metadata[1 - rank]) for rank in range(2)])
        results = ray.get([actor.run.remote(iterations, size_bytes) for actor in actors], timeout=120)
        for result in results:
            print(result)
            if result["values"] != result["expected"]:
                raise RuntimeError(result)
            if result["remote_gib_s"] < minimum_gib_s:
                raise RuntimeError(
                    f"rank {result['rank']} P2P bandwidth {result['remote_gib_s']:.2f} GiB/s "
                    f"is below the required {minimum_gib_s:.2f} GiB/s"
                )
        ray.get([actor.close.remote() for actor in actors])
    finally:
        ray.shutdown()
