"""Ray actor probe for Raylight's Windows Gloo CUDA backend."""

from __future__ import annotations

import importlib.util
import os
from datetime import timedelta
from pathlib import Path

os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_memory_usage_threshold"] = "1"
os.environ["USE_LIBUV"] = "0"

import ray


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "raylight"
    / "distributed_worker"
    / "windows_gloo.py"
)


@ray.remote(num_gpus=1)
class GlooCudaActor:
    def run(self, rank: int, port: int, module_path: str):
        import torch
        import torch.distributed as dist

        spec = importlib.util.spec_from_file_location(
            f"raylight_windows_gloo_rank_{rank}", module_path
        )
        windows_gloo = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(windows_gloo)

        host = windows_gloo.init_windows_gloo_process_group(
            rank=rank,
            world_size=2,
            master_addr="127.0.0.1",
            port=port,
            timeout=timedelta(seconds=60),
        )
        try:
            torch.cuda.set_device(0)
            source = torch.arange(8, device="cuda:0", dtype=torch.float32) + rank * 100
            output = torch.empty_like(source)
            dist.all_to_all_single(output, source)
            return {
                "rank": rank,
                "gpu": torch.cuda.get_device_name(0),
                "gloo_device": host,
                "result": output.cpu().tolist(),
            }
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    ray_temp = Path(os.environ["TEMP"]) / "raylight-ray-probe"
    ray.init(
        num_gpus=2,
        include_dashboard=False,
        _temp_dir=str(ray_temp),
        ignore_reinit_error=True,
    )
    try:
        actors = [GlooCudaActor.remote() for _ in range(2)]
        results = ray.get(
            [actor.run.remote(rank, 29660, str(MODULE_PATH)) for rank, actor in enumerate(actors)],
            timeout=90,
        )
        for result in results:
            print(result, flush=True)
    finally:
        ray.shutdown()
