"""Two-Ray-actor probe for xFuser subgroups on Raylight's Windows Gloo backend."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_memory_usage_threshold"] = "1"
os.environ["USE_LIBUV"] = "0"

import ray


RAYLIGHT_SRC = Path(__file__).resolve().parents[1] / "src"


@ray.remote(num_gpus=1)
class XFuserActor:
    def run(self, rank: int, port: int, raylight_src: str):
        import sys

        sys.path.insert(0, raylight_src)

        import torch
        import torch.distributed as dist
        from raylight.distributed_worker.windows_gloo import (
            BACKEND_NAME,
            init_windows_gloo_process_group,
        )
        from xfuser.core.distributed import (
            get_sp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )

        gloo_host = init_windows_gloo_process_group(
            rank=rank,
            world_size=2,
            master_addr="127.0.0.1",
            port=port,
            timeout=timedelta(seconds=60),
        )
        try:
            init_distributed_environment(
                rank=rank,
                world_size=2,
                backend=BACKEND_NAME,
            )
            initialize_model_parallel(
                data_parallel_degree=1,
                sequence_parallel_degree=2,
                classifier_free_guidance_degree=1,
                ring_degree=1,
                ulysses_degree=2,
                pipeline_parallel_degree=1,
                backend=BACKEND_NAME,
            )
            torch.cuda.set_device(0)
            source = torch.arange(8, device="cuda:0", dtype=torch.float32) + rank * 100
            output = torch.empty_like(source)
            dist.all_to_all_single(output, source, group=get_sp_group().device_group)
            return {
                "rank": rank,
                "gpu": torch.cuda.get_device_name(0),
                "gloo_device": gloo_host,
                "sp_world_size": get_sp_group().world_size,
                "result": output.cpu().tolist(),
            }
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    ray_temp = Path(os.environ["TEMP"]) / "raylight-xfuser-ray-probe"
    ray.init(
        num_gpus=2,
        include_dashboard=False,
        _temp_dir=str(ray_temp),
        ignore_reinit_error=True,
    )
    try:
        actors = [XFuserActor.remote() for _ in range(2)]
        results = ray.get(
            [
                actor.run.remote(rank, 29670, str(RAYLIGHT_SRC))
                for rank, actor in enumerate(actors)
            ],
            timeout=120,
        )
        for result in results:
            print(result, flush=True)
    finally:
        ray.shutdown()
