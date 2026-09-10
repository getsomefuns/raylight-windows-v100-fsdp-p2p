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

        def mark(stage: str):
            print(f"[XFUSER_PROBE] rank={rank} stage={stage}", flush=True)

        sys.path.insert(0, raylight_src)

        mark("imports_begin")

        import torch
        import torch.distributed as dist
        from raylight.distributed_worker.windows_gloo import (
            BACKEND_NAME,
            init_windows_gloo_process_group,
            install_windows_gloo_new_group_router,
        )
        from xfuser.core.distributed import (
            get_sp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )

        mark("imports_ok")

        mark("default_pg_begin")
        gloo_host = init_windows_gloo_process_group(
            rank=rank,
            world_size=2,
            master_addr="127.0.0.1",
            port=port,
            timeout=timedelta(seconds=60),
        )
        mark("default_pg_ok")
        try:
            original_new_group = dist.new_group
            group_counter = 0

            def traced_new_group(*args, **kwargs):
                nonlocal group_counter
                group_counter += 1
                label = f"new_group_{group_counter}_{kwargs.get('backend', 'default')}"
                mark(label + "_begin")
                group = original_new_group(*args, **kwargs)
                mark(label + "_ok")
                return group

            dist.new_group = traced_new_group
            install_windows_gloo_new_group_router(dist)
            mark("xfuser_environment_begin")
            init_distributed_environment(
                rank=rank,
                world_size=2,
                backend=BACKEND_NAME,
            )
            mark("xfuser_environment_ok")
            mark("model_parallel_begin")
            initialize_model_parallel(
                data_parallel_degree=1,
                sequence_parallel_degree=2,
                classifier_free_guidance_degree=1,
                ring_degree=1,
                ulysses_degree=2,
                pipeline_parallel_degree=1,
                backend=BACKEND_NAME,
            )
            mark("model_parallel_ok")
            torch.cuda.set_device(0)
            source = torch.arange(8, device="cuda:0", dtype=torch.float32) + rank * 100
            output = torch.empty_like(source)
            mark("all_to_all_begin")
            dist.all_to_all_single(output, source, group=get_sp_group().device_group)
            mark("all_to_all_ok")
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
