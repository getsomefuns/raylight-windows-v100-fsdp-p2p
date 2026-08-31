"""Real two-RayWorker probe for the external TP2 text-encoder GPU lease."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def _install_runtime(comfy_root: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, str(comfy_root))
    os.chdir(comfy_root)
    sys.argv = [sys.argv[0], "--disable-cuda-malloc", "--reserve-vram", "4"]
    os.environ.update(
        {
            "RAY_DEBUG_DISABLE_MEMORY_MONITOR": "1",
            "RAY_memory_usage_threshold": "1",
            "USE_LIBUV": "0",
            "MASTER_ADDR": "127.0.0.1",
            "RAYLIGHT_WINDOWS_P2P": "0",
        }
    )
    from comfy.options import enable_args_parsing

    enable_args_parsing()


def _parallel_dict() -> dict:
    return {
        "global_world_size": 2,
        "shard_size": 2,
        "use_group_process_group": False,
        "use_mmap": True,
        "pp_degree": 1,
        "dp_degree": 1,
        "clear_vram_after_sampling": False,
        "is_xdit": False,
        "is_fsdp": False,
        "sync_ulysses": False,
        "pipefusion_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=29720)
    args = parser.parse_args()
    comfy_root = args.comfy_root.resolve()
    _install_runtime(comfy_root)

    import ray

    from raylight.distributed_worker.ray_worker import make_ray_actor_fn
    from raylight.text_encoder_parallel.lifecycle import DualGpuLeaseCoordinator

    os.environ["MASTER_PORT"] = str(args.port)
    ray_temp = REPO_ROOT / ".local" / "ray" / "o10-lease-probe"
    ray_temp.mkdir(parents=True, exist_ok=True)
    runtime_pythonpath = os.pathsep.join((str(REPO_ROOT / "src"), str(comfy_root)))
    ray.init(
        num_gpus=2,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=str(ray_temp),
        runtime_env={
            "env_vars": {
                "PYTHONPATH": runtime_pythonpath,
                "USE_LIBUV": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(args.port),
                "RAYLIGHT_WINDOWS_P2P": "0",
                "RAY_DEBUG_DISABLE_MEMORY_MONITOR": "1",
                "RAY_memory_usage_threshold": "1",
                "RAYLIGHT_COMFY_ARGS": "--disable-cuda-malloc --reserve-vram 4",
            }
        },
    )
    actors = None
    started = time.perf_counter()
    try:
        actors = make_ray_actor_fn(2, _parallel_dict())()
        workers = actors["workers"]
        before = ray.get([worker.get_external_gpu_lease.remote() for worker in workers])
        coordinator = DualGpuLeaseCoordinator(workers, ray_module=ray)
        leases = coordinator.acquire("o10-gate-e-probe", "MiniMaxH3TP2")
        during = ray.get([worker.get_external_gpu_lease.remote() for worker in workers])

        blocked = []
        for worker in workers:
            try:
                ray.get(worker.load_unet.remote("unused", {}))
            except Exception as error:
                blocked.append(str(error))
        if len(blocked) != 2 or any("leased by MiniMaxH3TP2" not in error for error in blocked):
            raise AssertionError(f"worker loading entry point was not blocked by the lease: {blocked}")
        released = coordinator.release("o10-gate-e-probe", "MiniMaxH3TP2")
        after = ray.get([worker.get_external_gpu_lease.remote() for worker in workers])

        ranks = tuple(lease.rank for lease in leases)
        if ranks != (0, 1):
            raise AssertionError(f"lease ranks must be (0, 1), got {ranks}")
        if any(item["state"] != "idle" for item in before + after):
            raise AssertionError(f"workers did not begin/end idle: before={before}, after={after}")
        if any(item["state"] != "leased_for_text_encoder" for item in during):
            raise AssertionError(f"workers were not leased together: {during}")
        if any(lease.free_bytes <= 0 for lease in leases):
            raise AssertionError(f"invalid free-memory receipts: {leases}")

        payload = {
            "schema_version": 1,
            "elapsed_s": time.perf_counter() - started,
            "before": before,
            "leases": [lease.to_mapping() for lease in leases],
            "during": during,
            "released": released,
            "after": after,
            "blocked_loading_errors": blocked,
            "gate_e_remote_lease_pass": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return 0
    finally:
        if actors is not None:
            for actor in actors.get("workers", ()):
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass
        ray.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
