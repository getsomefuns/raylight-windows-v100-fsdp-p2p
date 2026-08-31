from __future__ import annotations

from contextlib import contextmanager
import gc

import ray

import comfy.model_management

from raylight.text_encoder_parallel.lifecycle import DualGpuLeaseCoordinator


def clear_ray_worker_vram_after_sampling(ray_actors: dict) -> bool:
    """Apply the existing worker cleanup policy after any Ray sampling path."""

    gpu_actors = list(ray_actors.get("workers", ()))
    if not gpu_actors:
        return False
    parallel_dict = ray.get(gpu_actors[0].get_parallel_dict.remote())
    if not parallel_dict.get("clear_vram_after_sampling", False):
        return False

    ray.get([actor.clear_sampling_vram.remote() for actor in gpu_actors])
    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    return True


@contextmanager
def external_text_encoder_gpu_lease(ray_actors: dict, operation_id: str, owner: str):
    """Lease both existing sampling workers to a main-process Text Encoder."""

    gpu_actors = list(ray_actors.get("workers", ()))
    coordinator = DualGpuLeaseCoordinator(gpu_actors, ray_module=ray)
    with coordinator.lease(operation_id, owner) as leases:
        yield leases
