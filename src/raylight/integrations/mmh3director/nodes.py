from __future__ import annotations

import ray

from .metadata import validate_rank_metadata
from .proxy import RayMMH3DirectorModelProxy


class RayMMH3DirectorModelAdapter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ray_actors": ("RAY_ACTORS",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "adapt_model"
    CATEGORY = "Raylight/integrations/MMH3Director"

    def adapt_model(self, ray_actors: dict):
        if not isinstance(ray_actors, dict) or "workers" not in ray_actors:
            raise TypeError("RAY_ACTORS payload must be a dict with workers")

        workers = list(ray_actors["workers"])
        if not workers:
            raise RuntimeError("MMH3Director adapter requires at least one loaded Ray worker")

        configured_world_size = ray_actors.get("world_size")
        if configured_world_size is not None and int(configured_world_size) != len(workers):
            raise ValueError(
                f"RAY_ACTORS world_size={configured_world_size} disagrees with {len(workers)} workers"
            )

        futures = [worker.get_mmh3director_model_metadata.remote() for worker in workers]
        rows = ray.get(futures)
        for rank, row in enumerate(rows):
            if row is None:
                raise RuntimeError(f"rank {rank} has no loaded model metadata")

        metadata = validate_rank_metadata(rows)
        if metadata.model_family != "minimax_h3":
            raise ValueError(
                f"MMH3Director adapter requires MiniMax H3 workers; found {metadata.model_family}"
            )

        return (
            RayMMH3DirectorModelProxy(
                ray_actors,
                metadata,
                session_id=ray_actors.get("session_id"),
                generation_id=ray_actors.get("generation_id"),
            ),
        )


NODE_CLASS_MAPPINGS = {
    "RayMMH3DirectorModelAdapter": RayMMH3DirectorModelAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RayMMH3DirectorModelAdapter": "Ray MODEL Adapter (MMH3Director)",
}
