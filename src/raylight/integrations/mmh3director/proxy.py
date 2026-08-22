from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch

import comfy.model_sampling

from .metadata import MMH3DirectorModelMetadata


@dataclass(frozen=True)
class _ModelConfigFacade:
    sampling_settings: dict[str, object]


@dataclass(frozen=True)
class _LatentFormatFacade:
    latent_channels: int
    latent_dimensions: int
    spacial_downscale_ratio: int
    temporal_downscale_ratio: int


class _ModelFacade:
    def __init__(self, metadata: MMH3DirectorModelMetadata):
        self.model_config = _ModelConfigFacade(copy.deepcopy(metadata.sampling_settings))
        self.latent_format = _LatentFormatFacade(
            latent_channels=metadata.latent_channels,
            latent_dimensions=metadata.latent_dimensions,
            spacial_downscale_ratio=metadata.spacial_downscale_ratio,
            temporal_downscale_ratio=metadata.temporal_downscale_ratio,
        )
        self.model_sampling = comfy.model_sampling.ModelSamplingAV(self.model_config)
        if metadata.noise_scale is not None:
            self.model_sampling.set_noise_scale(metadata.noise_scale)


class RayMMH3DirectorModelProxy:
    """A weightless ComfyUI MODEL facade backed by already-loaded Ray workers."""

    def __init__(
        self,
        ray_actors: dict,
        metadata: MMH3DirectorModelMetadata,
        *,
        session_id: str | None = None,
        generation_id: int | str | None = None,
    ):
        self.ray_actors = ray_actors
        self.metadata = metadata
        self.session_id = session_id
        self.generation_id = generation_id
        self.model = _ModelFacade(metadata)
        self.model_options: dict[str, Any] = {"transformer_options": {}}
        self.load_device = torch.device("cpu")
        self._object_patches: dict[str, object] = {}

    def clone(self) -> "RayMMH3DirectorModelProxy":
        cloned = type(self)(
            self.ray_actors,
            self.metadata,
            session_id=self.session_id,
            generation_id=self.generation_id,
        )
        cloned.model_options = copy.deepcopy(self.model_options)
        cloned._object_patches = self._object_patches.copy()
        return cloned

    def get_model_object(self, name: str):
        if name in self._object_patches:
            return self._object_patches[name]
        try:
            return getattr(self.model, name)
        except AttributeError as error:
            raise AttributeError(f"unknown MMH3Director model object: {name}") from error

    def add_object_patch(self, name: str, obj) -> None:
        self._object_patches[name] = obj
