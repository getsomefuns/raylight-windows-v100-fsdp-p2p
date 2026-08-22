from dataclasses import FrozenInstanceError

import pytest
import torch

from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
from raylight.integrations.mmh3director.metadata import (
    MMH3DirectorModelMetadata,
    validate_rank_metadata,
)
from raylight.integrations.mmh3director.proxy import RayMMH3DirectorModelProxy


def _rank_row(rank=0, **overrides):
    row = {
        "rank": rank,
        "model_family": "minimax_h3",
        "sampling_settings": {"shift": 12.0, "audio_shift": 3.0},
        "latent_channels": 32,
        "latent_dimensions": 3,
        "spacial_downscale_ratio": 16,
        "temporal_downscale_ratio": 4,
        "noise_scale": 0.75,
    }
    row.update(overrides)
    return row


def _metadata():
    return validate_rank_metadata([_rank_row(0), _rank_row(1)])


def _proxy():
    return RayMMH3DirectorModelProxy(
        ray_actors={"workers": ("rank-0", "rank-1")},
        metadata=_metadata(),
        session_id="session-a",
        generation_id=7,
    )


def test_matching_rank_metadata_builds_a_frozen_value_object():
    metadata = _metadata()

    assert metadata == MMH3DirectorModelMetadata(
        model_family="minimax_h3",
        sampling_settings={"shift": 12.0, "audio_shift": 3.0},
        latent_channels=32,
        latent_dimensions=3,
        spacial_downscale_ratio=16,
        temporal_downscale_ratio=4,
        noise_scale=0.75,
    )
    with pytest.raises(FrozenInstanceError):
        metadata.latent_channels = 64


@pytest.mark.parametrize(
    "field, mismatched_value",
    [
        ("model_family", "ltx"),
        ("sampling_settings", {"shift": 9.0, "audio_shift": 3.0}),
        ("latent_channels", 24),
        ("latent_dimensions", 4),
        ("spacial_downscale_ratio", 8),
        ("temporal_downscale_ratio", 8),
        ("noise_scale", 1.0),
    ],
)
def test_rank_metadata_mismatch_names_rank_and_field(field, mismatched_value):
    with pytest.raises(ValueError, match=rf"rank 1.*{field}"):
        validate_rank_metadata([_rank_row(0), _rank_row(1, **{field: mismatched_value})])


def test_rank_metadata_rejects_missing_or_duplicate_ranks():
    with pytest.raises(ValueError, match="at least one rank"):
        validate_rank_metadata([])
    with pytest.raises(ValueError, match="duplicate rank 0"):
        validate_rank_metadata([_rank_row(0), _rank_row(0)])


def test_proxy_contains_no_diffusion_weights_or_parameters():
    proxy = _proxy()

    assert not hasattr(proxy, "state_dict")
    assert not hasattr(proxy.model, "diffusion_model")
    assert not any(isinstance(value, torch.nn.Parameter) for value in vars(proxy).values())
    assert proxy.ray_actors == {"workers": ("rank-0", "rank-1")}


def test_clone_isolates_model_options_and_object_patch_maps():
    parent = _proxy()
    marker = object()
    parent.add_object_patch("custom", marker)
    parent.model_options["transformer_options"] = {"parent": True}

    cloned = parent.clone()
    cloned.model_options["transformer_options"]["child"] = True
    cloned.add_object_patch("child_only", object())

    assert cloned is not parent
    assert cloned.ray_actors is parent.ray_actors
    assert cloned.get_model_object("custom") is marker
    assert "child" not in parent.model_options["transformer_options"]
    with pytest.raises(AttributeError, match="child_only"):
        parent.get_model_object("child_only")


def test_proxy_exposes_only_supported_model_objects():
    proxy = _proxy()

    assert proxy.get_model_object("model_sampling").noise_scale == pytest.approx(0.75)
    assert proxy.get_model_object("latent_format") is proxy.model.latent_format
    assert proxy.model.latent_format.latent_channels == 32
    assert proxy.model.latent_format.latent_dimensions == 3
    assert proxy.model.latent_format.spacial_downscale_ratio == 16
    assert proxy.model.latent_format.temporal_downscale_ratio == 4
    with pytest.raises(AttributeError, match="unknown"):
        proxy.get_model_object("unknown")


def test_real_minimax_sigma_shift_patches_a_clone_without_mutating_parent():
    parent = _proxy()

    shifted = MiniMaxH3SigmaShift.execute(parent, 12.0, 3.0)[0]
    shifted_sampling = shifted.get_model_object("model_sampling")

    assert shifted is not parent
    assert shifted_sampling.shift == pytest.approx(12.0)
    assert shifted_sampling.audio_shift == pytest.approx(3.0)
    assert shifted_sampling.noise_scale == pytest.approx(0.75)
    assert shifted.model_options["transformer_options"] == {
        "minimax_h3_sigma_shift_video": 12.0,
        "minimax_h3_sigma_shift_audio": 3.0,
    }
    assert parent.model_options == {"transformer_options": {}}
    assert parent.get_model_object("model_sampling") is not shifted_sampling
