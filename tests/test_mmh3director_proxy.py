from dataclasses import FrozenInstanceError

import pytest
import torch
from comfy.nested_tensor import NestedTensor

from comfy.ldm.minimax.model import MiniMaxH3Model
from comfy.patcher_extension import WrappersMP
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


def test_proxy_exposes_a_weightless_minimax_diffusion_facade():
    proxy = _proxy()

    assert not hasattr(proxy, "state_dict")
    diffusion = proxy.get_model_object("diffusion_model")
    assert isinstance(diffusion, MiniMaxH3Model)
    assert len(diffusion.blocks) == 50
    assert not tuple(diffusion.parameters())
    assert not any(isinstance(value, torch.nn.Parameter) for value in vars(proxy).values())
    assert proxy.ray_actors == {"workers": ("rank-0", "rank-1")}


def test_proxy_supports_comfy_wrapper_lifecycle_and_clone_isolation():
    proxy = _proxy()
    wrapper = lambda executor, *args, **kwargs: executor(*args, **kwargs)

    proxy.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "te-speed", wrapper)
    cloned = proxy.clone()

    assert proxy.get_wrappers(WrappersMP.DIFFUSION_MODEL, "te-speed") == [wrapper]
    assert cloned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) == [wrapper]
    cloned.remove_wrappers_with_key(WrappersMP.DIFFUSION_MODEL, "te-speed")
    assert cloned.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) == []
    assert proxy.get_all_wrappers(WrappersMP.DIFFUSION_MODEL) == [wrapper]


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


def test_segment_lora_descriptors_are_clone_local():
    parent = _proxy()

    child = parent.with_segment_loras(
        [{"path": "a.safetensors", "strength_model": 0.7}]
    )

    assert parent.segment_loras == ()
    assert child.segment_loras == (
        {"path": "a.safetensors", "strength_model": 0.7},
    )
    child.segment_loras[0]["strength_model"] = 0.1
    assert parent.segment_loras == ()


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


def test_proxy_preserves_minimax_av_audio_scale_during_selflift_transition():
    proxy = _proxy()
    video = torch.tensor([1.0, 2.0])
    audio = torch.tensor([3.0, 4.0])
    samples = NestedTensor((video, audio))

    internal = proxy.model.process_latent_in(samples)
    internal_video, internal_audio = internal.unbind()
    restored = proxy.model.process_latent_out(internal)
    restored_video, restored_audio = restored.unbind()

    assert proxy.model.model_sampling.audio_scale == pytest.approx(4.0)
    torch.testing.assert_close(internal_video, video)
    torch.testing.assert_close(internal_audio, audio * 4.0)
    torch.testing.assert_close(restored_video, video)
    torch.testing.assert_close(restored_audio, audio)
