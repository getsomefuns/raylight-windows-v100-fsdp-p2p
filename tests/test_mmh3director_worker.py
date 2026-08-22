from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from raylight.distributed_worker import ray_worker


class _LatentFormat:
    latent_channels = 32
    latent_dimensions = 3
    spacial_downscale_ratio = 16
    temporal_downscale_ratio = 4


class _ModelConfig:
    unet_config = {"image_model": "minimax_h3"}
    sampling_settings = {"shift": 12.0, "audio_shift": 3.0}
    latent_format = _LatentFormat


class _ModelPatcher:
    def __init__(self):
        self.model = SimpleNamespace(model_config=_ModelConfig(), latent_format=_LatentFormat())
        self.model_options = {"transformer_options": {"existing": True}}
        self.sampling = SimpleNamespace(noise_scale=0.75)
        self.cleanup_calls = 0

    def get_model_object(self, name):
        if name == "model_sampling":
            return self.sampling
        if name == "latent_format":
            return self.model.latent_format
        raise AttributeError(name)

    def cleanup(self):
        self.cleanup_calls += 1


def _worker(is_fsdp=False):
    worker = ray_worker.RayWorker.__new__(ray_worker.RayWorker)
    worker.local_rank = 0
    worker.global_world_size = 2
    worker.model = _ModelPatcher()
    worker.is_model_loaded = True
    worker.parallel_dict = {"is_fsdp": is_fsdp}
    worker.cached_controlnet = None
    worker.vae_model = None
    worker.state_dict = {}
    worker._patch_fsdp_for_sampling = mock.Mock()
    worker._mmh3director_active_operation = None
    worker._mmh3director_last_status = None
    return worker


def _request(operation_id="operation-a", samples=None):
    latent = torch.zeros(1, 32, 2, 4, 4) if samples is None else samples
    return {
        "operation_id": operation_id,
        "noise": torch.randn_like(latent),
        "steps": 8,
        "cfg": 4.5,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": [["positive", {}]],
        "negative": [["negative", {}]],
        "latent_image": latent,
        "denoise": 0.65,
        "disable_noise": False,
        "start_step": 1,
        "last_step": 7,
        "force_full_denoise": True,
        "noise_mask": torch.ones(1, 1, 2, 4, 4),
        "sigmas": None,
        "disable_pbar": True,
        "seed": 12345,
        "model_options": {
            "transformer_options": {
                "minimax_h3_sigma_shift_video": 12.0,
                "minimax_h3_sigma_shift_audio": 3.0,
                "director_marker": "kept",
            }
        },
    }


def _unwrapped_sample(worker, request):
    implementation = inspect.unwrap(ray_worker.RayWorker.sample_mmh3director_request)
    return implementation(worker, request)


def _contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_worker_proxy_metadata_is_compact_and_contains_no_weights():
    metadata = _worker().get_mmh3director_proxy_metadata()

    assert metadata == {
        "rank": 0,
        "model_family": "minimax_h3",
        "sampling_settings": {"shift": 12.0, "audio_shift": 3.0},
        "latent_channels": 32,
        "latent_dimensions": 3,
        "spacial_downscale_ratio": 16,
        "temporal_downscale_ratio": 4,
        "noise_scale": 0.75,
    }
    assert not _contains_tensor(metadata)
    assert "state_dict" not in metadata
    assert "diffusion_model" not in metadata


def test_worker_proxy_metadata_returns_none_until_model_is_loaded():
    worker = _worker()
    worker.is_model_loaded = False
    assert worker.get_mmh3director_proxy_metadata() is None
    worker.is_model_loaded = True
    worker.model = None
    assert worker.get_mmh3director_proxy_metadata() is None


def test_worker_uses_director_noise_applies_shifts_and_disables_callback():
    worker = _worker(is_fsdp=True)
    request = _request()
    shifted_model = _ModelPatcher()
    sampled = request["latent_image"] + 2

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ) as sigma_shift,
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", return_value=sampled) as sample,
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        result = _unwrapped_sample(worker, request)

    sigma_shift.assert_called_once_with(worker.model, 12.0, 3.0)
    worker._patch_fsdp_for_sampling.assert_called_once_with()
    assert worker.state_dict is None
    call = sample.call_args
    assert call.args[0] is shifted_model
    assert call.args[1] is request["noise"]
    assert call.args[2:9] == (
        8,
        4.5,
        "euler",
        "simple",
        request["positive"],
        request["negative"],
        request["latent_image"],
    )
    assert call.kwargs["noise_mask"] is request["noise_mask"]
    assert call.kwargs["callback"] is None
    assert call.kwargs["seed"] == 12345
    assert shifted_model.model_options["transformer_options"] == {
        "existing": True,
        "minimax_h3_sigma_shift_video": 12.0,
        "minimax_h3_sigma_shift_audio": 3.0,
        "director_marker": "kept",
    }
    assert result["rank"] == 0
    assert result["operation_id"] == "operation-a"
    assert torch.equal(result["samples"], sampled)
    assert result["finite"] is True
    assert worker._mmh3director_active_operation is None
    assert worker.get_mmh3director_sample_status("operation-a")["state"] == "completed"


def test_worker_rejects_callback_in_request_and_clears_failed_status():
    worker = _worker()
    request = _request(operation_id="operation-callback")
    request["callback"] = object()

    with pytest.raises(ValueError, match="callback.*must not be serialized"):
        _unwrapped_sample(worker, request)

    assert worker._mmh3director_active_operation is None
    status = worker.get_mmh3director_sample_status("operation-callback")
    assert status["state"] == "failed"
    assert status["error_type"] == "ValueError"


def test_worker_non_finite_output_fails_and_records_bounded_status():
    worker = _worker()
    request = _request(operation_id="operation-nan")
    shifted_model = _ModelPatcher()

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", return_value=torch.tensor([float("nan")])),
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
        pytest.raises(ValueError, match="NaN/Inf"),
    ):
        _unwrapped_sample(worker, request)

    assert worker._mmh3director_active_operation is None
    status = worker.get_mmh3director_sample_status("operation-nan")
    assert status["state"] == "failed"
    assert status["error_type"] == "ValueError"
    assert set(vars(worker)) >= {"_mmh3director_active_operation", "_mmh3director_last_status"}


def test_failed_segment_does_not_pollute_the_next_hot_operation():
    worker = _worker()
    failed_request = _request(operation_id="operation-failed")
    next_request = _request(operation_id="operation-next")
    shifted_model = _ModelPatcher()
    next_samples = next_request["latent_image"] + 3

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch(
            "comfy.sample.sample",
            side_effect=[RuntimeError("first segment failed"), next_samples],
        ),
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        with pytest.raises(RuntimeError, match="first segment failed"):
            _unwrapped_sample(worker, failed_request)
        assert worker.get_mmh3director_sample_status("operation-failed")["state"] == "failed"

        result = _unwrapped_sample(worker, next_request)

    assert torch.equal(result["samples"], next_samples)
    assert worker._mmh3director_active_operation is None
    assert worker.get_mmh3director_sample_status("operation-next")["state"] == "completed"
    assert worker.get_mmh3director_sample_status("operation-failed")["state"] == "unknown"


def test_existing_common_ksampler_public_signature_is_unchanged():
    assert tuple(inspect.signature(ray_worker.RayWorker.common_ksampler).parameters) == (
        "self",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "positive",
        "negative",
        "latent",
        "denoise",
        "disable_noise",
        "start_step",
        "last_step",
        "force_full_denoise",
        "grouped_output",
    )
