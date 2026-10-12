from __future__ import annotations

import importlib
import inspect
from collections import OrderedDict
from pathlib import Path
import sys
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
        self.clone_calls = 0

    def get_model_object(self, name):
        if name == "model_sampling":
            return self.sampling
        if name == "latent_format":
            return self.model.latent_format
        raise AttributeError(name)

    def cleanup(self):
        self.cleanup_calls += 1

    def clone(self):
        self.clone_calls += 1
        cloned = _ModelPatcher()
        cloned.model_options = {
            "transformer_options": dict(self.model_options.get("transformer_options", {}))
        }
        return cloned


def _worker(is_fsdp=False, local_rank=0):
    worker = ray_worker.RayWorker.__new__(ray_worker.RayWorker)
    worker.local_rank = local_rank
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
        "apply_shift": True,
        "after_shift_capability": None,
        "director_latent_metadata": {},
        "enable_tiling": False,
        "tile_count": 2,
        "tile_overlap": 128,
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


def test_director_module_loader_falls_back_to_comfyui_custom_nodes(tmp_path):
    director_root = tmp_path / "ComfyUI_MiniMaxH3_Director" / "director"
    director_root.mkdir(parents=True)
    (director_root / "__init__.py").write_text("", encoding="utf-8")
    (director_root / "spatial_tiled_sampling.py").write_text(
        "SOURCE = 'installed-director'\n",
        encoding="utf-8",
    )
    folder_paths = SimpleNamespace(get_folder_paths=lambda name: [str(tmp_path)])
    real_import_module = importlib.import_module

    def import_without_installed_package(name, package=None):
        if name.startswith("ComfyUI_MiniMaxH3_Director."):
            missing = ModuleNotFoundError("No module named 'ComfyUI_MiniMaxH3_Director'")
            missing.name = "ComfyUI_MiniMaxH3_Director"
            raise missing
        return real_import_module(name, package)

    with (
        mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
        mock.patch.object(importlib, "import_module", side_effect=import_without_installed_package),
    ):
        module = ray_worker._import_mmh3director_module("spatial_tiled_sampling")

    assert module.SOURCE == "installed-director"
    assert Path(module.__file__).resolve() == (director_root / "spatial_tiled_sampling.py").resolve()


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


def test_worker_passes_external_sigmas_to_comfy_sampler_unchanged():
    worker = _worker()
    request = _request(operation_id="operation-sigmas")
    request["sigmas"] = torch.tensor([1.0, 0.7, 0.2, 0.0])
    shifted_model = _ModelPatcher()

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", return_value=request["latent_image"]) as sample,
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        _unwrapped_sample(worker, request)

    assert sample.call_args.kwargs["sigmas"] is request["sigmas"]


def test_worker_skips_sigma_shift_when_current_director_disables_it():
    worker = _worker()
    request = _request(operation_id="operation-no-shift")
    request["apply_shift"] = False

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
        ) as sigma_shift,
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", return_value=request["latent_image"]) as sample,
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        _unwrapped_sample(worker, request)

    sigma_shift.assert_not_called()
    assert worker.model.clone_calls == 1
    assert sample.call_args.args[0] is not worker.model


def test_worker_reconstructs_continue_remask_on_the_real_shifted_model():
    worker = _worker()
    request = _request(operation_id="operation-remask")
    request["sigmas"] = torch.tensor([1.0, 0.5, 0.0])
    request["after_shift_capability"] = "continue_prefix_remask_v1"
    request["director_latent_metadata"] = {
        "_director_continue_prefix_steps": 1,
        "_director_continue_seam_min": 0.1,
    }
    shifted_model = _ModelPatcher()
    remasked_model = _ModelPatcher()
    remask = mock.Mock(return_value=remasked_model)
    director_module = SimpleNamespace(install_continue_prefix_remask=remask)

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", return_value=request["latent_image"]) as sample,
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
        mock.patch.object(importlib, "import_module", return_value=director_module),
    ):
        _unwrapped_sample(worker, request)

    call = remask.call_args
    assert call.args[0] is shifted_model
    assert call.args[1]["samples"] is request["latent_image"]
    assert call.args[1]["noise_mask"] is request["noise_mask"]
    assert call.args[1]["_director_continue_prefix_steps"] == 1
    assert call.args[1]["_director_continue_seam_min"] == 0.1
    assert call.args[2] is request["sigmas"]
    assert sample.call_args.args[0] is remasked_model


def test_worker_runs_director_spatial_tiles_on_the_real_sampler_and_restores_it():
    worker = _worker()
    request = _request(operation_id="operation-tiles")
    request["sigmas"] = torch.tensor([1.0, 0.5, 0.0])
    request["enable_tiling"] = True
    request["tile_count"] = 3
    request["tile_overlap"] = 96
    shifted_model = _ModelPatcher()
    sampler = SimpleNamespace(sampler_function=object())
    restore = mock.Mock()
    wrap_tiles = mock.Mock(return_value=restore)
    director_module = SimpleNamespace(wrap_sampler_spatial_tiles=wrap_tiles)

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.samplers.sampler_object", return_value=sampler),
        mock.patch("comfy.sample.sample") as sample,
        mock.patch("comfy.sample.sample_custom", return_value=request["latent_image"]) as sample_custom,
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
        mock.patch.object(importlib, "import_module", return_value=director_module),
    ):
        _unwrapped_sample(worker, request)

    sample.assert_not_called()
    wrap_tiles.assert_called_once_with(sampler, n_tiles=3, overlap_pixels=96)
    call = sample_custom.call_args
    assert call.args[:8] == (
        shifted_model,
        request["noise"],
        request["cfg"],
        sampler,
        request["sigmas"],
        request["positive"],
        request["negative"],
        request["latent_image"],
    )
    assert call.kwargs["noise_mask"] is request["noise_mask"]
    assert call.kwargs["seed"] == request["seed"]
    restore.assert_called_once_with()


def test_worker_restores_spatial_sampler_when_tiled_sampling_fails():
    worker = _worker()
    request = _request(operation_id="operation-tiles-fail")
    request["sigmas"] = torch.tensor([1.0, 0.0])
    request["enable_tiling"] = True
    shifted_model = _ModelPatcher()
    restore = mock.Mock()
    director_module = SimpleNamespace(
        wrap_sampler_spatial_tiles=mock.Mock(return_value=restore)
    )

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.samplers.sampler_object", return_value=SimpleNamespace()),
        mock.patch("comfy.sample.sample_custom", side_effect=RuntimeError("tile failed")),
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
        mock.patch.object(importlib, "import_module", return_value=director_module),
        pytest.raises(RuntimeError, match="tile failed"),
    ):
        _unwrapped_sample(worker, request)

    restore.assert_called_once_with()


def test_rank_zero_publishes_step_preview_without_serializing_a_callback():
    worker = _worker(local_rank=0)
    request = _request(operation_id="operation-preview")
    request["preview_queue"] = object()
    request["preview_every"] = 2
    shifted_model = _ModelPatcher()
    event = {"step": 0, "total_steps": 4, "x0": torch.zeros(1)}

    def run_sample(*args, **kwargs):
        kwargs["callback"](0, request["latent_image"], request["latent_image"], 4)
        return request["latent_image"]

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", side_effect=run_sample),
        mock.patch(
            "raylight.integrations.mmh3director.preview_transport.extract_preview_event",
            return_value=event,
        ) as extract,
        mock.patch(
            "raylight.integrations.mmh3director.preview_transport.publish_preview_nonblocking",
            return_value=True,
        ) as publish,
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        _unwrapped_sample(worker, request)

    assert extract.call_args.kwargs["preview_every"] == 2
    publish.assert_called_once_with(request["preview_queue"], event)


def test_rank_zero_returns_final_full_selflift_step_state():
    worker = _worker(local_rank=0)
    request = _request(operation_id="operation-selflift-state")
    request["capture_step_state"] = True
    shifted_model = _ModelPatcher()
    final_x0 = request["latent_image"] + 3
    final_x = request["latent_image"] + 4

    def run_sample(*args, **kwargs):
        kwargs["callback"](6, final_x0, final_x, 8)
        return request["latent_image"] + 5

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", side_effect=run_sample),
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        result = _unwrapped_sample(worker, request)

    state = result["step_state"]
    assert (state["step"], state["total_steps"]) == (6, 8)
    assert state["x0"].device.type == "cpu"
    assert state["x"].device.type == "cpu"
    assert torch.equal(state["x0"], final_x0)
    assert torch.equal(state["x"], final_x)


def test_nonzero_rank_does_not_return_duplicate_selflift_state():
    worker = _worker(local_rank=1)
    request = _request(operation_id="operation-selflift-state-rank1")
    request["capture_step_state"] = True
    shifted_model = _ModelPatcher()

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", return_value=request["latent_image"]),
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        result = _unwrapped_sample(worker, request)

    assert "step_state" not in result


def test_nonzero_rank_never_builds_a_step_preview_callback():
    worker = _worker(local_rank=1)
    request = _request(operation_id="operation-preview-rank1")
    request["preview_queue"] = object()
    shifted_model = _ModelPatcher()

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            return_value=(shifted_model,),
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch("comfy.sample.sample", return_value=request["latent_image"]) as sample,
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        _unwrapped_sample(worker, request)

    assert sample.call_args.kwargs["callback"] is None


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


def test_worker_segment_loras_are_request_scoped_across_hot_operations():
    worker = _worker(is_fsdp=True)
    requests = [
        _request(operation_id="segment-a"),
        _request(operation_id="segment-none"),
        _request(operation_id="segment-b"),
    ]
    requests[0]["segment_loras"] = [
        {"path": "A.safetensors", "strength_model": 0.8}
    ]
    requests[1]["segment_loras"] = []
    requests[2]["segment_loras"] = [
        {"path": "B.safetensors", "strength_model": 0.6}
    ]
    shifted_models = [_ModelPatcher(), _ModelPatcher(), _ModelPatcher()]
    patched_a = _ModelPatcher()
    patched_b = _ModelPatcher()
    sampled_models = []

    def apply_segment(model, descriptors, **_kwargs):
        if not descriptors:
            return model
        return patched_a if descriptors[0]["path"] == "A.safetensors" else patched_b

    def sample(model, *_args, **_kwargs):
        sampled_models.append(model)
        return requests[len(sampled_models) - 1]["latent_image"]

    with (
        mock.patch(
            "comfy_extras.nodes_minimax_h3.MiniMaxH3SigmaShift.execute",
            side_effect=[(model,) for model in shifted_models],
        ),
        mock.patch.object(ray_worker, "_restore_controlnet_refs"),
        mock.patch.object(ray_worker, "_remap_conditioning_devices"),
        mock.patch.object(ray_worker, "_prepare_control_models"),
        mock.patch(
            "raylight.integrations.mmh3director.segment_lora.apply_worker_segment_loras",
            side_effect=apply_segment,
        ) as apply,
        mock.patch("comfy.sample.sample", side_effect=sample),
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        for request in requests:
            _unwrapped_sample(worker, request)

    assert sampled_models == [patched_a, shifted_models[1], patched_b]
    assert [call.args[1] for call in apply.call_args_list] == [
        requests[0]["segment_loras"],
        (),
        requests[2]["segment_loras"],
    ]
    assert worker.model is not patched_a
    assert worker.model is not patched_b


def test_segment_lora_state_cache_clears_on_global_lora_change_and_model_reset():
    worker = _worker()
    worker._mmh3director_segment_lora_cache = OrderedDict(
        [(('A.safetensors', 1, 1), {"A": 1})]
    )

    worker.set_lora_list([{"path": "global.safetensors", "strength_model": 1.0}])
    assert worker._mmh3director_segment_lora_cache == OrderedDict()

    worker._mmh3director_segment_lora_cache[("B.safetensors", 1, 1)] = {"B": 1}
    worker.model = None
    with (
        mock.patch("comfy.model_management.soft_empty_cache"),
        mock.patch.object(ray_worker.gc, "collect"),
    ):
        worker._reset_active_model()
    assert worker._mmh3director_segment_lora_cache == OrderedDict()


def test_segment_lora_state_cache_clears_before_worker_exit():
    worker = _worker()
    worker._mmh3director_segment_lora_cache = OrderedDict(
        [(('A.safetensors', 1, 1), {"A": 1})]
    )
    worker._free_cached_aux_models = mock.Mock()
    worker._invalidate_non_fsdp_cache = mock.Mock()
    worker._free_current_model = mock.Mock()
    worker.disable_windows_p2p = mock.Mock()
    worker._windows_p2p = None

    with (
        mock.patch.object(ray_worker.dist, "destroy_process_group"),
        mock.patch.object(ray_worker.ray.actor, "exit_actor"),
    ):
        worker.kill()

    assert worker._mmh3director_segment_lora_cache == OrderedDict()


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
