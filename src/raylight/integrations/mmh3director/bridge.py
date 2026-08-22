from __future__ import annotations

import copy
import importlib
import inspect
import logging
import sys
import uuid
from typing import Callable

import ray
import torch

import comfy.model_management
import comfy.sample

from raylight.distributed_worker.sampling_lifecycle import clear_ray_worker_vram_after_sampling

from .proxy import RayMMH3DirectorModelProxy


SUPPORTED_DIRECTOR_VERSION = "1.0.0"
SUPPORTED_DIRECTOR_COMMIT = "d655912262bcd70ad43c3a431879be9539d8e054"
EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS = (
    "model",
    "positive",
    "negative",
    "latent",
    "seed",
    "cfg",
    "steps",
    "sampler_name",
    "scheduler",
    "shift_video",
    "shift_audio",
    "on_phase",
    "on_step_preview",
    "preview_every",
    "denoise",
    "phase_name",
)

_BRIDGE_MARKER = "_raylight_mmh3director_sample_bridge"
_ORIGINAL_MARKER = "_raylight_mmh3director_original_sample"
_original_comfy_sample: Callable | None = None
_installed_wrapper: Callable | None = None
_LOG = logging.getLogger("raylight.mmh3director")


def validate_mmh3director_sample_single_stage(sample_single_stage: Callable) -> None:
    """Fail closed when Director's public sampling seam no longer matches O8."""

    parameters = inspect.signature(sample_single_stage).parameters
    missing = [name for name in EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS if name not in parameters]
    wrong_kind = [
        name
        for name in EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS
        if name in parameters and parameters[name].kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    if not missing and not wrong_kind:
        return

    failures = []
    if missing:
        failures.append(f"missing or renamed parameter(s): {', '.join(missing)}")
    if wrong_kind:
        failures.append(f"non-keyword parameter(s): {', '.join(wrong_kind)}")
    details = "; ".join(failures)
    raise RuntimeError(
        "Unsupported MiniMaxH3Director sample_single_stage contract: "
        f"{details}. This adapter supports Director {SUPPORTED_DIRECTOR_VERSION} at "
        f"commit {SUPPORTED_DIRECTOR_COMMIT[:7]}. Please update Director or this Raylight adapter "
        "to a mutually supported version."
    )


def _find_loaded_sample_single_stage() -> Callable:
    for module_name, module in tuple(sys.modules.items()):
        module_path = str(getattr(module, "__file__", ""))
        identity = f"{module_name} {module_path}".lower()
        if "minimaxh3_director" not in identity or not module_name.endswith("director.core_sampling"):
            continue
        sample_single_stage = getattr(module, "sample_single_stage", None)
        if callable(sample_single_stage):
            return sample_single_stage

    try:
        module = importlib.import_module("ComfyUI_MiniMaxH3_Director.director.core_sampling")
    except Exception as error:
        raise RuntimeError(
            "Ray MODEL Adapter (MMH3Director) requires the MiniMaxH3Director package at sampling time. "
            f"Validated target: Director {SUPPORTED_DIRECTOR_VERSION}, commit {SUPPORTED_DIRECTOR_COMMIT[:7]}."
        ) from error
    return module.sample_single_stage


def ensure_mmh3director_compatibility() -> None:
    validate_mmh3director_sample_single_stage(_find_loaded_sample_single_stage())


def _sample_streams(samples) -> list[torch.Tensor]:
    if getattr(samples, "is_nested", False):
        streams = list(samples.unbind())
    else:
        streams = [samples]
    if not streams or any(not isinstance(stream, torch.Tensor) for stream in streams):
        raise TypeError(f"MMH3Director worker returned unsupported samples type {type(samples).__name__}")
    return streams


def _validate_finite_samples(samples, rank: int) -> list[torch.Tensor]:
    streams = _sample_streams(samples)
    for stream_index, stream in enumerate(streams):
        if (stream.is_floating_point() or stream.is_complex()) and not bool(torch.isfinite(stream).all().item()):
            suffix = f" stream {stream_index}" if len(streams) > 1 else ""
            raise ValueError(f"rank {rank} returned NaN/Inf in MMH3Director samples{suffix}")
    return streams


def _assert_rank_samples_equal(reference, current, rank: int) -> None:
    reference_streams = _sample_streams(reference)
    current_streams = _sample_streams(current)
    if len(reference_streams) != len(current_streams):
        raise ValueError(f"rank {rank} samples differ from rank 0: stream count mismatch")
    for reference_stream, current_stream in zip(reference_streams, current_streams):
        if reference_stream.shape != current_stream.shape or reference_stream.dtype != current_stream.dtype:
            raise ValueError(f"rank {rank} samples differ from rank 0: shape or dtype mismatch")
        comparable = current_stream.to(device=reference_stream.device)
        if not torch.equal(reference_stream, comparable):
            raise ValueError(f"rank {rank} samples differ from rank 0")


def dispatch_mmh3director_sample(
    model,
    noise,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
    noise_mask=None,
    sigmas=None,
    callback=None,
    disable_pbar=False,
    seed=None,
):
    if not isinstance(model, RayMMH3DirectorModelProxy):
        raise TypeError("dispatch_mmh3director_sample requires RayMMH3DirectorModelProxy")
    if model.metadata.model_family != "minimax_h3":
        raise RuntimeError(
            "MMH3Director proxy sampling requires model_family=minimax_h3; "
            f"found {model.metadata.model_family}"
        )

    workers = list(model.ray_actors.get("workers", ()))
    if not workers:
        raise RuntimeError("MMH3Director proxy sampling requires live Ray workers")
    configured_world_size = model.ray_actors.get("world_size")
    if configured_world_size is not None and int(configured_world_size) != len(workers):
        raise RuntimeError(
            f"MMH3Director proxy world_size={configured_world_size} but has {len(workers)} live Ray workers"
        )

    ensure_mmh3director_compatibility()
    operation_id = uuid.uuid4().hex
    request = {
        "operation_id": operation_id,
        "session_id": model.session_id,
        "generation_id": model.generation_id,
        "noise": noise,
        "steps": int(steps),
        "cfg": float(cfg),
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "positive": positive,
        "negative": negative,
        "latent_image": latent_image,
        "denoise": float(denoise),
        "disable_noise": bool(disable_noise),
        "start_step": start_step,
        "last_step": last_step,
        "force_full_denoise": bool(force_full_denoise),
        "noise_mask": noise_mask,
        "sigmas": sigmas,
        "disable_pbar": bool(disable_pbar),
        "seed": seed,
        "model_options": copy.deepcopy(model.model_options),
    }

    print(
        f"[Raylight] MMH3Director remote sampling started operation={operation_id} ranks={len(workers)}",
        flush=True,
    )
    sampling_error = None
    try:
        try:
            futures = [worker.sample_mmh3director_request.remote(request) for worker in workers]
            results = ray.get(futures)
        except Exception as error:
            raise RuntimeError(
                f"MMH3Director remote sampling operation {operation_id} failed before rank validation"
            ) from error

        if not isinstance(results, (list, tuple)) or len(results) != len(workers):
            received = len(results) if isinstance(results, (list, tuple)) else 0
            raise RuntimeError(
                f"MMH3Director operation {operation_id} returned {received}/{len(workers)} rank results"
            )

        reference_samples = None
        for expected_rank, result in enumerate(results):
            if result is None:
                raise RuntimeError(f"MMH3Director operation {operation_id} missing result for rank {expected_rank}")
            if not isinstance(result, dict):
                raise TypeError(
                    f"MMH3Director operation {operation_id} rank {expected_rank} returned {type(result).__name__}"
                )
            actual_rank = result.get("rank")
            if actual_rank != expected_rank:
                raise ValueError(
                    f"MMH3Director operation {operation_id} expected rank {expected_rank}, got rank {actual_rank}"
                )
            if result.get("operation_id") != operation_id:
                raise ValueError(f"rank {expected_rank} returned a mismatched MMH3Director operation_id")
            if "samples" not in result:
                raise RuntimeError(f"MMH3Director operation {operation_id} missing samples for rank {expected_rank}")
            samples = result["samples"]
            _validate_finite_samples(samples, expected_rank)
            if reference_samples is None:
                reference_samples = samples
            else:
                _assert_rank_samples_equal(reference_samples, samples, expected_rank)

        normalized = reference_samples.to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        if callback is not None:
            callback(max(0, int(steps) - 1), normalized, normalized, int(steps))
        print(
            f"[Raylight] MMH3Director remote sampling completed operation={operation_id} ranks={len(workers)}",
            flush=True,
        )
        return normalized
    except BaseException as error:
        sampling_error = error
        print(
            f"[Raylight] MMH3Director remote sampling failed operation={operation_id} "
            f"error={type(error).__name__}",
            flush=True,
        )
        raise
    finally:
        try:
            clear_ray_worker_vram_after_sampling(model.ray_actors)
        except BaseException:
            if sampling_error is None:
                raise
            _LOG.exception(
                "MMH3Director worker cleanup failed after sampling operation %s; preserving %s",
                operation_id,
                type(sampling_error).__name__,
            )


def _raylight_sample_dispatch(model, *args, **kwargs):
    if isinstance(model, RayMMH3DirectorModelProxy):
        return dispatch_mmh3director_sample(model, *args, **kwargs)
    if _original_comfy_sample is None:
        raise RuntimeError("MMH3Director sample bridge lost its original comfy.sample.sample reference")
    return _original_comfy_sample(model, *args, **kwargs)


def install_mmh3director_sample_bridge() -> bool:
    global _installed_wrapper, _original_comfy_sample

    current = comfy.sample.sample
    if getattr(current, _BRIDGE_MARKER, False) is True:
        _installed_wrapper = current
        _original_comfy_sample = getattr(current, _ORIGINAL_MARKER)
        return False

    _original_comfy_sample = current
    setattr(_raylight_sample_dispatch, _BRIDGE_MARKER, True)
    setattr(_raylight_sample_dispatch, _ORIGINAL_MARKER, current)
    comfy.sample.sample = _raylight_sample_dispatch
    _installed_wrapper = _raylight_sample_dispatch
    return True


def uninstall_mmh3director_sample_bridge_for_test() -> None:
    global _installed_wrapper, _original_comfy_sample

    current = comfy.sample.sample
    if getattr(current, _BRIDGE_MARKER, False) is True:
        comfy.sample.sample = getattr(current, _ORIGINAL_MARKER)
    _installed_wrapper = None
    _original_comfy_sample = None
