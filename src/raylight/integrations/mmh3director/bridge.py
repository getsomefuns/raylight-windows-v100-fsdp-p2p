from __future__ import annotations

import copy
import functools
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


SUPPORTED_DIRECTOR_VERSION = "main"
SUPPORTED_DIRECTOR_COMMIT = "b8f721cb145490cb2c13c8944ff90f21551570f8"
LEGACY_DIRECTOR_VERSION = "1.0.0"
LEGACY_DIRECTOR_COMMIT = "d655912262bcd70ad43c3a431879be9539d8e054"
LEGACY_SAMPLE_SINGLE_STAGE_PARAMETERS = (
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
EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS = (
    *LEGACY_SAMPLE_SINGLE_STAGE_PARAMETERS,
    "sigmas",
    "apply_shift",
)

_BRIDGE_MARKER = "_raylight_mmh3director_sample_bridge"
_ORIGINAL_MARKER = "_raylight_mmh3director_original_sample"
_STAGE_BRIDGE_MARKER = "_raylight_mmh3director_stage_bridge"
_STAGE_ORIGINAL_MARKER = "_raylight_mmh3director_original_stage"
_original_comfy_sample: Callable | None = None
_installed_wrapper: Callable | None = None
_original_director_stage: Callable | None = None
_installed_stage_wrapper: Callable | None = None
_patched_stage_targets: list[tuple[object, Callable]] = []
_LOG = logging.getLogger("raylight.mmh3director")


def _contract_failures(sample_single_stage: Callable, expected: tuple[str, ...]) -> tuple[list[str], list[str]]:
    parameters = inspect.signature(sample_single_stage).parameters
    missing = [name for name in expected if name not in parameters]
    wrong_kind = [
        name
        for name in expected
        if name in parameters and parameters[name].kind is not inspect.Parameter.KEYWORD_ONLY
    ]
    return missing, wrong_kind


def validate_mmh3director_sample_single_stage(sample_single_stage: Callable) -> str:
    """Accept the legacy and current Director seams, and identify the route."""

    sample_single_stage = inspect.unwrap(sample_single_stage)
    current_missing, current_wrong_kind = _contract_failures(
        sample_single_stage,
        EXPECTED_SAMPLE_SINGLE_STAGE_PARAMETERS,
    )
    if not current_missing and not current_wrong_kind:
        return "stage"

    legacy_missing, legacy_wrong_kind = _contract_failures(
        sample_single_stage,
        LEGACY_SAMPLE_SINGLE_STAGE_PARAMETERS,
    )
    if not legacy_missing and not legacy_wrong_kind:
        return "legacy"

    failures = []
    if current_missing:
        failures.append(f"missing or renamed parameter(s): {', '.join(current_missing)}")
    if current_wrong_kind:
        failures.append(f"non-keyword parameter(s): {', '.join(current_wrong_kind)}")
    details = "; ".join(failures)
    raise RuntimeError(
        "Unsupported MiniMaxH3Director sample_single_stage contract: "
        f"{details}. This adapter supports Director {LEGACY_DIRECTOR_VERSION} at commit "
        f"{LEGACY_DIRECTOR_COMMIT[:7]} and Director {SUPPORTED_DIRECTOR_VERSION} at commit "
        f"{SUPPORTED_DIRECTOR_COMMIT[:7]}. Please update Director or this Raylight adapter "
        "to a mutually supported version."
    )


def _director_module_identity(module_name: str, module) -> bool:
    module_path = str(getattr(module, "__file__", ""))
    identity = f"{module_name} {module_path}".lower()
    return "minimaxh3_director" in identity


def _find_loaded_sample_single_stage(*, import_if_missing: bool = True) -> Callable | None:
    for module_name, module in tuple(sys.modules.items()):
        if not _director_module_identity(module_name, module) or not module_name.endswith("director.core_sampling"):
            continue
        sample_single_stage = getattr(module, "sample_single_stage", None)
        if callable(sample_single_stage):
            return getattr(sample_single_stage, _STAGE_ORIGINAL_MARKER, sample_single_stage)

    if not import_if_missing:
        return None

    try:
        module = importlib.import_module("ComfyUI_MiniMaxH3_Director.director.core_sampling")
    except Exception as error:
        raise RuntimeError(
            "Ray MODEL Adapter (MMH3Director) requires the MiniMaxH3Director package at sampling time. "
            f"Validated target: Director {SUPPORTED_DIRECTOR_VERSION}, commit {SUPPORTED_DIRECTOR_COMMIT[:7]}."
        ) from error
    sample_single_stage = module.sample_single_stage
    return getattr(sample_single_stage, _STAGE_ORIGINAL_MARKER, sample_single_stage)


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


def _release_main_process_models_before_remote_sampling() -> None:
    """Free Director preprocessing VRAM before both Ray ranks begin sampling."""

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _first_node_output(output):
    if hasattr(output, "args") and output.args:
        return output.args[0]
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    try:
        return output[0]
    except (KeyError, TypeError, IndexError) as error:
        raise RuntimeError(f"Unexpected Director node output type: {type(output).__name__}") from error


def dispatch_mmh3director_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    on_phase=None,
    on_step_preview=None,
    preview_every: int = 1,
    denoise: float = 1.0,
    phase_name: str = "sample",
    sigmas=None,
    apply_shift: bool = True,
):
    """Route the current Director stage seam to the existing Ray workers."""

    if not isinstance(model, RayMMH3DirectorModelProxy):
        raise TypeError("dispatch_mmh3director_stage requires RayMMH3DirectorModelProxy")
    latent_samples = latent.get("samples") if isinstance(latent, dict) else None
    is_sample_tensor = isinstance(latent_samples, torch.Tensor) or (
        getattr(latent_samples, "is_nested", False) is True
        and callable(getattr(latent_samples, "unbind", None))
    )
    if not is_sample_tensor:
        raise TypeError(
            "MMH3Director stage requires a latent dict containing tensor or AV nested samples"
        )

    if on_phase is not None:
        on_phase(phase_name, 0)

    model_use = model
    if apply_shift:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        model_use = _first_node_output(
            MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
        )

    latent_image = comfy.sample.fix_empty_latent_channels(
        model_use,
        latent["samples"],
        latent.get("downscale_ratio_spacial"),
        latent.get("downscale_ratio_temporal"),
    )
    noise = comfy.sample.prepare_noise(
        latent_image,
        int(seed),
        latent.get("batch_index"),
    )
    normalized_sigmas = sigmas
    if isinstance(sigmas, torch.Tensor):
        normalized_sigmas = sigmas.detach().float().cpu().reshape(-1)
    elif sigmas is not None:
        normalized_sigmas = torch.tensor([float(value) for value in sigmas], dtype=torch.float32)

    callback = None
    if on_step_preview is not None:
        every = max(1, int(preview_every))

        def callback(step, x0, _x, total_steps):
            last = max(0, int(total_steps) - 1)
            if int(preview_every) < 0:
                show = int(step) >= last
            else:
                show = int(step) % every == 0 or int(step) >= last
            if show:
                on_step_preview(int(step), int(total_steps), x0)

    samples = dispatch_mmh3director_sample(
        model_use,
        noise,
        int(steps),
        float(cfg),
        str(sampler_name),
        str(scheduler),
        positive,
        negative if negative else [],
        latent_image,
        denoise=float(max(0.0, min(1.0, denoise))),
        noise_mask=latent.get("noise_mask"),
        sigmas=normalized_sigmas,
        callback=callback,
        seed=int(seed),
        apply_shift=bool(apply_shift),
    )

    output = latent.copy()
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    if on_phase is not None:
        on_phase(phase_name, 1)
    return output


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
    apply_shift=True,
):
    if not isinstance(model, RayMMH3DirectorModelProxy):
        raise TypeError("dispatch_mmh3director_sample requires RayMMH3DirectorModelProxy")
    if model.metadata.model_family != "minimax_h3":
        raise RuntimeError(f"MMH3Director proxy sampling requires model_family=minimax_h3; found {model.metadata.model_family}")

    workers = list(model.ray_actors.get("workers", ()))
    if not workers:
        raise RuntimeError("MMH3Director proxy sampling requires live Ray workers")
    configured_world_size = model.ray_actors.get("world_size")
    if configured_world_size is not None and int(configured_world_size) != len(workers):
        raise RuntimeError(f"MMH3Director proxy world_size={configured_world_size} but has {len(workers)} live Ray workers")

    ensure_mmh3director_compatibility()
    _release_main_process_models_before_remote_sampling()
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
        "apply_shift": bool(apply_shift),
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
            raise RuntimeError(f"MMH3Director remote sampling operation {operation_id} failed before rank validation") from error

        if not isinstance(results, (list, tuple)) or len(results) != len(workers):
            received = len(results) if isinstance(results, (list, tuple)) else 0
            raise RuntimeError(f"MMH3Director operation {operation_id} returned {received}/{len(workers)} rank results")

        reference_samples = None
        for expected_rank, result in enumerate(results):
            if result is None:
                raise RuntimeError(f"MMH3Director operation {operation_id} missing result for rank {expected_rank}")
            if not isinstance(result, dict):
                raise TypeError(f"MMH3Director operation {operation_id} rank {expected_rank} returned {type(result).__name__}")
            actual_rank = result.get("rank")
            if actual_rank != expected_rank:
                raise ValueError(f"MMH3Director operation {operation_id} expected rank {expected_rank}, got rank {actual_rank}")
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
            f"[Raylight] MMH3Director remote sampling failed operation={operation_id} error={type(error).__name__}",
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


def _make_mmh3director_stage_wrapper(original: Callable) -> Callable:
    @functools.wraps(original)
    def stage_wrapper(*args, **kwargs):
        model = kwargs.get("model")
        if isinstance(model, RayMMH3DirectorModelProxy):
            return dispatch_mmh3director_stage(*args, **kwargs)
        return original(*args, **kwargs)

    setattr(stage_wrapper, _STAGE_BRIDGE_MARKER, True)
    setattr(stage_wrapper, _STAGE_ORIGINAL_MARKER, original)
    return stage_wrapper


def _install_mmh3director_stage_bridge(*, require_director: bool) -> bool:
    global _installed_stage_wrapper, _original_director_stage

    original = _find_loaded_sample_single_stage(import_if_missing=require_director)
    if original is None:
        return False
    route = validate_mmh3director_sample_single_stage(original)
    if route != "stage":
        return False

    wrapper = _installed_stage_wrapper
    if wrapper is None or getattr(wrapper, _STAGE_ORIGINAL_MARKER, None) is not original:
        wrapper = _make_mmh3director_stage_wrapper(original)
        _installed_stage_wrapper = wrapper
        _original_director_stage = original

    changed = False
    recorded_modules = {id(module) for module, _original in _patched_stage_targets}
    for module_name, module in tuple(sys.modules.items()):
        if not _director_module_identity(module_name, module):
            continue
        current = getattr(module, "sample_single_stage", None)
        if current is not original:
            continue
        setattr(module, "sample_single_stage", wrapper)
        if id(module) not in recorded_modules:
            _patched_stage_targets.append((module, original))
            recorded_modules.add(id(module))
        changed = True
    return changed


def install_mmh3director_sample_bridge(*, require_director: bool = False) -> bool:
    global _installed_wrapper, _original_comfy_sample

    changed = False
    current = comfy.sample.sample
    if getattr(current, _BRIDGE_MARKER, False) is True:
        _installed_wrapper = current
        _original_comfy_sample = getattr(current, _ORIGINAL_MARKER)
    else:
        _original_comfy_sample = current
        setattr(_raylight_sample_dispatch, _BRIDGE_MARKER, True)
        setattr(_raylight_sample_dispatch, _ORIGINAL_MARKER, current)
        comfy.sample.sample = _raylight_sample_dispatch
        _installed_wrapper = _raylight_sample_dispatch
        changed = True

    return _install_mmh3director_stage_bridge(require_director=require_director) or changed


def uninstall_mmh3director_sample_bridge_for_test() -> None:
    global _installed_stage_wrapper, _installed_wrapper
    global _original_comfy_sample, _original_director_stage

    current = comfy.sample.sample
    if getattr(current, _BRIDGE_MARKER, False) is True:
        comfy.sample.sample = getattr(current, _ORIGINAL_MARKER)
    for module, original in reversed(_patched_stage_targets):
        current_stage = getattr(module, "sample_single_stage", None)
        if getattr(current_stage, _STAGE_BRIDGE_MARKER, False) is True:
            module.sample_single_stage = original
    _patched_stage_targets.clear()
    _installed_stage_wrapper = None
    _original_director_stage = None
    _installed_wrapper = None
    _original_comfy_sample = None
