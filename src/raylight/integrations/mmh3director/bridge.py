from __future__ import annotations

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
from raylight.diagnostics.memory_recorder import MemoryRunRecorder

from .proxy import RayMMH3DirectorModelProxy
from .external_acceleration import serialize_external_acceleration
from .segment_lora import (
    SEGMENT_LORA_BRIDGE_MARKER,
    SEGMENT_LORA_ORIGINAL_MARKER,
    make_segment_lora_wrapper,
)
from . import preview_transport


SUPPORTED_DIRECTOR_VERSION = "main"
SUPPORTED_DIRECTOR_COMMIT = "4c6245e518dd8a494f3afc4f3cccd53f22d82d10"
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
    "after_shift",
    "enable_tiling",
    "tile_count",
    "tile_overlap",
    "shift_cache",
    "on_step_state",
    "zero_noise",
    "noise_override",
)

_CONTINUE_REMASK_CAPABILITY = "continue_prefix_remask_v1"
_CONTINUE_REMASK_MODULE_SUFFIX = ".director.h3_latent_continue"
_DIRECTOR_LATENT_METADATA_KEYS = (
    "_director_continue_prefix_steps",
    "_director_continue_seam_min",
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
_original_segment_lora_apply: Callable | None = None
_installed_segment_lora_wrapper: Callable | None = None
_patched_segment_lora_targets: list[tuple[object, Callable]] = []
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


def _find_loaded_segment_lora_apply(*, import_if_missing: bool = True) -> Callable | None:
    for module_name, module in tuple(sys.modules.items()):
        if not _director_module_identity(module_name, module) or not module_name.endswith("director.segment_loras"):
            continue
        apply_segment_loras = getattr(module, "apply_segment_loras", None)
        if callable(apply_segment_loras):
            return getattr(
                apply_segment_loras,
                SEGMENT_LORA_ORIGINAL_MARKER,
                apply_segment_loras,
            )

    if not import_if_missing:
        return None

    try:
        module = importlib.import_module("ComfyUI_MiniMaxH3_Director.director.segment_loras")
    except Exception as error:
        raise RuntimeError(
            "Ray MODEL Adapter (MMH3Director) requires Director's segment LoRA module. "
            f"Validated target: Director {SUPPORTED_DIRECTOR_VERSION}, commit {SUPPORTED_DIRECTOR_COMMIT[:7]}."
        ) from error
    apply_segment_loras = module.apply_segment_loras
    return getattr(
        apply_segment_loras,
        SEGMENT_LORA_ORIGINAL_MARKER,
        apply_segment_loras,
    )


def ensure_mmh3director_compatibility() -> None:
    validate_mmh3director_sample_single_stage(_find_loaded_sample_single_stage())


def serialize_director_after_shift(after_shift: Callable | None) -> str | None:
    """Convert a known Director hook into a stable worker-side capability."""

    if after_shift is None:
        return None
    callback = inspect.unwrap(after_shift)
    module_name = str(getattr(callback, "__module__", ""))
    callback_name = str(getattr(callback, "__name__", ""))
    if callback_name == "install_continue_prefix_remask" and (
        module_name == "director.h3_latent_continue"
        or module_name.endswith(_CONTINUE_REMASK_MODULE_SUFFIX)
    ):
        return _CONTINUE_REMASK_CAPABILITY
    raise RuntimeError(
        "Unsupported MiniMaxH3Director after_shift callback "
        f"{module_name}.{callback_name}; update the Raylight adapter before sampling."
    )


def _director_latent_metadata(latent: dict) -> dict[str, object]:
    return {
        key: latent[key]
        for key in _DIRECTOR_LATENT_METADATA_KEYS
        if key in latent
    }


def _zero_noise_like(samples):
    """Match Director's ZeroNoise behavior for tensor and AV nested latents."""

    if getattr(samples, "is_nested", False) and callable(getattr(samples, "unbind", None)):
        import comfy.nested_tensor

        return comfy.nested_tensor.NestedTensor(
            tuple(torch.zeros_like(stream) for stream in samples.unbind())
        )
    if isinstance(samples, torch.Tensor):
        return torch.zeros_like(samples)
    raise TypeError(f"Cannot build MMH3Director zero noise for {type(samples).__name__}")


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
    after_shift=None,
    enable_tiling: bool = False,
    tile_count: int = 2,
    tile_overlap: int = 128,
    shift_cache=None,
    on_step_state=None,
    zero_noise: bool = False,
    noise_override=None,
):
    """Route the current Director stage seam to the existing Ray workers."""

    # Director 19b6a73 reuses SigmaShift clones through this local cache.  The
    # real model lives in Ray workers, so retaining the main-process proxy clone
    # has no weight/VRAM benefit. Accept the contract while keeping worker-side
    # lifecycle ownership in Raylight.
    del shift_cache

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
    if noise_override is not None:
        noise = noise_override
    elif zero_noise:
        noise = _zero_noise_like(latent_image)
    else:
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

    after_shift_capability = serialize_director_after_shift(after_shift)

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
        step_state_callback=on_step_state,
        preview_every=int(preview_every),
        seed=int(seed),
        apply_shift=bool(apply_shift),
        after_shift_capability=after_shift_capability,
        director_latent_metadata=_director_latent_metadata(latent),
        enable_tiling=bool(enable_tiling),
        tile_count=max(1, min(8, int(tile_count))),
        tile_overlap=max(0, int(tile_overlap)),
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
    step_state_callback=None,
    preview_every=1,
    disable_pbar=False,
    seed=None,
    apply_shift=True,
    after_shift_capability=None,
    director_latent_metadata=None,
    enable_tiling=False,
    tile_count=2,
    tile_overlap=128,
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
    from .vae_decode import prepare_remote_vae_for_sampling

    prepare_remote_vae_for_sampling()
    _release_main_process_models_before_remote_sampling()
    operation_id = uuid.uuid4().hex
    preview_queue = preview_transport.create_preview_queue() if callback is not None else None
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
        "after_shift_capability": after_shift_capability,
        "director_latent_metadata": dict(director_latent_metadata or {}),
        "enable_tiling": bool(enable_tiling),
        "tile_count": max(1, min(8, int(tile_count))),
        "tile_overlap": max(0, int(tile_overlap)),
        "model_options": serialize_external_acceleration(model.model_options),
        "segment_loras": [dict(item) for item in model.segment_loras],
    }
    if step_state_callback is not None:
        request["capture_step_state"] = True
    if preview_queue is not None:
        request["preview_queue"] = preview_queue
        request["preview_every"] = int(preview_every)

    print(
        f"[Raylight] MMH3Director remote sampling started operation={operation_id} ranks={len(workers)}",
        flush=True,
    )
    sampling_error = None
    memory_record = MemoryRunRecorder(
        "mmh3director_main_sampling",
        operation_id=operation_id,
        fields={"steps": int(steps), "ranks": len(workers)},
    )
    memory_record.__enter__()
    last_preview_step = [-1]

    def forward_preview(step, x0, x, total_steps):
        last_preview_step[0] = max(last_preview_step[0], int(step))
        callback(step, x0, x, total_steps)

    try:
        try:
            futures = [worker.sample_mmh3director_request.remote(request) for worker in workers]
            if preview_queue is None:
                results = ray.get(futures)
            else:
                results = preview_transport.wait_for_worker_results(
                    futures,
                    preview_queue,
                    forward_preview,
                )
        except Exception as error:
            error_detail = str(error).strip()
            if "\n" in error_detail:
                error_detail = next(
                    (line.strip() for line in reversed(error_detail.splitlines()) if line.strip()),
                    type(error).__name__,
                )
            raise RuntimeError(
                f"MMH3Director remote sampling operation {operation_id} failed before rank validation: "
                f"{type(error).__name__}: {error_detail}"
            ) from error

        if not isinstance(results, (list, tuple)) or len(results) != len(workers):
            received = len(results) if isinstance(results, (list, tuple)) else 0
            raise RuntimeError(f"MMH3Director operation {operation_id} returned {received}/{len(workers)} rank results")

        reference_samples = None
        final_step_state = None
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
            if expected_rank == 0:
                final_step_state = result.get("step_state")
            elif result.get("step_state") is not None:
                raise ValueError(
                    f"rank {expected_rank} returned duplicate MMH3Director step state"
                )

        normalized = reference_samples.to(
            device=comfy.model_management.intermediate_device(),
            dtype=comfy.model_management.intermediate_dtype(),
        )
        if step_state_callback is not None:
            if not isinstance(final_step_state, dict):
                raise RuntimeError(
                    f"MMH3Director operation {operation_id} did not return final step state"
                )
            required_state = ("step", "total_steps", "x0", "x")
            missing_state = [name for name in required_state if name not in final_step_state]
            if missing_state:
                raise RuntimeError(
                    "MMH3Director final step state missing field(s): "
                    + ", ".join(missing_state)
                )
            step_state_callback(
                int(final_step_state["step"]),
                int(final_step_state["total_steps"]),
                final_step_state["x0"],
                final_step_state["x"],
            )
        if callback is not None and last_preview_step[0] < max(0, int(steps) - 1):
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
        preview_transport.close_preview_queue(preview_queue)
        memory_record.finish(sampling_error)
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


def _install_mmh3director_segment_lora_bridge(*, require_director: bool) -> bool:
    global _installed_segment_lora_wrapper, _original_segment_lora_apply

    # Director releases before per-segment LoRA do not provide this module.
    # A current Director imports it before the MODEL adapter executes, so do
    # not turn an optional compatibility seam into a legacy startup failure.
    del require_director
    original = _find_loaded_segment_lora_apply(import_if_missing=False)
    if original is None:
        return False

    wrapper = _installed_segment_lora_wrapper
    if wrapper is None or getattr(wrapper, SEGMENT_LORA_ORIGINAL_MARKER, None) is not original:
        wrapper = make_segment_lora_wrapper(original)
        _installed_segment_lora_wrapper = wrapper
        _original_segment_lora_apply = original

    changed = False
    recorded_modules = {id(module) for module, _original in _patched_segment_lora_targets}
    for module_name, module in tuple(sys.modules.items()):
        if not _director_module_identity(module_name, module):
            continue
        current = getattr(module, "apply_segment_loras", None)
        if current is not original:
            continue
        setattr(module, "apply_segment_loras", wrapper)
        if id(module) not in recorded_modules:
            _patched_segment_lora_targets.append((module, original))
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

    stage_changed = _install_mmh3director_stage_bridge(require_director=require_director)
    segment_lora_changed = _install_mmh3director_segment_lora_bridge(
        require_director=require_director
    )
    from .vae_decode import install_mmh3director_vae_decode_bridge

    vae_changed = install_mmh3director_vae_decode_bridge(
        require_director=require_director
    )
    return vae_changed or segment_lora_changed or stage_changed or changed


def uninstall_mmh3director_sample_bridge_for_test() -> None:
    global _installed_stage_wrapper, _installed_wrapper
    global _original_comfy_sample, _original_director_stage
    global _installed_segment_lora_wrapper, _original_segment_lora_apply

    from .vae_decode import uninstall_mmh3director_vae_decode_bridge_for_test

    uninstall_mmh3director_vae_decode_bridge_for_test()
    current = comfy.sample.sample
    if getattr(current, _BRIDGE_MARKER, False) is True:
        comfy.sample.sample = getattr(current, _ORIGINAL_MARKER)
    for module, original in reversed(_patched_stage_targets):
        current_stage = getattr(module, "sample_single_stage", None)
        if getattr(current_stage, _STAGE_BRIDGE_MARKER, False) is True:
            module.sample_single_stage = original
    _patched_stage_targets.clear()
    for module, original in reversed(_patched_segment_lora_targets):
        current_apply = getattr(module, "apply_segment_loras", None)
        if getattr(current_apply, SEGMENT_LORA_BRIDGE_MARKER, False) is True:
            module.apply_segment_loras = original
    _patched_segment_lora_targets.clear()
    _installed_stage_wrapper = None
    _original_director_stage = None
    _installed_segment_lora_wrapper = None
    _original_segment_lora_apply = None
    _installed_wrapper = None
    _original_comfy_sample = None
