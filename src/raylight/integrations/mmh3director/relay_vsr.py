"""Opt-in Director VSR routing; unrelated Director/VAE calls stay native."""
from contextvars import ContextVar
import functools
import importlib
import inspect
import logging
import math
import sys
import time

import torch

from raylight.relayvsr.admission import authorize_flash
from raylight.relayvsr.local import release_local_pool, run_local_flash
from raylight.relayvsr.runtime import _checkpoint_scale
from .proxy import RayMMH3DirectorModelProxy
from .vae_decode import extract_vae_source_path


_LOG = logging.getLogger(__name__)
_ACTIVE = ContextVar("raylight_director_vsr_workers", default=None)
_EXECUTE_MARKER = "_raylight_director_vsr_execute"
_SAMPLE_MARKER = "_raylight_director_vsr_sample"
_FINGERPRINT_MARKER = "_raylight_director_vsr_fingerprint"
_PATCHED_CLASSES = []
_PATCHED_ALIASES = []
_PATCHED_FINGERPRINTS = []


def current_director_vsr_workers():
    return _ACTIVE.get()


def make_execute_wrapper(original):
    signature = inspect.signature(original)
    @functools.wraps(original)
    def execute(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs).arguments
        model, pack = bound.get("model"), bound.get("vsr")
        if not isinstance(model, RayMMH3DirectorModelProxy) or not isinstance(pack, dict) or pack.get("enabled") is False:
            return original(*args, **kwargs)
        workers = tuple(model.ray_actors.get("workers", ()))
        if len(workers) != 2:
            raise RuntimeError("Director dual V100 RelayVSR requires both H3 workers")
        marker = _ACTIVE.set(workers)
        try:
            return original(*args, **kwargs)
        finally:
            _ACTIVE.reset(marker)
    setattr(execute, _EXECUTE_MARKER, True)
    return execute


def _release_main_gpu_models():
    import comfy.model_management as mm
    mm.unload_all_models()
    mm.soft_empty_cache()


def _unpack(output):
    return output.args if hasattr(output, "args") else output


def _sample_relay_vsr(images, pack):
    from comfy_extras.nodes_wan import RelayVSRConditioning
    from nodes import common_ksampler
    positive, latent = _unpack(RelayVSRConditioning.execute(pack["model"], images))
    return _unpack(common_ksampler(
        pack["model"], int(pack.get("seed", 0)), int(pack.get("steps", 1)),
        float(pack.get("cfg", 1.)), str(pack.get("sampler", "euler")),
        str(pack.get("scheduler", "simple")), positive, positive, latent,
        denoise=float(pack.get("denoise", 1.)),
    ))[0]


def _validate_inputs(frames, pack):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] != 3 or min(frames.shape[:3]) < 1:
        raise ValueError("Director RelayVSR needs nonempty F,H,W,3 frames")
    model, vae = pack.get("model"), pack.get("vae")
    diffusion = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion is None or "RelayVSR" not in type(diffusion).__name__:
        raise ValueError("Director RelayVSR requires its native RelayVSR model, not H3")
    path = extract_vae_source_path(vae)
    if not path:
        raise ValueError("Cannot identify the native RelayVSR Flash VAE checkpoint path")
    scale = _checkpoint_scale(path)
    if getattr(diffusion, "latent_stride", None) != 16 // scale:
        raise ValueError("Director RelayVSR model/Flash scale or latent stride mismatch")
    for key, default in (("detail", 1.), ("cfg", 1.), ("denoise", 1.)):
        value = float(pack.get(key, default))
        if not math.isfinite(value) or value < 0 or (key != "cfg" and value > 1):
            raise ValueError(f"Invalid Director RelayVSR {key}")
    if int(pack.get("steps", 1)) < 1:
        raise ValueError("Director RelayVSR requires at least one sampling step")
    return path, scale


@torch.inference_mode()
def run_director_relay_vsr(frames, pack, workers):
    path, scale = _validate_inputs(frames, pack)
    images = frames.detach().to(device="cpu", dtype=torch.float32).contiguous()
    def cleanup():
        release_local_pool()
        _release_main_gpu_models()
    started = time.perf_counter()
    with authorize_flash(workers, cleanup=cleanup):
        _release_main_gpu_models()
        sampled = _sample_relay_vsr(images, pack)
        output = run_local_flash(path, sampled["samples"], images, float(pack.get("detail", 1.)),
            worker_count=2, keep_workers=False, prepare=_release_main_gpu_models)
        expected = (images.shape[0], images.shape[1] * scale, images.shape[2] * scale, 3)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"Director RelayVSR output frame/size mismatch: {tuple(output.shape)} != {expected}")
    _LOG.info("[Raylight] Director dual-V100 RelayVSR x%d completed frames=%d seconds=%.3f",
              scale, images.shape[0], time.perf_counter() - started)
    return output


def make_relay_vsr_wrapper(original):
    @functools.wraps(original)
    def apply(frames, pack):
        workers = current_director_vsr_workers()
        if workers is None:
            return original(frames, pack)
        return run_director_relay_vsr(frames, pack, workers)
    setattr(apply, _SAMPLE_MARKER, True)
    return apply


def make_fingerprint_wrapper(original):
    @functools.wraps(original)
    def fingerprint(plan):
        result = original(plan)
        if current_director_vsr_workers() is None or not result.get("vsr"):
            return result
        result = dict(result)
        pack = plan.vsr
        # Upstream `value or 1` folds the legal value zero into one. Preserve
        # zeros only in our active Ray Director execution; leave native calls alone.
        for name in ("cfg", "denoise", "detail"):
            value = pack.get(name)
            result[f"vsr_{name}"] = round(float(1. if value is None else value), 4)
        return result
    setattr(fingerprint, _FINGERPRINT_MARKER, True)
    return fingerprint


def install_mmh3director_relay_vsr_bridge(*, require_director=False):
    from .vae_decode import _find_director_classes
    changed = False
    for cls in _find_director_classes(import_if_missing=require_director):
        if not cls.__module__.endswith(".nodes.director"):
            continue
        root = cls.__module__.removesuffix(".nodes.director")
        try:
            sample_module = importlib.import_module(f"{root}.director.relay_vsr.sample")
        except ModuleNotFoundError as error:
            # Older supported Directors have no VSR stage at all.
            if error.name and (error.name.endswith("relay_vsr") or error.name.endswith("relay_vsr.sample")):
                continue
            raise
        pack_module = importlib.import_module(f"{root}.director.relay_vsr.pack")
        original_fingerprint = pack_module.relay_vsr_fingerprint
        if not getattr(original_fingerprint, _FINGERPRINT_MARKER, False):
            if tuple(inspect.signature(original_fingerprint).parameters) != ("plan",):
                raise RuntimeError("Unvalidated Director RelayVSR fingerprint contract")
            wrapped_fingerprint = make_fingerprint_wrapper(original_fingerprint)
            pack_module.relay_vsr_fingerprint = wrapped_fingerprint
            _PATCHED_FINGERPRINTS.append((pack_module, original_fingerprint, wrapped_fingerprint))
            changed = True
        original = sample_module.apply_segment_relay_vsr
        if not getattr(original, _SAMPLE_MARKER, False):
            if tuple(inspect.signature(original).parameters) != ("frames", "pack"):
                raise RuntimeError("Unvalidated Director RelayVSR sampling contract")
            wrapped = make_relay_vsr_wrapper(original)
            for name, module in tuple(sys.modules.items()):
                if name.startswith(f"{root}.director") and getattr(module, "apply_segment_relay_vsr", None) is original:
                    setattr(module, "apply_segment_relay_vsr", wrapped)
                    _PATCHED_ALIASES.append((module, original, wrapped))
            changed = True
        if not getattr(cls.execute, _EXECUTE_MARKER, False):
            original_execute = cls.execute
            wrapper = make_execute_wrapper(original_execute)
            cls.execute = wrapper
            _PATCHED_CLASSES.append((cls, original_execute, wrapper))
            changed = True
    return changed


def uninstall_mmh3director_relay_vsr_bridge_for_test():
    for cls, original, wrapped in reversed(_PATCHED_CLASSES):
        if cls.execute is wrapped:
            cls.execute = original
    _PATCHED_CLASSES.clear()
    for module, original, wrapped in reversed(_PATCHED_ALIASES):
        if getattr(module, "apply_segment_relay_vsr", None) is wrapped:
            module.apply_segment_relay_vsr = original
    _PATCHED_ALIASES.clear()
    for module, original, wrapped in reversed(_PATCHED_FINGERPRINTS):
        if module.relay_vsr_fingerprint is wrapped:
            module.relay_vsr_fingerprint = original
    _PATCHED_FINGERPRINTS.clear()
