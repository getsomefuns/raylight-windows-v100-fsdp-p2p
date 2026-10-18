from __future__ import annotations

import contextvars
import functools
import importlib
import inspect
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

import ray
import torch

from raylight.distributed_worker.ray_worker_vae import (
    build_mmh3_temporal_plan,
    decode_mmh3_temporal_chunks,
    merge_mmh3_temporal_chunks,
)

from .proxy import RayMMH3DirectorModelProxy


_LOG = logging.getLogger("raylight.mmh3director.vae")
_EXECUTE_MARKER = "_raylight_mmh3_vae_execute_bridge"
_EXECUTE_ORIGINAL = "_raylight_mmh3_vae_original_execute"
_DECODE_MARKER = "_raylight_mmh3_vae_decode_bridge"
_DECODE_ORIGINAL = "_raylight_mmh3_vae_original_decode"


@dataclass
class DirectorVaeDecodeContext:
    remote_worker: object
    vae_path: str
    vae_identity: int | None = None
    workers: tuple[object, ...] = ()
    needs_gpu_cleanup: bool = True


_ACTIVE_CONTEXT: contextvars.ContextVar[DirectorVaeDecodeContext | None] = (
    contextvars.ContextVar("raylight_mmh3_vae_context", default=None)
)
_patched_director_classes: list[tuple[type, Callable]] = []
_original_vae_decode: Callable | None = None
_installed_vae_decode: Callable | None = None


def current_mmh3_vae_context() -> DirectorVaeDecodeContext | None:
    return _ACTIVE_CONTEXT.get()


@contextmanager
def activate_mmh3_vae_context(context: DirectorVaeDecodeContext):
    token = _ACTIVE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _ACTIVE_CONTEXT.reset(token)


def extract_vae_source_path(vae) -> str | None:
    cached = getattr(getattr(vae, "patcher", None), "cached_patcher_init", None)
    if not isinstance(cached, (tuple, list)) or len(cached) < 2:
        return None
    args = cached[1]
    if not isinstance(args, (tuple, list)) or not args:
        return None
    path = args[0]
    return str(path) if isinstance(path, (str, bytes)) else None


def _bind_execute_arguments(original: Callable, args, kwargs):
    try:
        return inspect.signature(original).bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        return {}


def make_director_execute_wrapper(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        bound = _bind_execute_arguments(original, args, kwargs)
        model = bound.get("model")
        vae = bound.get("video_vae")
        if not isinstance(model, RayMMH3DirectorModelProxy):
            return original(*args, **kwargs)
        workers = list(model.ray_actors.get("workers", ()))
        vae_path = extract_vae_source_path(vae)
        if len(workers) < 2 or vae_path is None:
            return original(*args, **kwargs)

        remote_worker = workers[1]
        try:
            ray.get(remote_worker.start_mmh3_vae_preload.remote(vae_path))
        except Exception as error:
            _LOG.warning(
                "MiniMax H3 VideoVAE background preload unavailable; using native decode: %s: %s",
                type(error).__name__,
                error,
            )
            return original(*args, **kwargs)

        context = DirectorVaeDecodeContext(
            remote_worker=remote_worker,
            vae_path=vae_path,
            vae_identity=id(vae),
            workers=tuple(workers),
        )
        with activate_mmh3_vae_context(context):
            try:
                return original(*args, **kwargs)
            finally:
                try:
                    ray.get(remote_worker.clear_sampling_vram.remote())
                except Exception as error:
                    _LOG.warning(
                        "MiniMax H3 VideoVAE final GPU cleanup failed: %s: %s",
                        type(error).__name__,
                        error,
                    )

    setattr(wrapped, _EXECUTE_MARKER, True)
    setattr(wrapped, _EXECUTE_ORIGINAL, original)
    return wrapped


def _video_stream(samples):
    if getattr(samples, "is_nested", False):
        streams = samples.unbind()
        if not streams:
            raise ValueError("MiniMax H3 AV latent contains no streams.")
        samples = streams[0]
    if not isinstance(samples, torch.Tensor) or samples.ndim != 5:
        raise TypeError("MiniMax H3 VideoVAE bridge expects a 5-D video latent.")
    return samples


def hybrid_decode_mmh3_video(vae, samples, context: DirectorVaeDecodeContext):
    video = _video_stream(samples)
    model = vae.first_stage_model
    plan = build_mmh3_temporal_plan(model, video.shape[2])
    if plan["num_chunks"] < 2:
        raise ValueError("MiniMax H3 VideoVAE bridge needs at least two temporal chunks.")

    local_indices = list(range(0, plan["num_chunks"], 2))
    remote_indices = list(range(1, plan["num_chunks"], 2))
    if not remote_indices:
        raise ValueError("MiniMax H3 VideoVAE bridge has no remote temporal chunks.")

    started = time.perf_counter()
    workers = context.workers or (context.remote_worker,)
    if context.needs_gpu_cleanup:
        ray.get([worker.clear_sampling_vram.remote() for worker in workers])
    remote_future = context.remote_worker.ray_mmh3_vae_decode_chunks.remote(
        video,
        remote_indices,
        context.vae_path,
    )
    local = decode_mmh3_temporal_chunks(vae, video, local_indices)
    remote = ray.get(remote_future)
    if tuple(remote.get("input_shape", ())) != tuple(local["input_shape"]):
        raise ValueError("MiniMax H3 VideoVAE ranks returned different input shapes.")
    chunks = list(local["chunks"]) + list(remote.get("chunks", ()))
    merged = merge_mmh3_temporal_chunks(model, local["input_shape"], chunks)
    context.needs_gpu_cleanup = False
    elapsed = time.perf_counter() - started
    print(
        "[Raylight] MMH3Director dual-V100 VideoVAE decode completed "
        f"chunks={plan['num_chunks']} seconds={elapsed:.3f}",
        flush=True,
    )
    return merged.movedim(1, -1)


def make_vae_decode_wrapper(original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapped(self, vae, samples):
        context = current_mmh3_vae_context()
        if context is None or (
            context.vae_identity is not None and context.vae_identity != id(vae)
        ):
            return original(self, vae, samples)
        latent = samples.get("samples") if isinstance(samples, dict) else None
        try:
            images = hybrid_decode_mmh3_video(vae, latent, context)
        except Exception as error:
            error_name = type(error).__name__
            error_message = str(error)
        else:
            if images.ndim == 5:
                images = images.reshape(
                    -1,
                    images.shape[-3],
                    images.shape[-2],
                    images.shape[-1],
                )
            return (images,)

        _LOG.warning(
            "MiniMax H3 dual-V100 VideoVAE decode failed; retrying native decode: %s: %s",
            error_name,
            error_message,
        )
        return original(self, vae, samples)

    setattr(wrapped, _DECODE_MARKER, True)
    setattr(wrapped, _DECODE_ORIGINAL, original)
    return wrapped


def prepare_remote_vae_for_sampling() -> bool:
    context = current_mmh3_vae_context()
    if context is None:
        return False
    ray.get(context.remote_worker.clear_sampling_vram.remote())
    context.needs_gpu_cleanup = True
    return True


def _director_module(module_name: str, module) -> bool:
    identity = f"{module_name} {getattr(module, '__file__', '')}".lower()
    return "minimaxh3_director" in identity and module_name.endswith("nodes.director")


def _find_director_classes(*, import_if_missing: bool) -> list[type]:
    classes = []
    for module_name, module in tuple(sys.modules.items()):
        if _director_module(module_name, module):
            cls = getattr(module, "MiniMaxH3Director", None)
            if isinstance(cls, type) and cls not in classes:
                classes.append(cls)
    if classes or not import_if_missing:
        return classes
    try:
        module = importlib.import_module("ComfyUI_MiniMaxH3_Director.nodes.director")
    except Exception:
        return []
    cls = getattr(module, "MiniMaxH3Director", None)
    return [cls] if isinstance(cls, type) else []


def install_mmh3director_vae_decode_bridge(*, require_director: bool = False) -> bool:
    global _original_vae_decode, _installed_vae_decode

    changed = False
    for cls in _find_director_classes(import_if_missing=require_director):
        current = cls.execute
        if getattr(current, _EXECUTE_MARKER, False):
            continue
        wrapper = make_director_execute_wrapper(current)
        cls.execute = wrapper
        _patched_director_classes.append((cls, current))
        changed = True

    try:
        import nodes as comfy_nodes
    except Exception:
        return changed
    current_decode = comfy_nodes.VAEDecode.decode
    if getattr(current_decode, _DECODE_MARKER, False):
        _installed_vae_decode = current_decode
        _original_vae_decode = getattr(current_decode, _DECODE_ORIGINAL)
        return changed
    _original_vae_decode = current_decode
    _installed_vae_decode = make_vae_decode_wrapper(current_decode)
    comfy_nodes.VAEDecode.decode = _installed_vae_decode
    return True


def uninstall_mmh3director_vae_decode_bridge_for_test() -> None:
    global _original_vae_decode, _installed_vae_decode

    for cls, original in reversed(_patched_director_classes):
        if getattr(cls.execute, _EXECUTE_MARKER, False):
            cls.execute = original
    _patched_director_classes.clear()
    try:
        import nodes as comfy_nodes
    except Exception:
        comfy_nodes = None
    if comfy_nodes is not None and getattr(comfy_nodes.VAEDecode.decode, _DECODE_MARKER, False):
        comfy_nodes.VAEDecode.decode = getattr(comfy_nodes.VAEDecode.decode, _DECODE_ORIGINAL)
    _original_vae_decode = None
    _installed_vae_decode = None
