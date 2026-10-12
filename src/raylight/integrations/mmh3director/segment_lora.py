from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

from .proxy import RayMMH3DirectorModelProxy


MAX_SEGMENT_LORAS = 8
MIN_LORA_STRENGTH = -10.0
MAX_LORA_STRENGTH = 10.0
LORA_EPSILON = 1e-6
WORKER_LORA_STATE_CACHE_LIMIT = 2
SEGMENT_LORA_BRIDGE_MARKER = "_raylight_mmh3director_segment_lora_bridge"
SEGMENT_LORA_ORIGINAL_MARKER = "_raylight_mmh3director_original_segment_lora"


def _coerce_strength(value: Any) -> float:
    try:
        strength = float(value)
    except (TypeError, ValueError):
        strength = 1.0
    if not math.isfinite(strength):
        strength = 1.0
    return max(MIN_LORA_STRENGTH, min(MAX_LORA_STRENGTH, strength))


def _effective_rows(raw: Any) -> list[tuple[str, float]]:
    if not raw:
        return []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes)):
        return []

    rows: list[tuple[str, float]] = []
    for item in raw:
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, Mapping):
            continue
        name = str(
            item.get("name")
            or item.get("lora")
            or item.get("lora_name")
            or ""
        ).strip()
        active = item.get("active", item.get("enabled", True))
        strength = _coerce_strength(
            item.get("strength", item.get("strength_model", 1.0))
        )
        if not name or not bool(active) or abs(strength) <= LORA_EPSILON:
            continue
        rows.append((name, strength))
        if len(rows) >= MAX_SEGMENT_LORAS:
            break
    return rows


def resolve_segment_lora_descriptors(raw: Any) -> tuple[dict[str, object], ...]:
    """Resolve active Director rows without loading weight tensors in ComfyUI."""

    import folder_paths

    descriptors: list[dict[str, object]] = []
    for name, strength in _effective_rows(raw):
        try:
            path = folder_paths.get_full_path_or_raise("loras", name)
        except Exception as error:
            raise ValueError(
                f"MiniMax H3 Director: segment LoRA '{name}' not found under models/loras ({error})."
            ) from error
        descriptors.append(
            {
                "path": str(path),
                "strength_model": strength,
            }
        )
    return tuple(descriptors)


def make_segment_lora_wrapper(original: Callable) -> Callable:
    """Keep stock Director behavior except for the weightless Ray MODEL proxy."""

    @functools.wraps(original)
    def wrapper(model, rows, *, state_dict_cache=None):
        if not isinstance(model, RayMMH3DirectorModelProxy):
            return original(model, rows, state_dict_cache=state_dict_cache)
        return model.with_segment_loras(resolve_segment_lora_descriptors(rows))

    setattr(wrapper, SEGMENT_LORA_BRIDGE_MARKER, True)
    setattr(wrapper, SEGMENT_LORA_ORIGINAL_MARKER, original)
    return wrapper


def _lora_state_cache_key(path: str) -> tuple[str, int, int]:
    resolved = Path(path)
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def _load_lora_state(
    path: str,
    *,
    state_cache: MutableMapping[tuple[str, int, int], object] | None = None,
):
    import comfy.utils

    if state_cache is None:
        return comfy.utils.load_torch_file(path, safe_load=True)

    key = _lora_state_cache_key(path)
    cached = state_cache.get(key)
    if cached is not None:
        if hasattr(state_cache, "move_to_end"):
            state_cache.move_to_end(key)
        return cached

    state = comfy.utils.load_torch_file(path, safe_load=True)
    state_cache[key] = state
    if hasattr(state_cache, "move_to_end"):
        state_cache.move_to_end(key)
    while len(state_cache) > WORKER_LORA_STATE_CACHE_LIMIT:
        if hasattr(state_cache, "popitem"):
            try:
                state_cache.popitem(last=False)
                continue
            except TypeError:
                pass
        state_cache.pop(next(iter(state_cache)))
    return state


def apply_worker_segment_loras(
    model,
    descriptors,
    *,
    is_fsdp: bool,
    is_quant: bool,
    dynamic_sidecar: bool,
    state_cache: MutableMapping[tuple[str, int, int], object] | None = None,
):
    """Stack Director segment LoRAs on an already request-local model clone."""

    if not descriptors:
        return model
    if not isinstance(descriptors, (list, tuple)):
        raise ValueError("MMH3Director segment_loras must be a list or tuple")
    if len(descriptors) > MAX_SEGMENT_LORAS:
        raise ValueError(
            f"MMH3Director segment_loras exceeds the {MAX_SEGMENT_LORAS}-item limit"
        )

    patched = model
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"MMH3Director segment_loras[{index}] must be a mapping")
        path = str(descriptor.get("path") or "").strip()
        strength = _coerce_strength(descriptor.get("strength_model", 1.0))
        if not path:
            raise ValueError(f"MMH3Director segment_loras[{index}] requires path")
        if abs(strength) <= LORA_EPSILON:
            continue

        lora_state = _load_lora_state(path, state_cache=state_cache)
        if is_fsdp:
            from raylight.comfy_dist.sd import (
                load_lora_for_models as ray_load_lora_for_models,
                load_lora_for_models_quantized as ray_load_lora_for_models_quantized,
            )

            if is_quant or dynamic_sidecar:
                patched = ray_load_lora_for_models_quantized(
                    patched,
                    lora_state,
                    strength,
                    dynamic_sidecar=dynamic_sidecar,
                    defer_device_move=True,
                    fallback_to_patches=not is_quant,
                )
            else:
                patched = ray_load_lora_for_models(
                    patched,
                    lora_state,
                    strength,
                )
        else:
            import comfy.sd

            patched = comfy.sd.load_lora_for_models(
                patched,
                None,
                lora_state,
                strength,
                0.0,
            )[0]
    return patched


__all__ = [
    "SEGMENT_LORA_BRIDGE_MARKER",
    "SEGMENT_LORA_ORIGINAL_MARKER",
    "apply_worker_segment_loras",
    "make_segment_lora_wrapper",
    "resolve_segment_lora_descriptors",
]
