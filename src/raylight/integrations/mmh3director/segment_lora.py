from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .proxy import RayMMH3DirectorModelProxy


MAX_SEGMENT_LORAS = 8
MIN_LORA_STRENGTH = -10.0
MAX_LORA_STRENGTH = 10.0
LORA_EPSILON = 1e-6
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


__all__ = [
    "SEGMENT_LORA_BRIDGE_MARKER",
    "SEGMENT_LORA_ORIGINAL_MARKER",
    "make_segment_lora_wrapper",
    "resolve_segment_lora_descriptors",
]
