from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable


RAYLIGHT_TE_SPEED_CONFIG_KEY = "raylight_external_te_speed"
RAYLIGHT_SOLATTN_CONFIG_KEY = "raylight_external_solattn"

_TE_RUNTIME_KEY = "te_speed_minimax_h3_runtime"
_SOL_OVERRIDE_KEY = "optimized_attention_override"
_SOL_OPTION_KEYS = (
    "solattn_sparse",
    "solattn_tau",
    "solattn_start_percent",
    "solattn_end_percent",
    "solattn_min_tokens",
    "solattn_dense_blocks",
    "solattn_prefix_tokens",
    "solattn_topk",
)


def _te_mode(runtime: object) -> str:
    variant = getattr(runtime, "turbo_variant", None)
    if variant == "four":
        return "4-step LoRA"
    if variant == "eight":
        return "8-step LoRA"
    return "standard"


def _dense_blocks_spec(value: object) -> str:
    if value is None:
        return ""
    values = sorted({int(index) for index in value})
    if not values:
        return ""
    groups: list[str] = []
    start = previous = values[0]
    for index in values[1:]:
        if index == previous + 1:
            previous = index
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = index
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def serialize_external_acceleration(model_options: dict[str, Any]) -> dict[str, Any]:
    """Convert third-party in-process patch objects into Ray-safe plain values."""

    serialized = copy.deepcopy(model_options)
    source_transformer = model_options.get("transformer_options", {})
    transformer = serialized.setdefault("transformer_options", {})

    runtime = source_transformer.get(_TE_RUNTIME_KEY)
    if runtime is not None:
        transformer.pop(_TE_RUNTIME_KEY, None)
        transformer[RAYLIGHT_TE_SPEED_CONFIG_KEY] = {
            "processing_control_value": float(runtime.threshold),
            "processing_percent_1": float(runtime.start_percent),
            "processing_percent_2": float(runtime.end_percent),
            "mcs": int(runtime.max_consecutive_skips),
            "device": str(runtime.cache_device),
            "mode": _te_mode(runtime),
        }

    override = source_transformer.get(_SOL_OVERRIDE_KEY)
    if override is not None and getattr(override, "_solattn_override", False):
        transformer.pop(_SOL_OVERRIDE_KEY, None)
        transformer[RAYLIGHT_SOLATTN_CONFIG_KEY] = {
            # Raylight already supplies its tested MiniMax H3 safe-FP16 path.
            "fp16_safe": False,
            "tau": float(source_transformer.get("solattn_tau", 0.75)),
            "start_percent": float(
                source_transformer.get("solattn_start_percent", 0.2)
            ),
            "end_percent": float(source_transformer.get("solattn_end_percent", 0.9)),
            "min_tokens": int(source_transformer.get("solattn_min_tokens", 1024)),
            "dense_blocks": _dense_blocks_spec(
                source_transformer.get("solattn_dense_blocks", ())
            ),
            "h3_prefix_tokens": int(
                source_transformer.get("solattn_prefix_tokens", 1024)
            ),
            "topk_blocks": int(source_transformer.get("solattn_topk", 32)),
        }
        for key in _SOL_OPTION_KEYS:
            transformer.pop(key, None)

    return serialized


def _custom_node_roots() -> list[Path]:
    import folder_paths

    return [Path(path) for path in folder_paths.get_folder_paths("custom_nodes")]


def _find_node_package(fragment: str) -> Path:
    lowered = fragment.lower()
    for root in _custom_node_roots():
        if not root.is_dir():
            continue
        for candidate in root.iterdir():
            if candidate.is_dir() and lowered in candidate.name.lower():
                init_file = candidate / "__init__.py"
                if init_file.is_file():
                    return candidate
    raise RuntimeError(
        f"Raylight could not find the required custom node package containing '{fragment}'"
    )


def _load_package(module_name: str, package_path: Path):
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_path / "__init__.py",
        submodule_search_locations=[str(package_path)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load custom node package at {package_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _te_speed_factory():
    package = _load_package(
        "_raylight_external_te_speed",
        _find_node_package("TE-Speed-MiniMaxH3"),
    )
    return package.NODE_CLASS_MAPPINGS["TESpeedMiniMaxH3"]()


def _solattn_factory():
    package = _load_package(
        "_raylight_external_solattn",
        _find_node_package("SolAttn-V100"),
    )
    return package.NODE_CLASS_MAPPINGS["SolAttnV100"]()


class _TESpeedExactH3Adapter:
    """Expose an exact MiniMaxH3Model view without replacing the FSDP module."""

    def __init__(self, model, diffusion_view):
        self._model = model
        self._diffusion_view = diffusion_view

    def get_model_object(self, name: str):
        if name == "diffusion_model":
            return self._diffusion_view
        return self._model.get_model_object(name)

    def __getattr__(self, name: str):
        return getattr(self._model, name)


def _te_speed_model_argument(model):
    """Bridge TE-Speed's exact-type guard to PyTorch's dynamic FSDP subclass."""
    from comfy.ldm.minimax.model import MiniMaxH3Model

    get_model_object = getattr(model, "get_model_object", None)
    if not callable(get_model_object):
        return model
    diffusion = get_model_object("diffusion_model")
    if type(diffusion) is MiniMaxH3Model:
        return model
    if not isinstance(diffusion, MiniMaxH3Model):
        return model

    # Composable FSDP changes ``__class__`` in-place to a generated subclass.
    # TE-Speed 3.3 accepts only the exact base type.  This weightless view shares
    # the live module dictionary, so its blocks are still the real FSDP blocks.
    view = object.__new__(MiniMaxH3Model)
    view.__dict__ = diffusion.__dict__
    return _TESpeedExactH3Adapter(model, view)


def apply_external_acceleration(
    model,
    *,
    te_speed_factory: Callable[[], object] | None = None,
    solattn_factory: Callable[[], object] | None = None,
):
    """Replay requested third-party MODEL patches inside each Ray worker."""

    transformer = model.model_options.setdefault("transformer_options", {})
    te_config = transformer.pop(RAYLIGHT_TE_SPEED_CONFIG_KEY, None)
    sol_config = transformer.pop(RAYLIGHT_SOLATTN_CONFIG_KEY, None)

    patched = model
    if te_config is not None:
        factory = te_speed_factory or _te_speed_factory
        te_argument = _te_speed_model_argument(patched)
        te_result = factory().patch(te_argument, **te_config)[0]
        patched = patched if te_result is te_argument else te_result
    if sol_config is not None:
        factory = solattn_factory or _solattn_factory
        patched = factory().patch(patched, **sol_config)[0]
    return patched
