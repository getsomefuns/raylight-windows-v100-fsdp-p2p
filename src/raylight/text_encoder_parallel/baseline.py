"""Opt-in, read-only MiniMax H3 text-encoder benchmark records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager
from functools import wraps
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping


BASELINE_ENV = "RAYLIGHT_TEXT_ENCODER_BENCHMARK"
BASELINE_EVENT_PATH_ENV = "RAYLIGHT_TEXT_ENCODER_BENCHMARK_EVENT_PATH"
BASELINE_CHECKPOINT_ENV = "RAYLIGHT_TEXT_ENCODER_CHECKPOINT"
BASELINE_WORKFLOW_SHA256_ENV = "RAYLIGHT_TEXT_ENCODER_WORKFLOW_SHA256"
BASELINE_MODE_ENV = "RAYLIGHT_TEXT_ENCODER_TEMPERATURE"
BASELINE_RUN_INDEX_ENV = "RAYLIGHT_TEXT_ENCODER_RUN_INDEX"
REQUIRED_FIELDS = (
    "checkpoint",
    "token_count",
    "te_total_s",
    "gpu0_peak_mib",
    "ram_peak_mib",
    "pagefile_peak_mib",
)


def _required(mapping: Mapping[str, Any], name: str) -> Any:
    if name not in mapping:
        raise ValueError(f"missing required field: {name}")
    return mapping[name]


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return result


@dataclass(frozen=True)
class BaselineRecord:
    """Stable JSON/CSV row for cold/warm Text Encoder comparisons."""

    schema_version: int
    checkpoint: str
    workflow_sha256: str
    mode: str
    run_index: int
    token_count: int
    te_total_s: float
    gpu0_peak_mib: float
    ram_peak_mib: float
    pagefile_peak_mib: float

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "BaselineRecord":
        for name in REQUIRED_FIELDS:
            _required(mapping, name)
        schema_version = mapping.get("schema_version", 1)
        if schema_version != 1:
            raise ValueError("schema_version must be 1")
        checkpoint = _nonempty_string(mapping["checkpoint"], "checkpoint")
        workflow_sha256 = _nonempty_string(
            _required(mapping, "workflow_sha256"), "workflow_sha256"
        )
        if len(workflow_sha256) != 64 or any(c not in "0123456789abcdef" for c in workflow_sha256.lower()):
            raise ValueError("workflow_sha256 must be a 64-character hexadecimal digest")
        mode = _nonempty_string(_required(mapping, "mode"), "mode").lower()
        if mode not in {"cold", "warm"}:
            raise ValueError("mode must be cold or warm")
        run_index = _positive_int(_required(mapping, "run_index"), "run_index")
        return cls(
            schema_version=1,
            checkpoint=checkpoint,
            workflow_sha256=workflow_sha256.lower(),
            mode=mode,
            run_index=run_index,
            token_count=_positive_int(mapping["token_count"], "token_count"),
            te_total_s=_finite_float(mapping["te_total_s"], "te_total_s", positive=True),
            gpu0_peak_mib=_finite_float(mapping["gpu0_peak_mib"], "gpu0_peak_mib"),
            ram_peak_mib=_finite_float(mapping["ram_peak_mib"], "ram_peak_mib"),
            pagefile_peak_mib=_finite_float(mapping["pagefile_peak_mib"], "pagefile_peak_mib"),
        )

    @classmethod
    def from_json(cls, text: str) -> "BaselineRecord":
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("baseline JSON must contain an object")
        return cls.from_mapping(payload)

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_mapping(), ensure_ascii=False, sort_keys=True)


def stage_timing_enabled(environ: Mapping[str, str] | None = None) -> bool:
    value = (environ or os.environ).get(BASELINE_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def count_tokens(tokens: Any) -> int:
    """Count token positions without assuming a specific ComfyUI tokenizer wrapper."""

    if isinstance(tokens, Mapping):
        return sum(count_tokens(value) for value in tokens.values())
    if isinstance(tokens, (list, tuple)):
        if tokens and all(
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[1], (int, float))
            for item in tokens
        ):
            return len(tokens)
        return sum(count_tokens(value) for value in tokens)
    shape = getattr(tokens, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[-1])
    return 0


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")


@contextmanager
def timed_stage(stage: str):
    """Record one opt-in TP2 stage without changing normal execution behavior."""

    event_path = os.environ.get(BASELINE_EVENT_PATH_ENV, "").strip()
    enabled = stage_timing_enabled() and bool(event_path)
    started = time.perf_counter()
    begin_ns = time.time_ns()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        if enabled:
            append_event(
                Path(event_path),
                {
                    "event": "text_encoder_stage",
                    "stage": stage,
                    "elapsed_s": time.perf_counter() - started,
                    "begin_time_ns": begin_ns,
                    "time_ns": time.time_ns(),
                    "failed": failed,
                },
            )


def _synchronize_clip_device(clip: Any) -> None:
    patcher = getattr(clip, "patcher", None)
    device = getattr(patcher, "load_device", None)
    if device is None or getattr(device, "type", None) != "cuda":
        return
    import torch

    torch.cuda.synchronize(device)


def install_clip_timing_hook(clip_class: type[Any]) -> bool:
    """Patch one CLIP class only while the explicit benchmark flag is enabled."""

    if not stage_timing_enabled():
        return False
    if getattr(clip_class, "_raylight_text_encoder_baseline_hook", False):
        return False
    original = clip_class.encode_from_tokens_scheduled

    @wraps(original)
    def timed_encode(clip: Any, tokens: Any, *args: Any, **kwargs: Any) -> Any:
        event_path = os.environ.get(BASELINE_EVENT_PATH_ENV, "").strip()
        metadata = {
            "checkpoint": os.environ.get(BASELINE_CHECKPOINT_ENV, ""),
            "workflow_sha256": os.environ.get(BASELINE_WORKFLOW_SHA256_ENV, ""),
            "mode": os.environ.get(BASELINE_MODE_ENV, ""),
            "run_index": os.environ.get(BASELINE_RUN_INDEX_ENV, ""),
            "token_count": count_tokens(tokens),
        }
        _synchronize_clip_device(clip)
        begin_time_ns = time.time_ns()
        started = time.perf_counter()
        output = original(clip, tokens, *args, **kwargs)
        _synchronize_clip_device(clip)
        elapsed = time.perf_counter() - started
        complete_time_ns = time.time_ns()
        if event_path:
            path = Path(event_path)
            append_event(
                path,
                {"event": "text_encoder_begin", **metadata, "time_ns": begin_time_ns},
            )
            append_event(
                path,
                {
                    "event": "text_encoder_complete",
                    **metadata,
                    "te_total_s": elapsed,
                    "time_ns": complete_time_ns,
                },
            )
        return output

    clip_class.encode_from_tokens_scheduled = timed_encode
    clip_class._raylight_text_encoder_baseline_hook = True
    return True


def install_comfy_clip_timing_from_environment() -> bool:
    """Install stock and TP2 CLIP hooks; normal launches remain untouched."""

    if not stage_timing_enabled():
        return False
    from comfy.sd import CLIP
    from .clip_proxy import MiniMaxH3TP2ClipProxy

    stock_installed = install_clip_timing_hook(CLIP)
    tp2_installed = install_clip_timing_hook(MiniMaxH3TP2ClipProxy)
    return stock_installed or tp2_installed
