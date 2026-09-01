"""Attention backend policy for the MiniMax H3 TP2 text encoder."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F


class TextEncoderAttentionBackend(str, Enum):
    """Stable user-facing names for text-encoder attention implementations."""

    AUTO = "AUTO"
    XFORMERS_GQA = "XFORMERS_GQA"
    TORCH_EFFICIENT = "TORCH_EFFICIENT"
    MATH_DEBUG = "MATH_DEBUG"


class AttentionBackendUnavailable(RuntimeError):
    """Raised when a requested production attention backend cannot run."""


ProbeResult = bool | tuple[bool, str]
BackendProbe = Callable[[TextEncoderAttentionBackend], ProbeResult]


def _parse_backend(requested: str | TextEncoderAttentionBackend) -> TextEncoderAttentionBackend:
    if isinstance(requested, TextEncoderAttentionBackend):
        return requested
    normalized = str(requested).strip().upper()
    try:
        return TextEncoderAttentionBackend(normalized)
    except ValueError as exc:
        choices = ", ".join(backend.value for backend in TextEncoderAttentionBackend)
        raise ValueError(f"Unknown text encoder attention backend {requested!r}; expected one of: {choices}") from exc


def _normalize_probe_result(result: ProbeResult) -> tuple[bool, str]:
    if isinstance(result, tuple):
        available, reason = result
        return bool(available), str(reason)
    return bool(result), "available" if result else "unavailable"


def _default_probe(
    backend: TextEncoderAttentionBackend,
    *,
    probe_device: torch.device | str | None,
) -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    device = torch.device(probe_device) if probe_device is not None else torch.device("cuda", torch.cuda.current_device())
    try:
        query = torch.randn(1, 8, 16, 32, device=device, dtype=torch.float16)
        key = torch.randn(1, 1, 16, 32, device=device, dtype=torch.float16)
        value = torch.randn(1, 1, 16, 32, device=device, dtype=torch.float16)
        output = run_text_encoder_attention(query, key, value, backend=backend, is_causal=True)
        if not bool(torch.isfinite(output).all().item()):
            return False, "probe output contains NaN or Inf"
        return True, "probe succeeded"
    except Exception as exc:  # the error text is surfaced in the strict selection failure
        return False, f"{type(exc).__name__}: {exc}"


def resolve_attention_backend(
    requested: str | TextEncoderAttentionBackend,
    *,
    probe_device: torch.device | str | None = None,
    probe: BackendProbe | None = None,
) -> TextEncoderAttentionBackend:
    """Resolve one backend without silently falling back to quadratic Math SDPA."""

    selected = _parse_backend(requested)
    if selected is TextEncoderAttentionBackend.MATH_DEBUG:
        return selected

    candidates = (
        (TextEncoderAttentionBackend.XFORMERS_GQA, TextEncoderAttentionBackend.TORCH_EFFICIENT)
        if selected is TextEncoderAttentionBackend.AUTO
        else (selected,)
    )
    probe_backend = probe or (lambda backend: _default_probe(backend, probe_device=probe_device))
    rejection_reasons: list[str] = []
    for candidate in candidates:
        available, reason = _normalize_probe_result(probe_backend(candidate))
        if available:
            return candidate
        rejection_reasons.append(f"{candidate.value}: {reason}")

    details = "; ".join(rejection_reasons)
    raise AttentionBackendUnavailable(f"No requested Text Encoder attention backend is available ({details})")


def _validate_gqa_tensors(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use [batch, heads, sequence, head_dim]")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError("query, key, and value must share batch, sequence, and head_dim")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query head count must be divisible by key/value head count")


def run_text_encoder_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    backend: str | TextEncoderAttentionBackend,
    is_causal: bool = True,
    xformers_ops: Any | None = None,
) -> torch.Tensor:
    """Run the selected backend using TP2-local GQA tensors."""

    del xformers_ops  # reserved for the dependency-injected xFormers implementation
    selected = _parse_backend(backend)
    if selected is TextEncoderAttentionBackend.AUTO:
        selected = resolve_attention_backend(selected, probe_device=query.device)

    _validate_gqa_tensors(query, key, value)
    sequence = query.shape[2]
    if selected is TextEncoderAttentionBackend.MATH_DEBUG:
        if sequence > 2048:
            raise ValueError("MATH_DEBUG is limited to 2048 tokens to prevent quadratic attention allocation")
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=is_causal,
            enable_gqa=query.shape[1] != key.shape[1],
        )

    raise AttentionBackendUnavailable(f"{selected.value} is not implemented yet")


__all__ = [
    "AttentionBackendUnavailable",
    "TextEncoderAttentionBackend",
    "resolve_attention_backend",
    "run_text_encoder_attention",
]
