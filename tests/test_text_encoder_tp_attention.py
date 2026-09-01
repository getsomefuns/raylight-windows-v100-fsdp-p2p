import pytest
import torch

from raylight.text_encoder_parallel.attention_backends import (
    AttentionBackendUnavailable,
    TextEncoderAttentionBackend,
    resolve_attention_backend,
    run_text_encoder_attention,
)


def _cpu_gqa(sequence: int, *, dtype: torch.dtype = torch.float32):
    query = torch.randn(1, 8, sequence, 8, dtype=dtype)
    key = torch.randn(1, 1, sequence, 8, dtype=dtype)
    value = torch.randn(1, 1, sequence, 8, dtype=dtype)
    return query, key, value


def test_attention_backend_values_are_stable():
    assert [backend.value for backend in TextEncoderAttentionBackend] == [
        "AUTO",
        "XFORMERS_GQA",
        "TORCH_EFFICIENT",
        "MATH_DEBUG",
    ]


def test_auto_prefers_xformers_before_torch_efficient():
    attempts = []

    def probe(backend):
        attempts.append(backend)
        return True

    selected = resolve_attention_backend("AUTO", probe=probe)

    assert selected is TextEncoderAttentionBackend.XFORMERS_GQA
    assert attempts == [TextEncoderAttentionBackend.XFORMERS_GQA]


def test_auto_uses_torch_efficient_when_xformers_is_unavailable():
    attempts = []

    def probe(backend):
        attempts.append(backend)
        return backend is TextEncoderAttentionBackend.TORCH_EFFICIENT

    selected = resolve_attention_backend("auto", probe=probe)

    assert selected is TextEncoderAttentionBackend.TORCH_EFFICIENT
    assert attempts == [
        TextEncoderAttentionBackend.XFORMERS_GQA,
        TextEncoderAttentionBackend.TORCH_EFFICIENT,
    ]


def test_auto_never_resolves_to_math():
    with pytest.raises(AttentionBackendUnavailable, match="XFORMERS_GQA.*TORCH_EFFICIENT"):
        resolve_attention_backend("AUTO", probe=lambda _: False)


def test_forced_backend_fails_instead_of_falling_back():
    with pytest.raises(AttentionBackendUnavailable, match="XFORMERS_GQA"):
        resolve_attention_backend("XFORMERS_GQA", probe=lambda _: (False, "unsupported kernel"))


def test_invalid_backend_is_rejected():
    with pytest.raises(ValueError, match="UNKNOWN"):
        resolve_attention_backend("UNKNOWN", probe=lambda _: True)


def test_math_debug_rejects_long_sequence_before_allocating_attention_matrix():
    query, key, value = _cpu_gqa(sequence=2049)

    with pytest.raises(ValueError, match="2048"):
        run_text_encoder_attention(query, key, value, backend="MATH_DEBUG")

