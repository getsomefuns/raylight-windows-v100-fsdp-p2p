from contextlib import nullcontext

import pytest
import torch

import raylight.text_encoder_parallel.attention_backends as attention_module
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


class _FakeXformersOps:
    class LowerTriangularMask:
        pass

    def __init__(self):
        self.call = None
        self.calls = []

    def memory_efficient_attention(self, query, key, value, *, attn_bias):
        self.call = (query, key, value, attn_bias)
        self.calls.append(self.call)
        return torch.zeros_like(query)


def test_xformers_gqa_uses_expand_views_in_five_dimensions():
    query, key, value = _cpu_gqa(sequence=16)
    fake_ops = _FakeXformersOps()

    output = run_text_encoder_attention(
        query,
        key,
        value,
        backend="XFORMERS_GQA",
        xformers_ops=fake_ops,
    )

    shaped_query, shaped_key, shaped_value, bias = fake_ops.call
    assert shaped_query.shape == (1, 16, 1, 8, 8)
    assert shaped_key.shape == (1, 16, 1, 8, 8)
    assert shaped_value.shape == (1, 16, 1, 8, 8)
    assert shaped_key.stride(3) == 0
    assert shaped_value.stride(3) == 0
    assert isinstance(bias, _FakeXformersOps.LowerTriangularMask)
    assert output.shape == query.shape


def test_xformers_long_sequence_splits_independent_kv_groups(monkeypatch):
    query = torch.randn(1, 32, 10, 8)
    key = torch.randn(1, 4, 10, 8)
    value = torch.randn(1, 4, 10, 8)
    fake_ops = _FakeXformersOps()
    monkeypatch.setattr(attention_module, "XFORMERS_GROUP_CHUNK_SEQUENCE", 8)

    output = run_text_encoder_attention(
        query,
        key,
        value,
        backend="XFORMERS_GQA",
        xformers_ops=fake_ops,
    )

    assert len(fake_ops.calls) == 2
    assert [call[0].shape[2] for call in fake_ops.calls] == [2, 2]
    assert [call[1].shape[2] for call in fake_ops.calls] == [2, 2]
    assert all(isinstance(call[3], _FakeXformersOps.LowerTriangularMask) for call in fake_ops.calls)
    assert output.shape == query.shape
    assert output.data_ptr() == query.data_ptr()


def test_torch_efficient_physically_expands_kv_heads(monkeypatch):
    query, key, value = _cpu_gqa(sequence=16)
    captured = {}

    def fake_sdpa(q, k, v, *, is_causal):
        captured.update(query=q, key=k, value=v, is_causal=is_causal)
        return torch.zeros_like(q)

    monkeypatch.setattr(attention_module, "sdpa_kernel", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", fake_sdpa)

    output = run_text_encoder_attention(query, key, value, backend="TORCH_EFFICIENT")

    assert captured["query"].shape == captured["key"].shape == captured["value"].shape
    assert captured["key"].shape[1] == 8
    assert captured["key"].stride(1) != 0
    assert captured["value"].stride(1) != 0
    assert captured["is_causal"] is True
    assert output.shape == query.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel validation requires a GPU")
@pytest.mark.parametrize("backend", ["XFORMERS_GQA", "TORCH_EFFICIENT"])
@pytest.mark.parametrize("sequence", [64, 221, 256])
def test_v100_backends_match_causal_math_gqa(backend, sequence):
    torch.manual_seed(20260901 + sequence)
    device = torch.device("cuda:0")
    query = torch.randn(1, 32, sequence, 128, device=device, dtype=torch.float16) * 0.1
    key = torch.randn(1, 4, sequence, 128, device=device, dtype=torch.float16) * 0.1
    value = torch.randn(1, 4, sequence, 128, device=device, dtype=torch.float16) * 0.1

    expected = run_text_encoder_attention(query, key, value, backend="MATH_DEBUG")
    actual = run_text_encoder_attention(query, key, value, backend=backend)
    difference = (actual.float() - expected.float()).abs()

    assert bool(torch.isfinite(actual).all().item())
    # 1.220703125e-4 is the observed FP16 quantization step at this output scale.
    assert float(difference.max().item()) <= 1.25e-4
    assert float(difference.mean().item()) <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel validation requires a GPU")
@pytest.mark.parametrize("backend", ["XFORMERS_GQA", "TORCH_EFFICIENT"])
def test_v100_backends_do_not_see_future_values(backend):
    torch.manual_seed(17)
    query = torch.randn(1, 32, 64, 128, device="cuda:0", dtype=torch.float16) * 0.1
    key = torch.randn(1, 4, 64, 128, device="cuda:0", dtype=torch.float16) * 0.1
    value = torch.randn(1, 4, 64, 128, device="cuda:0", dtype=torch.float16) * 0.1
    changed_value = value.clone()
    changed_value[:, :, 32:, :] += 50

    original = run_text_encoder_attention(query, key, value, backend=backend)
    changed = run_text_encoder_attention(query, key, changed_value, backend=backend)

    torch.testing.assert_close(original[:, :, :32], changed[:, :, :32], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel validation requires a GPU")
@pytest.mark.parametrize("backend", ["XFORMERS_GQA", "TORCH_EFFICIENT"])
def test_v100_backends_match_float32_math_strictly(backend):
    torch.manual_seed(29)
    query = torch.randn(1, 32, 64, 128, device="cuda:0", dtype=torch.float32) * 0.1
    key = torch.randn(1, 4, 64, 128, device="cuda:0", dtype=torch.float32) * 0.1
    value = torch.randn(1, 4, 64, 128, device="cuda:0", dtype=torch.float32) * 0.1

    expected = run_text_encoder_attention(query, key, value, backend="MATH_DEBUG")
    actual = run_text_encoder_attention(query, key, value, backend=backend)
    difference = (actual - expected).abs()

    assert float(difference.max().item()) <= 1e-4
    assert float(difference.mean().item()) <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernel validation requires a GPU")
def test_v100_group_chunked_xformers_matches_float32_math(monkeypatch):
    torch.manual_seed(31)
    monkeypatch.setattr(attention_module, "XFORMERS_GROUP_CHUNK_SEQUENCE", 17)
    query = torch.randn(1, 32, 64, 128, device="cuda:0", dtype=torch.float32) * 0.1
    key = torch.randn(1, 4, 64, 128, device="cuda:0", dtype=torch.float32) * 0.1
    value = torch.randn(1, 4, 64, 128, device="cuda:0", dtype=torch.float32) * 0.1

    expected = run_text_encoder_attention(query, key, value, backend="MATH_DEBUG")
    actual = run_text_encoder_attention(query, key, value, backend="XFORMERS_GQA")
    difference = (actual - expected).abs()

    assert float(difference.max().item()) <= 1e-4
    assert float(difference.mean().item()) <= 1e-5
