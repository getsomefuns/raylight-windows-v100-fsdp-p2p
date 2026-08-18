from types import SimpleNamespace

import torch

from raylight.distributed_modules.attention import single_ring_ulysses_attention


class _RecordingAllToAll:
    calls = []

    @classmethod
    def apply(cls, group, tensor, scatter_idx, gather_idx, use_sync=False):
        cls.calls.append((group, tensor.clone(), scatter_idx, gather_idx, use_sync))
        return tensor + len(cls.calls)


def test_single_ring_attention_discards_unused_lse_and_reverses_ulysses_layout():
    _RecordingAllToAll.calls = []
    seen = {}

    def kernel(query, key, value, **kwargs):
        seen["query"] = query
        seen["key"] = key
        seen["value"] = value
        seen["kwargs"] = kwargs
        output = query + key + value
        lse = torch.full((1,), 123.0, dtype=torch.float32)
        return output, lse

    query = torch.ones((1, 2, 2, 1))
    key = query * 10
    value = query * 100
    result = single_ring_ulysses_attention(
        query,
        key,
        value,
        ulysses_group="ulysses",
        attention_kernel=kernel,
        all_to_all=_RecordingAllToAll,
        softmax_scale=0.25,
        use_sync=True,
    )

    assert [(call[2], call[3]) for call in _RecordingAllToAll.calls] == [
        (2, 1),
        (2, 1),
        (2, 1),
        (1, 2),
    ]
    assert all(call[4] is True for call in _RecordingAllToAll.calls)
    torch.testing.assert_close(seen["query"], query + 1)
    torch.testing.assert_close(seen["key"], key + 2)
    torch.testing.assert_close(seen["value"], value + 3)
    assert seen["kwargs"]["causal"] is False
    assert seen["kwargs"]["softmax_scale"] == 0.25
    torch.testing.assert_close(result, (query + 1) + (key + 2) + (value + 3) + 4)


def test_single_ring_attention_accepts_tensor_only_kernel_result_and_restores_dtype():
    _RecordingAllToAll.calls = []
    query = torch.ones((1, 1, 1, 1), dtype=torch.float16)

    result = single_ring_ulysses_attention(
        query,
        query,
        query,
        ulysses_group=SimpleNamespace(name="ulysses"),
        attention_kernel=lambda q, _k, _v, **_kwargs: q.float(),
        all_to_all=_RecordingAllToAll,
    )

    assert result.dtype is query.dtype
    torch.testing.assert_close(result, query + 5)


class _FakeXFuserAttention:
    def __init__(self, use_sync, attn_type):
        self.use_sync = use_sync
        self.attn_type = attn_type
        self.ring_pg = "ring"
        self.ulysses_pg = "ulysses"
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return args[1]


def _make_attention(monkeypatch, ring_world_size):
    import raylight.distributed_modules.attention as attention

    instances = []

    def construct(*args, **kwargs):
        instance = _FakeXFuserAttention(*args, **kwargs)
        instances.append(instance)
        return instance

    monkeypatch.setattr(attention, "xFuserLongContextAttention", construct)
    monkeypatch.setattr(attention.dist, "get_world_size", lambda _group: ring_world_size)
    monkeypatch.setattr(
        attention,
        "select_flash_attn_impl",
        lambda *_args, **_kwargs: lambda q, _k, _v, **_kw: (q, torch.zeros(1)),
    )
    return attention, attention.make_xfuser_attention("TORCH_EFFICIENT", True), instances


def test_factory_routes_single_ring_to_direct_path(monkeypatch):
    attention, forward, instances = _make_attention(monkeypatch, ring_world_size=1)
    direct_calls = []
    monkeypatch.setattr(
        attention,
        "single_ring_ulysses_attention",
        lambda query, *_args, **kwargs: direct_calls.append((query, kwargs)) or query,
    )
    q = torch.ones((1, 2, 3, 4))

    result = forward(
        q,
        q,
        q,
        heads=2,
        skip_reshape=True,
        skip_output_reshape=True,
    )

    assert result.shape == q.shape
    assert len(direct_calls) == 1
    assert direct_calls[0][1]["use_sync"] is True
    assert instances[0].calls == []


def test_factory_keeps_multi_ring_on_xfuser(monkeypatch):
    _attention, forward, instances = _make_attention(monkeypatch, ring_world_size=2)
    q = torch.ones((1, 2, 3, 4))

    forward(q, q, q, heads=2, skip_reshape=True)

    assert len(instances[0].calls) == 1


def test_factory_keeps_join_attention_on_xfuser(monkeypatch):
    _attention, forward, instances = _make_attention(monkeypatch, ring_world_size=1)
    q = torch.ones((1, 2, 3, 4))

    forward(
        q,
        q,
        q,
        heads=2,
        join_q=q,
        join_k=q,
        join_v=q,
        skip_reshape=True,
    )

    assert len(instances[0].calls) == 1
