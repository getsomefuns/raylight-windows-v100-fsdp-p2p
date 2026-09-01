import torch


def test_xformers_ulysses_backend_is_exposed_to_ray_initializer():
    import raylight.distributed_modules.attention as attention

    assert attention.XFORMERS_MEMORY_EFFICIENT in attention.xfuser_attention_choices()


def test_xformers_ulysses_releases_local_layout_between_synchronous_transfers(monkeypatch):
    import raylight.distributed_modules.attention as attention

    calls = []

    class FakeAllToAll:
        @staticmethod
        def apply(group, value, scatter_idx, gather_idx, use_sync):
            calls.append((group, tuple(value.shape), scatter_idx, gather_idx, use_sync))
            return value.clone()

    class FakeProcessGroups:
        ULYSSES_PG = object()
        RING_PG = object()

    seen = {}

    def fake_xformers(query, key, value, **kwargs):
        seen.update(kwargs)
        return query + key + value

    monkeypatch.setattr(attention, "SeqAllToAll4D", FakeAllToAll)
    monkeypatch.setattr(attention, "PROCESS_GROUP", FakeProcessGroups)
    monkeypatch.setattr(attention.dist, "get_world_size", lambda group: 1 if group is FakeProcessGroups.RING_PG else 2)
    monkeypatch.setattr(attention.xops, "memory_efficient_attention", fake_xformers)

    operation = attention.make_xfuser_attention(
        attention.XFORMERS_MEMORY_EFFICIENT,
        True,
    )
    query = torch.ones(1, 2, 4, 3)
    result = operation(
        query,
        query,
        query,
        heads=2,
        skip_reshape=True,
        scale=0.25,
    )

    assert result.shape == (1, 4, 6)
    assert seen == {"attn_bias": None, "p": 0.0, "scale": 0.25}
    assert [(call[2], call[3], call[4]) for call in calls] == [
        (2, 1, True),
        (2, 1, True),
        (2, 1, True),
        (1, 2, True),
    ]


def test_xformers_ulysses_rejects_ring_parallelism(monkeypatch):
    import raylight.distributed_modules.attention as attention

    class FakeProcessGroups:
        ULYSSES_PG = object()
        RING_PG = object()

    monkeypatch.setattr(attention, "PROCESS_GROUP", FakeProcessGroups)
    monkeypatch.setattr(attention.dist, "get_world_size", lambda _group: 2)
    operation = attention.make_xfuser_attention(
        attention.XFORMERS_MEMORY_EFFICIENT,
        True,
    )
    query = torch.ones(1, 2, 4, 3)

    try:
        operation(query, query, query, heads=2, skip_reshape=True)
    except ValueError as error:
        assert "ring degree 1" in str(error)
    else:
        raise AssertionError("expected ring degree validation")


def test_owned_xformers_ulysses_consumes_all_qkv_references(monkeypatch):
    import raylight.distributed_modules.attention as attention

    class FakeAllToAll:
        @staticmethod
        def apply(_group, value, _scatter_idx, _gather_idx, _use_sync):
            return value.clone()

    class FakeProcessGroups:
        ULYSSES_PG = object()
        RING_PG = object()

    monkeypatch.setattr(attention, "SeqAllToAll4D", FakeAllToAll)
    monkeypatch.setattr(attention, "PROCESS_GROUP", FakeProcessGroups)
    monkeypatch.setattr(attention.dist, "get_world_size", lambda _group: 1)
    monkeypatch.setattr(
        attention.xops,
        "memory_efficient_attention",
        lambda query, _key, _value, **_kwargs: query.clone(),
    )
    owned = [torch.ones(1, 2, 4, 3) for _ in range(3)]

    output = attention.xformers_ulysses_attention_owned(owned, use_sync=True)

    assert owned == []
    assert output.shape == (1, 4, 2, 3)
