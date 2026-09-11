from raylight.distributed_worker import utils


class _FakeRayBar:
    def __init__(self, iterable, **_kwargs):
        self._iterable = iterable
        self.closed = False
        self.close_calls = 0

    def __iter__(self):
        yield from self._iterable

    def close(self):
        self.close_calls += 1
        self.closed = True


def test_iterating_ray_tqdm_compat_closes_remote_bar(monkeypatch):
    created = []

    def fake_ray_tqdm(iterable, **kwargs):
        bar = _FakeRayBar(iterable, **kwargs)
        created.append(bar)
        return bar

    monkeypatch.setattr(utils, "ray_tqdm", fake_ray_tqdm)

    compat = utils._RayTqdmCompat(range(3))
    assert list(compat) == [0, 1, 2]
    assert len(created) == 1
    assert created[0].closed is True

    compat.close()
    assert created[0].close_calls == 1
