from dataclasses import dataclass

import pytest

from raylight.text_encoder_parallel.collectives import CudaPeerGroup2


@dataclass(frozen=True)
class FakeDevice:
    type: str
    index: int


class FakeTensor:
    def __init__(
        self,
        values,
        *,
        device: int,
        dtype: str = "float16",
        contiguous: bool = True,
        shape=None,
    ):
        self.values = tuple(values)
        self.device = FakeDevice("cuda", device)
        self.dtype = dtype
        self.shape = tuple(shape or (len(self.values),))
        self.is_cuda = True
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous

    def numel(self):
        return len(self.values)

    def element_size(self):
        return 2 if self.dtype == "float16" else 4


class FakeBackend:
    def __init__(self):
        self.validate_calls = 0
        self.allocate_calls = 0
        self.close_calls = 0

    def validate(self, devices):
        self.validate_calls += 1
        assert devices == (0, 1)

    def allocate_buffers(self, rank0, rank1):
        self.allocate_calls += 1
        return object()

    def all_reduce_sum(self, rank0, rank1, buffers):
        assert buffers is not None
        result = tuple(a + b for a, b in zip(rank0.values, rank1.values))
        return result, result, 1.25

    def close(self):
        self.close_calls += 1


def _pair(**kwargs):
    return (
        FakeTensor((1, 2, 3), device=0, **kwargs),
        FakeTensor((10, 20, 30), device=1, **kwargs),
    )


def test_all_reduce_sum_validates_once_and_reuses_buffers() -> None:
    backend = FakeBackend()
    group = CudaPeerGroup2(backend=backend)
    rank0, rank1 = _pair()

    first = group.all_reduce_sum(rank0, rank1, operation_id=1)
    second = group.all_reduce_sum(rank0, rank1, operation_id=2)

    assert first[0] == (11, 22, 33)
    assert first[1] == first[0]
    assert second[0] == first[0]
    assert first[2].bytes_sent == 12
    assert first[2].operation_id == 1
    assert backend.validate_calls == 1
    assert backend.allocate_calls == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda a, b: (FakeTensor(a.values, device=1), b), "device 0"),
        (lambda a, b: (a, FakeTensor(b.values, device=0)), "device 1"),
        (lambda a, b: (FakeTensor(a.values, device=0, dtype="float32"), b), "dtype"),
        (lambda a, b: (FakeTensor((1, 2), device=0), b), "shape"),
        (lambda a, b: (FakeTensor(a.values, device=0, contiguous=False), b), "contiguous"),
    ],
)
def test_rejects_invalid_tensor_pair(mutate, message: str) -> None:
    group = CudaPeerGroup2(backend=FakeBackend())
    rank0, rank1 = mutate(*_pair())

    with pytest.raises(ValueError, match=message):
        group.all_reduce_sum(rank0, rank1, operation_id=1)


@pytest.mark.parametrize("operation_id", [0, -1, 1.5, True])
def test_rejects_invalid_operation_id(operation_id) -> None:
    group = CudaPeerGroup2(backend=FakeBackend())

    with pytest.raises(ValueError, match="operation_id"):
        group.all_reduce_sum(*_pair(), operation_id=operation_id)


def test_rejects_replayed_or_out_of_order_operation_id() -> None:
    group = CudaPeerGroup2(backend=FakeBackend())
    group.all_reduce_sum(*_pair(), operation_id=2)

    with pytest.raises(ValueError, match="strictly increase"):
        group.all_reduce_sum(*_pair(), operation_id=2)
    with pytest.raises(ValueError, match="strictly increase"):
        group.all_reduce_sum(*_pair(), operation_id=1)


def test_profile_snapshot_aggregates_calls_bytes_and_slowest_operation() -> None:
    group = CudaPeerGroup2(backend=FakeBackend())
    group.all_reduce_sum(*_pair(), operation_id=1)
    group.all_reduce_sum(*_pair(dtype="float32"), operation_id=2)

    snapshot = group.profile_snapshot()

    assert snapshot == {
        "calls": 2,
        "total_bytes_sent": 36,
        "total_elapsed_ms": 2.5,
        "slowest_elapsed_ms": 1.25,
        "slowest_operation_id": 1,
    }


def test_close_releases_backend_and_refuses_future_calls() -> None:
    backend = FakeBackend()
    group = CudaPeerGroup2(backend=backend)
    group.close()

    assert backend.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        group.all_reduce_sum(*_pair(), operation_id=1)
