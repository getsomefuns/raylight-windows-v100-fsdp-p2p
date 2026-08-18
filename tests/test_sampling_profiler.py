from contextlib import nullcontext

import pytest

from raylight.distributed_worker.sampling_profiler import run_with_optional_profile


class _FakeAverages:
    def table(self, **kwargs):
        assert kwargs == {"sort_by": "self_cuda_time_total", "row_limit": 40}
        return "CUDA PROFILE TABLE"


class _FakeProfile:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def key_averages(self):
        return _FakeAverages()


def test_sampling_profiler_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("RAYLIGHT_TORCH_PROFILE", raising=False)
    called = []

    result = run_with_optional_profile(
        lambda: "sampled",
        rank=0,
        invocation=1,
        profile_factory=lambda **_kwargs: called.append(True) or nullcontext(),
    )

    assert result == "sampled"
    assert called == []


def test_sampling_profiler_records_rank_zero_and_emits_cuda_table(monkeypatch):
    monkeypatch.setenv("RAYLIGHT_TORCH_PROFILE", "1")
    emitted = []
    factory_kwargs = []

    result = run_with_optional_profile(
        lambda: "sampled",
        rank=0,
        invocation=3,
        profile_factory=lambda **kwargs: factory_kwargs.append(kwargs) or _FakeProfile(),
        activities=("cpu", "cuda"),
        synchronize_fn=lambda: emitted.append("synchronized"),
        emit_fn=emitted.append,
    )

    assert result == "sampled"
    assert factory_kwargs == [
        {
            "activities": ("cpu", "cuda"),
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        }
    ]
    assert emitted == [
        "synchronized",
        "[RAYLIGHT_TORCH_PROFILE] rank=0 invocation=3 sort=self_cuda_time_total",
        "CUDA PROFILE TABLE",
    ]


def test_sampling_profiler_skips_nonzero_rank(monkeypatch):
    monkeypatch.setenv("RAYLIGHT_TORCH_PROFILE", "1")
    called = []

    result = run_with_optional_profile(
        lambda: "sampled",
        rank=1,
        invocation=1,
        profile_factory=lambda **_kwargs: called.append(True) or nullcontext(),
    )

    assert result == "sampled"
    assert called == []


def test_sampling_profiler_setup_failure_falls_back_before_sampling(monkeypatch):
    monkeypatch.setenv("RAYLIGHT_TORCH_PROFILE", "1")
    calls = []
    emitted = []

    result = run_with_optional_profile(
        lambda: calls.append("sample") or "sampled",
        rank=0,
        invocation=1,
        profile_factory=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("CUPTI unavailable")),
        emit_fn=emitted.append,
    )

    assert result == "sampled"
    assert calls == ["sample"]
    assert emitted == [
        "[RAYLIGHT_TORCH_PROFILE] rank=0 invocation=1 setup_failed=RuntimeError: CUPTI unavailable"
    ]


def test_sampling_profiler_never_retries_started_sampling(monkeypatch):
    monkeypatch.setenv("RAYLIGHT_TORCH_PROFILE", "1")
    calls = []

    def failing_sample():
        calls.append("sample")
        raise ValueError("sampling failed")

    with pytest.raises(ValueError, match="sampling failed"):
        run_with_optional_profile(
            failing_sample,
            rank=0,
            invocation=1,
            profile_factory=lambda **_kwargs: _FakeProfile(),
        )

    assert calls == ["sample"]


def test_sampling_profiler_propagates_cuda_synchronize_failure(monkeypatch):
    monkeypatch.setenv("RAYLIGHT_TORCH_PROFILE", "1")
    calls = []

    with pytest.raises(RuntimeError, match="asynchronous CUDA failure"):
        run_with_optional_profile(
            lambda: calls.append("sample") or "sampled",
            rank=0,
            invocation=1,
            profile_factory=lambda **_kwargs: _FakeProfile(),
            synchronize_fn=lambda: (_ for _ in ()).throw(
                RuntimeError("asynchronous CUDA failure")
            ),
        )

    assert calls == ["sample"]


def test_sampling_profiler_report_failure_keeps_valid_result(monkeypatch):
    monkeypatch.setenv("RAYLIGHT_TORCH_PROFILE", "1")
    emitted = []

    class BrokenAverages:
        def table(self, **_kwargs):
            raise RuntimeError("table failed")

    class BrokenReportProfile(_FakeProfile):
        def key_averages(self):
            return BrokenAverages()

    result = run_with_optional_profile(
        lambda: "sampled",
        rank=0,
        invocation=2,
        profile_factory=lambda **_kwargs: BrokenReportProfile(),
        synchronize_fn=lambda: None,
        emit_fn=emitted.append,
    )

    assert result == "sampled"
    assert emitted[-1] == (
        "[RAYLIGHT_TORCH_PROFILE] rank=0 invocation=2 report_failed=RuntimeError: table failed"
    )
