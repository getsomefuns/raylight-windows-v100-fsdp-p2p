from __future__ import annotations

from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from raylight.distributed_worker.ray_worker_vae import (
    BackgroundVaePreloader,
    build_mmh3_temporal_plan,
    decode_mmh3_temporal_chunks,
    merge_mmh3_temporal_chunks,
)


class FakeMiniMaxVAE(torch.nn.Module):
    tokens_chunk_size = 5
    token_overlap = 2
    token_drop = 3
    vae_ratio_t = 4
    frame_pre_padding = 3
    frame_overlap = 5
    decoder = SimpleNamespace(out_channels=3)

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("latents_mean", torch.zeros(24))
        self.register_buffer("latents_std", torch.ones(24))
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1, 1))
        self.decoded_starts: list[int] = []
        self.inference_modes: list[bool] = []

    def _decode_temporal_chunks(self, z_len: int) -> tuple[int, int]:
        pseudo_total_tokens = z_len + self.token_drop
        pad_tokens = (-pseudo_total_tokens) % self.tokens_chunk_size
        pseudo_total_tokens += pad_tokens
        return pad_tokens, pseudo_total_tokens // self.tokens_chunk_size - 1

    def _decode_temporal_frame_plan(self, z_len: int, num_chunks: int, pad_tokens: int) -> int:
        del z_len, pad_tokens
        return num_chunks * 17 + 5

    def decode_output_shape(self, input_shape) -> tuple[int, int, int, int, int]:
        b, _c, t, h, w = input_shape
        pad, chunks = self._decode_temporal_chunks(t)
        return (
            b,
            self.decoder.out_channels,
            self._decode_temporal_frame_plan(t + pad, chunks, pad),
            h,
            w,
        )

    def _adaptive_decode(self, clip: torch.Tensor) -> torch.Tensor:
        self.inference_modes.append(torch.is_inference_mode_enabled())
        start = int(clip[0, 0, 0, 0, 0].item())
        self.decoded_starts.append(start)
        return torch.full((clip.shape[0], 3, 28, 1, 1), float(start), dtype=clip.dtype)

    def _finalize_pixels(self, part: torch.Tensor) -> torch.Tensor:
        return part.float().clamp(0.0, 1.0)

    def blend(self, a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        positions = torch.arange(extent, dtype=b.dtype)
        shape = [1] * b.ndim
        shape[dim] = extent
        wb = (positions / extent).view(shape)
        wa = 1 - wb
        prefix = a.narrow(dim, a.shape[dim] - extent, extent) * wa + b.narrow(dim, 0, extent) * wb
        return torch.cat((prefix, b.narrow(dim, extent, b.shape[dim] - extent)), dim=dim)


def test_temporal_plan_matches_official_107_and_73_frame_boundaries() -> None:
    model = FakeMiniMaxVAE()

    plan_107 = build_mmh3_temporal_plan(model, 32)
    plan_73 = build_mmh3_temporal_plan(model, 22)

    assert plan_107["pad_tokens"] == 0
    assert plan_107["num_chunks"] == 6
    assert plan_107["output_frames"] == 107
    assert plan_107["chunks"][0] == {"index": 0, "start": 0, "end": 7}
    assert plan_107["chunks"][-1] == {"index": 5, "start": 25, "end": 32}
    assert plan_73["num_chunks"] == 4
    assert plan_73["output_frames"] == 73


def test_selected_chunk_decode_only_runs_requested_temporal_chunk() -> None:
    import comfy.model_management as model_management

    model = FakeMiniMaxVAE()
    vae = SimpleNamespace(
        first_stage_model=model,
        vae_dtype=torch.float32,
        device=torch.device("cpu"),
        patcher=object(),
        disable_offload=True,
        memory_used_decode=lambda shape, dtype: 1,
    )
    latent = torch.arange(12, dtype=torch.float32).view(1, 1, 12, 1, 1).expand(1, 24, 12, 1, 1).clone()

    with patch.object(model_management, "load_models_gpu", lambda *args, **kwargs: None), patch.object(
        model_management, "cuda_device_context", lambda device: nullcontext()
    ):
        result = decode_mmh3_temporal_chunks(vae, latent, [1])

    assert model.decoded_starts == [5]
    assert model.inference_modes == [True]
    assert result["input_shape"] == (1, 24, 12, 1, 1)
    assert result["chunks"][0]["chunk_index"] == 1
    assert result["chunks"][0]["main"].shape == (1, 3, 17, 1, 1)
    assert result["chunks"][0]["overlap"].shape == (1, 3, 5, 1, 1)


def _chunk(index: int, main: float, overlap: float) -> dict:
    return {
        "chunk_index": index,
        "main": torch.full((1, 1, 17, 1, 1), main, dtype=torch.float32),
        "overlap": torch.full((1, 1, 5, 1, 1), overlap, dtype=torch.float32),
    }


def test_merge_accepts_out_of_order_chunks_and_reproduces_overlap_weights() -> None:
    model = FakeMiniMaxVAE()
    model.decoder = SimpleNamespace(out_channels=1)
    merged = merge_mmh3_temporal_chunks(
        model,
        (1, 24, 12, 1, 1),
        [_chunk(1, 20.0, 30.0), _chunk(0, 0.0, 10.0)],
        finalize=False,
    )

    assert merged.shape == (1, 1, 39, 1, 1)
    assert torch.equal(merged[0, 0, :17, 0, 0], torch.zeros(17))
    assert torch.equal(merged[0, 0, 17:22, 0, 0], torch.tensor([10.0, 12.0, 14.0, 16.0, 18.0]))
    assert torch.equal(merged[0, 0, 22:34, 0, 0], torch.full((12,), 20.0))
    assert torch.equal(merged[0, 0, 34:, 0, 0], torch.full((5,), 30.0))


@pytest.mark.parametrize(
    ("chunks", "message"),
    [
        ([_chunk(0, 0.0, 1.0)], "missing"),
        ([_chunk(0, 0.0, 1.0), _chunk(0, 0.0, 1.0)], "duplicate"),
    ],
)
def test_merge_rejects_missing_and_duplicate_chunks(chunks: list[dict], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        merge_mmh3_temporal_chunks(FakeMiniMaxVAE(), (1, 24, 12, 1, 1), chunks)


def test_merge_rejects_non_finite_chunk_output() -> None:
    chunks = [_chunk(0, 0.0, 1.0), _chunk(1, 2.0, 3.0)]
    chunks[1]["main"][0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        merge_mmh3_temporal_chunks(FakeMiniMaxVAE(), (1, 24, 12, 1, 1), chunks)


def test_background_preloader_returns_immediately_and_deduplicates_same_path() -> None:
    entered = Event()
    release = Event()
    calls = []

    def loader(path: str):
        calls.append(path)
        entered.set()
        assert release.wait(2)
        return {"path": path}

    cache = BackgroundVaePreloader(loader)
    assert cache.start("a.safetensors") == "started"
    assert entered.wait(2)
    assert cache.start("a.safetensors") == "loading"
    assert calls == ["a.safetensors"]
    release.set()
    assert cache.wait("a.safetensors") == {"path": "a.safetensors"}
    assert cache.start("a.safetensors") == "ready"
    assert calls == ["a.safetensors"]


def test_background_preloader_replaces_path_and_clear_drops_cached_object() -> None:
    cache = BackgroundVaePreloader(lambda path: {"path": path})
    cache.start("a.safetensors")
    first = cache.wait("a.safetensors")
    cache.start("b.safetensors")
    second = cache.wait("b.safetensors")

    assert first == {"path": "a.safetensors"}
    assert second == {"path": "b.safetensors"}
    assert second is not first
    cache.clear()
    assert cache.snapshot() == {"path": None, "state": "empty"}


def test_background_preloader_surfaces_loader_error_at_wait_boundary() -> None:
    def fail(_path: str):
        raise RuntimeError("broken VAE")

    cache = BackgroundVaePreloader(fail)
    cache.start("broken.safetensors")
    with pytest.raises(RuntimeError, match="broken VAE"):
        cache.wait("broken.safetensors")
    assert cache.snapshot()["state"] == "failed"


def test_ray_worker_preload_and_decode_delegate_to_shared_helpers() -> None:
    from raylight.distributed_worker.ray_worker import RayWorker

    cache = SimpleNamespace(
        start=lambda path: f"started:{path}",
        wait=lambda path: f"vae:{path}",
        clear=lambda: None,
    )
    worker = object.__new__(RayWorker)
    worker._vae_preloader = cache
    worker.vae_model = None
    worker._cached_vae_path = None
    worker.local_rank = 1
    worker.device = torch.device("cpu")

    assert worker.start_mmh3_vae_preload("video.safetensors") == "started:video.safetensors"
    with patch(
        "raylight.distributed_worker.ray_worker.decode_mmh3_temporal_chunks",
        return_value={"chunks": []},
    ) as decode:
        result = worker.ray_mmh3_vae_decode_chunks(
            torch.zeros(1, 24, 12, 1, 1),
            [1],
            "video.safetensors",
        )

    assert result == {"chunks": []}
    assert worker.vae_model == "vae:video.safetensors"
    decode.assert_called_once()


class _RemoteCall:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        return self.result


def _proxy_without_init(worker):
    from raylight.integrations.mmh3director.proxy import RayMMH3DirectorModelProxy

    proxy = object.__new__(RayMMH3DirectorModelProxy)
    proxy.ray_actors = {"workers": [object(), worker], "world_size": 2}
    return proxy


def _source_vae(path="video.safetensors"):
    return SimpleNamespace(
        patcher=SimpleNamespace(cached_patcher_init=(object(), (path, {"meta": 1}, None)))
    )


def test_extract_vae_source_path_uses_comfy_reload_factory() -> None:
    from raylight.integrations.mmh3director.vae_decode import extract_vae_source_path

    assert extract_vae_source_path(_source_vae()) == "video.safetensors"
    assert extract_vae_source_path(SimpleNamespace(patcher=SimpleNamespace())) is None


def test_director_execute_wrapper_preloads_rank1_and_clears_context_on_error() -> None:
    from raylight.integrations.mmh3director import vae_decode

    preload = _RemoteCall("started")
    clear = _RemoteCall(True)
    worker = SimpleNamespace(
        start_mmh3_vae_preload=preload,
        clear_sampling_vram=clear,
    )
    proxy = _proxy_without_init(worker)
    seen = []

    def original(_self, model, video_vae):
        seen.append(vae_decode.current_mmh3_vae_context())
        assert model is proxy
        assert video_vae is not None
        raise RuntimeError("director stopped")

    wrapped = vae_decode.make_director_execute_wrapper(original)
    with patch.object(vae_decode.ray, "get", side_effect=lambda value: value):
        with pytest.raises(RuntimeError, match="director stopped"):
            wrapped(object(), proxy, _source_vae())

    assert preload.calls == [("video.safetensors",)]
    assert clear.calls == [()]
    assert seen[0].remote_worker is worker
    assert vae_decode.current_mmh3_vae_context() is None


def test_hybrid_decode_clears_sampling_workers_once_then_reuses_resident_vaes() -> None:
    from raylight.integrations.mmh3director import vae_decode

    clear0 = _RemoteCall(True)
    clear1 = _RemoteCall(True)
    remote_result = {
        "input_shape": (1, 24, 12, 1, 1),
        "chunks": [_chunk(1, 0.2, 0.3)],
    }
    worker0 = SimpleNamespace(clear_sampling_vram=clear0)
    worker1 = SimpleNamespace(
        clear_sampling_vram=clear1,
        ray_mmh3_vae_decode_chunks=_RemoteCall(remote_result),
    )
    context = vae_decode.DirectorVaeDecodeContext(
        remote_worker=worker1,
        vae_path="video.safetensors",
        workers=(worker0, worker1),
    )
    local_result = {
        "input_shape": (1, 24, 12, 1, 1),
        "chunks": [_chunk(0, 0.0, 0.1)],
    }
    vae = SimpleNamespace(first_stage_model=FakeMiniMaxVAE())

    with patch.object(vae_decode.ray, "get", side_effect=lambda value: value), patch.object(
        vae_decode,
        "decode_mmh3_temporal_chunks",
        return_value=local_result,
    ), patch.object(
        vae_decode,
        "merge_mmh3_temporal_chunks",
        return_value=torch.zeros(1, 3, 39, 1, 1),
    ):
        output = vae_decode.hybrid_decode_mmh3_video(
            vae,
            torch.zeros(1, 24, 12, 1, 1),
            context,
        )
        second = vae_decode.hybrid_decode_mmh3_video(
            vae,
            torch.zeros(1, 24, 12, 1, 1),
            context,
        )

    assert output.shape == (1, 39, 1, 1, 3)
    assert second.shape == output.shape
    assert clear0.calls == [()]
    assert clear1.calls == [()]
    assert context.needs_gpu_cleanup is False


def test_sampling_marks_remote_vae_for_cleanup_before_the_next_decode() -> None:
    from raylight.integrations.mmh3director import vae_decode

    clear = _RemoteCall(True)
    context = vae_decode.DirectorVaeDecodeContext(
        remote_worker=SimpleNamespace(clear_sampling_vram=clear),
        vae_path="video.safetensors",
        needs_gpu_cleanup=False,
    )

    with vae_decode.activate_mmh3_vae_context(context), patch.object(
        vae_decode.ray, "get", side_effect=lambda value: value
    ):
        assert vae_decode.prepare_remote_vae_for_sampling() is True

    assert clear.calls == [()]
    assert context.needs_gpu_cleanup is True


def test_director_execute_wrapper_leaves_non_proxy_execution_untouched() -> None:
    from raylight.integrations.mmh3director import vae_decode

    original = Mock(return_value="native")
    wrapped = vae_decode.make_director_execute_wrapper(original)

    assert wrapped(object(), object(), _source_vae()) == "native"
    original.assert_called_once()
    assert vae_decode.current_mmh3_vae_context() is None


def test_vae_decode_wrapper_uses_hybrid_inside_context_and_preserves_image_layout() -> None:
    from raylight.integrations.mmh3director import vae_decode

    native = Mock(return_value=("native",))
    wrapped = vae_decode.make_vae_decode_wrapper(native)
    context = vae_decode.DirectorVaeDecodeContext(
        remote_worker=object(),
        vae_path="video.safetensors",
    )
    video = torch.zeros(1, 2, 3, 4, 3)
    nested = SimpleNamespace(is_nested=True, unbind=lambda: (torch.zeros(1, 24, 12, 1, 1), object()))

    with vae_decode.activate_mmh3_vae_context(context), patch.object(
        vae_decode,
        "hybrid_decode_mmh3_video",
        return_value=video,
    ) as hybrid:
        result = wrapped(object(), object(), {"samples": nested})

    assert result[0].shape == (2, 3, 4, 3)
    hybrid.assert_called_once()
    native.assert_not_called()


def test_vae_decode_wrapper_falls_back_to_native_on_parallel_failure() -> None:
    from raylight.integrations.mmh3director import vae_decode

    native = Mock(return_value=("native",))
    wrapped = vae_decode.make_vae_decode_wrapper(native)
    context = vae_decode.DirectorVaeDecodeContext(object(), "video.safetensors")

    with vae_decode.activate_mmh3_vae_context(context), patch.object(
        vae_decode,
        "hybrid_decode_mmh3_video",
        side_effect=RuntimeError("remote failed"),
    ):
        assert wrapped(object(), object(), {"samples": torch.zeros(1, 24, 12, 1, 1)}) == ("native",)

    native.assert_called_once()
