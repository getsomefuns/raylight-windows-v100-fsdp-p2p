from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from raylight.distributed_worker.ray_worker_vae import (
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
