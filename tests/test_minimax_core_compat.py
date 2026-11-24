"""Catch dropped per-token mask modulation and wrong rank-local row indices."""
import copy
import inspect

import pytest
import torch

from tests.test_minimax_adaln_curves import make_model, parallel  # noqa: F401


def test_split_slices_per_token_modulation_rows(parallel, monkeypatch):
    monkeypatch.setattr(parallel, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel, "get_sequence_parallel_rank", lambda: 1)
    hidden = torch.zeros(12, 4)
    rows = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12])
    _, _, segments = parallel._split_packed_sequence(hidden, torch.zeros(1, 12, 4), [(2, 10, rows)])
    assert segments[0][:2] == (0, 4)
    assert torch.equal(segments[0][2], torch.tensor([9, 10, 11, 12]))


@pytest.mark.parametrize("rank", [0, 1])
def test_final_projection_uses_only_this_rank_timestep_rows(parallel, rank):
    from types import SimpleNamespace
    head = torch.nn.Linear(3, 2)
    norm = torch.nn.RMSNorm(3)
    hidden = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    shift = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    scale = shift / 20
    indices = torch.tensor([0, 1, 2, 3, 4, 0])
    segment = (1, 7, indices)
    offset = max(1, rank * 4) - 1
    count = 3
    local = hidden[1:4] if rank == 0 else hidden[:3]
    selected = indices[offset:offset + count]
    expected = head(norm(local) * (1 + scale[selected]) + shift[selected])
    actual, counts = parallel._project_local_final_segment(
        SimpleNamespace(norm=norm), hidden, shift, scale, segment, head,
        rank=rank, world_size=2, chunk_tokens=2,
    )
    assert counts == [3, 3]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("video_strength,audio_strength", [(0., 0.), (.25, .5), (1., 1.)])
def test_masked_forward_matches_new_native_reference(parallel, video_strength, audio_strength):
    model = make_model()
    if "denoise_mask" not in inspect.signature(model._forward).parameters:
        pytest.skip("this legacy native core predates the mask reference API")
    reference = copy.deepcopy(model)
    video = torch.ones(1, 24, 2, 4, 4)
    audio = torch.ones(1, 32, 2, 4)
    video_mask = torch.ones(1, 1, 2, 4, 4)
    video_mask[:, :, 0] = video_strength
    audio_mask = torch.tensor([audio_strength, 1., .75, 0.] * 2).reshape(1, 1, 2, 4)
    kwargs = dict(denoise_mask=video_mask, audio_denoise_mask=audio_mask)
    with torch.inference_mode():
        expected = reference._forward([video, audio], torch.tensor([650.]), torch.ones(1, 3, 96), **kwargs)
        actual = parallel.usp_dit_forward(model, [video, audio], torch.tensor([650.]), torch.ones(1, 3, 96), **kwargs)
    for observed, wanted in zip(actual, expected):
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, wanted, rtol=1e-5, atol=1e-5)


def test_safe_fp16_guard_accepts_installed_native_core():
    import comfy.ldm.minimax.model as native
    from raylight.comfy_dist.minimax_h3_fp16 import validate_minimax_h3_api
    validate_minimax_h3_api(native)
