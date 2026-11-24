"""The break these tests catch: losing overlap history or misplacing async windows."""
import importlib

import pytest
import torch


def load_planning():
    try:
        return importlib.import_module("raylight.relayvsr.planning")
    except ModuleNotFoundError:
        pytest.fail("RelayVSR window planner has not been implemented")


@pytest.mark.parametrize("count,expected", [
    (1, [(0, 0, 1, 0, 1, 0)]),
    (16, [(0, 0, 16, 0, 2, 0)]),
    (17, [(0, 0, 16, 0, 2, 0), (1, 15, 17, 1, 3, 1)]),
    (31, [(0, 0, 16, 0, 2, 0), (1, 15, 31, 1, 3, 1)]),
])
def test_native_window_boundaries(count, expected):
    planning = load_planning()
    jobs = planning.plan_relay_windows(count)
    assert [(j.index, j.pixel_start, j.pixel_stop, j.latent_start,
             j.latent_stop, j.discard_prefix) for j in jobs] == expected


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_frame_count_rejected_before_work(count):
    planning = load_planning()
    with pytest.raises((ValueError, TypeError)):
        planning.plan_relay_windows(count)


def test_240_frames_merge_out_of_order_exactly_once():
    planning = load_planning()
    jobs = planning.plan_relay_windows(240)
    assert len(jobs) == 16
    assert (jobs[-1].pixel_start, jobs[-1].pixel_stop,
            jobs[-1].latent_start, jobs[-1].latent_stop, jobs[-1].discard_prefix) == (225, 240, 15, 17, 1)
    output = torch.empty(240, 2, 3, 3)
    written = torch.zeros(240, dtype=torch.bool)
    for job in reversed(jobs):
        values = torch.arange(job.pixel_start, job.pixel_stop, dtype=torch.float32)
        result = values[:, None, None, None].expand(-1, 2, 3, 3)
        planning.merge_window_result(output, written, job, result)
    assert written.all()
    assert torch.equal(output[:, 0, 0, 0], torch.arange(240, dtype=torch.float32))


def test_duplicate_window_result_is_not_silently_accepted():
    planning = load_planning()
    job = planning.plan_relay_windows(16)[0]
    output = torch.empty(16, 2, 2, 3)
    written = torch.zeros(16, dtype=torch.bool)
    result = torch.ones_like(output)
    planning.merge_window_result(output, written, job, result)
    with pytest.raises(ValueError, match="duplicate|already"):
        planning.merge_window_result(output, written, job, result)


@pytest.mark.parametrize("bad_shape", [(15, 2, 2, 3), (16, 3, 2, 3), (16, 2, 2, 4)])
def test_malformed_result_does_not_mark_frames_written(bad_shape):
    planning = load_planning()
    job = planning.plan_relay_windows(16)[0]
    written = torch.zeros(16, dtype=torch.bool)
    with pytest.raises(ValueError):
        planning.merge_window_result(torch.empty(16, 2, 2, 3), written, job, torch.zeros(bad_shape))
    assert not written.any()


def test_nonfinite_result_cannot_escape_to_video_saving():
    planning = load_planning()
    job = planning.plan_relay_windows(1)[0]
    result = torch.full((1, 2, 2, 3), float("nan"))
    written = torch.zeros(1, dtype=torch.bool)
    with pytest.raises(ValueError, match="finite|NaN"):
        planning.merge_window_result(torch.empty_like(result), written, job, result)
    assert not written.any()


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("count", [1, 16, 17, 31])
def test_real_flash_window_recurrence_matches_whole_decode(scale, count):
    planning = load_planning()
    native = pytest.importorskip("comfy.ldm.wan.model_vsr", reason="older core has no RelayVSR")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(481)
            model = native.RelayFlash(scale, dim=128, depth=1, num_heads=2,
                                      kv_groups=1, hidden_dim=64, head_dim=64).float().eval()
            for name, parameter in model.named_parameters():
                if "norm" in name and name.endswith("weight"):
                    parameter.fill_(1)
                else:
                    parameter.normal_(0, 0.01)
            side = 64 if scale == 2 else 32
            images = torch.rand(count, side, side, 3)
            key_count = len(list(range(0, count - 1, 15))) + 1
            latents = torch.randn(1, 48, key_count, 8, 8) * 0.01
            whole = model.decode(latents, lq_images=images).movedim(1, -1).reshape(count, side*scale, side*scale, 3)
            merged = torch.empty_like(whole)
            written = torch.zeros(count, dtype=torch.bool)
            for job in reversed(planning.plan_relay_windows(count)):
                window = model.decode(latents[:, :, job.latent_start:job.latent_stop].contiguous(),
                                      lq_images=images[job.pixel_start:job.pixel_stop])
                frames = window.movedim(1, -1).reshape(-1, side*scale, side*scale, 3)
                planning.merge_window_result(merged, written, job, frames)
            assert written.all()
            assert torch.equal(merged, whole)
    finally:
        torch.set_num_threads(previous_threads)

