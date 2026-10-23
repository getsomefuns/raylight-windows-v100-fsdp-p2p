from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import av
import pytest
import torch

from raylight.integrations.mmh3director import vae_decode
from raylight.integrations.mmh3director.proxy import RayMMH3DirectorModelProxy


def director_result(audio):
    images = [torch.zeros(6, 32, 48, 3)]
    return (images, [audio], 24.0, 6, images, "report", images, None)


def execute_proxy(result, *, preload_error=False, preload_success=False):
    proxy = object.__new__(RayMMH3DirectorModelProxy)
    worker = SimpleNamespace(
        start_mmh3_vae_preload=SimpleNamespace(remote=lambda path: "preloaded"),
        clear_sampling_vram=SimpleNamespace(remote=lambda: None),
    )
    proxy.ray_actors = {"workers": [object(), worker] if preload_error or preload_success else []}

    def original(_self, model, video_vae):
        return result

    wrapped = vae_decode.make_director_execute_wrapper(original)
    vae = SimpleNamespace(patcher=SimpleNamespace(cached_patcher_init=(None, ("vae.safetensors",))))
    get_result = RuntimeError("preload unavailable") if preload_error else lambda value: value
    with patch.object(vae_decode.ray, "get", side_effect=get_result):
        return wrapped(object(), proxy, vae)


@pytest.mark.parametrize("preload_error", [False, True])
def test_empty_director_audio_becomes_frame_aligned_silence(preload_error):
    audio = {"waveform": torch.empty(1, 1, 0), "sample_rate": 32000}
    original = director_result(audio)
    result = execute_proxy(original, preload_error=preload_error)

    assert result[1][0]["waveform"].shape == (1, 1, 8000)
    assert torch.count_nonzero(result[1][0]["waveform"]) == 0
    assert result[1][0]["sample_rate"] == 32000
    assert audio["waveform"].shape == (1, 1, 0)
    assert result[0] is original[0]
    assert result[6] is original[6]


def test_normal_audio_is_passed_through_without_copy_or_value_changes():
    audio = {"waveform": torch.rand(1, 2, 12000), "sample_rate": 48000}
    original = director_result(audio)
    result = execute_proxy(original)
    assert result is original
    assert result[1][0] is audio


def test_successful_dual_vae_execution_also_normalizes_empty_audio():
    audio = {"waveform": torch.empty(1, 2, 0), "sample_rate": 48000}
    result = execute_proxy(director_result(audio), preload_success=True)
    assert result[1][0]["waveform"].shape == (1, 2, 12000)


def test_first_pass_confirmation_uses_pre_refine_images_for_silent_duration():
    from comfy_execution.graph import ExecutionBlocker

    audio = {"waveform": torch.empty(1, 1, 0), "sample_rate": 32000}
    original = director_result(audio)
    blocked = ExecutionBlocker(None)
    result = execute_proxy((blocked, *original[1:]))
    assert result[0] is blocked
    assert result[1][0]["waveform"].shape[-1] == 8000


def test_non_ray_director_audio_is_untouched():
    audio = {"waveform": torch.empty(1, 1, 0), "sample_rate": 32000}
    original = director_result(audio)

    def execute(_self, model, video_vae):
        return original

    wrapped = vae_decode.make_director_execute_wrapper(execute)
    assert wrapped(object(), object(), None) is original


def test_mixed_segment_audio_preserves_nonempty_entries_and_list_alignment():
    sound = {"waveform": torch.ones(1, 2, 1000), "sample_rate": 48000}
    empty = {"waveform": torch.empty(1, 2, 0), "sample_rate": 48000}
    images = [torch.zeros(2, 16, 16, 3), torch.zeros(3, 16, 16, 3)]
    original = (images, [sound, empty], 24.0, 5, images, "report", images, None)
    result = execute_proxy(original)
    assert result[1][0] is sound
    assert result[1][1]["waveform"].shape == (1, 2, 6000)
    assert len(result[1]) == 2


@pytest.mark.parametrize("normal_audio", [False, True])
def test_official_video_save_encodes_empty_or_normal_director_audio(tmp_path, normal_audio):
    from comfy_api.latest._input_impl.video_types import VideoFromComponents
    from comfy_api.latest._util import VideoComponents

    if normal_audio:
        t = torch.arange(12000) / 48000
        wave = (0.1 * torch.sin(2 * torch.pi * 440 * t))[None, None, :].repeat(1, 2, 1)
        audio = {"waveform": wave, "sample_rate": 48000}
    else:
        audio = {"waveform": torch.empty(1, 1, 0), "sample_rate": 32000}
    result = execute_proxy(director_result(audio))
    video = VideoFromComponents(VideoComponents(
        images=result[0][0], audio=result[1][0], frame_rate=Fraction(24, 1)
    ))
    path = tmp_path / "director.mp4"
    video.save_to(str(path))
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 6
        assert len(container.streams.audio) == 1
    with av.open(str(path)) as container:
        decoded = [frame.to_ndarray() for frame in container.decode(audio=0)]
        assert sum(part.shape[-1] for part in decoded) >= 8000
        peak = max(abs(part).max() for part in decoded)
        assert peak > 0.01 if normal_audio else peak == 0
