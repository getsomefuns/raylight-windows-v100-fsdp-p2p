from __future__ import annotations

import math

import torch


def normalize_director_audio_output(result):
    """Make empty Director AUDIO safe for stock video encoders on Ray executions."""
    if not isinstance(result, tuple) or len(result) < 3:
        return result
    audios = result[1]
    if not isinstance(audios, list):
        return result
    fps = float(result[2])
    if not math.isfinite(fps) or fps <= 0:
        return result

    normalized = None
    for index, audio in enumerate(audios):
        if not isinstance(audio, dict):
            continue
        wave = audio.get("waveform")
        if (
            not isinstance(wave, torch.Tensor)
            or wave.ndim != 3
            or wave.shape[-1] != 0
            or wave.shape[0] < 1
            or wave.shape[1] < 1
        ):
            continue
        # During first-pass confirmation the main image output is blocked;
        # images_pre_refine still carries the clip that the user can save.
        frame_count = 0
        for slot in (0, 6, 7):
            batches = result[slot] if slot < len(result) else None
            if not isinstance(batches, (list, tuple)) or index >= len(batches):
                continue
            images = batches[index]
            if isinstance(images, torch.Tensor) and images.ndim == 4:
                frame_count = max(frame_count, int(images.shape[0]))
        rate = int(audio.get("sample_rate") or 0)
        if frame_count <= 0 or rate <= 0:
            continue
        if normalized is None:
            normalized = list(audios)
        normalized[index] = {
            **audio,
            "waveform": torch.zeros(
                (*wave.shape[:-1], math.ceil(frame_count * rate / fps)),
                dtype=wave.dtype,
                device="cpu",
            ),
        }

    if normalized is None:
        return result
    return (result[0], normalized, *result[2:])
