"""Worker-side MiniMax H3 Director continuity compatibility.

Director prepares motion-context keyframes in the main ComfyUI process, while
Raylight samples in independent Ray worker processes. Stock ComfyUI replaces
keyframe condition latents with reference condition latents when both are
present. This module restores the combined payload and the Director timeline
inside each worker without importing or patching the Director custom node.
"""

from __future__ import annotations

from typing import Any


CONTEXT_FRAME_KEY = "director_context_index"
CONTEXT_AUDIO_END_KEY = "director_context_audio_end"


def _has_director_markers(payload: dict[str, Any]) -> bool:
    return any(CONTEXT_FRAME_KEY in item for item in payload.get("keyframes") or ()) or any(
        CONTEXT_AUDIO_END_KEY in item for item in payload.get("refs") or ()
    )


def _target_video_span(layout) -> tuple[int, int]:
    for start, end, kind in reversed(layout.segments):
        if kind == "video":
            if end <= start:
                break
            return int(start), int(end)
    raise RuntimeError("Raylight Director continuity expected a non-empty target video segment")


def _rewrite_context_keyframe_times(payload, layout, frame_rescale: float) -> None:
    keyframes = list(payload.get("keyframes") or ())
    marked = [item for item in keyframes if CONTEXT_FRAME_KEY in item]
    if not marked:
        return

    spans = [(start, end) for start, end, kind in layout.segments if kind == "cond"]
    if len(spans) != len(keyframes):
        raise RuntimeError(
            f"Raylight Director continuity keyframe/layout mismatch: {len(keyframes)} keyframes but {len(spans)} condition segments"
        )

    target_start, target_end = _target_video_span(layout)
    target_origin = float(layout.position_ids[target_start, 0])
    target_last = float(layout.position_ids[target_end - 1, 0])
    frame_count = payload.get("frame_count")

    for (start, end), keyframe in zip(spans, keyframes):
        pixel_index = keyframe.get(CONTEXT_FRAME_KEY)
        if pixel_index is None:
            continue
        pixel_index = int(pixel_index)
        if pixel_index < 0 or (frame_count is not None and pixel_index >= int(frame_count)):
            raise RuntimeError(
                f"Raylight Director continuity keyframe index is outside the target clip: index={pixel_index}, frame_count={frame_count}"
            )
        if frame_count is not None and pixel_index == int(frame_count) - 1:
            time_value = target_last
        else:
            time_value = target_origin + float(frame_rescale) * pixel_index
        layout.position_ids[start:end, 0] = time_value


def _reference_segments(layout, refs: list[dict[str, Any]]) -> dict[int, dict[str, tuple[int, int]]]:
    available = [(start, end, kind) for start, end, kind in layout.segments if kind in {"ref_img", "ref_audio"}]
    expected: list[tuple[int, str]] = []
    for index, ref in enumerate(refs):
        kind = ref.get("kind")
        if kind == "image":
            expected.append((index, "ref_img"))
        elif kind == "audio":
            if int(ref.get("ref_audio_t", 0) or 0) > 0:
                expected.append((index, "ref_audio"))
        elif kind in {"video", "video_audio"}:
            if int(ref.get("ref_audio_t", 0) or 0) > 0:
                expected.append((index, "ref_audio"))
            expected.append((index, "ref_img"))
        else:
            raise RuntimeError(f"Raylight Director continuity found unknown reference kind {kind!r}")

    if len(expected) != len(available):
        raise RuntimeError(
            "Raylight Director continuity reference/layout mismatch: "
            f"expected {len(expected)} reference segments but layout has {len(available)}"
        )

    result: dict[int, dict[str, tuple[int, int]]] = {}
    for (index, expected_kind), (start, end, actual_kind) in zip(expected, available):
        if actual_kind != expected_kind:
            raise RuntimeError(
                f"Raylight Director continuity reference segment order mismatch: expected {expected_kind}, got {actual_kind}"
            )
        result.setdefault(index, {})[actual_kind] = (int(start), int(end))
    return result


def _rewrite_context_audio_time(payload, layout, frame_rescale: float) -> None:
    refs = list(payload.get("refs") or ())
    marked = [(index, ref) for index, ref in enumerate(refs) if CONTEXT_AUDIO_END_KEY in ref]
    if not marked:
        return
    if len(marked) != 1:
        raise RuntimeError(f"Raylight Director continuity requires exactly one marked audio reference, got {len(marked)}")

    index, ref = marked[0]
    if ref.get("kind") != "audio":
        raise RuntimeError("Raylight Director continuity audio marker is attached to a non-audio reference")
    audio_steps = int(ref.get("ref_audio_t", 0) or 0)
    if audio_steps <= 0:
        return
    span = _reference_segments(layout, refs).get(index, {}).get("ref_audio")
    if span is None:
        raise RuntimeError("Raylight Director continuity marked audio reference produced no layout segment")
    start, end = span
    if end - start != 2 * audio_steps:
        raise RuntimeError(f"Raylight Director continuity audio row mismatch: {end - start} rows for {audio_steps} stereo steps")

    target_start, _ = _target_video_span(layout)
    target_origin = float(layout.position_ids[target_start, 0])
    current_start = float(layout.position_ids[start, 0])
    desired_start = target_origin + float(frame_rescale) * float(ref[CONTEXT_AUDIO_END_KEY]) - float(audio_steps)
    layout.position_ids[start:end, 0] += desired_start - current_start


def prepare_director_continuity(payload: dict[str, Any], layout, *, frame_rescale: float) -> dict[str, Any]:
    """Restore Director keyframe/reference coexistence for a Ray worker.

    Unmarked MiniMax payloads are returned by identity, keeping the existing
    I2V/FL2VA/REF2VA paths unchanged.
    """
    if not _has_director_markers(payload):
        return payload

    prepared = dict(payload)
    keyframes = list(payload.get("keyframes") or ())
    refs = list(payload.get("refs") or ())
    prepared["cond_video_latents"] = [item["latent"] for item in keyframes if "latent" in item] + [
        item["latent"] for item in refs if "latent" in item
    ]
    prepared["cond_audio_latents"] = [item["audio_latent"] for item in refs if item.get("audio_latent") is not None]

    _rewrite_context_keyframe_times(prepared, layout, frame_rescale)
    _rewrite_context_audio_time(prepared, layout, frame_rescale)
    return prepared
