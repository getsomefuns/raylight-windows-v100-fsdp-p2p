from types import SimpleNamespace

import torch

from raylight.diffusion_models.minimax.director_continuity import (
    prepare_director_continuity,
)


def _layout():
    # Seven prior-segment anchors plus one reference image, one row each.
    segments = [(index, index + 1, "cond") for index in range(7)]
    segments += [(7, 8, "ref_img"), (8, 10, "ref_audio"), (10, 12, "audio"), (12, 15, "video")]
    position_ids = torch.zeros(15, 3, dtype=torch.float64)
    position_ids[7, 0] = 20.0
    position_ids[8:10, 0] = torch.tensor([21.0, 21.5])
    position_ids[12:15, 0] = torch.tensor([30.0, 31.0, 32.0])
    return SimpleNamespace(segments=segments, position_ids=position_ids)


def test_director_continuity_merges_keyframes_before_reference_latents():
    keyframes = [
        {
            "resolved_frame_index": 0,
            "director_context_index": index,
            "latent": torch.tensor([float(index)]),
        }
        for index in range(7)
    ]
    ref = {"kind": "image", "latent": torch.tensor([99.0])}
    payload = {
        "keyframes": keyframes,
        "refs": [ref],
        # This is the broken stock-ComfyUI result: refs overwrite keyframes.
        "cond_video_latents": [ref["latent"]],
        "frame_count": 13,
    }

    prepared = prepare_director_continuity(payload, _layout(), frame_rescale=0.5)

    assert [item.item() for item in prepared["cond_video_latents"]] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        99.0,
    ]


def test_director_continuity_places_context_anchors_on_target_timeline():
    layout = _layout()
    payload = {
        "keyframes": [
            {
                "resolved_frame_index": 0,
                "director_context_index": index,
                "latent": torch.tensor([float(index)]),
            }
            for index in range(7)
        ],
        "refs": [{"kind": "image", "latent": torch.tensor([99.0])}],
        "cond_video_latents": [torch.tensor([99.0])],
        "frame_count": 13,
    }

    prepare_director_continuity(payload, layout, frame_rescale=0.5)

    assert layout.position_ids[:7, 0].tolist() == [30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0]


def test_director_continuity_repositions_marked_audio_reference():
    layout = _layout()
    audio = torch.tensor([7.0])
    payload = {
        "keyframes": [
            {
                "resolved_frame_index": 0,
                "director_context_index": index,
                "latent": torch.tensor([float(index)]),
            }
            for index in range(7)
        ],
        "refs": [
            {"kind": "image", "latent": torch.tensor([99.0])},
            {
                "kind": "audio",
                "ref_audio_t": 1,
                "audio_latent": audio,
                "director_context_audio_end": 6,
            },
        ],
        "cond_video_latents": [torch.tensor([99.0])],
        "cond_audio_latents": [audio],
        "frame_count": 13,
    }

    prepare_director_continuity(payload, layout, frame_rescale=0.5)

    # Target origin 30 + end frame 6 * 0.5 - one audio step.
    assert layout.position_ids[8:10, 0].tolist() == [32.0, 32.5]


def test_unmarked_minimax_payload_is_untouched():
    layout = _layout()
    original_positions = layout.position_ids.clone()
    original = [torch.tensor([99.0])]
    payload = {
        "keyframes": [{"resolved_frame_index": 0, "latent": torch.tensor([0.0])}],
        "refs": [{"kind": "image", "latent": original[0]}],
        "cond_video_latents": original,
    }

    prepared = prepare_director_continuity(payload, layout, frame_rescale=0.5)

    assert prepared is payload
    assert prepared["cond_video_latents"] is original
    assert torch.equal(layout.position_ids, original_positions)
