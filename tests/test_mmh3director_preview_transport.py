from __future__ import annotations

import importlib
from unittest import mock

import torch
from ray.util.queue import Empty, Full

import comfy.nested_tensor


def _transport():
    return importlib.import_module(
        "raylight.integrations.mmh3director.preview_transport"
    )


def test_preview_event_extracts_only_video_and_caps_time_on_cpu():
    transport = _transport()
    video = torch.arange(1 * 24 * 40 * 3 * 5, dtype=torch.float32).reshape(
        1, 24, 40, 3, 5
    )
    audio = torch.zeros(1, 32, 2, 50)
    nested = comfy.nested_tensor.NestedTensor((video, audio))

    event = transport.extract_preview_event(
        nested,
        step=2,
        total_steps=4,
        preview_every=1,
        max_frames=16,
    )

    assert event["step"] == 2
    assert event["total_steps"] == 4
    assert event["x0"].shape == (1, 24, 16, 3, 5)
    assert event["x0"].device.type == "cpu"
    assert torch.equal(event["x0"][:, :, 0], video[:, :, 0])
    assert torch.equal(event["x0"][:, :, -1], video[:, :, -1])


def test_preview_event_honors_interval_and_always_keeps_last_step():
    transport = _transport()
    x0 = torch.zeros(1, 24, 4, 2, 2)

    assert transport.extract_preview_event(
        x0, step=1, total_steps=4, preview_every=2
    ) is None
    assert transport.extract_preview_event(
        x0, step=3, total_steps=4, preview_every=2
    )["step"] == 3
    assert transport.extract_preview_event(
        x0, step=1, total_steps=4, preview_every=-1
    ) is None
    assert transport.extract_preview_event(
        x0, step=3, total_steps=4, preview_every=-1
    )["step"] == 3


def test_preview_publish_drops_when_queue_is_full_without_blocking():
    transport = _transport()
    queue = mock.Mock()
    queue.put.side_effect = Full
    event = {"step": 1, "total_steps": 4, "x0": torch.zeros(1)}

    assert transport.publish_preview_nonblocking(queue, event) is False
    queue.put.assert_called_once_with(event, block=False)


def test_preview_drain_forwards_all_available_events_in_order():
    transport = _transport()
    callback = mock.Mock()
    first = {"step": 0, "total_steps": 4, "x0": torch.zeros(1)}
    second = {"step": 1, "total_steps": 4, "x0": torch.ones(1)}
    queue = mock.Mock()
    queue.get.side_effect = [first, second, Empty]

    assert transport.drain_preview_queue(queue, callback) == 2
    assert callback.call_args_list == [
        mock.call(0, first["x0"], first["x0"], 4),
        mock.call(1, second["x0"], second["x0"], 4),
    ]
    assert all(call.kwargs == {"block": False} for call in queue.get.call_args_list)

