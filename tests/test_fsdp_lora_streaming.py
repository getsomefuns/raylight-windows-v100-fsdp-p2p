from __future__ import annotations

import os
import unittest
from unittest import mock
import torch
import torch.nn.functional as F

from comfy.weight_adapter.bypass import BypassForwardHook
from raylight.comfy_dist.sd import defer_adapter_device_move
from raylight.comfy_dist.weight_adapter.lora import LoRAAdapter


def verify_deferred_fsdp_lora_does_not_eagerly_move_weights_and_keeps_math():
    up = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    down = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]],
        dtype=torch.float32,
    )
    adapter = LoRAAdapter(
        {"up", "down"},
        (up, down, 4.0, None, None, None),
    )
    deferred = defer_adapter_device_move(adapter)

    layer = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.zero_()

    # If the hook used the regular eager path, the adapter tensors would be
    # replaced by meta-device tensors during injection. The deferred wrapper
    # must leave the real payload on CPU until h() consumes the current layer.
    hook = BypassForwardHook(layer, deferred, multiplier=0.5)
    with mock.patch("comfy.model_management.get_torch_device", return_value=torch.device("meta")):
        hook.inject()

    assert adapter.weights[0].device.type == "cpu"
    assert adapter.weights[1].device.type == "cpu"

    x = torch.tensor([[1.0, -1.0, 2.0, 0.5]], dtype=torch.float32)
    actual = layer(x)
    expected = F.linear(F.linear(x, down), up) * (4.0 / 2.0) * 0.5
    torch.testing.assert_close(actual, expected)

    hook.eject()


def verify_deferred_fsdp_lora_chunks_and_accumulates_large_linear_output():
    up = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    down = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]],
        dtype=torch.float32,
    )
    adapter = LoRAAdapter(
        {"up", "down"},
        (up, down, 4.0, None, None, None),
    )
    deferred = defer_adapter_device_move(adapter)

    layer = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    hook = BypassForwardHook(layer, deferred, multiplier=0.5)
    hook.inject()

    x = torch.arange(40, dtype=torch.float32).reshape(10, 4) / 10
    original_linear = F.linear
    up_call_rows = []

    def tracking_linear(value, weight, bias=None):
        if tuple(weight.shape) == tuple(up.shape):
            up_call_rows.append(value.reshape(-1, value.shape[-1]).shape[0])
        return original_linear(value, weight, bias)

    # Three float32 output values use 12 bytes per row, so a 24-byte
    # temporary budget must split ten rows into chunks of at most two.
    with mock.patch.dict(os.environ, {"RAYLIGHT_FSDP_LORA_CHUNK_BYTES": "24"}):
        with mock.patch(
            "raylight.comfy_dist.weight_adapter.lora.F.linear",
            side_effect=tracking_linear,
        ):
            with torch.no_grad():
                actual = layer(x)

    expected = original_linear(original_linear(x, down), up) * (4.0 / 2.0) * 0.5
    torch.testing.assert_close(actual, expected)
    assert len(up_call_rows) == 5
    assert max(up_call_rows) <= 2
    hook.eject()

def verify_chunked_lora_retries_with_smaller_rows_after_cuda_oom():
    up = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        dtype=torch.float32,
    )
    down = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]],
        dtype=torch.float32,
    )
    adapter = LoRAAdapter(
        {"up", "down"},
        (up, down, 4.0, None, None, None),
    )
    x = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    base_out = torch.zeros(6, 3, dtype=torch.float32)
    original_linear = F.linear
    attempted_rows = []

    def oom_above_one_row(value, weight, bias=None):
        rows = value.reshape(-1, value.shape[-1]).shape[0]
        if tuple(weight.shape) == tuple(down.shape):
            attempted_rows.append(rows)
            if rows > 1:
                raise torch.OutOfMemoryError("simulated LoRA temporary OOM")
        return original_linear(value, weight, bias)

    with mock.patch(
        "raylight.comfy_dist.weight_adapter.lora.F.linear",
        side_effect=oom_above_one_row,
    ):
        actual = adapter.add_to_base_chunked(x, base_out, max_temp_bytes=24)

    expected = original_linear(original_linear(x, down), up) * (4.0 / 2.0)
    torch.testing.assert_close(actual, expected)
    assert attempted_rows[0] == 2
    assert 1 in attempted_rows



class FSDPLoRAStreamingTests(unittest.TestCase):
    def test_deferred_fsdp_lora_does_not_eagerly_move_weights_and_keeps_math(self):
        verify_deferred_fsdp_lora_does_not_eagerly_move_weights_and_keeps_math()

    def test_deferred_fsdp_lora_chunks_and_accumulates_large_linear_output(self):
        verify_deferred_fsdp_lora_chunks_and_accumulates_large_linear_output()
    def test_chunked_lora_retries_with_smaller_rows_after_cuda_oom(self):
        verify_chunked_lora_retries_with_smaller_rows_after_cuda_oom()



if __name__ == "__main__":
    unittest.main()
