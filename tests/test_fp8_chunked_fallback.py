import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from raylight.comfy_dist.kitchen_patches.fp8 import (
    allocate_output_with_cache_retry,
    fp8_addmm_fallback_chunked,
    fp8_linear_fallback_chunked,
    should_use_v100_chunked_fp8,
)


class FP8ChunkedFallbackTests(unittest.TestCase):
    def test_chunked_fallback_limits_weight_and_output_temporaries(self):
        input_tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
        qweight = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 20
        scale = torch.tensor(0.5, dtype=torch.float32)
        bias = torch.arange(6, dtype=torch.float32) / 100
        expected = F.linear(input_tensor, qweight * scale, bias)
        original_linear = F.linear
        weight_rows = []

        def tracking_linear(value, weight, chunk_bias=None):
            weight_rows.append(weight.shape[0])
            return original_linear(value, weight, chunk_bias)

        # Per output channel the dequantized weight consumes 16 bytes and
        # the five-row output consumes 20 bytes. A 72-byte budget therefore
        # permits at most two output channels per temporary chunk.
        with mock.patch(
            "raylight.comfy_dist.kitchen_patches.fp8.torch.nn.functional.linear",
            side_effect=tracking_linear,
        ):
            actual = fp8_linear_fallback_chunked(
                input_tensor,
                qweight,
                scale,
                bias,
                torch.float32,
                max_temp_bytes=72,
            )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(weight_rows, [2, 2, 2])

    def test_addmm_fallback_chunks_quantized_rhs_columns_with_bias(self):
        left = torch.arange(20, dtype=torch.float32).reshape(5, 4) / 10
        qright = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 20
        scale = torch.tensor(0.5, dtype=torch.float32)
        bias = torch.arange(6, dtype=torch.float32) / 100
        expected = torch.addmm(bias, left, qright * scale)
        original_addmm = torch.addmm
        right_columns = []

        def tracking_addmm(chunk_bias, a, b):
            right_columns.append(b.shape[1])
            return original_addmm(chunk_bias, a, b)

        with mock.patch(
            "raylight.comfy_dist.kitchen_patches.fp8.torch.addmm",
            side_effect=tracking_addmm,
        ):
            actual = fp8_addmm_fallback_chunked(
                bias,
                left,
                qright,
                scale,
                torch.float32,
                max_temp_bytes=72,
            )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(right_columns, [2, 2, 2])

    def test_chunked_fallback_handles_no_bias(self):
        x = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        qweight = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
        actual = fp8_linear_fallback_chunked(x, qweight, torch.tensor(0.25), None, torch.float32, 16)
        torch.testing.assert_close(actual, F.linear(x, qweight * 0.25))

    def test_output_allocation_releases_cached_blocks_and_retries_one_oom(self):
        expected = object()
        with (
            mock.patch(
                "raylight.comfy_dist.kitchen_patches.fp8.torch.empty",
                side_effect=[torch.OutOfMemoryError("fragmented"), expected],
            ) as empty,
            mock.patch(
                "raylight.comfy_dist.kitchen_patches.fp8.torch.cuda.empty_cache"
            ) as empty_cache,
        ):
            actual = allocate_output_with_cache_retry(
                (4, 8), device=torch.device("cuda"), dtype=torch.bfloat16
            )

        self.assertIs(actual, expected)
        self.assertEqual(empty.call_count, 2)
        empty_cache.assert_called_once_with()

    def test_v100_detection_routes_cuda_capability_7_before_input_quantization(self):
        fake_cuda_tensor = mock.Mock()
        fake_cuda_tensor.is_cuda = True
        fake_cuda_tensor.device = torch.device("cuda:0")

        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 0)):
            self.assertTrue(should_use_v100_chunked_fp8(fake_cuda_tensor))

        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            self.assertFalse(should_use_v100_chunked_fp8(fake_cuda_tensor))

    def test_v100_detection_never_routes_cpu_tensor(self):
        with mock.patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("CPU tensors must not query CUDA capability"),
        ):
            self.assertFalse(should_use_v100_chunked_fp8(torch.ones(1)))


if __name__ == "__main__":
    unittest.main()
