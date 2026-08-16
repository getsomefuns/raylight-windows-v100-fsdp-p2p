from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from raylight.comfy_dist import quant_ops


class FP8OOMRetryTests(unittest.TestCase):
    def setUp(self):
        quant_ops._FP8_FALLBACK_LOGGED = False

    def test_fp8_dequantize_releases_cached_blocks_and_retries_one_oom(self):
        expected = torch.tensor([7.0])
        qdata = torch.tensor([1], dtype=torch.uint8)
        params = SimpleNamespace(scale=torch.tensor(1.0))

        with mock.patch.object(
            quant_ops.ck,
            "dequantize_per_tensor_fp8",
            side_effect=[torch.OutOfMemoryError("fragmented"), expected],
        ) as dequantize:
            with mock.patch.object(torch.cuda, "empty_cache") as empty_cache:
                actual = quant_ops.dequantize_ray_temp_fix_fp8(
                    qdata,
                    params,
                    torch.float16,
                )

        self.assertIs(actual, expected)
        self.assertEqual(dequantize.call_count, 2)
        empty_cache.assert_called_once_with()

    def test_fp8_dequantize_propagates_a_second_oom(self):
        qdata = torch.tensor([1], dtype=torch.uint8)
        params = SimpleNamespace(scale=torch.tensor(1.0))

        with mock.patch.object(
            quant_ops.ck,
            "dequantize_per_tensor_fp8",
            side_effect=torch.OutOfMemoryError("still full"),
        ) as dequantize:
            with mock.patch.object(torch.cuda, "empty_cache") as empty_cache:
                with self.assertRaises(torch.OutOfMemoryError):
                    quant_ops.dequantize_ray_temp_fix_fp8(
                        qdata,
                        params,
                        torch.float16,
                    )

        self.assertEqual(dequantize.call_count, 2)
        empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
