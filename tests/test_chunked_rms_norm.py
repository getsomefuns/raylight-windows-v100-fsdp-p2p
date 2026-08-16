import os
import types
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from raylight.comfy_dist.quant_ops import (
    apply_rms_norm_chunked,
    should_use_v100_chunked_rms_norm,
    v100_rms_norm_chunk_bytes,
)


class ChunkedRMSNormTests(unittest.TestCase):
    def test_rms_norm_is_chunked_without_destroying_residual_input(self):
        value = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10 + 0.1
        original_value = value.clone()
        weight = torch.tensor([0.8, 0.9, 1.1, 1.2])
        expected = F.rms_norm(value.clone(), (4,), weight, 1e-5)
        rows_seen = []
        original = F.rms_norm

        def tracking_rms_norm(chunk, normalized_shape, chunk_weight=None, eps=None):
            rows_seen.append(chunk.shape[0])
            return original(chunk, normalized_shape, chunk_weight, eps)

        actual = apply_rms_norm_chunked(
            value,
            (4,),
            weight,
            1e-5,
            tracking_rms_norm,
            max_temp_bytes=32,
        )

        self.assertNotEqual(actual.data_ptr(), value.data_ptr())
        self.assertEqual(rows_seen, [2, 2, 2, 2])
        torch.testing.assert_close(value, original_value)
        torch.testing.assert_close(actual, expected)

    def test_non_contiguous_input_uses_safe_out_of_place_fallback(self):
        value = torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1)
        actual = apply_rms_norm_chunked(
            value, (4,), None, 1e-5, F.rms_norm, max_temp_bytes=32
        )
        expected = F.rms_norm(value, (4,), None, 1e-5)
        self.assertNotEqual(actual.data_ptr(), value.data_ptr())
        torch.testing.assert_close(actual, expected)

    def test_only_v100_fsdp_selects_chunked_rms_norm(self):
        worker = types.SimpleNamespace(parallel_dict={"is_fsdp": True})
        self.assertTrue(
            should_use_v100_chunked_rms_norm(worker, cuda_capability=(7, 0))
        )
        self.assertFalse(
            should_use_v100_chunked_rms_norm(worker, cuda_capability=(8, 0))
        )
        worker.parallel_dict["is_fsdp"] = False
        self.assertFalse(
            should_use_v100_chunked_rms_norm(worker, cuda_capability=(7, 0))
        )

    def test_v100_rms_norm_default_budget_is_16_mib_and_can_be_overridden(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(v100_rms_norm_chunk_bytes(), 16 * 1024 * 1024)
        with mock.patch.dict(
            os.environ,
            {"RAYLIGHT_RMS_NORM_CHUNK_BYTES": str(8 * 1024 * 1024)},
            clear=True,
        ):
            self.assertEqual(v100_rms_norm_chunk_bytes(), 8 * 1024 * 1024)
        with mock.patch.dict(os.environ, {"RAYLIGHT_RMS_NORM_CHUNK_BYTES": "bad"}, clear=True):
            self.assertEqual(v100_rms_norm_chunk_bytes(), 16 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
