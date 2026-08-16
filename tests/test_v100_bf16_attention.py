import os
import types
import unittest
from unittest import mock

import torch

from raylight.comfy_dist.quant_ops import (
    should_use_v100_bf16_chunked_attention,
    temporary_v100_bf16_chunked_attention,
)


class V100BF16AttentionTests(unittest.TestCase):
    @staticmethod
    def _worker(dtype=torch.bfloat16, is_fsdp=True):
        return types.SimpleNamespace(
            parallel_dict={"is_fsdp": is_fsdp},
            model=types.SimpleNamespace(fsdp_param_dtype=dtype),
        )

    def test_v100_bf16_fsdp_selects_chunked_attention(self):
        self.assertTrue(
            should_use_v100_bf16_chunked_attention(
                self._worker(), cuda_capability=(7, 0)
            )
        )

    def test_non_bf16_non_fsdp_and_newer_gpu_keep_existing_backend(self):
        cases = (
            (self._worker(dtype=torch.float32), (7, 0)),
            (self._worker(is_fsdp=False), (7, 0)),
            (self._worker(), (8, 0)),
        )
        for worker, capability in cases:
            with self.subTest(worker=worker, capability=capability):
                self.assertFalse(
                    should_use_v100_bf16_chunked_attention(
                        worker, cuda_capability=capability
                    )
                )

    def test_context_replaces_both_backends_and_restores_after_error(self):
        import comfy.ldm.modules.attention as attention

        old_unmasked = object()
        old_masked = object()
        sub_quad_backend = mock.Mock(return_value="sub_quad")
        split_backend = mock.Mock(return_value="split")
        efficient_backend = mock.Mock(return_value="efficient")
        with (
            mock.patch.object(attention, "optimized_attention", old_unmasked),
            mock.patch.object(attention, "optimized_attention_masked", old_masked),
            mock.patch.object(attention, "attention_sub_quad", sub_quad_backend),
            mock.patch.object(attention, "attention_split", split_backend),
            mock.patch.object(attention, "efficient_dot_product_attention", efficient_backend),
            mock.patch.dict(os.environ, {"RAYLIGHT_ATTENTION_LONG_THRESHOLD_BYTES": "16"}),
            mock.patch(
                "raylight.comfy_dist.quant_ops.torch.cuda.get_device_capability",
                return_value=(7, 0),
            ),
            mock.patch(
                "raylight.comfy_dist.quant_ops.torch.cuda.is_available",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "sample failed"):
                with temporary_v100_bf16_chunked_attention(self._worker()) as enabled:
                    self.assertTrue(enabled)
                    backend = attention.optimized_attention
                    self.assertIs(attention.optimized_attention_masked, backend)
                    self.assertEqual(
                        backend(torch.zeros(2), None, None, 1), "sub_quad"
                    )
                    self.assertEqual(
                        backend(torch.zeros(8), None, None, 1), "sub_quad"
                    )
                    key_t = torch.zeros((1, 1, 5))
                    value = torch.zeros((1, 5, 1))
                    self.assertEqual(
                        attention.efficient_dot_product_attention(
                            torch.zeros(8), key_t, value,
                            query_chunk_size=1024, kv_chunk_size=None,
                        ),
                        "efficient",
                    )
                    _, call_kwargs = efficient_backend.call_args
                    self.assertEqual(call_kwargs["query_chunk_size"], 64)
                    self.assertEqual(call_kwargs["kv_chunk_size"], 5)
                    raise RuntimeError("sample failed")

            self.assertIs(attention.optimized_attention, old_unmasked)
            self.assertIs(attention.optimized_attention_masked, old_masked)
            self.assertIs(attention.efficient_dot_product_attention, efficient_backend)


if __name__ == "__main__":
    unittest.main()
