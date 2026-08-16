import unittest

import torch
import torch.nn.functional as F

from raylight.comfy_dist.quant_ops import (
    apply_activation_inplace_chunked,
    bf16_linear_fp32_chunked,
)


class ChunkedActivationTests(unittest.TestCase):
    def test_activation_is_chunked_and_reuses_input_storage(self):
        value = torch.linspace(-3, 3, 40, dtype=torch.float32).reshape(10, 4)
        expected = F.gelu(value.clone(), approximate="tanh")
        original_ptr = value.data_ptr()
        observed_rows = []

        def activation(chunk):
            observed_rows.append(chunk.shape[0])
            return F.gelu(chunk, approximate="tanh")

        # Four float32 values consume 16 bytes per row. A 32-byte budget
        # must process ten rows as five two-row temporary chunks.
        actual = apply_activation_inplace_chunked(
            value,
            activation,
            max_temp_bytes=32,
        )

        self.assertIs(actual, value)
        self.assertEqual(actual.data_ptr(), original_ptr)
        self.assertEqual(observed_rows, [2, 2, 2, 2, 2])
        torch.testing.assert_close(actual, expected)

    def test_activation_rejects_a_non_positive_budget(self):
        value = torch.zeros((1, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "positive"):
            apply_activation_inplace_chunked(value, torch.relu, 0)

    def test_shape_changing_activation_returns_bounded_output(self):
        value = torch.linspace(-3, 3, 48, dtype=torch.float32).reshape(6, 8)

        def swiglu(chunk):
            gate, up = chunk.chunk(2, dim=-1)
            return F.silu(gate) * up

        expected = swiglu(value.clone())
        original_ptr = value.data_ptr()
        observed_rows = []

        def observed_swiglu(chunk):
            observed_rows.append(chunk.shape[0])
            return swiglu(chunk)

        actual = apply_activation_inplace_chunked(
            value,
            observed_swiglu,
            max_temp_bytes=64,
        )

        self.assertEqual(actual.shape, (6, 4))
        self.assertNotEqual(actual.data_ptr(), original_ptr)
        self.assertEqual(observed_rows, [2, 2, 2])
        torch.testing.assert_close(actual, expected)

    def test_activation_requires_inference_mode(self):
        value = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "inference"):
            apply_activation_inplace_chunked(value, torch.relu, 16)


    def test_bf16_dense_linear_casts_output_rows_in_bounded_chunks(self):
        torch.manual_seed(7)
        value = torch.randn(2, 3, 4, dtype=torch.float32)
        weight = torch.randn(7, 4, dtype=torch.bfloat16)
        bias = torch.randn(7, dtype=torch.bfloat16)
        observed_rows = []

        def linear(input_tensor, weight_chunk, bias_chunk):
            observed_rows.append(weight_chunk.shape[0])
            return F.linear(input_tensor, weight_chunk, bias_chunk)

        actual = bf16_linear_fp32_chunked(
            value,
            weight,
            bias,
            max_temp_bytes=80,
            linear_fn=linear,
        )
        expected = F.linear(value, weight.float(), bias.float())

        self.assertEqual(observed_rows, [2, 2, 2, 1])
        torch.testing.assert_close(actual, expected)

    def test_bf16_dense_linear_rejects_non_positive_budget(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            bf16_linear_fp32_chunked(
                torch.zeros(1, 2), torch.zeros(3, 2, dtype=torch.bfloat16), None, 0
            )


if __name__ == "__main__":
    unittest.main()
