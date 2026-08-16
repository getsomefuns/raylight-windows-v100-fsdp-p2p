import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[3]))

from comfy_kitchen.tensor.base import QuantizedTensor
from raylight.expansion.comfyui_lazytensors.lazy_tensor import wrap_state_dict_lazy
from raylight.expansion.comfyui_lazytensors.ops import SafetensorOps


class LazyQuantizedSafetensorOpsTests(unittest.TestCase):
    def test_fp8_linear_preserves_weight_scale_and_quant_metadata(self):
        qweight = torch.tensor(
            [[4.0, -2.0], [1.0, 3.0]], dtype=torch.float8_e4m3fn
        )
        scale = torch.tensor(0.25, dtype=torch.float32)
        bias = torch.tensor([0.5, -0.25], dtype=torch.float32)
        state_dict = wrap_state_dict_lazy(
            {
                "weight": qweight,
                "bias": bias,
                "weight_scale": scale,
                "comfy_quant": torch.tensor(
                    list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")),
                    dtype=torch.uint8,
                ),
            }
        )
        layer = SafetensorOps.Linear(
            2, 2, bias=True, device=torch.device("cpu"), dtype=torch.float32
        )

        layer.load_state_dict(state_dict, strict=True)
        value = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
        actual = layer(value)
        expected = torch.nn.functional.linear(
            value, qweight.float() * scale, bias
        )

        self.assertIsInstance(layer.weight, QuantizedTensor)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()