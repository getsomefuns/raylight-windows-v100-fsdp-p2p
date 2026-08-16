from contextlib import nullcontext
import unittest

import torch

from raylight.comfy_dist import sd as ray_sd
from raylight.comfy_dist import fsdp_utils


class _ToyDiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(4, 8, dtype=torch.float32)
        self.quant = torch.nn.Linear(4, 8, dtype=torch.float32)


class _CastingModel:
    """Reproduces Comfy mixed-precision loading of plain weights."""

    def __init__(self):
        self.diffusion_model = _ToyDiffusionModel()

    def load_model_weights(self, state_dict, prefix, assign=False):
        dense = state_dict.pop("dense.weight")
        quant = state_dict.pop("quant.weight")
        state_dict.pop("quant.comfy_quant")
        self.diffusion_model.dense.weight = torch.nn.Parameter(
            dense.to(torch.float32), requires_grad=False
        )
        self.diffusion_model.quant.weight = torch.nn.Parameter(
            quant.to(torch.float32), requires_grad=False
        )
        return self


class FSDPStateStorageTests(unittest.TestCase):
    def test_plain_bf16_checkpoint_weight_survives_compute_dtype_loader(self):
        model = _CastingModel()
        dense_checkpoint = torch.randn(8, 4, dtype=torch.bfloat16)
        quant_checkpoint = torch.randn(8, 4, dtype=torch.bfloat16)
        state_dict = {
            "dense.weight": dense_checkpoint,
            "quant.weight": quant_checkpoint,
            "quant.comfy_quant": torch.tensor([1], dtype=torch.uint8),
        }

        load_preserving_storage = getattr(
            ray_sd,
            "_load_model_weights_preserving_plain_bf16_storage",
            lambda model, state_dict, prefix, assign=False: model.load_model_weights(
                state_dict, prefix, assign=assign
            ),
        )
        changed = load_preserving_storage(model, state_dict, "", assign=True)

        self.assertEqual(changed, 1)
        self.assertIs(model.diffusion_model.dense.weight.dtype, torch.bfloat16)
        torch.testing.assert_close(
            model.diffusion_model.dense.weight,
            dense_checkpoint,
        )
        self.assertIs(model.diffusion_model.quant.weight.dtype, torch.float32)
        self.assertEqual(state_dict, {})


    def test_plain_bf16_state_assign_bypasses_compute_dtype_recast(self):
        import comfy.ops

        operations = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.float32)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = operations.Linear(4, 8, bias=True)

        model = Tiny()
        state_dict = {
            "dense.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "dense.bias": torch.randn(8, dtype=torch.bfloat16),
        }
        model.load_state_dict(state_dict.copy(), strict=False, assign=True)
        self.assertIs(model.dense.weight.dtype, torch.float32)

        preserve_assign = getattr(
            fsdp_utils,
            "plain_bf16_state_dict_assign",
            lambda model, state_dict: nullcontext(),
        )

        with preserve_assign(model, state_dict):
            model.load_state_dict(state_dict.copy(), strict=False, assign=True)

        self.assertIs(model.dense.weight.dtype, torch.bfloat16)
        self.assertIs(model.dense.bias.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
