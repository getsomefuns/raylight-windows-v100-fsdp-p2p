import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "src/raylight/comfy_dist/fsdp_utils.py"


def load_module():
    spec = importlib.util.spec_from_file_location("raylight_fsdp_utils_topology_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Linear(8, 8, bias=False)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(8, 16, bias=False),
            torch.nn.Linear(16, 8, bias=False),
        )

    def forward(self, value):
        return self.ff(self.attn(value))


class LTXLikeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([TransformerBlock(), TransformerBlock()])
        self.proj_out = torch.nn.Linear(8, 8, bias=False)

    def forward(self, value):
        for block in self.transformer_blocks:
            value = block(value)
        return self.proj_out(value)


class FusedFeedForward(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 16, bias=False),
            torch.nn.Linear(16, 8, bias=False),
        )

    def forward(self, value):
        return self.net[1](torch.nn.functional.gelu(self.net[0](value)))


class FusedPathModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FusedFeedForward()

    def forward(self, value):
        return self.ff(value)


class RootOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(8, 8))

    def forward(self, value):
        return value @ self.weight


class LTXFSDPTopologyTests(unittest.TestCase):
    def test_ltx_like_topology_has_non_root_wrappers_before_root(self):
        module = load_module()
        shard_order = module.collect_bottom_up_shard_order(LTXLikeModel())
        names = [name for name, _wrapped in shard_order]

        self.assertEqual(names[-1], "")
        self.assertTrue(any(name.startswith("transformer_blocks.0") for name in names[:-1]))
        self.assertTrue(any(name.startswith("transformer_blocks.1") for name in names[:-1]))
        self.assertIn("transformer_blocks.0.attn", names)
        self.assertIn("transformer_blocks.0.ff.0", names)
        self.assertIn("transformer_blocks.0.ff.1", names)

        self.assertGreater(module.validate_inference_shard_order(shard_order), 0)

    def test_storage_sequential_bypassed_by_fused_forward_is_not_wrapped(self):
        module = load_module()
        shard_order = module.collect_bottom_up_shard_order(FusedPathModel())
        names = [name for name, _wrapped in shard_order]

        self.assertIn("ff", names)
        self.assertNotIn("ff.net", names)
        self.assertIn("ff.net.0", names)
        self.assertIn("ff.net.1", names)

    def test_root_only_topology_is_rejected_before_fsdp_mutation(self):
        module = load_module()
        shard_order = module.collect_bottom_up_shard_order(RootOnlyModel())

        self.assertEqual([name for name, _wrapped in shard_order], [""])
        with self.assertRaisesRegex(ValueError, "root-only"):
            module.validate_inference_shard_order(shard_order)

    def test_root_only_model_is_rejected_before_fully_shard_is_called(self):
        module = load_module()

        with mock.patch.object(module, "fully_shard") as fully_shard:
            with self.assertRaisesRegex(ValueError, "root-only"):
                module.fully_shard_bottom_up(
                    RootOnlyModel(),
                    fsdp_kwargs={"reshard_after_forward": True},
                    native_ignore_scale=True,
                )

        fully_shard.assert_not_called()

    def test_bottom_up_wrapper_count_matches_every_selected_unit(self):
        module = load_module()
        model = LTXLikeModel()
        expected = len(module.collect_bottom_up_shard_order(model))

        with mock.patch.object(module, "fully_shard") as fully_shard:
            actual = module.fully_shard_bottom_up(
                model,
                fsdp_kwargs={"reshard_after_forward": True},
                native_ignore_scale=True,
            )

        self.assertEqual(actual, expected)
        self.assertEqual(fully_shard.call_count, expected)

    def test_low_peak_unshard_uses_default_stream_for_every_fsdp_wrapper(self):
        module = load_module()

        class FakeFSDPModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = torch.nn.Linear(2, 2)
                self.values = []

            def _set_unshard_async_op(self, enabled):
                self.values.append(enabled)

        model = FakeFSDPModule()
        with mock.patch.object(module, "FSDPModule", FakeFSDPModule):
            count = module.enable_low_peak_fsdp_unshard(model)

        self.assertEqual(count, 1)
        self.assertEqual(model.values, [True])
        with mock.patch.object(module, "FSDPModule", FakeFSDPModule):
            self.assertEqual(module.enable_low_peak_fsdp_unshard(torch.nn.Linear(2, 2)), 0)



if __name__ == "__main__":
    unittest.main()
