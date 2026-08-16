import inspect
import os
import types
import unittest
from unittest import mock

import torch

from raylight.comfy_dist.model_patcher import (
    FSDPModelPatcher,
    _align_dense_meta_dtypes_from_state_dict,
    _can_release_fsdp_state_dict_during_load,
    patch_fsdp,
    select_fsdp_cpu_offload_policy,
    select_fsdp_mixed_precision_policy,
)


class FSDPMixedPrecisionTests(unittest.TestCase):
    def test_explicit_bf16_manual_cast_selects_bf16_unshard_and_forward(self):
        model_patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(manual_cast_dtype=torch.bfloat16)
        )

        policy = select_fsdp_mixed_precision_policy(model_patcher)

        self.assertIsNotNone(policy)
        self.assertIs(policy.param_dtype, torch.bfloat16)
        self.assertIsNone(policy.output_dtype)
        self.assertTrue(policy.cast_forward_inputs)

    def test_native_bf16_model_dtype_selects_bf16_when_manual_cast_is_none(self):
        base_model = types.SimpleNamespace(
            manual_cast_dtype=None,
            get_dtype=lambda: torch.bfloat16,
        )
        model_patcher = types.SimpleNamespace(model=base_model)

        policy = select_fsdp_mixed_precision_policy(model_patcher)

        self.assertIsNotNone(policy)
        self.assertIs(policy.param_dtype, torch.bfloat16)

    def test_explicit_fsdp_dtype_marker_survives_quant_model_dtype_fallback(self):
        base_model = types.SimpleNamespace(
            manual_cast_dtype=None,
            get_dtype=lambda: torch.float32,
        )
        model_patcher = types.SimpleNamespace(
            model=base_model,
            fsdp_param_dtype=torch.bfloat16,
        )

        policy = select_fsdp_mixed_precision_policy(model_patcher)
        self.assertIsNotNone(policy)
        self.assertIs(policy.param_dtype, torch.bfloat16)

    def test_default_or_fp32_does_not_change_fsdp_parameter_dtype(self):
        for dtype in (None, torch.float32):
            with self.subTest(dtype=dtype):
                model_patcher = types.SimpleNamespace(
                    model=types.SimpleNamespace(manual_cast_dtype=dtype)
                )
                self.assertIsNone(select_fsdp_mixed_precision_policy(model_patcher))

    def test_fp16_is_not_enabled_by_the_v100_bf16_safety_path(self):
        model_patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(manual_cast_dtype=torch.float16)
        )
        self.assertIsNone(select_fsdp_mixed_precision_policy(model_patcher))

    def test_cpu_offload_disabled_does_not_install_an_offload_policy(self):
        model_patcher = types.SimpleNamespace(is_cpu_offload=False)

        self.assertIsNone(select_fsdp_cpu_offload_policy(model_patcher))

    def test_cpu_offload_uses_pageable_memory_by_default_on_windows(self):
        model_patcher = types.SimpleNamespace(is_cpu_offload=True)

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAYLIGHT_FSDP_CPU_OFFLOAD_PIN_MEMORY", None)
            policy = select_fsdp_cpu_offload_policy(model_patcher, platform="win32")

        self.assertIsNotNone(policy)
        self.assertFalse(policy.pin_memory)

    def test_cpu_offload_pin_memory_can_be_enabled_explicitly(self):
        model_patcher = types.SimpleNamespace(is_cpu_offload=True)

        with mock.patch.dict(
            os.environ,
            {"RAYLIGHT_FSDP_CPU_OFFLOAD_PIN_MEMORY": "1"},
        ):
            policy = select_fsdp_cpu_offload_policy(model_patcher, platform="win32")

        self.assertTrue(policy.pin_memory)

    def test_quant_state_dict_is_released_incrementally_without_excluded_modules(self):
        self.assertTrue(_can_release_fsdp_state_dict_during_load(set()))
        self.assertIn("release_sd=release_state_dict", inspect.getsource(patch_fsdp))
        self.assertIn("self.fsdp_state_dict = None", inspect.getsource(patch_fsdp))

    def test_quant_state_dict_is_retained_when_excluded_modules_need_it_later(self):
        self.assertFalse(_can_release_fsdp_state_dict_during_load({object()}))


    def test_lora_clone_preserves_explicit_fsdp_parameter_dtype(self):
        patcher = object.__new__(FSDPModelPatcher)
        patcher.rank = 0
        patcher.fsdp_state_dict = {"weight": object()}
        patcher.device_mesh = object()
        patcher.is_cpu_offload = False
        patcher._has_quantized_dtensor_shards = True
        patcher.fsdp_param_dtype = torch.bfloat16
        parent_clone = object.__new__(FSDPModelPatcher)

        with mock.patch(
            "comfy.model_patcher.ModelPatcher.clone",
            return_value=parent_clone,
        ):
            cloned = FSDPModelPatcher.clone(patcher)

        self.assertIs(cloned.fsdp_param_dtype, torch.bfloat16)
        self.assertEqual(cloned.rank, 0)
        self.assertIs(cloned.fsdp_state_dict, patcher.fsdp_state_dict)


    def test_dense_meta_parameters_keep_checkpoint_bf16_but_quant_weight_stays_logical(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Linear(4, 8, device="meta", dtype=torch.float32)
                self.quant = torch.nn.Linear(4, 8, device="meta", dtype=torch.float32)

        model = Tiny()
        state = {
            "dense.weight": torch.empty(8, 4, dtype=torch.bfloat16),
            "dense.bias": torch.empty(8, dtype=torch.bfloat16),
            "quant.weight": torch.empty(8, 4, dtype=torch.bfloat16),
            "quant.bias": torch.empty(8, dtype=torch.bfloat16),
            "quant.weight_scale": torch.ones((), dtype=torch.float32),
        }

        changed = _align_dense_meta_dtypes_from_state_dict(model, state)

        self.assertEqual(changed, 3)
        self.assertIs(model.dense.weight.dtype, torch.bfloat16)
        self.assertIs(model.dense.bias.dtype, torch.bfloat16)
        self.assertIs(model.quant.weight.dtype, torch.float32)
        self.assertIs(model.quant.bias.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
