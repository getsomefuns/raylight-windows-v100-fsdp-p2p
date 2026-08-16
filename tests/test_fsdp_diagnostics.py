import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


MODULE_PATH = Path(__file__).parents[1] / "src/raylight/comfy_dist/fsdp_utils.py"


def load_module():
    spec = importlib.util.spec_from_file_location("raylight_fsdp_utils_diagnostics_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class FakeQuantizedLocal:
    def __init__(self):
        self._qdata = torch.zeros(4, dtype=torch.uint8)
        self._layout_cls = "TensorCoreFP8Layout"


class FakeDTensor:
    dtype = torch.float16

    def __init__(self):
        self._local = FakeQuantizedLocal()

    def numel(self):
        return 16

    def element_size(self):
        return 2

    def to_local(self):
        return self._local


class FakeModel:
    def __init__(self):
        self._params = [FakeDTensor(), torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))]

    def parameters(self):
        return iter(self._params)


class FSDPDiagnosticsTests(unittest.TestCase):
    def test_parameter_diagnostics_separates_logical_local_and_unsharded_bytes(self):
        module = load_module()

        with mock.patch.object(module, "DTensor", FakeDTensor):
            result = module.summarize_fsdp_parameters(FakeModel())

        self.assertEqual(result["parameter_count"], 2)
        self.assertEqual(result["dtensor_count"], 1)
        self.assertEqual(result["logical_parameter_bytes"], 44)
        self.assertEqual(result["local_payload_bytes"], 16)
        self.assertEqual(result["unsharded_parameter_bytes"], 12)
        self.assertEqual(result["layouts"], {"TensorCoreFP8Layout": 1})
        self.assertEqual(result["logical_dtypes"], {"float16": 1, "float32": 1})
        self.assertEqual(result["storage_dtypes"], {"float32": 1, "uint8": 1})

    def test_format_diagnostics_is_compact_and_stable(self):
        module = load_module()
        diagnostics = {
            "parameter_count": 2,
            "dtensor_count": 1,
            "logical_parameter_bytes": 2 * 1024 * 1024,
            "local_payload_bytes": 1 * 1024 * 1024,
            "unsharded_parameter_bytes": 4096,
            "layouts": {"TensorCoreFP8Layout": 1},
            "logical_dtypes": {"float16": 1, "float32": 1},
            "storage_dtypes": {"float32": 1, "uint8": 1},
        }

        self.assertEqual(
            module.format_fsdp_diagnostics(diagnostics),
            "params=2 dtensors=1 logical=2.00MiB local_payload=1.00MiB "
            "unsharded=0.00MiB layouts=TensorCoreFP8Layout:1 "
            "logical_dtypes=float16:1,float32:1 storage_dtypes=float32:1,uint8:1",
        )

    def test_mixed_dtype_selection_replicates_smaller_dtype_payload(self):
        module = load_module()
        primary = torch.nn.Parameter(torch.zeros(1024, dtype=torch.bfloat16))
        auxiliary = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))

        replicated = module.select_mixed_dtype_ignored_params([primary, auxiliary], set())

        self.assertEqual(replicated, {auxiliary})

    def test_mixed_dtype_selection_keeps_uniform_or_already_ignored_params(self):
        module = load_module()
        first = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
        second = torch.nn.Parameter(torch.zeros(8, dtype=torch.bfloat16))
        auxiliary = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        self.assertEqual(module.select_mixed_dtype_ignored_params([first, second], set()), set())
        self.assertEqual(
            module.select_mixed_dtype_ignored_params([first, auxiliary], {auxiliary}),
            set(),
        )

    def test_selected_parameter_summary_reports_dtype_suffix_and_largest_names(self):
        module = load_module()

        class MixedModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(16, dtype=torch.bfloat16))
                self.bias = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))

        model = MixedModule()
        result = module.summarize_selected_parameters(
            model,
            {model.weight, model.bias},
            largest_limit=1,
        )

        self.assertEqual(result["parameter_count"], 2)
        self.assertEqual(result["parameter_bytes"], 48)
        self.assertEqual(result["dtypes"]["bfloat16"], {"count": 1, "bytes": 32})
        self.assertEqual(result["dtypes"]["float32"], {"count": 1, "bytes": 16})
        self.assertEqual(result["suffixes"], {"bias": 1, "weight": 1})
        self.assertEqual(result["largest"][0]["name"], "weight")
        self.assertEqual(result["largest"][0]["bytes"], 32)




    def test_quantized_state_dict_load_reuses_identical_storage_only_in_context(self):
        module = load_module()
        layout_name = "TensorCoreFP8E4M3Layout"
        layout_cls = module.get_layout_class(layout_name)
        qdata = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
        params = layout_cls.Params(
            scale=torch.ones((), dtype=torch.float32),
            orig_dtype=torch.bfloat16,
            orig_shape=(8, 8),
        )
        quantized = module.QuantizedTensor(qdata, layout_name, params)
        storage_pointer = quantized._qdata.data_ptr()

        copied = torch.ops.aten._to_copy.default(
            quantized, device=torch.device("cpu"), dtype=torch.bfloat16
        )
        self.assertNotEqual(copied._qdata.data_ptr(), storage_pointer)

        with module.quantized_state_dict_zero_copy():
            reused = torch.ops.aten._to_copy.default(
                quantized, device=torch.device("cpu"), dtype=torch.bfloat16
            )
            self.assertEqual(reused._qdata.data_ptr(), storage_pointer)

        copied_after = torch.ops.aten._to_copy.default(
            quantized, device=torch.device("cpu"), dtype=torch.bfloat16
        )
        self.assertNotEqual(copied_after._qdata.data_ptr(), storage_pointer)

    def test_bf16_policy_aligns_fp8_logical_dtype_without_copying_storage(self):
        module = load_module()
        layout_name = "TensorCoreFP8E4M3Layout"
        layout_cls = module.get_layout_class(layout_name)
        qdata = torch.zeros((8, 8), dtype=torch.float8_e4m3fn)
        params = layout_cls.Params(
            scale=torch.ones((), dtype=torch.float32),
            orig_dtype=torch.float32,
            orig_shape=(8, 8),
        )
        quantized = module.QuantizedTensor(qdata, layout_name, params)

        class QuantLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "weight",
                    torch.nn.Parameter(quantized, requires_grad=False),
                )
                self.bias = torch.nn.Parameter(
                    torch.zeros(8, dtype=torch.bfloat16),
                    requires_grad=False,
                )

        model = QuantLinear()
        storage_pointer = model.weight._qdata.data_ptr()
        aligned = module.align_fp8_logical_dtype(model, torch.bfloat16)

        self.assertEqual(aligned, 1)
        self.assertIs(model.weight.dtype, torch.bfloat16)
        self.assertIs(model.weight._params.orig_dtype, torch.bfloat16)
        self.assertIs(model.weight.storage_dtype, torch.float8_e4m3fn)
        self.assertEqual(model.weight._qdata.data_ptr(), storage_pointer)
        self.assertEqual(
            module.select_mixed_dtype_ignored_params(list(model.parameters()), set()),
            set(),
        )
    def test_all_gather_input_summary_uses_storage_dtype_and_world_size(self):
        module = load_module()
        inputs = [
            torch.empty(72 * 1024 * 1024, dtype=torch.float8_e4m3fn),
            torch.empty(1024, dtype=torch.bfloat16),
        ]

        summary = module.summarize_all_gather_inputs(inputs, world_size=2)

        self.assertEqual(summary[0]["dtype"], "float8_e4m3fn")
        self.assertEqual(summary[0]["output_bytes"], 144 * 1024 * 1024)
        self.assertEqual(summary[1]["output_bytes"], 4096)


if __name__ == "__main__":
    unittest.main()
