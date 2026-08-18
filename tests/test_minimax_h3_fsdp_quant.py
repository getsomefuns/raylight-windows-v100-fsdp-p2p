import types
import unittest
from unittest import mock

import torch

from raylight.comfy_dist.fsdp_utils import (
    collect_odd_dim0_ignored_params,
    collect_scalar_ignored_params,
)
from raylight.comfy_dist.kitchen_patches.fp8 import (
    fp8_linear_fallback_chunked,
    install_fp8_patches,
    restore_fp8_patches,
)
from raylight.comfy_dist.model_patcher import select_fsdp_mixed_precision_policy
from raylight.distributed_worker.parallel_group_manager import validate_hybrid_topology
from raylight.distributed_worker.windows_p2p import make_all_gather_into_tensor_router

try:
    from comfy.quant_ops import QuantizedTensor, get_layout_class
except ImportError:  # ComfyUI versions before comfy.quant_ops
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class


class MiniMaxH3FSDPQuantTests(unittest.TestCase):
    def tearDown(self):
        restore_fp8_patches()

    def test_fp8_rows_reconstruct_minimax_hidden_and_ffn_dimensions(self):
        layout_name = "TensorCoreFP8E4M3Layout"
        layout = get_layout_class(layout_name)
        scale = torch.tensor(0.25, dtype=torch.float32)
        install_fp8_patches()

        for global_rows in (5376, 14336):
            with self.subTest(global_rows=global_rows):
                local_rows = global_rows // 2
                local_qdata = torch.zeros(
                    (local_rows, 1), dtype=torch.float8_e4m3fn
                )
                local = QuantizedTensor(
                    local_qdata,
                    layout_name,
                    layout.Params(
                        scale=scale,
                        orig_dtype=torch.float32,
                        orig_shape=(local_rows, 1),
                    ),
                )
                gathered_qdata = torch.cat((local_qdata, local_qdata), dim=0)

                gathered, _inner = local.fsdp_post_all_gather(
                    (gathered_qdata,), (scale,), torch.float32
                )

                self.assertEqual(tuple(gathered.shape), (global_rows, 1))
                self.assertEqual(gathered._params.orig_shape, (global_rows, 1))
                self.assertEqual(tuple(gathered._qdata.shape), (global_rows, 1))

    def test_odd_and_scalar_fp32_islands_are_replicated_but_even_weights_shard(self):
        class MiniMaxShapeProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = torch.nn.Parameter(
                    torch.empty((5376, 1), device="meta", dtype=torch.float32)
                )
                self.ffn = torch.nn.Parameter(
                    torch.empty((14336, 1), device="meta", dtype=torch.float32)
                )
                self.odd_aux = torch.nn.Parameter(
                    torch.empty((5375, 1), device="meta", dtype=torch.float32)
                )
                self.scalar = torch.nn.Parameter(
                    torch.empty((), device="meta", dtype=torch.float32)
                )

        model = MiniMaxShapeProbe()
        odd = collect_odd_dim0_ignored_params(model)
        scalar = collect_scalar_ignored_params(model)

        self.assertEqual(odd, {model.odd_aux})
        self.assertEqual(scalar, {model.scalar})
        self.assertNotIn(model.hidden, odd | scalar)
        self.assertNotIn(model.ffn, odd | scalar)

    def test_v100_fp32_compute_keeps_fp8_storage_without_bf16_mixed_policy(self):
        model_patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(manual_cast_dtype=torch.float32)
        )

        self.assertIsNone(select_fsdp_mixed_precision_policy(model_patcher))

    def test_v100_safe_fp16_compute_keeps_fp8_storage_without_bf16_mixed_policy(self):
        model_patcher = types.SimpleNamespace(
            model=types.SimpleNamespace(manual_cast_dtype=torch.float16)
        )
        qweight = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0]], dtype=torch.float8_e4m3fn
        )
        original_storage = qweight.clone()
        x = torch.tensor([[2.0, -1.0]], dtype=torch.float16)

        result = fp8_linear_fallback_chunked(
            x,
            qweight,
            torch.tensor(0.5, dtype=torch.float32),
            None,
            torch.float16,
            max_temp_bytes=16,
        )

        self.assertIsNone(select_fsdp_mixed_precision_policy(model_patcher))
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(qweight.float(), original_storage.float())
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(torch.isfinite(result).all())

    def test_dual_v100_minimax_hybrid_topology_is_accepted(self):
        config = {
            "is_fsdp": True,
            "is_xdit": True,
            "ulysses_degree": 2,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "shard_size": 2,
        }

        self.assertEqual(validate_hybrid_topology(2, 2, config), "hybrid")

    def test_cuda_all_gather_routes_to_p2p_instead_of_gloo(self):
        endpoint = mock.Mock()
        endpoint.world_size = 2
        endpoint.all_gather_into_tensor.return_value = "p2p-work"
        fallback = mock.Mock(name="gloo_fallback")
        router = make_all_gather_into_tensor_router(endpoint, fallback)
        output = mock.Mock(is_cuda=True)
        source = mock.Mock(is_cuda=True)

        with mock.patch(
            "raylight.distributed_worker.windows_p2p.dist.get_world_size",
            return_value=2,
        ):
            result = router(output, source, group="fsdp", async_op=True)

        self.assertEqual(result, "p2p-work")
        endpoint.all_gather_into_tensor.assert_called_once_with(
            output, source, async_op=True
        )
        fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
