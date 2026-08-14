import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src/raylight/distributed_worker/windows_p2p.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("raylight_windows_p2p_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class WindowsP2PTests(unittest.TestCase):
    def test_copy_plan_matches_two_rank_all_to_all_layout(self):
        module = load_module()

        self.assertEqual(
            module.copy_plan(rank=0, total_bytes=16),
            ((0, 8, "input", 0, 8), (8, 16, "peer", 0, 8)),
        )
        self.assertEqual(
            module.copy_plan(rank=1, total_bytes=16),
            ((0, 8, "peer", 0, 8), (8, 16, "input", 8, 16)),
        )
        self.assertEqual(module.send_slice(rank=0, total_bytes=16), (8, 16))
        self.assertEqual(module.send_slice(rank=1, total_bytes=16), (0, 8))

    def test_copy_plan_rejects_invalid_rank_or_uneven_payload(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "rank must be 0 or 1"):
            module.copy_plan(rank=2, total_bytes=16)
        with self.assertRaisesRegex(ValueError, "divisible by 2"):
            module.copy_plan(rank=0, total_bytes=15)

    def test_validate_collective_accepts_contiguous_equal_cuda_contract(self):
        module = load_module()
        source = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float32,
            numel=lambda: 8,
            element_size=lambda: 4,
        )
        output = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float32,
            numel=lambda: 8,
            element_size=lambda: 4,
        )

        self.assertEqual(module.validate_collective(output, source, capacity_bytes=16), 32)

    def test_validate_collective_rejects_shape_dtype_capacity_and_split(self):
        module = load_module()
        source = mock.Mock(
            is_cuda=False,
            is_contiguous=lambda: True,
            dtype=torch.float32,
            numel=lambda: 8,
            element_size=lambda: 4,
        )
        output = mock.Mock(
            is_cuda=False,
            is_contiguous=lambda: True,
            dtype=torch.float32,
            numel=lambda: 8,
            element_size=lambda: 4,
        )

        with self.assertRaisesRegex(ValueError, "CUDA tensors"):
            module.validate_collective(output, source, capacity_bytes=32)

        source.is_cuda = output.is_cuda = True
        with self.assertRaisesRegex(ValueError, "same dtype"):
            output.dtype = torch.float16
            module.validate_collective(output, source, capacity_bytes=32)
        output.dtype = torch.float32
        with self.assertRaisesRegex(ValueError, "same number of elements"):
            output.numel = lambda: 4
            module.validate_collective(output, source, capacity_bytes=32)
        output.numel = lambda: 8
        with self.assertRaisesRegex(ValueError, "capacity"):
            module.validate_collective(output, source, capacity_bytes=15)
        with self.assertRaisesRegex(ValueError, "divisible by 2"):
            odd = mock.Mock(
                is_cuda=True,
                is_contiguous=lambda: True,
                dtype=torch.uint8,
                numel=lambda: 3,
                element_size=lambda: 1,
            )
            module.validate_collective(odd, odd, capacity_bytes=32)

    def test_poisoned_endpoint_rejects_future_collectives(self):
        module = load_module()
        endpoint = module.CudaP2PAllToAll.__new__(module.CudaP2PAllToAll)
        endpoint._poisoned_reason = "peer timeout"

        with self.assertRaisesRegex(module.P2PGroupError, "peer timeout"):
            endpoint._ensure_healthy()

    def test_router_uses_p2p_only_for_supported_sync_cuda_collective(self):
        module = load_module()
        endpoint = mock.Mock()
        fallback = mock.Mock(return_value="gloo")
        router = module.make_all_to_all_router(endpoint, fallback)
        source = mock.Mock(is_cuda=True)
        output = mock.Mock(is_cuda=True)

        with mock.patch.object(module.dist, "get_world_size", return_value=2):
            self.assertIsNone(router(output, source, group="sequence"))
            self.assertEqual(endpoint.all_to_all_single.call_args.args, (output, source))

            self.assertEqual(router(output, source, async_op=True), "gloo")
            self.assertEqual(
                router(output, source, output_split_sizes=[1, 1]),
                "gloo",
            )
            source.is_cuda = False
            self.assertEqual(router(output, source), "gloo")
            module.dist.get_world_size.return_value = 1
            source.is_cuda = True
            self.assertEqual(router(output, source), "gloo")

        self.assertEqual(fallback.call_count, 4)

    def test_spin_control_wraps_slots_but_keeps_absolute_operation_ids(self):
        module = load_module()

        self.assertEqual(module.WindowsSpinControl._slot(1), 1)
        self.assertEqual(
            module.WindowsSpinControl._slot(module.WindowsSpinControl._SLOT_COUNT + 1),
            1,
        )
        self.assertEqual(
            module.WindowsSpinControl._offset(0, module.WindowsSpinControl._SLOT_COUNT + 1),
            module.WindowsSpinControl._offset(0, 1),
        )

    def test_spin_control_rejects_non_positive_operation_ids(self):
        module = load_module()

        with self.assertRaisesRegex(ValueError, "operation id must be positive"):
            module.WindowsSpinControl._slot(0)


if __name__ == "__main__":
    unittest.main()
