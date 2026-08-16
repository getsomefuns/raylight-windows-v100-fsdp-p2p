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

    def test_chunk_ranges_covers_payload_with_literal_half_open_ranges(self):
        module = load_module()

        cases = (
            (0, 8, ()),
            (1, 8, ((0, 1),)),
            (8, 8, ((0, 8),)),
            (9, 8, ((0, 8), (8, 9))),
            (20, 8, ((0, 8), (8, 16), (16, 20))),
        )
        for total_bytes, capacity_bytes, expected in cases:
            with self.subTest(total_bytes=total_bytes, capacity_bytes=capacity_bytes):
                self.assertEqual(
                    module.chunk_ranges(total_bytes, capacity_bytes), expected
                )


    def test_chunk_ranges_rejects_invalid_sizes(self):
        module = load_module()

        for invalid_capacity in (0, -1):
            with self.subTest(capacity_bytes=invalid_capacity):
                with self.assertRaisesRegex(
                    ValueError, "capacity_bytes must be positive"
                ):
                    module.chunk_ranges(8, invalid_capacity)

        with self.assertRaisesRegex(
            ValueError, "total_bytes must be non-negative"
        ):
            module.chunk_ranges(-1, 8)


    def test_all_gather_copy_plan_places_local_and_peer_shards_by_rank(self):
        module = load_module()

        self.assertEqual(
            module.all_gather_copy_plan(rank=0, shard_bytes=16),
            ((0, 16, "input", 0, 16), (16, 32, "peer", 0, 16)),
        )
        self.assertEqual(
            module.all_gather_copy_plan(rank=1, shard_bytes=16),
            ((0, 16, "peer", 0, 16), (16, 32, "input", 0, 16)),
        )
    def test_all_gather_chunk_copy_plan_uses_absolute_destination_offsets(self):
        module = load_module()

        self.assertEqual(
            module.all_gather_chunk_copy_plan(
                rank=0, shard_bytes=20, chunk_start=8, chunk_end=16
            ),
            ((8, 16, "input", 8, 16), (28, 36, "peer", 0, 8)),
        )
        self.assertEqual(
            module.all_gather_chunk_copy_plan(
                rank=1, shard_bytes=20, chunk_start=8, chunk_end=16
            ),
            ((8, 16, "peer", 0, 8), (28, 36, "input", 8, 16)),
        )



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

    def test_validate_all_gather_accepts_one_shard_and_two_shard_output(self):
        module = load_module()
        source = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 8,
            element_size=lambda: 2,
        )
        output = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 16,
            element_size=lambda: 2,
        )

        self.assertEqual(
            module.validate_all_gather(
                output,
                source,
                capacity_bytes=16,
            ),
            16,
        )

    def test_validate_all_gather_rejects_output_that_is_not_two_shards(self):
        module = load_module()
        source = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 8,
            element_size=lambda: 2,
        )
        output = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 8,
            element_size=lambda: 2,
        )

        with self.assertRaisesRegex(ValueError, "exactly twice"):
            module.validate_all_gather(output, source, capacity_bytes=16)

    def test_validate_all_gather_accepts_shard_larger_than_p2p_buffer(self):
        module = load_module()
        source = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 8,
            element_size=lambda: 2,
        )
        output = mock.Mock(
            is_cuda=True,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 16,
            element_size=lambda: 2,
        )

        self.assertEqual(
            module.validate_all_gather(output, source, capacity_bytes=15), 16
        )

    def test_validate_all_gather_requires_matching_contiguous_cuda_tensors(self):
        module = load_module()
        source = mock.Mock(
            is_cuda=False,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 8,
            element_size=lambda: 2,
        )
        output = mock.Mock(
            is_cuda=False,
            is_contiguous=lambda: True,
            dtype=torch.float16,
            numel=lambda: 16,
            element_size=lambda: 2,
        )

        with self.assertRaisesRegex(ValueError, "CUDA tensors"):
            module.validate_all_gather(output, source, capacity_bytes=16)

        source.is_cuda = output.is_cuda = True
        source.is_contiguous = lambda: False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            module.validate_all_gather(output, source, capacity_bytes=16)

        source.is_contiguous = lambda: True
        output.dtype = torch.float32
        with self.assertRaisesRegex(ValueError, "same dtype"):
            module.validate_all_gather(output, source, capacity_bytes=16)

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

    def test_all_gather_router_routes_async_two_rank_cuda_to_p2p(self):
        module = load_module()
        endpoint = mock.Mock()
        endpoint.all_gather_into_tensor.return_value = "p2p-work"
        fallback = mock.Mock(return_value="gloo-work")
        router = module.make_all_gather_into_tensor_router(endpoint, fallback)
        source = mock.Mock(is_cuda=True)
        output = mock.Mock(is_cuda=True)

        with mock.patch.object(module.dist, "get_world_size", return_value=2):
            self.assertEqual(
                router(output, source, group="fsdp", async_op=True),
                "p2p-work",
            )

        endpoint.all_gather_into_tensor.assert_called_once_with(
            output,
            source,
            async_op=True,
        )
        fallback.assert_not_called()

    def test_all_gather_router_refuses_cuda_fallback(self):
        module = load_module()
        endpoint = mock.Mock()
        fallback = mock.Mock(return_value="gloo-work")
        router = module.make_all_gather_into_tensor_router(endpoint, fallback)
        source = mock.Mock(is_cuda=True)
        output = mock.Mock(is_cuda=True)

        with mock.patch.object(module.dist, "get_world_size", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "refuses CUDA fallback"):
                router(output, source, group="fsdp", async_op=True)

        endpoint.all_gather_into_tensor.assert_not_called()
        fallback.assert_not_called()

    def test_collective_router_lifecycle_installs_and_restores_both_operations(self):
        module = load_module()
        endpoint = mock.Mock()
        original_all_to_all = mock.Mock(name="original_all_to_all")
        original_all_gather = mock.Mock(name="original_all_gather")
        dist_module = mock.Mock(
            all_to_all_single=original_all_to_all,
            all_gather_into_tensor=original_all_gather,
        )

        originals = module.install_collective_routers(endpoint, dist_module)
        self.assertIsNot(dist_module.all_to_all_single, original_all_to_all)
        self.assertIsNot(dist_module.all_gather_into_tensor, original_all_gather)

        module.restore_collective_routers(dist_module, originals)
        self.assertIs(dist_module.all_to_all_single, original_all_to_all)
        self.assertIs(dist_module.all_gather_into_tensor, original_all_gather)

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

    def test_endpoint_exposes_aggregate_profile_snapshot(self):
        module = load_module()
        endpoint = module.CudaP2PAllToAll.__new__(module.CudaP2PAllToAll)
        endpoint._profiler = mock.Mock()
        endpoint._profiler.snapshot.return_value = {"enabled": True, "totals": {"calls": 3}}

        result = endpoint.profile_snapshot(reset=True)

        self.assertEqual(result["totals"]["calls"], 3)
        endpoint._profiler.snapshot.assert_called_once_with(reset=True)



if __name__ == "__main__":
    unittest.main()
