import json
import unittest
from unittest import mock

from raylight.comfy_extra_dist import nodes_custom_sampler


class MainCudaReleaseTests(unittest.TestCase):
    def test_release_unloads_before_gc_and_cuda_cache_cleanup(self):
        calls = []
        with (
            mock.patch.object(
                nodes_custom_sampler.comfy.model_management,
                "unload_all_models",
                side_effect=lambda: calls.append("unload"),
            ),
            mock.patch.object(
                nodes_custom_sampler.gc,
                "collect",
                side_effect=lambda: calls.append("gc"),
            ),
            mock.patch.object(
                nodes_custom_sampler.comfy.model_management,
                "soft_empty_cache",
                side_effect=lambda: calls.append("empty_cache"),
            ),
            mock.patch.dict(nodes_custom_sampler.os.environ, {"RAYLIGHT_RANK_DIAG": "0"}),
        ):
            nodes_custom_sampler._release_main_cuda_for_ray_sampling()

        self.assertEqual(calls, ["unload", "gc", "empty_cache"])

    def test_diagnostic_reports_each_release_phase_duration(self):
        with (
            mock.patch.object(
                nodes_custom_sampler.comfy.model_management,
                "unload_all_models",
            ),
            mock.patch.object(nodes_custom_sampler.gc, "collect"),
            mock.patch.object(
                nodes_custom_sampler.comfy.model_management,
                "soft_empty_cache",
            ),
            mock.patch.object(
                nodes_custom_sampler.time,
                "perf_counter_ns",
                side_effect=[
                    0,
                    0,
                    2_000_000_000,
                    2_000_000_000,
                    3_000_000_000,
                    3_000_000_000,
                    6_000_000_000,
                    7_000_000_000,
                ],
            ),
            mock.patch.object(nodes_custom_sampler.torch.cuda, "is_available", return_value=True),
            mock.patch.object(nodes_custom_sampler.torch.cuda, "current_device", return_value=0),
            mock.patch.object(nodes_custom_sampler.torch.cuda, "memory_allocated", return_value=11),
            mock.patch.object(nodes_custom_sampler.torch.cuda, "memory_reserved", return_value=22),
            mock.patch.dict(nodes_custom_sampler.os.environ, {"RAYLIGHT_RANK_DIAG": "1"}),
            mock.patch("builtins.print") as print_mock,
        ):
            nodes_custom_sampler._release_main_cuda_for_ray_sampling()

        prefix, payload = print_mock.call_args.args[0].split(" ", 1)
        diagnostic = json.loads(payload)
        self.assertEqual(prefix, "[RAYLIGHT_MAIN_CUDA_DIAG]")
        self.assertEqual(diagnostic["unload_seconds"], 2.0)
        self.assertEqual(diagnostic["gc_seconds"], 1.0)
        self.assertEqual(diagnostic["empty_cache_seconds"], 3.0)
        self.assertEqual(diagnostic["total_seconds"], 7.0)


if __name__ == "__main__":
    unittest.main()
