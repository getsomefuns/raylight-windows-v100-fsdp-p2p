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


if __name__ == "__main__":
    unittest.main()
