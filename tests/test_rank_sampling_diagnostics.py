from __future__ import annotations

import unittest

import torch

from raylight.comfy_extra_dist.nodes_custom_sampler import summarize_rank_sampling_results


class RankSamplingDiagnosticsTests(unittest.TestCase):
    def test_reports_matching_and_divergent_rank_latents(self):
        rank0 = ({"samples": torch.tensor([[1.0, 2.0]])}, {"samples": torch.tensor([[3.0]])})
        rank1 = ({"samples": torch.tensor([[1.0, 2.0]])}, {"samples": torch.tensor([[4.0]])})

        summary = summarize_rank_sampling_results([rank0, rank1])

        self.assertEqual(summary["rank_count"], 2)
        self.assertTrue(summary["outputs"][0]["comparisons"][0]["exact"])
        self.assertFalse(summary["outputs"][1]["comparisons"][0]["exact"])
        self.assertEqual(summary["outputs"][1]["comparisons"][0]["sample_max_abs"], 1.0)

    def test_handles_nested_like_samples_as_multiple_streams(self):
        class NestedLike:
            is_nested = True

            def __init__(self, values):
                self.values = values

            def unbind(self):
                return tuple(self.values)

        rank0 = ({"samples": NestedLike([torch.ones(2), torch.zeros(3)])}, {"samples": torch.ones(1)})
        rank1 = ({"samples": NestedLike([torch.ones(2), torch.zeros(3)])}, {"samples": torch.ones(1)})

        summary = summarize_rank_sampling_results([rank0, rank1])

        self.assertEqual(len(summary["outputs"][0]["streams"]), 2)
        self.assertTrue(all(row["exact"] for row in summary["outputs"][0]["comparisons"]))


    def test_result_diagnostic_is_only_in_advanced_single_group_sampler(self):
        from raylight.comfy_extra_dist import nodes_custom_sampler

        from pathlib import Path

        source = Path(nodes_custom_sampler.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("[RAYLIGHT_RESULT_DIAG]"), 1)

if __name__ == "__main__":
    unittest.main()
