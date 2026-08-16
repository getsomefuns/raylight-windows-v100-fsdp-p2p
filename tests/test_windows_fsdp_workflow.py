import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).parents[1]
BASELINE_WORKFLOW = REPO_ROOT / "example_workflows/LTX2_3_i2v_Raylight_Windows_P2P.json"
FSDP_WORKFLOW = REPO_ROOT / "example_workflows/LTX2_3_i2v_Raylight_Windows_FSDP_5s.json"


def load_workflow(path):
    return json.loads(path.read_text(encoding="utf-8"))


def one_node(workflow, node_type):
    matches = [node for node in workflow["nodes"] if node["type"] == node_type]
    if len(matches) != 1:
        raise AssertionError(f"expected one {node_type}, found {len(matches)}")
    return matches[0]


class WindowsFSDPWorkflowTests(unittest.TestCase):
    def test_fsdp_workflow_uses_one_replica_sharded_across_two_workers(self):
        workflow = load_workflow(FSDP_WORKFLOW)
        initializer = one_node(workflow, "RayInitializer")

        self.assertEqual(
            initializer["widgets_values"],
            [
                "local",
                "default",
                2,
                0,
                0,
                0,
                1,
                False,
                True,
                True,
                False,
                "TORCH_EFFICIENT",
                True,
                True,
            ],
        )

    def test_fsdp_workflow_preserves_5s_media_and_model_inputs(self):
        baseline = load_workflow(BASELINE_WORKFLOW)
        workflow = load_workflow(FSDP_WORKFLOW)

        self.assertEqual(one_node(workflow, "EmptyLTXVLatentVideo")["widgets_values"], [768, 512, 97, 1])
        self.assertEqual(one_node(workflow, "CreateVideo")["widgets_values"], [24])
        self.assertEqual(one_node(workflow, "LoadImage")["widgets_values"], ["LTX2_3_i2v_Raylight.jpg", "image"])
        self.assertEqual(
            one_node(workflow, "RayUNETLoader")["widgets_values"],
            one_node(baseline, "RayUNETLoader")["widgets_values"],
        )
        self.assertEqual(
            [node["widgets_values"] for node in workflow["nodes"] if node["type"] == "RayLoraLoader"],
            [node["widgets_values"] for node in baseline["nodes"] if node["type"] == "RayLoraLoader"],
        )

    def test_fsdp_workflow_uses_bounded_video_vae_tiles(self):
        workflow = load_workflow(FSDP_WORKFLOW)

        self.assertEqual(
            one_node(workflow, "VAEDecodeTiled")["widgets_values"],
            [384, 64, 64, 8],
        )


    def test_baseline_stays_ulysses_and_fsdp_workflow_keeps_custom_sampling_chain(self):
        baseline = load_workflow(BASELINE_WORKFLOW)
        workflow = load_workflow(FSDP_WORKFLOW)

        self.assertEqual(one_node(baseline, "RayInitializer")["widgets_values"][3:11], [2, 1, 1, 1, True, True, False, False])
        self.assertEqual(
            len([node for node in workflow["nodes"] if node["type"] == "XFuserSamplerCustomAdvanced"]),
            2,
        )


if __name__ == "__main__":
    unittest.main()
