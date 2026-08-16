import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).parents[1]
WORKER_PATH = REPO_ROOT / "src/raylight/distributed_worker/ray_worker.py"
NODES_PATH = REPO_ROOT / "src/raylight/nodes.py"


def class_node(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def method_node(class_definition, name):
    return next(node for node in class_definition.body if isinstance(node, ast.FunctionDef) and node.name == name)


class FSDPPreflightEntrypointTests(unittest.TestCase):
    def test_worker_preflight_patches_fsdp_and_returns_structured_metrics(self):
        worker = class_node(WORKER_PATH, "RayWorker")
        method = method_node(worker, "fsdp_preflight")
        source = ast.unparse(method)

        self.assertIn("self._patch_fsdp_for_sampling()", source)
        self.assertIn("summarize_fsdp_parameters", source)
        self.assertIn("p2p_next_operation_id", source)
        self.assertIn("cuda_allocated_bytes", source)

    def test_preflight_node_calls_all_workers_and_is_registered(self):
        node = class_node(NODES_PATH, "RayFSDPPreflight")
        method = method_node(node, "preflight")
        source = ast.unparse(method)
        module_source = NODES_PATH.read_text(encoding="utf-8")

        self.assertIn("actor.fsdp_preflight.remote()", source)
        self.assertIn('"RayFSDPPreflight": RayFSDPPreflight', module_source)
        self.assertIn('"RayFSDPPreflight": "FSDP Preflight (Raylight)"', module_source)


if __name__ == "__main__":
    unittest.main()
