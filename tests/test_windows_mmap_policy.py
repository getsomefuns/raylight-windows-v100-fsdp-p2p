import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src/raylight/distributed_worker/windows_p2p.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("raylight_windows_mmap_policy_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class WindowsMmapPolicyTests(unittest.TestCase):
    def test_explicit_mmap_allows_quantized_safetensors(self):
        module = load_module()

        self.assertTrue(
            module.should_use_safetensors_mmap(
                {"use_mmap": True, "is_quant": True},
                "ltx-2.3-fp8.safetensors",
            )
        )

    def test_mmap_stays_disabled_when_not_requested_or_not_safetensors(self):
        module = load_module()

        self.assertFalse(
            module.should_use_safetensors_mmap(
                {"use_mmap": False, "is_quant": True},
                "ltx-2.3-fp8.safetensors",
            )
        )
        self.assertFalse(
            module.should_use_safetensors_mmap(
                {"use_mmap": True, "is_quant": True},
                "ltx-2.3-fp8.gguf",
            )
        )


if __name__ == "__main__":
    unittest.main()
