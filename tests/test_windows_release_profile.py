import json
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).parents[1]
START_SCRIPT = REPO_ROOT / "scripts" / "start-comfyui-windows-p2p.ps1"
ENVIRONMENT_MATRIX = REPO_ROOT / "environment-windows-v100.json"
WORKFLOW = (
    REPO_ROOT
    / "example_workflows"
    / "LTX2_3_i2v_Raylight_Windows_P2P.json"
)


class WindowsReleaseProfileTests(unittest.TestCase):
    def test_environment_matrix_records_release_capacity_and_timeout(self):
        matrix = json.loads(ENVIRONMENT_MATRIX.read_text(encoding="utf-8"))
        gates = matrix["release_gates"]

        self.assertEqual(gates["p2p_capacity_bytes_per_rank"], 134217728)
        self.assertEqual(gates["maximum_validated_collective_input_bytes"], 230686720)
        self.assertEqual(gates["p2p_timeout_seconds"], 10)

    def test_worker_default_matches_validated_10s_capacity(self):
        from raylight.distributed_worker.windows_p2p import (
            DEFAULT_WINDOWS_P2P_CAPACITY_BYTES,
        )

        self.assertEqual(DEFAULT_WINDOWS_P2P_CAPACITY_BYTES, 134217728)
    def test_validate_only_uses_capacity_required_by_validated_10s_workflow(self):
        python_path = Path(__file__).parents[4] / "Python310" / "python.exe"
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(START_SCRIPT),
                "-PythonPath",
                str(python_path),
                "-ValidateOnly",
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("P2P capacity: 134217728 bytes", result.stdout)

    def test_example_workflow_enables_mmap_for_quantized_safetensors(self):
        workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        initializer = next(
            node
            for node in workflow["nodes"]
            if node["type"] == "RayInitializer"
        )

        self.assertEqual(initializer["widgets_values"][-1], True)


if __name__ == "__main__":
    unittest.main()
