import json
from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).parents[1]
START_SCRIPT = REPO_ROOT / "scripts" / "start-comfyui-windows-p2p.ps1"
ENVIRONMENT_MATRIX = REPO_ROOT / "environment-windows-v100.json"
WINDOWS_REQUIREMENTS = REPO_ROOT / "requirements-windows-v100.txt"
WORKFLOW = (
    REPO_ROOT
    / "example_workflows"
    / "LTX2_3_i2v_Raylight_Windows_P2P.json"
)


class WindowsReleaseProfileTests(unittest.TestCase):
    def test_windows_requirements_pin_benchmark_websocket_client(self):
        requirements = WINDOWS_REQUIREMENTS.read_text(encoding="utf-8").splitlines()

        self.assertIn("websocket-client==1.9.0", requirements)

    def test_environment_matrix_records_release_capacity_and_timeout(self):
        matrix = json.loads(ENVIRONMENT_MATRIX.read_text(encoding="utf-8"))
        gates = matrix["release_gates"]

        self.assertEqual(gates["p2p_capacity_bytes_per_rank"], 268435456)
        self.assertEqual(gates["launcher_default_p2p_capacity_mib_per_rank"], 256)
        self.assertEqual(gates["worker_fallback_p2p_capacity_bytes_per_rank"], 134217728)
        self.assertEqual(gates["maximum_validated_collective_input_bytes"], 230686720)
        self.assertEqual(gates["p2p_timeout_seconds"], 10)

    def test_worker_default_matches_validated_10s_capacity(self):
        from raylight.distributed_worker.windows_p2p import (
            DEFAULT_WINDOWS_P2P_CAPACITY_BYTES,
        )

        self.assertEqual(DEFAULT_WINDOWS_P2P_CAPACITY_BYTES, 134217728)

    def run_launcher(self, *arguments):
        python_path = REPO_ROOT.parent / "Python310" / "python.exe"
        comfy_root = REPO_ROOT.parent / "ComfyUI"
        return subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(START_SCRIPT),
                "-PythonPath",
                str(python_path),
                "-ComfyRoot",
                str(comfy_root),
                *arguments,
                "-ValidateOnly",
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

    def test_validate_only_defaults_to_256_mib_with_diagnostics_disabled(self):
        result = self.run_launcher()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("P2P capacity choices: 128, 256, 512 MiB", result.stdout)
        self.assertIn(
            "Selected P2P capacity: 256 MiB (268435456 bytes per GPU)",
            result.stdout,
        )
        self.assertIn("Rank/P2P diagnostics: disabled", result.stdout)
        self.assertIn("Reserved VRAM: 2 GiB", result.stdout)

    def test_validate_only_accepts_each_documented_mib_choice(self):
        for capacity_mib, capacity_bytes in (
            (128, 134217728),
            (256, 268435456),
            (512, 536870912),
        ):
            with self.subTest(capacity_mib=capacity_mib):
                result = self.run_launcher(
                    "-P2PCapacityMiB",
                    str(capacity_mib),
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(
                    f"Selected P2P capacity: {capacity_mib} MiB "
                    f"({capacity_bytes} bytes per GPU)",
                    result.stdout,
                )

    def test_validate_only_rejects_undocumented_mib_choice(self):
        result = self.run_launcher("-P2PCapacityMiB", "384")

        self.assertNotEqual(result.returncode, 0)

    def test_validate_only_enables_diagnostics_only_when_requested(self):
        result = self.run_launcher("-EnableDiagnostics")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Rank/P2P diagnostics: enabled", result.stdout)

    def test_legacy_capacity_bytes_parameter_remains_compatible(self):
        result = self.run_launcher("-P2PCapacityBytes", "268435456")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Selected P2P capacity: 256 MiB (268435456 bytes per GPU)",
            result.stdout,
        )

    def test_legacy_positional_arguments_remain_compatible(self):
        python_path = REPO_ROOT.parent / "Python310" / "python.exe"
        comfy_root = REPO_ROOT.parent / "ComfyUI"
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(START_SCRIPT),
                str(python_path),
                str(comfy_root),
                "127.0.0.1",
                "0,1",
                "8188",
                "29500",
                "268435456",
                "50",
                "128",
                "2",
                "-ValidateOnly",
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Selected P2P capacity: 256 MiB (268435456 bytes per GPU)",
            result.stdout,
        )

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
