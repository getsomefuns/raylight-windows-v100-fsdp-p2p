import copy
import unittest

import psutil
from unittest import mock

import importlib.util
from pathlib import Path

_BENCHMARK_PATH = Path(__file__).with_name("windows_ltx_mode_benchmark.py")
_SPEC = importlib.util.spec_from_file_location("raylight_windows_ltx_mode_benchmark", _BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(benchmark)
prepare_prompt = benchmark.prepare_prompt
validate_mode_prompt = benchmark.validate_mode_prompt


def minimal_prompt(mode):
    prompt = {
        "75": {"class_type": "SaveVideo", "inputs": {"filename_prefix": "old"}},
        "369": {"class_type": "VAEDecodeTiled", "inputs": {"tile_size": 384}},
        "374": {"class_type": "ComfyMathExpression", "inputs": {"expression": "a * b + 1"}},
        "383": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise_seed": 1}},
        "389": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise_seed": 2}},
    }
    if mode == "single":
        prompt["396"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}}
        prompt["397"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 2}}
        prompt["383"]["inputs"]["noise"] = ["396", 0]
        prompt["389"]["inputs"]["noise"] = ["397", 0]
    if mode in ("ray-single", "ulysses", "fsdp"):
        prompt["381"] = {
            "class_type": "RayInitializer",
            "inputs": {
                "GPU": 1 if mode == "ray-single" else 2,
                "ulysses_degree": 2 if mode == "ulysses" else 0,
                "ring_degree": 1 if mode == "ulysses" else 0,
                "cfg_degree": 1 if mode == "ulysses" else 0,
                "dp_degree": 1,
                "FSDP": mode == "fsdp",
            },
        }
        prompt["383"]["class_type"] = "XFuserSamplerCustomAdvanced"
        prompt["389"]["class_type"] = "XFuserSamplerCustomAdvanced"
    return prompt


class WindowsLTXModeBenchmarkTests(unittest.TestCase):
    def test_prepare_prompt_uses_matching_per_run_seeds_without_mutating_template(self):
        template = minimal_prompt("single")
        original = copy.deepcopy(template)

        prepared = prepare_prompt(template, "single", 2)

        self.assertEqual(template, original)
        self.assertEqual(prepared["396"]["inputs"]["noise_seed"], 426531806528257)
        self.assertEqual(prepared["397"]["inputs"]["noise_seed"], 513480589783141)
        self.assertEqual(
            prepared["75"]["inputs"]["filename_prefix"],
            "raylight/f4_single_run2",
        )

    def test_mode_validation_accepts_only_the_fixed_matched_topologies(self):
        for mode in ("single", "ulysses", "fsdp"):
            with self.subTest(mode=mode):
                validate_mode_prompt(mode, minimal_prompt(mode))

        invalid = minimal_prompt("ulysses")
        invalid["381"]["inputs"]["dp_degree"] = 2
        with self.assertRaisesRegex(ValueError, "dp_degree"):
            validate_mode_prompt("ulysses", invalid)

    def test_ray_single_control_uses_ray_nodes_without_distributed_topology(self):
        prompt = minimal_prompt("ray-single")

        validate_mode_prompt("ray-single", prompt)
        environment = benchmark.server_environment("ray-single")

        self.assertEqual(prompt["381"]["inputs"]["GPU"], 1)
        self.assertEqual(prompt["381"]["inputs"]["ulysses_degree"], 0)
        self.assertFalse(prompt["381"]["inputs"]["FSDP"])
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(environment["RAYLIGHT_WINDOWS_P2P"], "0")
        self.assertEqual(environment["RAYLIGHT_P2P_PROFILE"], "0")
    def test_mode_validation_rejects_nonmatching_vae_tile(self):
        prompt = minimal_prompt("single")
        prompt["369"]["inputs"]["tile_size"] = 512
        with self.assertRaisesRegex(ValueError, "tile_size=384"):
            validate_mode_prompt("single", prompt)

    def test_dual_gpu_workflow_uses_capacity_for_upsampled_stage(self):
        environment = benchmark.server_environment("ulysses")

        self.assertEqual(
            environment["RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES"],
            str(128 * 1024 * 1024),
        )
    def test_server_environment_uses_loopback_or_explicit_gloo_host(self):
        with mock.patch.dict(benchmark.os.environ, {}, clear=True):
            default_environment = benchmark.server_environment("fsdp")
        self.assertEqual(default_environment["RAYLIGHT_GLOO_HOST"], "127.0.0.1")

        with mock.patch.dict(
            benchmark.os.environ,
            {"RAYLIGHT_GLOO_HOST": "203.0.113.10"},
            clear=True,
        ):
            explicit_environment = benchmark.server_environment("fsdp")
        self.assertEqual(explicit_environment["RAYLIGHT_GLOO_HOST"], "203.0.113.10")

    def test_default_payloads_are_distributed_with_the_repository(self):
        self.assertEqual(
            benchmark.PAYLOAD_ROOT,
            benchmark.REPO_ROOT / "benchmark_payloads",
        )
        for name in ("single-5s.json", "ray-single-5s.json", "ulysses-5s.json", "fsdp-5s.json"):
            with self.subTest(name=name):
                self.assertTrue((benchmark.PAYLOAD_ROOT / name).is_file())

    def test_history_poll_tolerates_one_transient_request_timeout(self):
        prompt_id = "prompt-1"
        process = mock.Mock()
        process.poll.return_value = None
        completed = {
            prompt_id: {
                "status": {"completed": True, "status_str": "success"},
                "outputs": {},
            }
        }
        with (
            mock.patch.object(
                benchmark,
                "request_json",
                side_effect=[TimeoutError("busy"), completed],
            ) as request_mock,
            mock.patch.object(benchmark.time, "sleep"),
        ):
            result = benchmark.wait_prompt(
                "http://127.0.0.1:8188", prompt_id, process, timeout=30
            )

        self.assertEqual(result, completed[prompt_id])
        self.assertEqual(request_mock.call_count, 2)



    def test_prompt_monitor_stops_when_submission_fails(self):
        process = mock.Mock()
        stop_event = mock.Mock()
        monitor = mock.Mock()
        with (
            mock.patch.object(benchmark.threading, "Event", return_value=stop_event),
            mock.patch.object(benchmark.threading, "Thread", return_value=monitor),
            mock.patch.object(
                benchmark,
                "request_json",
                side_effect=RuntimeError("submission failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                benchmark.submit_prompt_and_wait(
                    "http://127.0.0.1:8188",
                    {},
                    process,
                    [],
                )

        monitor.start.assert_called_once_with()
        stop_event.set.assert_called_once_with()
        monitor.join.assert_called_once_with(timeout=15)

    def test_terminate_process_tree_tolerates_child_exit_race(self):
        process = mock.Mock(pid=100)
        process.poll.return_value = None
        parent = mock.Mock()
        child = mock.Mock()
        child.terminate.side_effect = psutil.NoSuchProcess(101)
        parent.children.return_value = [child]

        with mock.patch.object(benchmark.psutil, "Process", return_value=parent), mock.patch.object(
            benchmark.psutil, "wait_procs", return_value=([], [])
        ):
            benchmark.terminate_process_tree(process)

        parent.terminate.assert_called_once_with()

if __name__ == "__main__":
    unittest.main()
