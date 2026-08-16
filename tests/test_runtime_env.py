import os
from pathlib import Path
import sys
import unittest
from unittest import mock


COMFY_ROOT = Path(__file__).parents[3]
RAYLIGHT_SRC = Path(__file__).parents[1] / "src"


class RuntimeEnvironmentTests(unittest.TestCase):
    def test_local_runtime_env_forwards_windows_collective_settings(self):
        sys.path[:0] = [str(COMFY_ROOT), str(RAYLIGHT_SRC)]
        try:
            from raylight import nodes
        finally:
            del sys.path[:2]

        expected = {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RAYLIGHT_GLOO_HOST": "192.0.2.10",
            "RAYLIGHT_A2A_TRACE_DIR": r"C:\raylight-trace",
            "USE_LIBUV": "0",
            "RAY_DEBUG_DISABLE_MEMORY_MONITOR": "1",
            "RAY_memory_usage_threshold": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "RAYLIGHT_WINDOWS_P2P": "1",
            "RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES": "134217728",
            "RAYLIGHT_WINDOWS_P2P_MIN_GIB_S": "50",
            "RAYLIGHT_WINDOWS_P2P_TIMEOUT_SECONDS": "10",
            "RAYLIGHT_RANK_DIAG": "1",
            "RAYLIGHT_P2P_PROFILE": "1",
        }
        with mock.patch.dict(os.environ, expected, clear=False):
            runtime_env = nodes._build_local_runtime_env(
                RAYLIGHT_SRC / "raylight",
                COMFY_ROOT,
                RAYLIGHT_SRC / "_ray_runtime_env",
            )

        env_vars = runtime_env["env_vars"]
        for key, value in expected.items():
            self.assertEqual(env_vars[key], value)

    def test_local_runtime_env_explicitly_forwards_native_allocator_tuning(self):
        sys.path[:0] = [str(COMFY_ROOT), str(RAYLIGHT_SRC)]
        try:
            from raylight import nodes
        finally:
            del sys.path[:2]

        with mock.patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"},
            clear=True,
        ):
            runtime_env = nodes._build_local_runtime_env(
                RAYLIGHT_SRC / "raylight",
                COMFY_ROOT,
                RAYLIGHT_SRC / "_ray_runtime_env",
            )

        self.assertEqual(
            runtime_env["env_vars"]["PYTORCH_CUDA_ALLOC_CONF"],
            "max_split_size_mb:128",
        )

    def test_worker_allocator_tuning_keeps_native_options_when_async_is_removed(self):
        sys.path[:0] = [str(COMFY_ROOT), str(RAYLIGHT_SRC)]
        try:
            from raylight import nodes
        finally:
            del sys.path[:2]

        with mock.patch.dict(
            os.environ,
            {"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,max_split_size_mb:128"},
            clear=True,
        ):
            self.assertEqual(nodes._sanitized_worker_alloc_conf(), "max_split_size_mb:128")


if __name__ == "__main__":
    unittest.main()
