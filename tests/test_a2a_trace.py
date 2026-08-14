import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

import torch
import torch.distributed as dist


MODULE_PATH = Path(__file__).parents[1] / "src/raylight/distributed_worker/a2a_trace.py"


def load_module():
    spec = importlib.util.spec_from_file_location("raylight_a2a_trace_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class A2ATraceTests(unittest.TestCase):
    def test_disabled_tracer_does_not_replace_collective(self):
        module = load_module()
        original = dist.all_to_all_single

        with mock.patch.dict(os.environ, {}, clear=True):
            tracer = module.create_a2a_tracer(rank=0)

        self.assertFalse(tracer.enabled)
        self.assertIs(dist.all_to_all_single, original)

    def test_capture_aggregates_only_calls_inside_window(self):
        module = load_module()
        original = dist.all_to_all_single

        def local_copy(output, input_tensor, **_kwargs):
            output.copy_(input_tensor)

        with tempfile.TemporaryDirectory() as trace_dir:
            with mock.patch.object(dist, "all_to_all_single", local_copy):
                with mock.patch.dict(os.environ, {"RAYLIGHT_A2A_TRACE_DIR": trace_dir}, clear=True):
                    tracer = module.create_a2a_tracer(rank=1)
                try:
                    source = torch.arange(8, dtype=torch.float16).reshape(2, 4)
                    output = torch.empty_like(source)
                    dist.all_to_all_single(output, source)
                    with tracer.capture("custom_sampler"):
                        for _ in range(2):
                            dist.all_to_all_single(output, source)
                    dist.all_to_all_single(output, source)
                finally:
                    tracer.close()

            rows = [json.loads(line) for path in Path(trace_dir).glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["label"], "custom_sampler")
        self.assertEqual(rows[0]["call_count"], 2)
        self.assertEqual(rows[0]["total_input_bytes"], 32)
        self.assertEqual(rows[0]["total_output_bytes"], 32)
        self.assertEqual(len(rows[0]["groups"]), 1)
        group = rows[0]["groups"][0]
        self.assertEqual(group["count"], 2)
        self.assertEqual(group["dtype"], "torch.float16")
        self.assertEqual(group["input_bytes"], 16)
        self.assertEqual(group["input_shape"], [2, 4])
        self.assertEqual(group["output_bytes"], 16)
        self.assertEqual(group["output_shape"], [2, 4])
        self.assertGreaterEqual(group["max_elapsed_seconds"], group["min_elapsed_seconds"])
        self.assertGreaterEqual(group["total_elapsed_seconds"], group["max_elapsed_seconds"])
        self.assertIs(dist.all_to_all_single, original)

    def test_capture_flushes_partial_summary_when_sampling_fails(self):
        module = load_module()

        def local_copy(output, input_tensor, **_kwargs):
            output.copy_(input_tensor)

        with tempfile.TemporaryDirectory() as trace_dir:
            with mock.patch.object(dist, "all_to_all_single", local_copy):
                with mock.patch.dict(os.environ, {"RAYLIGHT_A2A_TRACE_DIR": trace_dir}, clear=True):
                    tracer = module.create_a2a_tracer(rank=0)
                source = torch.ones(4)
                output = torch.empty_like(source)
                with self.assertRaisesRegex(RuntimeError, "sampling failed"):
                    with tracer.capture("failed_sample"):
                        dist.all_to_all_single(output, source)
                        raise RuntimeError("sampling failed")
                tracer.close()

            rows = [json.loads(line) for path in Path(trace_dir).glob("*.jsonl") for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(rows[0]["label"], "failed_sample")
        self.assertEqual(rows[0]["call_count"], 1)
        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error_type"], "RuntimeError")

    def test_capture_decorator_uses_method_name_and_preserves_result(self):
        module = load_module()
        events = []

        class RecordingTracer:
            @module.contextmanager
            def capture(self, label):
                events.append(("enter", label))
                try:
                    yield
                finally:
                    events.append(("exit", label))

        class Worker:
            def __init__(self):
                self._a2a_tracer = RecordingTracer()

            @module.trace_a2a_capture
            def sample_method(self, value):
                events.append(("body", value))
                return value + 1

        self.assertEqual(Worker().sample_method(4), 5)
        self.assertEqual(events, [
            ("enter", "sample_method"),
            ("body", 4),
            ("exit", "sample_method"),
        ])

    def test_ray_worker_kill_closes_tracer_before_process_group(self):
        comfy_root = MODULE_PATH.parents[5]
        raylight_src = MODULE_PATH.parents[2]
        sys.path[:0] = [str(comfy_root), str(raylight_src)]
        try:
            from raylight.distributed_worker import ray_worker
        finally:
            del sys.path[:2]

        events = []

        class RecordingTracer:
            def close(self):
                events.append("close_trace")

        worker = ray_worker.RayWorker.__new__(ray_worker.RayWorker)
        worker._free_cached_aux_models = lambda: events.append("free_aux")
        worker._invalidate_non_fsdp_cache = lambda: events.append("invalidate_cache")
        worker.model = object()
        worker._a2a_tracer = RecordingTracer()

        with (
            mock.patch.object(
                ray_worker.dist,
                "destroy_process_group",
                side_effect=lambda: events.append("destroy_group"),
            ),
            mock.patch.object(
                ray_worker.ray.actor,
                "exit_actor",
                side_effect=lambda: events.append("exit_actor"),
            ),
        ):
            worker.kill()

        self.assertEqual(events, [
            "free_aux",
            "invalidate_cache",
            "close_trace",
            "destroy_group",
            "exit_actor",
        ])


if __name__ == "__main__":
    unittest.main()
