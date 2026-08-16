import unittest
import contextlib
import io
import json
import types
from unittest import mock

import raylight.distributed_worker.ray_worker as ray_worker


class RayWorkerWindowsP2PTests(unittest.TestCase):
    def test_prepare_windows_p2p_accepts_two_rank_fsdp_inference(self):
        worker = ray_worker.RayWorker.__new__(ray_worker.RayWorker)
        worker.global_world_size = 2
        worker.shard_size = 2
        worker.local_rank = 0
        worker.parallel_dict = {"is_fsdp": True}

        endpoint = mock.Mock()
        endpoint.local_ipc_metadata.return_value = {"buffer": "ipc"}
        with (
            mock.patch.object(ray_worker, "is_windows", return_value=True),
            mock.patch.object(ray_worker, "WindowsSpinControl", return_value=mock.Mock()),
            mock.patch.object(ray_worker, "CudaP2PAllToAll", return_value=endpoint),
        ):
            metadata = worker.prepare_windows_p2p("fsdp-test", 128 * 1024 * 1024)

        self.assertEqual(metadata, {"buffer": "ipc"})


    def test_enable_and_disable_windows_p2p_routes_both_cuda_collectives(self):
        worker = ray_worker.RayWorker.__new__(ray_worker.RayWorker)
        worker._windows_p2p = mock.Mock()
        worker._original_all_to_all_single = None
        original_all_to_all = ray_worker.dist.all_to_all_single
        original_all_gather = ray_worker.dist.all_gather_into_tensor

        try:
            worker.enable_windows_p2p()
            self.assertIsNot(ray_worker.dist.all_to_all_single, original_all_to_all)
            self.assertIsNot(ray_worker.dist.all_gather_into_tensor, original_all_gather)
        finally:
            worker.disable_windows_p2p()
            ray_worker.dist.all_to_all_single = original_all_to_all
            ray_worker.dist.all_gather_into_tensor = original_all_gather

        self.assertIs(ray_worker.dist.all_to_all_single, original_all_to_all)
        self.assertIs(ray_worker.dist.all_gather_into_tensor, original_all_gather)

    def test_windows_p2p_launch_topology_accepts_fsdp(self):
        ray_worker.validate_windows_p2p_launch(
            world_size=2,
            shard_size=2,
            parallel_dict={"is_fsdp": True},
        )


    def test_windows_p2p_launch_rejects_invalid_hybrid_before_actor_start(self):
        with self.assertRaisesRegex(ValueError, "dp_degree=1"):
            ray_worker.validate_windows_p2p_launch(
                world_size=2,
                shard_size=2,
                parallel_dict={
                    "is_fsdp": True,
                    "is_xdit": True,
                    "ulysses_degree": 2,
                    "ring_degree": 1,
                    "cfg_degree": 1,
                    "dp_degree": 2,
                },
            )
    def test_windows_p2p_health_default_window_is_long_enough_for_windows_jitter(self):
        iterations = ray_worker.windows_p2p_health_iterations(
            size_bytes=36 * 1024 * 1024,
        )

        self.assertEqual(iterations, 5689)

    def test_windows_p2p_health_iterations_uses_long_measurement_window(self):
        iterations = ray_worker.windows_p2p_health_iterations(
            size_bytes=44 * 1024 * 1024,
            target_remote_bytes=20 * 1024**3,
            minimum_iterations=100,
        )

        self.assertEqual(iterations, 931)

    def test_windows_p2p_health_iterations_keeps_minimum_for_large_payload(self):
        iterations = ray_worker.windows_p2p_health_iterations(
            size_bytes=4 * 1024**3,
            target_remote_bytes=20 * 1024**3,
            minimum_iterations=100,
        )

        self.assertEqual(iterations, 100)

    def test_windows_p2p_warmup_requires_both_ranks_to_reach_measurement_gate(self):
        self.assertFalse(
            ray_worker.windows_p2p_warmup_ready(
                [
                    {"rank": 0, "remote_gib_s": 52.0},
                    {"rank": 1, "remote_gib_s": 49.9},
                ],
                minimum_gib_s=50.0,
            )
        )
        self.assertTrue(
            ray_worker.windows_p2p_warmup_ready(
                [
                    {"rank": 0, "remote_gib_s": 52.0},
                    {"rank": 1, "remote_gib_s": 51.9},
                ],
                minimum_gib_s=50.0,
            )
        )

    def test_windows_p2p_warmup_rejects_duplicate_or_missing_ranks(self):
        with self.assertRaisesRegex(ValueError, "exactly ranks 0 and 1"):
            ray_worker.windows_p2p_warmup_ready(
                [{"rank": 0, "remote_gib_s": 52.0}],
                minimum_gib_s=50.0,
            )

    def test_windows_p2p_health_uses_per_rank_median(self):
        trials = [
            [
                {"rank": 0, "remote_gib_s": 48.0},
                {"rank": 1, "remote_gib_s": 47.0},
            ],
            [
                {"rank": 0, "remote_gib_s": 56.0},
                {"rank": 1, "remote_gib_s": 57.0},
            ],
            [
                {"rank": 0, "remote_gib_s": 55.0},
                {"rank": 1, "remote_gib_s": 54.0},
            ],
        ]

        summary = ray_worker.summarize_windows_p2p_health(trials)

        self.assertEqual(summary[0]["samples_gib_s"], [48.0, 56.0, 55.0])
        self.assertEqual(summary[0]["median_gib_s"], 55.0)
        self.assertEqual(summary[1]["median_gib_s"], 54.0)

    def test_windows_p2p_health_keeps_sustained_slow_rank_below_gate(self):
        trials = [
            [{"rank": 0, "remote_gib_s": value}]
            for value in (48.0, 49.0, 55.0)
        ]

        summary = ray_worker.summarize_windows_p2p_health(trials)

        self.assertLess(summary[0]["median_gib_s"], 50.0)

    def test_prepare_sampling_trim_runs_after_prepare_and_synchronizes(self):
        events = []

        def prepare(value):
            events.append("prepare")
            return value + 1

        with (
            mock.patch.object(ray_worker.torch.cuda, "synchronize", side_effect=lambda: events.append("sync")),
            mock.patch.object(ray_worker.torch.cuda, "empty_cache", side_effect=lambda: events.append("empty")),
        ):
            result = ray_worker.prepare_sampling_with_cuda_trim(prepare, 4)

        self.assertEqual(result, 5)
        self.assertEqual(events, ["prepare", "sync", "empty"])

    def test_sampler_profile_decorator_emits_one_aggregate_json_line(self):
        endpoint = mock.Mock()
        endpoint.profile_snapshot.side_effect = [
            {"enabled": True, "collectives": {}, "totals": {"calls": 0}},
            {
                "enabled": True,
                "collectives": {"all_gather": {"calls": 4}},
                "totals": {"calls": 4, "remote_bytes": 128},
            },
        ]
        worker = types.SimpleNamespace(
            _windows_p2p=endpoint,
            _sampler_invocation=0,
            local_rank=1,
        )

        @ray_worker.trace_windows_p2p_profile
        def sample(owner):
            owner._sampler_invocation += 1
            return "ok"

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = sample(worker)

        self.assertEqual(result, "ok")
        lines = [line for line in output.getvalue().splitlines() if line]
        self.assertEqual(len(lines), 1)
        prefix = "[RAYLIGHT_P2P_PROFILE] "
        self.assertTrue(lines[0].startswith(prefix))
        payload = json.loads(lines[0][len(prefix):])
        self.assertEqual(payload["rank"], 1)
        self.assertEqual(payload["invocation"], 1)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["profile"]["totals"]["calls"], 4)

if __name__ == "__main__":

    unittest.main()
