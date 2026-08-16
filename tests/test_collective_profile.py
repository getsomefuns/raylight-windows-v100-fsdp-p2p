import os
import unittest
from unittest import mock

from raylight.distributed_worker.collective_profile import (
    CollectiveProfiler,
    create_collective_profiler,
)


class CollectiveProfilerTests(unittest.TestCase):
    def test_records_all_gather_and_all_to_all_separately(self):
        profiler = CollectiveProfiler(enabled=True)
        profiler.record(
            "all_gather",
            payload_bytes=600,
            remote_bytes=300,
            chunks=3,
            control_wait_ns=11,
            submit_ns=21,
        )
        profiler.record(
            "all_gather",
            payload_bytes=800,
            remote_bytes=400,
            chunks=4,
            control_wait_ns=13,
            submit_ns=23,
        )
        profiler.record(
            "all_to_all",
            payload_bytes=200,
            remote_bytes=100,
            chunks=1,
            control_wait_ns=7,
            submit_ns=17,
        )

        snapshot = profiler.snapshot()
        gather = snapshot["collectives"]["all_gather"]
        all_to_all = snapshot["collectives"]["all_to_all"]
        self.assertEqual(gather["calls"], 2)
        self.assertEqual(gather["payload_bytes"], 1400)
        self.assertEqual(gather["remote_bytes"], 700)
        self.assertEqual(gather["chunks"], 7)
        self.assertEqual(gather["control_wait_ns"], 24)
        self.assertEqual(gather["submit_ns"], 44)
        self.assertEqual(gather["max_payload_bytes"], 800)
        self.assertEqual(all_to_all["calls"], 1)
        self.assertEqual(snapshot["totals"]["calls"], 3)
        self.assertEqual(snapshot["totals"]["remote_bytes"], 800)

    def test_snapshot_reset_is_atomic_from_the_callers_view(self):
        profiler = CollectiveProfiler(enabled=True)
        profiler.record("all_to_all", 32, 16, 1, 5, 9)

        captured = profiler.snapshot(reset=True)

        self.assertEqual(captured["totals"]["calls"], 1)
        self.assertEqual(profiler.snapshot()["totals"]["calls"], 0)

    def test_disabled_profiler_does_not_validate_or_accumulate(self):
        profiler = CollectiveProfiler(enabled=False)
        profiler.record("not-a-collective", -1, -1, -1, -1, -1)

        snapshot = profiler.snapshot()

        self.assertFalse(snapshot["enabled"])
        self.assertEqual(snapshot["collectives"], {})
        self.assertEqual(snapshot["totals"]["calls"], 0)

    def test_environment_factory_only_enables_exact_opt_in(self):
        with mock.patch.dict(os.environ, {"RAYLIGHT_P2P_PROFILE": "1"}, clear=False):
            self.assertTrue(create_collective_profiler().enabled)
        with mock.patch.dict(os.environ, {"RAYLIGHT_P2P_PROFILE": "0"}, clear=False):
            self.assertFalse(create_collective_profiler().enabled)

    def test_enabled_profiler_rejects_invalid_values(self):
        profiler = CollectiveProfiler(enabled=True)
        with self.assertRaisesRegex(ValueError, "kind"):
            profiler.record("bad", 1, 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            profiler.record("all_gather", 1, -1, 1, 1, 1)


if __name__ == "__main__":
    unittest.main()
