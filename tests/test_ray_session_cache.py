import os
from pathlib import Path
import sys
import unittest
from unittest import mock

from tests.path_helpers import comfy_root


COMFY_ROOT = comfy_root()
RAYLIGHT_SRC = Path(__file__).parents[1] / "src"


class RaySessionCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [str(COMFY_ROOT), str(RAYLIGHT_SRC)]
        try:
            from raylight import nodes

            cls.nodes = nodes
        finally:
            del sys.path[:2]

    def setUp(self):
        self.nodes._clear_active_ray_session()

    def tearDown(self):
        self.nodes._clear_active_ray_session()

    def test_reuses_matching_live_session(self):
        payload = [{"workers": [object(), object()]}, object()]
        self.nodes._cache_active_ray_session("same", payload)

        with (
            mock.patch.object(self.nodes.ray, "is_initialized", return_value=True),
            mock.patch.object(self.nodes, "_probe_ray_actor_payload", return_value=True) as probe,
        ):
            reused = self.nodes._reuse_active_ray_session("same", expected_world_size=2)

        self.assertIs(reused, payload)
        probe.assert_called_once_with(payload, 2)

    def test_does_not_reuse_changed_configuration(self):
        payload = [{"workers": [object(), object()]}, object()]
        self.nodes._cache_active_ray_session("old", payload)

        with (
            mock.patch.object(self.nodes.ray, "is_initialized", return_value=True),
            mock.patch.object(self.nodes, "_probe_ray_actor_payload") as probe,
        ):
            reused = self.nodes._reuse_active_ray_session("new", expected_world_size=2)

        self.assertIsNone(reused)
        probe.assert_not_called()

    def test_dead_session_is_evicted(self):
        payload = [{"workers": [object(), object()]}, object()]
        self.nodes._cache_active_ray_session("same", payload)

        with (
            mock.patch.object(self.nodes.ray, "is_initialized", return_value=True),
            mock.patch.object(self.nodes, "_probe_ray_actor_payload", return_value=False),
        ):
            reused = self.nodes._reuse_active_ray_session("same", expected_world_size=2)

        self.assertIsNone(reused)
        self.assertIsNone(self.nodes._ACTIVE_RAY_SESSION)

    def test_fsdp_model_transition_reuses_only_an_exact_active_model(self):
        self.assertEqual(
            self.nodes._fsdp_model_transition(already_loaded=True, model_loaded=True),
            "reuse",
        )
        self.assertEqual(
            self.nodes._fsdp_model_transition(already_loaded=False, model_loaded=True),
            "recycle",
        )
        self.assertEqual(
            self.nodes._fsdp_model_transition(already_loaded=False, model_loaded=False),
            "load",
        )

    def test_recycle_replaces_cached_payload_workers(self):
        old_workers = [object(), object()]
        new_workers = [object(), object()]
        new_actors = {"workers": new_workers}
        actor_fn = mock.Mock(return_value=new_actors)
        payload = [{"workers": old_workers}, actor_fn]

        with (
            mock.patch.object(self.nodes.ray, "kill") as kill,
            mock.patch.object(self.nodes, "_wait_for_ray_workers_exit"),
        ):
            ray_actors, gpu_actors = self.nodes._recycle_ray_actor_payload(payload)

        self.assertIs(ray_actors, new_actors)
        self.assertIs(gpu_actors, new_workers)
        self.assertIs(payload[0], new_actors)
        actor_fn.assert_called_once_with()
        self.assertEqual(kill.call_count, 2)
        kill.assert_has_calls(
            [
                mock.call(old_workers[0], no_restart=True),
                mock.call(old_workers[1], no_restart=True),
            ]
        )

    def test_recycle_rejects_immutable_payload_before_killing_workers(self):
        old_workers = [object(), object()]
        actor_fn = mock.Mock()
        payload = ({"workers": old_workers}, actor_fn)

        with mock.patch.object(self.nodes.ray, "kill") as kill:
            with self.assertRaisesRegex(TypeError, "must be mutable"):
                self.nodes._recycle_ray_actor_payload(payload)

        kill.assert_not_called()
        actor_fn.assert_not_called()

    def test_session_key_changes_with_p2p_environment(self):
        base = {"GPU": 2, "parallel": {"ulysses_degree": 2}}
        with mock.patch.dict(os.environ, {"RAYLIGHT_WINDOWS_P2P": "1"}, clear=False):
            enabled = self.nodes._ray_session_key(base)
        with mock.patch.dict(os.environ, {"RAYLIGHT_WINDOWS_P2P": "0"}, clear=False):
            disabled = self.nodes._ray_session_key(base)

        self.assertNotEqual(enabled, disabled)

    def test_session_key_changes_with_collective_profile_setting(self):
        base = {"GPU": 2, "parallel": {"ulysses_degree": 2}}
        with mock.patch.dict(os.environ, {"RAYLIGHT_P2P_PROFILE": "1"}, clear=False):
            enabled = self.nodes._ray_session_key(base)
        with mock.patch.dict(os.environ, {"RAYLIGHT_P2P_PROFILE": "0"}, clear=False):
            disabled = self.nodes._ray_session_key(base)

        self.assertNotEqual(enabled, disabled)



if __name__ == "__main__":
    unittest.main()
