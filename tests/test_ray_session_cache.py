import os
from pathlib import Path
import sys
import unittest
from unittest import mock


COMFY_ROOT = Path(__file__).parents[3]
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
