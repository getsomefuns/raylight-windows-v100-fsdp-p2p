import unittest

from raylight.distributed_worker.parallel_group_manager import validate_hybrid_topology


def hybrid_config(**overrides):
    config = {
        "is_fsdp": True,
        "is_xdit": True,
        "ulysses_degree": 2,
        "ring_degree": 1,
        "cfg_degree": 1,
        "dp_degree": 1,
        "shard_size": 2,
    }
    config.update(overrides)
    return config


class HybridTopologyTests(unittest.TestCase):
    def test_dual_rank_fsdp2_ulysses2_uses_the_same_rank_pair(self):
        self.assertEqual(
            validate_hybrid_topology(2, 2, hybrid_config()),
            "hybrid",
        )

    def test_existing_modes_keep_their_classification(self):
        self.assertEqual(
            validate_hybrid_topology(1, 1, {"is_fsdp": False, "is_xdit": False}),
            "single",
        )
        self.assertEqual(
            validate_hybrid_topology(2, 2, {"is_fsdp": True, "is_xdit": False}),
            "fsdp",
        )
        self.assertEqual(
            validate_hybrid_topology(2, 2, {"is_fsdp": False, "is_xdit": True}),
            "ulysses",
        )

    def test_hybrid_rejects_unsupported_two_v100_degrees(self):
        cases = (
            (4, 2, hybrid_config(), "exactly two workers"),
            (2, 1, hybrid_config(shard_size=1), "shard_size=2"),
            (2, 2, hybrid_config(dp_degree=2), "dp_degree=1"),
            (2, 2, hybrid_config(ulysses_degree=1), "ulysses_degree=2"),
            (2, 2, hybrid_config(ring_degree=2), "ring_degree=1"),
            (2, 2, hybrid_config(cfg_degree=2), "cfg_degree=1"),
        )
        for world_size, shard_size, config, message in cases:
            with self.subTest(config=config, world_size=world_size, shard_size=shard_size):
                with self.assertRaisesRegex(ValueError, message):
                    validate_hybrid_topology(world_size, shard_size, config)


if __name__ == "__main__":
    unittest.main()
