import importlib.util
from pathlib import Path
import sys
import threading
import unittest


MODULE_PATH = (
    Path(__file__).parents[1]
    / "src/raylight/distributed_worker/windows_p2p.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("raylight_windows_p2p_model_sync_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


class TwoRankMinimum:
    def __init__(self):
        self._condition = threading.Condition()
        self._values = []

    def reduce(self, value):
        with self._condition:
            self._values.append(value)
            if len(self._values) == 2:
                self._condition.notify_all()
            else:
                self._condition.wait_for(lambda: len(self._values) == 2)
            return min(self._values)


class ModelLoadSynchronizationTests(unittest.TestCase):
    def test_both_ranks_load_with_the_minimum_vram_budget(self):
        module = load_module()
        reducer = TwoRankMinimum()
        barrier = threading.Barrier(2)
        loaded_budgets = [None, None]

        def run(rank, local_budget):
            module.synchronized_model_load(
                local_budget,
                lambda budget: loaded_budgets.__setitem__(rank, budget),
                reducer.reduce,
                barrier.wait,
            )

        threads = [
            threading.Thread(target=run, args=(0, 13.5)),
            threading.Thread(target=run, args=(1, 13.9)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=1)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(loaded_budgets, [13.5, 13.5])

    def test_fast_rank_does_not_continue_until_slow_rank_finishes_loading(self):
        module = load_module()
        reducer = TwoRankMinimum()
        barrier = threading.Barrier(2)
        slow_load_started = threading.Event()
        release_slow_load = threading.Event()
        fast_load_finished = threading.Event()
        fast_rank_returned = threading.Event()

        def slow_load(_budget):
            slow_load_started.set()
            release_slow_load.wait(timeout=1)

        def fast_load(_budget):
            fast_load_finished.set()

        slow_thread = threading.Thread(
            target=lambda: module.synchronized_model_load(13.5, slow_load, reducer.reduce, barrier.wait)
        )

        def run_fast_rank():
            module.synchronized_model_load(13.9, fast_load, reducer.reduce, barrier.wait)
            fast_rank_returned.set()

        fast_thread = threading.Thread(target=run_fast_rank)
        slow_thread.start()
        fast_thread.start()

        self.assertTrue(slow_load_started.wait(timeout=1))
        self.assertTrue(fast_load_finished.wait(timeout=1))
        self.assertFalse(fast_rank_returned.is_set())

        release_slow_load.set()
        slow_thread.join(timeout=1)
        fast_thread.join(timeout=1)

        self.assertFalse(slow_thread.is_alive())
        self.assertFalse(fast_thread.is_alive())
        self.assertTrue(fast_rank_returned.is_set())


if __name__ == "__main__":
    unittest.main()
