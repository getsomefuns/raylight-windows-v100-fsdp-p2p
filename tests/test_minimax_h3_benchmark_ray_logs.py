from __future__ import annotations

import importlib.util
from pathlib import Path
import re


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark_ray_logs", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_server_environment_never_deduplicates_rank_diagnostics():
    environment = benchmark.server_environment({})
    pattern = re.compile(environment["RAY_DEDUP_LOGS_ALLOW_REGEX"])

    assert pattern.search('[RAYLIGHT_RANK_DIAG] {"rank": 1, "event": "sampler_return"}')
    assert not pattern.search("ordinary Ray worker message")
