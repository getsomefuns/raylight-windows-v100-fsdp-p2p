from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark_windows_job", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object only")
def test_windows_job_close_kills_assigned_process():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    job = benchmark.WindowsKillOnCloseJob()
    try:
        job.assign(process)
        job.close()
        process.wait(timeout=10)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
