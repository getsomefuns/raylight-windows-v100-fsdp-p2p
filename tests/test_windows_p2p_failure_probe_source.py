import importlib.util
from pathlib import Path
import sys
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "tests" / "windows_p2p_failure_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "raylight_windows_p2p_failure_probe_test",
        PROBE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_second_chunk_injector_changes_only_target_rank_operation():
    probe = _load_probe()
    delegate = mock.Mock()
    control = probe.SecondChunkMismatchControl(
        delegate,
        rank=1,
        target_operation_id=3,
    )

    control.publish_ready(1, 2, 128)
    control.publish_ready(1, 3, 128)
    control.publish_ready(0, 3, 128)

    assert delegate.publish_ready.call_args_list == [
        mock.call(1, 2, 128),
        mock.call(1, 3, 127),
        mock.call(0, 3, 128),
    ]


def test_failure_probe_runs_a_multichunk_second_chunk_mismatch():
    source = PROBE_PATH.read_text(encoding="utf-8")

    assert 'mode == "second_chunk_mismatch"' in source
    assert "total_bytes = 8 * 1024 * 1024" in source
    assert 'run_mode("second_chunk_mismatch")' in source
    assert "future_rejected = False" in source
    assert "or not rank0[3]" in source
