import importlib.util
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "tests" / "windows_p2p_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("raylight_windows_p2p_probe_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_probe_geometry_keeps_capacity_independent_from_payload():
    probe = _load_probe()

    capacity, sizes = probe.resolve_probe_geometry(
        capacity_bytes=128 * 1024**2,
        size_bytes=(239_826_944 * 2,),
        stress_iterations=0,
    )

    assert capacity == 128 * 1024**2
    assert sizes == (479_653_888,)


def test_resolve_probe_geometry_preserves_legacy_defaults():
    probe = _load_probe()

    capacity, sizes = probe.resolve_probe_geometry(
        capacity_bytes=None,
        size_bytes=None,
        stress_iterations=0,
    )

    assert sizes == probe.REAL_SIZES
    assert capacity == max(probe.REAL_SIZES) // 2


@pytest.mark.parametrize(
    ("capacity_bytes", "size_bytes", "message"),
    (
        (0, (1024,), "capacity_bytes must be positive"),
        (1024, (0,), "size_bytes must be positive"),
        (1024, (1023,), "size_bytes must be divisible by 2"),
    ),
)
def test_resolve_probe_geometry_rejects_invalid_values(
    capacity_bytes,
    size_bytes,
    message,
):
    probe = _load_probe()

    with pytest.raises(ValueError, match=message):
        probe.resolve_probe_geometry(
            capacity_bytes=capacity_bytes,
            size_bytes=size_bytes,
            stress_iterations=0,
        )


def test_probe_cli_exposes_capacity_and_payload_bytes():
    source = PROBE_PATH.read_text(encoding="utf-8")

    assert 'parser.add_argument("--capacity-bytes", type=int)' in source
    assert 'parser.add_argument("--size-bytes", type=int, nargs="+")' in source


def test_probe_reports_endpoint_and_payload_cuda_memory():
    source = PROBE_PATH.read_text(encoding="utf-8")

    assert "endpoint_allocated_bytes = torch.cuda.memory_allocated()" in source
    assert "payload_allocated_bytes = torch.cuda.memory_allocated()" in source
    assert 'f"endpoint_allocated_bytes={endpoint_allocated_bytes} "' in source
    assert 'f"payload_allocated_bytes={payload_allocated_bytes} "' in source
