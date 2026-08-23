from __future__ import annotations

import re
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "start-comfyui-windows-p2p.ps1"


def test_windows_launcher_keeps_mmh3_multireference_headroom() -> None:
    """Two 0.9 MP MMH3 references need more than the old 2 GiB reserve on V100."""

    script = LAUNCHER.read_text(encoding="utf-8")
    match = re.search(r"\[double\]\$ReserveVramGiB\s*=\s*([0-9.]+)", script)

    assert match is not None, "launcher must expose ReserveVramGiB"
    assert float(match.group(1)) >= 3.0
    assert '"--reserve-vram", $reserveVramText' in script
