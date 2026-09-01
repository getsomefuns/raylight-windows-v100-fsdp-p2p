from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "name",
    [
        "windows_minimax_h3_qwen3vl_tp2_block_probe.py",
        "windows_minimax_h3_tp2_clip_proxy_probe.py",
    ],
)
def test_standalone_windows_probes_compile_and_use_current_o11_interfaces(name: str) -> None:
    path = ROOT / name
    source = path.read_text(encoding="utf-8")

    compile(source, str(path), "exec")
    assert "attention_mask=mask" not in source
    assert "backbone_factory=lambda:" not in source
