from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "start-comfyui-windows-p2p.ps1"
POWERSHELL = shutil.which("powershell.exe")


def _make_environment(root: Path, *runtime_names: str) -> Path:
    comfy = root / "ComfyUI"
    comfy.mkdir(parents=True)
    (comfy / "main.py").write_text("# launcher fixture\n", encoding="utf-8")
    for runtime_name in runtime_names:
        python = root / runtime_name / "python.exe"
        python.parent.mkdir()
        python.write_bytes(b"fixture")
    return comfy


def _run_launcher(comfy: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-ComfyRoot",
            str(comfy),
            *arguments,
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is required")
def test_launcher_prefers_python312_without_removing_python310(tmp_path: Path) -> None:
    comfy = _make_environment(tmp_path, "Python310", "Python312")

    result = _run_launcher(comfy, "-ValidateOnly")

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Python: {tmp_path / 'Python312' / 'python.exe'}" in result.stdout


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is required")
def test_launcher_falls_back_to_python310_when_candidate_is_absent(tmp_path: Path) -> None:
    comfy = _make_environment(tmp_path, "Python310")

    result = _run_launcher(comfy, "-ValidateOnly")

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Python: {tmp_path / 'Python310' / 'python.exe'}" in result.stdout


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is required")
def test_launcher_disables_unsupported_v100_allocators_for_real_python(tmp_path: Path) -> None:
    comfy = _make_environment(tmp_path)
    (comfy / "main.py").write_text(
        "import sys\nprint('LAUNCH_ARGS=' + '|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    result = _run_launcher(comfy, "-PythonPath", sys.executable, "-ReserveVramGiB", "0")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--disable-cuda-malloc" in result.stdout
    assert "--disable-dynamic-vram" in result.stdout
