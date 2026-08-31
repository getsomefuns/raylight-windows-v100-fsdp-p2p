from pathlib import Path
import shutil
import subprocess

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install-environment-launcher.ps1"
POWERSHELL = shutil.which("powershell.exe")


def _run_installer(environment_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALLER),
            "-EnvironmentRoot",
            str(environment_root),
            *extra,
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is required")
def test_installer_creates_thin_forwarder_without_a_second_capacity_default(tmp_path):
    environment_root = tmp_path / "environment"
    environment_root.mkdir()

    result = _run_installer(environment_root)

    assert result.returncode == 0, result.stdout + result.stderr
    wrapper = environment_root / "start_raylight_p2p.ps1"
    source = wrapper.read_text(encoding="utf-8-sig")
    assert "ComfyUI\\custom_nodes\\raylight\\scripts\\start-comfyui-windows-p2p.ps1" in source
    assert "& $launcher @args" in source
    assert "P2PCapacityMiB" not in source
    assert "128" not in source
    assert "256" not in source
    assert "512" not in source


@pytest.mark.skipif(POWERSHELL is None, reason="Windows PowerShell is required")
def test_validate_only_describes_target_without_writing(tmp_path):
    environment_root = tmp_path / "environment"
    environment_root.mkdir()

    result = _run_installer(environment_root, "-ValidateOnly")

    assert result.returncode == 0, result.stdout + result.stderr
    assert str(environment_root / "start_raylight_p2p.ps1") in result.stdout
    assert "Validation-only mode" in result.stdout
    assert not (environment_root / "start_raylight_p2p.ps1").exists()
