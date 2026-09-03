import json
from pathlib import Path
import hashlib
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFY = REPO_ROOT / "scripts" / "runtime" / "verify_comfy_aimdo_tcc.py"
BUILD = REPO_ROOT / "scripts" / "runtime" / "build-comfy-aimdo-tcc.ps1"
INSTALL = REPO_ROOT / "scripts" / "runtime" / "install-comfy-aimdo-tcc.ps1"


def _run_policy(*, wddm: bool, nvml: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY),
            "policy",
            "--wddm",
            "1" if wddm else "0",
            "--nvml",
            "1" if nvml else "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _payload(completed: subprocess.CompletedProcess[str]) -> dict:
    assert completed.stdout, completed.stderr
    return json.loads(completed.stdout)


def test_backend_policy_prefers_wddm_when_available():
    completed = _run_policy(wddm=True, nvml=True)

    assert completed.returncode == 0, completed.stderr
    assert _payload(completed) == {
        "allowed": True,
        "backend": "wddm",
        "pressure_headroom_bytes": 512 * 1024 * 1024,
    }


def test_backend_policy_allows_tcc_only_with_nvml_pressure_monitoring():
    completed = _run_policy(wddm=False, nvml=True)

    assert completed.returncode == 0, completed.stderr
    assert _payload(completed) == {
        "allowed": True,
        "backend": "nvml",
        "pressure_headroom_bytes": 512 * 1024 * 1024,
    }


def test_backend_policy_fails_closed_without_wddm_or_nvml():
    completed = _run_policy(wddm=False, nvml=False)

    assert completed.returncode == 1
    assert _payload(completed) == {
        "allowed": False,
        "backend": None,
        "pressure_headroom_bytes": 512 * 1024 * 1024,
    }


def test_local_version_satisfies_current_comfyui_exact_pin():
    completed = subprocess.run(
        [sys.executable, str(VERIFY), "metadata", "--version", "0.4.13+tcc1"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert _payload(completed) == {
        "upstream_version": "0.4.13",
        "local_version": "0.4.13+tcc1",
        "satisfies_exact_pin": True,
    }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _write_manifest_fixture(tmp_path: Path) -> Path:
    artifacts = {
        "patch": b"patch-v1",
        "dll": b"dll-v1",
        "wheel": b"wheel-v1",
    }
    entries = {}
    for name, payload in artifacts.items():
        path = tmp_path / name
        path.write_bytes(payload)
        entries[name] = {"path": name, "sha256": _sha256(payload)}
    manifest = {
        "schema": 1,
        "version": "0.4.13+tcc1",
        "pressure_headroom_bytes": 512 * 1024 * 1024,
        "upstream": {
            "repository": "https://github.com/Comfy-Org/comfy-aimdo.git",
            "tag": "v0.4.13",
            "commit": "3f4259a502e7df7508dca7c30bd07eb936edf5a6",
        },
        "detours": {
            "repository": "https://github.com/microsoft/Detours.git",
            "tag": "v4.0.1",
            "commit": "e4bfd6b03e50de46b47abfbd1e46b384f0c5f833",
        },
        "artifacts": entries,
    }
    path = tmp_path / "BUILD-MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_manifest_accepts_pinned_sources_and_matching_artifacts(tmp_path):
    manifest = _write_manifest_fixture(tmp_path)

    completed = subprocess.run(
        [sys.executable, str(VERIFY), "manifest", "--manifest", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert _payload(completed) == {"errors": [], "passed": True}


def test_build_manifest_rejects_an_artifact_changed_after_build(tmp_path):
    manifest = _write_manifest_fixture(tmp_path)
    (tmp_path / "dll").write_bytes(b"tampered")

    completed = subprocess.run(
        [sys.executable, str(VERIFY), "manifest", "--manifest", str(manifest)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 1
    assert _payload(completed) == {
        "errors": ["artifact dll SHA-256 mismatch"],
        "passed": False,
    }


def test_build_script_reports_the_reproducible_source_pins():
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD),
            "-PrintConfiguration",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert _payload(completed) == {
        "aimdo_commit": "3f4259a502e7df7508dca7c30bd07eb936edf5a6",
        "aimdo_tag": "v0.4.13",
        "detours_commit": "e4bfd6b03e50de46b47abfbd1e46b384f0c5f833",
        "detours_build_scope": "library-only",
        "detours_tag": "v4.0.1",
        "detours_target_processor": "X64",
        "local_version": "0.4.13+tcc1",
        "nvml_pressure_headroom_bytes": 512 * 1024 * 1024,
        "python_build_isolation": False,
        "python_build_dependencies": {
            "build": "1.6.0",
            "packaging": "25.0",
            "setuptools": "80.9.0",
            "setuptools-scm": "8.3.1",
            "tomli": "2.3.0",
            "wheel": "0.45.1",
        },
        "windows_sdk_nuget_version": "10.0.26100.1",
    }


def test_install_script_reports_guarded_reproducible_configuration():
    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALL),
            "-PrintConfiguration",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    assert _payload(completed) == {
        "canonical_requires_explicit_override": True,
        "expected_version": "0.4.13+tcc1",
        "pip_arguments": ["--no-deps", "--force-reinstall"],
    }


def test_install_script_refuses_canonical_runtime_without_override(tmp_path):
    python_path = tmp_path / "Python310" / "python.exe"
    python_path.parent.mkdir()
    python_path.write_bytes(b"")

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(INSTALL),
            "-EnvironmentRoot",
            str(tmp_path),
            "-PythonPath",
            str(python_path),
            "-ManifestPath",
            str(tmp_path / "missing-manifest.json"),
            "-WheelPath",
            str(tmp_path / "missing.whl"),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "Refusing to modify canonical Python310" in completed.stderr
