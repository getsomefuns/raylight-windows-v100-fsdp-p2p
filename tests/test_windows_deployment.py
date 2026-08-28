from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "deploy-to-comfyui.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def make_source_repository(path: Path) -> None:
    (path / "src" / "raylight").mkdir(parents=True)
    (path / "__init__.py").write_text('__version__ = "test"\n', encoding="utf-8")
    (path / "src" / "raylight" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (path / "README.md").write_text("tracked deployment fixture\n", encoding="utf-8")
    assert run(["git", "init"], path).returncode == 0
    assert run(["git", "add", "__init__.py", "src/raylight/example.py", "README.md"], path).returncode == 0


def invoke_deploy(source: Path, comfy_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None, "PowerShell is required for deployment tests"
    return run(
        [
            POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(DEPLOY_SCRIPT),
            "-SourceRoot",
            str(source),
            "-ComfyUIRoot",
            str(comfy_root),
            *extra,
        ]
    )


def test_deploy_replaces_stale_tree_with_exact_tracked_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    comfy_root = tmp_path / "ComfyUI"
    target = comfy_root / "custom_nodes" / "raylight"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("must disappear\n", encoding="utf-8")
    (target / ".git" / "objects").mkdir(parents=True)
    (target / ".git" / "config").write_text("must not deploy\n", encoding="utf-8")
    read_only_object = target / ".git" / "objects" / "read-only-object"
    read_only_object.write_text("git object\n", encoding="utf-8")
    assert run(["attrib", "+H", str(target / ".git")]).returncode == 0
    assert run(["attrib", "+R", str(read_only_object)]).returncode == 0
    make_source_repository(source)

    result = invoke_deploy(source, comfy_root)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not (target / "stale.txt").exists()
    assert not (target / ".git").exists()
    for relative in ("__init__.py", "src/raylight/example.py", "README.md"):
        assert (target / relative).read_bytes() == (source / relative).read_bytes()


def test_verify_only_detects_and_reports_deployment_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    comfy_root = tmp_path / "ComfyUI"
    (comfy_root / "custom_nodes").mkdir(parents=True)
    make_source_repository(source)
    deployed = invoke_deploy(source, comfy_root)
    assert deployed.returncode == 0, deployed.stdout + deployed.stderr

    clean = invoke_deploy(source, comfy_root, "-VerifyOnly")
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "DEPLOYMENT_VERIFIED" in clean.stdout

    target = comfy_root / "custom_nodes" / "raylight"
    cache_file = target / "src" / "raylight" / "__pycache__" / "example.cpython-310.pyc"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"runtime bytecode cache")
    (target / ".pytest_cache").mkdir()
    (target / ".pytest_cache" / "README.md").write_text("pytest cache\n", encoding="utf-8")
    (target / ".ruff_cache").mkdir()
    (target / ".ruff_cache" / "cache").write_text("ruff cache\n", encoding="utf-8")
    runtime_env_file = target / "src" / "_ray_runtime_env" / "session" / "worker-package.zip"
    runtime_env_file.parent.mkdir(parents=True)
    runtime_env_file.write_bytes(b"generated Ray runtime environment")

    caches_are_ignored = invoke_deploy(source, comfy_root, "-VerifyOnly")
    assert caches_are_ignored.returncode == 0, caches_are_ignored.stdout + caches_are_ignored.stderr

    (target / "src" / "raylight" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")
    (target / "unmanaged.log").write_text("drift\n", encoding="utf-8")

    drift = invoke_deploy(source, comfy_root, "-VerifyOnly")
    assert drift.returncode != 0
    output = drift.stdout + drift.stderr
    assert "CHANGED: src/raylight/example.py" in output
    assert "EXTRA: unmanaged.log" in output


def test_deploy_refuses_to_replace_the_source_repository(tmp_path: Path) -> None:
    comfy_root = tmp_path / "ComfyUI"
    source = comfy_root / "custom_nodes" / "raylight"
    make_source_repository(source)

    result = invoke_deploy(source, comfy_root)

    assert result.returncode != 0
    assert "must be different" in (result.stdout + result.stderr)
    assert (source / ".git").is_dir()


def test_deploy_uses_directory_level_rename_for_tree_swaps() -> None:
    script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert "[IO.Directory]::Move($target, $backup)" in script
    assert "[IO.Directory]::Move($stage, $target)" in script
    assert "Move-Item -LiteralPath $target" not in script


def test_deploy_scopes_safe_directory_to_manifest_command() -> None:
    script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert 'git -c "safe.directory=$RepositoryRoot" -C $RepositoryRoot ls-files' in script
    assert "git config --global" not in script
