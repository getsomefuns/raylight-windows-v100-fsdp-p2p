from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "minimax-h3" / "deploy-to-comfyui.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def run(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Raylight Test",
            "GIT_AUTHOR_EMAIL": "raylight@example.invalid",
            "GIT_COMMITTER_NAME": "Raylight Test",
            "GIT_COMMITTER_EMAIL": "raylight@example.invalid",
        }
    )
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def make_repo(root: Path) -> str:
    (root / "src" / "raylight").mkdir(parents=True)
    (root / "__init__.py").write_text("ENTRY = True\n", encoding="utf-8")
    (root / "src" / "raylight" / "runtime.py").write_text(
        "RUNTIME = True\n", encoding="utf-8"
    )
    (root / "README.md").write_text("fixture\n", encoding="utf-8")
    assert run("git", "init", cwd=root).returncode == 0
    assert run("git", "add", ".", cwd=root).returncode == 0
    assert run("git", "commit", "-m", "fixture", cwd=root).returncode == 0
    return run("git", "rev-parse", "HEAD", cwd=root).stdout.strip()


def invoke(source: Path, destination: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    assert POWERSHELL is not None
    return run(
        POWERSHELL,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(DEPLOY_SCRIPT),
        "-SourceRoot",
        str(source),
        "-DestinationRoot",
        str(destination),
        *extra,
    )


def test_deploy_plan_describes_plain_copy_without_existing_destination_repo(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "ComfyUI" / "custom_nodes" / "raylight"
    head = make_repo(source)
    destination.parent.mkdir(parents=True)

    result = invoke(source, destination, "-PlanOnly", "-Json")

    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["schema_version"] == 2
    assert plan["deployment_kind"] == "plain-copy"
    assert plan["source_commit"] == head
    assert plan["destination_root"] == str(destination.resolve())


def test_deploy_creates_non_git_runtime_copy_with_root_marker(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "ComfyUI" / "custom_nodes" / "raylight"
    head = make_repo(source)
    destination.parent.mkdir(parents=True)

    result = invoke(source, destination)

    assert result.returncode == 0, result.stdout + result.stderr
    assert (destination / "src" / "raylight" / "runtime.py").is_file()
    assert not (destination / ".git").exists()
    assert (destination / ".raylight-deployed-commit").read_text(
        encoding="utf-8-sig"
    ).strip() == head


def test_deploy_refuses_dirty_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "ComfyUI" / "custom_nodes" / "raylight"
    make_repo(source)
    destination.parent.mkdir(parents=True)
    (source / "src" / "raylight" / "runtime.py").write_text(
        "DIRTY = True\n", encoding="utf-8"
    )

    result = invoke(source, destination)

    assert result.returncode != 0
    assert "tracked modifications" in (result.stdout + result.stderr)
    assert not destination.exists()


def test_legacy_deployer_delegates_without_mutating_git() -> None:
    script = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert "..\\deploy-to-comfyui.ps1" in script
    assert "git add" not in script
    assert "git commit" not in script
