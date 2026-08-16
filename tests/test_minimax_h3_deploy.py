import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = REPO_ROOT / "scripts" / "minimax-h3" / "deploy-to-comfyui.ps1"


def run(*args, cwd=None):
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
        check=False,
    )


def make_repo(root: Path):
    (root / "src" / "raylight").mkdir(parents=True)
    (root / "__init__.py").write_text("ENTRY = True\n", encoding="utf-8")
    (root / "icon.png").write_bytes(b"icon")
    (root / "src" / "raylight" / "runtime.py").write_text(
        "RUNTIME = True\n", encoding="utf-8"
    )
    (root / "tests").mkdir()
    (root / "tests" / "not_runtime.py").write_text("TEST = True\n", encoding="utf-8")
    (root / "docs").mkdir()
    (root / "docs" / "not_runtime.md").write_text("docs\n", encoding="utf-8")
    assert run("git", "init", str(root)).returncode == 0
    assert run("git", "add", ".", cwd=root).returncode == 0
    assert run("git", "commit", "-m", "fixture", cwd=root).returncode == 0


def test_deploy_plan_contains_only_tracked_runtime_files(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    make_repo(source)
    make_repo(destination)
    before = run("git", "status", "--short", cwd=destination).stdout

    result = run(
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(DEPLOY_SCRIPT),
        "-SourceRoot",
        str(source),
        "-DestinationRoot",
        str(destination),
        "-PlanOnly",
        "-Json",
    )

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["copy_files"] == [
        "__init__.py",
        "icon.png",
        "src/raylight/runtime.py",
    ]
    assert plan["remove_files"] == []
    assert plan["source_commit"] == run("git", "rev-parse", "HEAD", cwd=source).stdout.strip()
    assert run("git", "status", "--short", cwd=destination).stdout == before


def test_deploy_refuses_tracked_destination_changes(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    make_repo(source)
    make_repo(destination)
    (destination / "src" / "raylight" / "runtime.py").write_text(
        "USER_CHANGE = True\n", encoding="utf-8"
    )

    result = run(
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(DEPLOY_SCRIPT),
        "-SourceRoot",
        str(source),
        "-DestinationRoot",
        str(destination),
        "-PlanOnly",
    )

    assert result.returncode != 0
    assert "tracked modifications" in (result.stdout + result.stderr)


def test_deploy_applies_only_runtime_files_and_leaves_destination_clean(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    make_repo(source)
    make_repo(destination)
    (source / "src" / "raylight" / "runtime.py").write_text(
        "SOURCE_RUNTIME = True\n", encoding="utf-8"
    )
    (source / "tests" / "source_only.py").write_text(
        "MUST_NOT_DEPLOY = True\n", encoding="utf-8"
    )
    assert run("git", "add", ".", cwd=source).returncode == 0
    assert run("git", "commit", "-m", "source update", cwd=source).returncode == 0
    source_commit = run("git", "rev-parse", "HEAD", cwd=source).stdout.strip()

    result = run(
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(DEPLOY_SCRIPT),
        "-SourceRoot",
        str(source),
        "-DestinationRoot",
        str(destination),
    )

    assert result.returncode == 0, result.stderr
    assert (destination / "src" / "raylight" / "runtime.py").read_text(
        encoding="utf-8"
    ) == "SOURCE_RUNTIME = True\n"
    assert not (destination / "tests" / "source_only.py").exists()
    assert run("git", "status", "--short", cwd=destination).stdout == ""
    marker = destination / ".git" / "raylight-deployed-commit"
    assert marker.read_text(encoding="utf-8-sig").strip() == source_commit
