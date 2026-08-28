from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from unittest import mock

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "minimax-h3" / "benchmark_cold_warm.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_benchmark_repro", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def make_runtime_repository(root: Path, content: str = "RUNTIME = True\n") -> str:
    runtime = root / "src" / "raylight" / "runtime.py"
    runtime.parent.mkdir(parents=True)
    (root / "__init__.py").write_text("ENTRY = True\n", encoding="utf-8")
    runtime.write_text(content, encoding="utf-8")
    assert benchmark.subprocess.run(["git", "init"], cwd=root).returncode == 0
    assert benchmark.subprocess.run(
        ["git", "add", "__init__.py", "src/raylight/runtime.py"], cwd=root
    ).returncode == 0
    assert (
        benchmark.subprocess.run(
            [
                "git",
                "-c",
                "user.name=Raylight Test",
                "-c",
                "user.email=raylight@example.invalid",
                "commit",
                "-m",
                "fixture",
            ],
            cwd=root,
        ).returncode
        == 0
    )
    return benchmark._git_output(root, "rev-parse", "HEAD")


def committed_bytes(root: Path, relative: str) -> bytes:
    completed = benchmark.subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=root,
        capture_output=True,
    )
    assert completed.returncode == 0
    return completed.stdout


def test_attach_job_closes_candidate_when_assignment_fails():
    candidate = mock.Mock()
    candidate.assign.side_effect = OSError("cannot assign")

    with pytest.raises(OSError, match="cannot assign"):
        benchmark.attach_kill_on_close_job(mock.Mock(), factory=lambda: candidate)

    candidate.close.assert_called_once_with()


def test_read_deployment_marker_accepts_authoritative_source_sha(tmp_path):
    marker = tmp_path / "raylight-deployed-commit"
    marker.write_text("a" * 40 + "\n", encoding="utf-8")

    assert benchmark.read_deployed_source_commit(marker) == "a" * 40


def test_read_deployment_marker_rejects_invalid_value(tmp_path):
    marker = tmp_path / "raylight-deployed-commit"
    marker.write_text("not-a-commit", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing or invalid"):
        benchmark.read_deployed_source_commit(marker)


def test_plain_deployment_identity_uses_root_marker(tmp_path):
    source = tmp_path / "source"
    deployed = tmp_path / "raylight"
    head = make_runtime_repository(source)
    (deployed / "src" / "raylight").mkdir(parents=True)
    (deployed / "__init__.py").write_bytes(committed_bytes(source, "__init__.py"))
    (deployed / "src" / "raylight" / "runtime.py").write_bytes(
        committed_bytes(source, "src/raylight/runtime.py")
    )
    marker = deployed / ".raylight-deployed-commit"
    marker.write_text(head + "\n", encoding="utf-8")

    identity = benchmark.collect_installed_identity(deployed, source)

    assert identity == {
        "kind": "plain-copy",
        "head": head,
        "status": "",
        "dirty": False,
    }


def test_plain_deployment_identity_detects_modified_runtime_file(tmp_path):
    source = tmp_path / "source"
    deployed = tmp_path / "raylight"
    head = make_runtime_repository(source)
    (deployed / "src" / "raylight").mkdir(parents=True)
    (deployed / "__init__.py").write_bytes(committed_bytes(source, "__init__.py"))
    (deployed / "src" / "raylight" / "runtime.py").write_text(
        "RUNTIME = False\n", encoding="utf-8"
    )
    (deployed / ".raylight-deployed-commit").write_text(head + "\n", encoding="utf-8")

    identity = benchmark.collect_installed_identity(deployed, source)

    assert identity["kind"] == "plain-copy"
    assert identity["dirty"] is True
    assert "CHANGED: src/raylight/runtime.py" in identity["status"]


def test_plain_deployment_applies_git_path_filters_to_line_endings(tmp_path):
    source = tmp_path / "source"
    deployed = tmp_path / "raylight"
    make_runtime_repository(source, content="LINE_ONE = 1\nLINE_TWO = 2\n")
    (source / ".gitattributes").write_text("*.py text\n", encoding="utf-8")
    assert benchmark.subprocess.run(["git", "add", ".gitattributes"], cwd=source).returncode == 0
    assert (
        benchmark.subprocess.run(
            [
                "git",
                "-c",
                "user.name=Raylight Test",
                "-c",
                "user.email=raylight@example.invalid",
                "commit",
                "-m",
                "line ending attributes",
            ],
            cwd=source,
        ).returncode
        == 0
    )
    head = benchmark._git_output(source, "rev-parse", "HEAD")
    (deployed / "src" / "raylight").mkdir(parents=True)
    (deployed / "__init__.py").write_bytes(
        committed_bytes(source, "__init__.py").replace(b"\n", b"\r\n")
    )
    committed_runtime = committed_bytes(source, "src/raylight/runtime.py")
    (deployed / "src" / "raylight" / "runtime.py").write_bytes(
        committed_runtime.replace(b"\n", b"\r\n")
    )
    (deployed / ".raylight-deployed-commit").write_text(head + "\n", encoding="utf-8")

    identity = benchmark.collect_installed_identity(deployed, source)

    assert identity["dirty"] is False
    assert identity["status"] == ""


def test_clean_git_clone_identity_does_not_require_private_marker(tmp_path):
    deployed = tmp_path / "raylight"
    deployed.mkdir()
    (deployed / "tracked.txt").write_text("fixture\n", encoding="utf-8")
    assert benchmark.subprocess.run(["git", "init"], cwd=deployed).returncode == 0
    assert benchmark.subprocess.run(["git", "add", "tracked.txt"], cwd=deployed).returncode == 0
    assert (
        benchmark.subprocess.run(
            [
                "git",
                "-c",
                "user.name=Raylight Test",
                "-c",
                "user.email=raylight@example.invalid",
                "commit",
                "-m",
                "fixture",
            ],
            cwd=deployed,
        ).returncode
        == 0
    )
    head = benchmark._git_output(deployed, "rev-parse", "HEAD")

    identity = benchmark.collect_installed_identity(deployed)

    assert identity == {
        "kind": "git-checkout",
        "head": head,
        "status": "",
        "dirty": False,
    }


def test_git_checkout_marker_cannot_hide_dirty_runtime(tmp_path):
    deployed = tmp_path / "raylight"
    head = make_runtime_repository(deployed)
    (deployed / ".raylight-deployed-commit").write_text(head + "\n", encoding="utf-8")
    (deployed / "src" / "raylight" / "runtime.py").write_text(
        "RUNTIME = False\n", encoding="utf-8"
    )

    identity = benchmark.collect_installed_identity(deployed)

    assert identity["kind"] == "git-checkout"
    assert identity["dirty"] is True
    assert "src/raylight/runtime.py" in identity["status"]


def test_clean_git_checkout_ignores_stale_private_marker(tmp_path):
    deployed = tmp_path / "raylight"
    head = make_runtime_repository(deployed)
    private_marker = deployed / ".git" / "raylight-deployed-commit"
    private_marker.write_text("c" * 40 + "\n", encoding="utf-8")

    identity = benchmark.collect_installed_identity(deployed)

    assert identity == {
        "kind": "git-checkout",
        "head": head,
        "status": "",
        "dirty": False,
    }


def test_runtime_source_dirty_scopes_safe_directory_to_the_repository(tmp_path):
    completed = mock.Mock(returncode=0, stderr="")
    with mock.patch.object(benchmark.subprocess, "run", return_value=completed) as run:
        assert benchmark._runtime_source_dirty(tmp_path) is False

    command = run.call_args.args[0]
    assert command[:4] == ["git", "-c", f"safe.directory={tmp_path}", "-C"]
    assert command[4] == str(tmp_path)


def test_write_prompt_artifact_uses_canonical_sha256(tmp_path):
    prompt = {"2": {"inputs": {"b": 2, "a": 1}}, "1": {"class_type": "Node"}}

    digest = benchmark.write_prompt_artifact(tmp_path / "prompt.json", prompt)

    encoded = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert json.loads((tmp_path / "prompt.json").read_text(encoding="utf-8")) == prompt


def test_benchmark_input_identity_hashes_every_reproducibility_input(tmp_path):
    files = []
    for name in ("benchmark.py", "build.py", "convert.py", "workflow.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        files.append(path)

    identity = benchmark.hash_input_files(files)

    assert set(identity) == {str(path) for path in files}
    assert all(len(digest) == 64 for digest in identity.values())
