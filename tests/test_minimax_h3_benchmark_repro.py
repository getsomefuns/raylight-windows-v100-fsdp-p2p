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
