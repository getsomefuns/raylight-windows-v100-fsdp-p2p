import json


def test_memory_recorder_writes_summary_when_operation_fails(monkeypatch, tmp_path):
    from raylight.diagnostics.memory_recorder import MemoryRunRecorder

    monkeypatch.setenv("RAYLIGHT_MEMORY_RECORD", "1")
    samples = iter(
        [
            {
                "system_available_bytes": 90,
                "system_commit_bytes": 30,
                "process_rss_bytes": 10,
                "process_private_bytes": 12,
            },
            {
                "system_available_bytes": 70,
                "system_commit_bytes": 50,
                "process_rss_bytes": 20,
                "process_private_bytes": 24,
            },
        ]
    )
    output = tmp_path / "memory.jsonl"
    recorder = MemoryRunRecorder(
        "mmh3director_sampling",
        operation_id="operation-1",
        rank=0,
        output_path=output,
        snapshot=lambda: next(samples),
        start_thread=False,
    )

    try:
        with recorder:
            recorder.poll_once()
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    payload = json.loads(output.read_text(encoding="utf-8").strip())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["peak_system_commit_bytes"] == 50
    assert payload["min_system_available_bytes"] == 70
    assert payload["peak_process_rss_bytes"] == 20
    assert payload["peak_process_private_bytes"] == 24


def test_memory_recorder_can_be_disabled(monkeypatch, tmp_path):
    from raylight.diagnostics.memory_recorder import MemoryRunRecorder

    monkeypatch.setenv("RAYLIGHT_MEMORY_RECORD", "0")
    output = tmp_path / "memory.jsonl"

    with MemoryRunRecorder("sampling", output_path=output, start_thread=False):
        pass

    assert not output.exists()
