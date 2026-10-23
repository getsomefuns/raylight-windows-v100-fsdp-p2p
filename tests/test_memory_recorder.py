import json


def test_capture_memory_snapshot_includes_physical_cuda_usage(monkeypatch):
    from raylight.diagnostics import memory_recorder

    process_memory = type(
        "ProcessMemory",
        (),
        {"rss": 10, "vms": 20, "private": 30, "pagefile": 40},
    )()
    system_memory = type(
        "SystemMemory",
        (),
        {"total": 100, "available": 60, "used": 40},
    )()
    monkeypatch.setattr(
        memory_recorder.psutil,
        "Process",
        lambda: type("Process", (), {"memory_full_info": lambda self: process_memory})(),
    )
    monkeypatch.setattr(memory_recorder.psutil, "virtual_memory", lambda: system_memory)
    monkeypatch.setattr(memory_recorder, "_windows_commit_memory", lambda: {})
    monkeypatch.setattr(memory_recorder.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(memory_recorder.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(memory_recorder.torch.cuda, "mem_get_info", lambda _device: (2, 10))
    monkeypatch.setattr(memory_recorder.torch.cuda, "memory_allocated", lambda _device: 3)
    monkeypatch.setattr(memory_recorder.torch.cuda, "memory_reserved", lambda _device: 4)
    monkeypatch.setattr(memory_recorder.torch.cuda, "max_memory_allocated", lambda _device: 5)
    monkeypatch.setattr(memory_recorder.torch.cuda, "max_memory_reserved", lambda _device: 6)

    snapshot = memory_recorder.capture_memory_snapshot()

    assert snapshot["cuda_device"] == 1
    assert snapshot["cuda_device_total_bytes"] == 10
    assert snapshot["cuda_device_free_bytes"] == 2
    assert snapshot["cuda_device_used_bytes"] == 8


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
                "cuda_device_free_bytes": 40,
                "cuda_device_used_bytes": 60,
            },
            {
                "system_available_bytes": 70,
                "system_commit_bytes": 50,
                "process_rss_bytes": 20,
                "process_private_bytes": 24,
                "cuda_device_free_bytes": 20,
                "cuda_device_used_bytes": 80,
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
    assert payload["min_cuda_device_free_bytes"] == 20
    assert payload["peak_cuda_device_used_bytes"] == 80


def test_memory_recorder_can_be_disabled(monkeypatch, tmp_path):
    from raylight.diagnostics.memory_recorder import MemoryRunRecorder

    monkeypatch.setenv("RAYLIGHT_MEMORY_RECORD", "0")
    output = tmp_path / "memory.jsonl"

    with MemoryRunRecorder("sampling", output_path=output, start_thread=False):
        pass

    assert not output.exists()


def test_memory_recorder_resets_cuda_peaks_before_the_start_snapshot(monkeypatch, tmp_path):
    from raylight.diagnostics import memory_recorder

    events = []
    monkeypatch.setenv("RAYLIGHT_MEMORY_RECORD", "1")
    monkeypatch.setattr(memory_recorder.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        memory_recorder.torch.cuda,
        "reset_peak_memory_stats",
        lambda: events.append("reset"),
    )

    def snapshot():
        events.append("snapshot")
        return {"process_rss_bytes": 1}

    with memory_recorder.MemoryRunRecorder(
        "sampling",
        output_path=tmp_path / "memory.jsonl",
        snapshot=snapshot,
        start_thread=False,
    ):
        pass

    assert events[:2] == ["reset", "snapshot"]
