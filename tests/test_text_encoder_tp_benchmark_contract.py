import json
import time

import pytest

from raylight.text_encoder_parallel.baseline import BaselineRecord


def _valid_record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "checkpoint": "qwen_3_vl_32b_fp4mixed.safetensors",
        "workflow_sha256": "a" * 64,
        "mode": "cold",
        "run_index": 1,
        "token_count": 4096,
        "te_total_s": 123.5,
        "gpu0_peak_mib": 15_321.0,
        "ram_peak_mib": 83_000.0,
        "pagefile_peak_mib": 11_200.0,
    }


@pytest.mark.parametrize(
    "missing",
    [
        "checkpoint",
        "token_count",
        "te_total_s",
        "gpu0_peak_mib",
        "ram_peak_mib",
        "pagefile_peak_mib",
    ],
)
def test_record_rejects_missing_required_metric(missing: str) -> None:
    payload = _valid_record()
    del payload[missing]

    with pytest.raises(ValueError, match=missing):
        BaselineRecord.from_mapping(payload)


def test_record_round_trips_json_without_losing_contract_fields() -> None:
    record = BaselineRecord.from_mapping(_valid_record())

    restored = BaselineRecord.from_json(record.to_json())

    assert restored == record
    assert json.loads(record.to_json())["workflow_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint", ""),
        ("token_count", 0),
        ("te_total_s", 0.0),
        ("gpu0_peak_mib", -1.0),
        ("ram_peak_mib", -1.0),
        ("pagefile_peak_mib", -1.0),
    ],
)
def test_record_rejects_invalid_measurements(field: str, value: object) -> None:
    payload = _valid_record()
    payload[field] = value

    with pytest.raises(ValueError, match=field):
        BaselineRecord.from_mapping(payload)


def test_stage_timing_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from raylight.text_encoder_parallel import baseline

    monkeypatch.delenv(baseline.BASELINE_ENV, raising=False)
    assert baseline.stage_timing_enabled() is False

    monkeypatch.setenv(baseline.BASELINE_ENV, "1")
    assert baseline.stage_timing_enabled() is True


def test_opt_in_hook_records_timing_without_changing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from raylight.text_encoder_parallel import baseline

    event_path = tmp_path / "events.jsonl"
    monkeypatch.setenv(baseline.BASELINE_ENV, "1")
    monkeypatch.setenv(baseline.BASELINE_EVENT_PATH_ENV, str(event_path))

    class FakeClip:
        def encode_from_tokens_scheduled(self, tokens, marker=None):
            return {"tokens": tokens, "marker": marker}

    assert baseline.install_clip_timing_hook(FakeClip) is True
    tokens = {"qwen3vl": [[(1, 1.0), (2, 1.0), (3, 1.0)]]}

    output = FakeClip().encode_from_tokens_scheduled(tokens, marker="unchanged")

    assert output == {"tokens": tokens, "marker": "unchanged"}
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events] == [
        "text_encoder_begin",
        "text_encoder_complete",
    ]
    assert events[0]["token_count"] == 3
    assert events[1]["token_count"] == 3
    assert events[1]["te_total_s"] > 0
    assert events[1]["time_ns"] >= events[0]["time_ns"]


def test_disabled_hook_does_not_patch_class(monkeypatch: pytest.MonkeyPatch) -> None:
    from raylight.text_encoder_parallel import baseline

    monkeypatch.delenv(baseline.BASELINE_ENV, raising=False)

    class FakeClip:
        def encode_from_tokens_scheduled(self, tokens):
            return tokens

    original = FakeClip.encode_from_tokens_scheduled
    assert baseline.install_clip_timing_hook(FakeClip) is False
    assert FakeClip.encode_from_tokens_scheduled is original


def test_comfy_installer_patches_stock_and_tp2_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from raylight.text_encoder_parallel import baseline
    class FakeStockClip:
        def encode_from_tokens_scheduled(self, tokens):
            return tokens

    class FakeTP2Clip:
        def encode_from_tokens_scheduled(self, tokens):
            return tokens

    monkeypatch.setenv(baseline.BASELINE_ENV, "1")
    monkeypatch.setattr("comfy.sd.CLIP", FakeStockClip)
    monkeypatch.setattr(
        "raylight.text_encoder_parallel.clip_proxy.MiniMaxH3TP2ClipProxy",
        FakeTP2Clip,
    )
    stock_original = FakeStockClip.encode_from_tokens_scheduled
    tp2_original = FakeTP2Clip.encode_from_tokens_scheduled

    assert baseline.install_comfy_clip_timing_from_environment() is True

    assert FakeStockClip.encode_from_tokens_scheduled is not stock_original
    assert FakeTP2Clip.encode_from_tokens_scheduled is not tp2_original


def test_count_tokens_counts_minimax_vision_entry_as_one_token() -> None:
    from raylight.text_encoder_parallel.baseline import count_tokens

    class FakeImage:
        shape = (1, 736, 1280, 3)

    tokens = {
        "qwen3vl_32b": [
            [
                (151652, 1.0),
                ({"type": "image", "data": FakeImage()}, 1.0),
                (151653, 1.0),
                (42, 1.0),
            ]
        ]
    }

    assert count_tokens(tokens) == 4


def test_timed_stage_is_opt_in_and_records_elapsed_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from raylight.text_encoder_parallel import baseline

    event_path = tmp_path / "events.jsonl"
    monkeypatch.setenv(baseline.BASELINE_ENV, "1")
    monkeypatch.setenv(baseline.BASELINE_EVENT_PATH_ENV, str(event_path))

    with baseline.timed_stage("language_backbone"):
        time.sleep(0.001)

    event = json.loads(event_path.read_text(encoding="utf-8").strip())
    assert event["event"] == "text_encoder_stage"
    assert event["stage"] == "language_backbone"
    assert event["elapsed_s"] > 0
