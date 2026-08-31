from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "minimax-h3"
    / "summarize_director_segment_resources.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("director_segment_resources", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _line(prefix: str, payload: dict) -> str:
    return f"worker {prefix} {json.dumps(payload)}"


def test_summarizes_each_two_rank_sampling_window():
    module = _module()
    lines = []
    for segment, (start, end) in enumerate(((100, 300), (500, 800))):
        for rank in (0, 1):
            lines.append(
                _line(
                    "[RAYLIGHT_RANK_DIAG]",
                    {
                        "event": "prepare_sampling_begin",
                        "rank": rank,
                        "time_ns": start + rank,
                    },
                )
            )
        for rank in (0, 1):
            lines.append(
                _line(
                    "[RAYLIGHT_P2P_PROFILE]",
                    {
                        "status": "success",
                        "rank": rank,
                        "elapsed_ns": 50 + segment,
                        "time_ns": end + rank,
                    },
                )
            )
    samples = [
        {
            "time_ns": 150,
            "physical_used_mib": 10.0,
            "committed_mib": 20.0,
            "pagefile_used_mib": 1.0,
            "gpus": [{"index": 0, "memory_mib": 7.0}, {"index": 1, "memory_mib": 8.0}],
        },
        {
            "time_ns": 250,
            "physical_used_mib": 12.0,
            "committed_mib": 22.0,
            "pagefile_used_mib": 1.5,
            "gpus": [{"index": 0, "memory_mib": 9.0}, {"index": 1, "memory_mib": 6.0}],
        },
        {
            "time_ns": 600,
            "physical_used_mib": 30.0,
            "committed_mib": 40.0,
            "pagefile_used_mib": 2.0,
            "gpus": [{"index": 0, "memory_mib": 11.0}, {"index": 1, "memory_mib": 12.0}],
        },
    ]

    rows = module.summarize_segment_resources("\n".join(lines), samples)

    assert len(rows) == 2
    assert rows[0]["gpu0_peak_mib"] == 9.0
    assert rows[0]["gpu1_peak_mib"] == 8.0
    assert rows[0]["physical_peak_mib"] == 12.0
    assert rows[0]["committed_peak_mib"] == 22.0
    assert rows[0]["pagefile_peak_mib"] == 1.5
    assert rows[1]["gpu0_peak_mib"] == 11.0
    assert rows[1]["sampling_seconds"] == 51e-9


def test_allows_ray_log_dedup_to_hide_one_final_profile_rank():
    module = _module()
    lines = [
        _line(
            "[RAYLIGHT_RANK_DIAG]",
            {"event": "prepare_sampling_begin", "rank": rank, "time_ns": 100 + rank},
        )
        for rank in (0, 1)
    ]
    lines.append(
        _line(
            "[RAYLIGHT_P2P_PROFILE]",
            {"status": "success", "rank": 1, "elapsed_ns": 50, "time_ns": 300},
        )
    )
    samples = [
        {
            "time_ns": 200,
            "physical_used_mib": 10.0,
            "committed_mib": 20.0,
            "pagefile_used_mib": 1.0,
            "gpus": [{"index": 0, "memory_mib": 7.0}, {"index": 1, "memory_mib": 8.0}],
        }
    ]

    rows = module.summarize_segment_resources("\n".join(lines), samples)

    assert rows[0]["profiled_ranks"] == [1]
