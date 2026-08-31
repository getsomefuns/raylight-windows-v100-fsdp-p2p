#!/usr/bin/env python3
"""Summarize per-segment Director resource peaks from benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


DIAG_PATTERN = re.compile(r"\[RAYLIGHT_RANK_DIAG\] (\{.*\})")
PROFILE_PATTERN = re.compile(r"\[RAYLIGHT_P2P_PROFILE\] (\{.*\})")


def _payloads(pattern: re.Pattern[str], text: str) -> list[dict]:
    result = []
    for line in text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            result.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return result


def _segment_windows(log_text: str) -> list[dict]:
    starts_by_rank: dict[int, list[dict]] = {0: [], 1: []}
    for payload in _payloads(DIAG_PATTERN, log_text):
        rank = int(payload.get("rank", -1))
        if payload.get("event") == "prepare_sampling_begin" and rank in starts_by_rank:
            starts_by_rank[rank].append(payload)
    for values in starts_by_rank.values():
        values.sort(key=lambda item: int(item["time_ns"]))
    counts = {rank: len(values) for rank, values in starts_by_rank.items()}
    if not counts[0] or counts[0] != counts[1]:
        raise RuntimeError(f"expected equal two-rank segment starts, got {counts}")

    profiles = [
        payload
        for payload in _payloads(PROFILE_PATTERN, log_text)
        if payload.get("status") == "success" and int(payload.get("rank", -1)) in (0, 1)
    ]
    profiles.sort(key=lambda item: int(item["time_ns"]))
    groups: list[dict[int, dict]] = []
    for profile in profiles:
        rank = int(profile["rank"])
        if not groups or rank in groups[-1]:
            groups.append({rank: profile})
        else:
            groups[-1][rank] = profile
    if len(groups) != counts[0] or any(not group for group in groups):
        raise RuntimeError(
            f"expected {counts[0]} profile groups, got {len(groups)}"
        )

    windows = []
    for index, group in enumerate(groups):
        start_ns = min(
            int(starts_by_rank[rank][index]["time_ns"]) for rank in (0, 1)
        )
        end_ns = max(int(profile["time_ns"]) for profile in group.values())
        if end_ns < start_ns:
            raise RuntimeError(f"segment {index + 1} ends before it starts")
        windows.append(
            {
                "segment": index + 1,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "sampling_seconds": max(
                    int(profile["elapsed_ns"]) for profile in group.values()
                )
                / 1e9,
                "profiled_ranks": sorted(group),
            }
        )
    return windows


def summarize_segment_resources(log_text: str, samples: list[dict]) -> list[dict]:
    rows = []
    for window in _segment_windows(log_text):
        selected = [
            sample
            for sample in samples
            if window["start_ns"] <= int(sample["time_ns"]) <= window["end_ns"]
        ]
        if not selected:
            raise RuntimeError(f"no monitor samples for segment {window['segment']}")

        def gpu_peak(index: int) -> float:
            values = []
            for sample in selected:
                by_index = {int(gpu["index"]): gpu for gpu in sample["gpus"]}
                if index not in by_index:
                    raise RuntimeError(f"monitor sample is missing GPU {index}")
                values.append(float(by_index[index]["memory_mib"]))
            return max(values)

        rows.append(
            {
                **window,
                "sample_count": len(selected),
                "gpu0_peak_mib": gpu_peak(0),
                "gpu1_peak_mib": gpu_peak(1),
                "physical_peak_mib": max(
                    float(sample["physical_used_mib"]) for sample in selected
                ),
                "committed_peak_mib": max(
                    float(sample["committed_mib"]) for sample in selected
                ),
                "pagefile_peak_mib": max(
                    float(sample["pagefile_used_mib"]) for sample in selected
                ),
            }
        )
    return rows


def _load_monitor(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = []
        for source in csv.DictReader(handle):
            rows.append(
                {
                    "time_ns": int(source["time_ns"]),
                    "physical_used_mib": float(source["physical_used_mib"]),
                    "committed_mib": float(source["committed_mib"]),
                    "pagefile_used_mib": float(source["pagefile_used_mib"]),
                    "gpus": json.loads(source["gpus_json"]),
                }
            )
    return rows


def _write_plot(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    segments = [row["segment"] for row in rows]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(segments, [row["gpu0_peak_mib"] for row in rows], "o-", label="GPU0")
    axes[0, 0].plot(segments, [row["gpu1_peak_mib"] for row in rows], "o-", label="GPU1")
    axes[0, 0].set_ylabel("VRAM peak (MiB)")
    axes[0, 0].legend()
    axes[0, 1].plot(segments, [row["physical_peak_mib"] for row in rows], "o-", label="RAM")
    axes[0, 1].plot(segments, [row["committed_peak_mib"] for row in rows], "o-", label="Commit")
    axes[0, 1].set_ylabel("System memory peak (MiB)")
    axes[0, 1].legend()
    axes[1, 0].plot(segments, [row["pagefile_peak_mib"] for row in rows], "o-")
    axes[1, 0].set_ylabel("Pagefile used peak (MiB)")
    axes[1, 1].plot(segments, [row["sampling_seconds"] for row in rows], "o-")
    axes[1, 1].set_ylabel("Dual-rank sampling (s)")
    for axis in axes.flat:
        axis.set_xlabel("Segment")
        axis.set_xticks(segments)
        axis.grid(alpha=0.25)
    figure.suptitle("MiniMax H3 Director TP2/FSDP per-segment resources")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    rows = summarize_segment_resources(
        (artifact_dir / "comfyui.out.log").read_text(encoding="utf-8", errors="replace"),
        _load_monitor(artifact_dir / "run0-monitor.csv"),
    )
    (artifact_dir / "segment-resources.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fieldnames = list(rows[0])
    with (artifact_dir / "segment-resources.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_plot(artifact_dir / "segment-resources.png", rows)
    print(json.dumps(rows, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
