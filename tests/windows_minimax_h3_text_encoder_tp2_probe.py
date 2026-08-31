#!/usr/bin/env python3
"""Real dual-V100 probe for the O10 single-process CUDA P2P SUM primitive."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from raylight.text_encoder_parallel.collectives import CudaPeerGroup2


def _make_inputs(size_bytes: int, dtype: torch.dtype):
    element_size = torch.empty((), dtype=dtype).element_size()
    if size_bytes % element_size:
        raise ValueError(f"size_bytes must be divisible by {element_size}")
    elements = size_bytes // element_size
    cpu0 = ((torch.arange(elements, dtype=torch.int32) % 97) - 48).to(dtype) / 32
    cpu1 = ((torch.arange(elements, dtype=torch.int32) % 89) - 44).to(dtype) / 16
    expected = (cpu0.float() + cpu1.float()).to(dtype)
    return cpu0.to("cuda:0"), cpu1.to("cuda:1"), expected


def run_probe(*, sizes_mib: tuple[int, ...], iterations: int) -> dict:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    group = CudaPeerGroup2()
    group.validate()
    operation_id = 1
    cases = []
    started_ns = time.time_ns()
    try:
        for dtype in (torch.float16, torch.float32):
            for size_mib in sizes_mib:
                size_bytes = size_mib * 2**20
                rank0, rank1, expected = _make_inputs(size_bytes, dtype)
                before = {
                    "0": torch.cuda.memory_allocated(0),
                    "1": torch.cuda.memory_allocated(1),
                }
                warm0, warm1, _ = group.all_reduce_sum(rank0, rank1, operation_id)
                operation_id += 1
                torch.testing.assert_close(warm0.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(warm1.cpu(), expected, rtol=0, atol=0)
                first_outputs = None
                steady_allocated = None
                elapsed_ms = 0.0
                last_outputs = None
                for iteration in range(iterations):
                    out0, out1, stats = group.all_reduce_sum(rank0, rank1, operation_id)
                    if iteration == 0:
                        first_outputs = (out0.clone(), out1.clone())
                        steady_allocated = {
                            "0": torch.cuda.memory_allocated(0),
                            "1": torch.cuda.memory_allocated(1),
                        }
                    last_outputs = (out0, out1)
                    elapsed_ms += stats.elapsed_ms
                    operation_id += 1
                assert first_outputs is not None and last_outputs is not None
                for output in (*first_outputs, *last_outputs):
                    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                    if not torch.isfinite(output).all().item():
                        raise AssertionError("P2P SUM output contains NaN or Inf")
                after = {
                    "0": torch.cuda.memory_allocated(0),
                    "1": torch.cuda.memory_allocated(1),
                }
                assert steady_allocated is not None
                setup_growth = {device: steady_allocated[device] - before[device] for device in ("0", "1")}
                iteration_growth = {
                    device: after[device] - steady_allocated[device] for device in ("0", "1")
                }
                if any(growth > 2**20 for growth in iteration_growth.values()):
                    raise AssertionError(
                        f"allocated memory grew across iterations: {iteration_growth}"
                    )
                aggregate_bytes = size_bytes * 2 * iterations
                cases.append(
                    {
                        "dtype": str(dtype).removeprefix("torch."),
                        "size_mib_per_rank": size_mib,
                        "iterations": iterations,
                        "warmup_iterations": 1,
                        "average_elapsed_ms": elapsed_ms / iterations,
                        "aggregate_gib_s": aggregate_bytes / 2**30 / (elapsed_ms / 1000),
                        "setup_growth_bytes": setup_growth,
                        "iteration_growth_bytes": iteration_growth,
                        "finite": True,
                        "exact": True,
                    }
                )
                del rank0, rank1, expected, first_outputs, last_outputs, out0, out1, warm0, warm1
    finally:
        snapshot = group.profile_snapshot()
        group.close()
        torch.cuda.empty_cache()
    if not all(math.isfinite(case["aggregate_gib_s"]) for case in cases):
        raise AssertionError("probe produced a non-finite bandwidth result")
    return {
        "schema_version": 1,
        "devices": [torch.cuda.get_device_name(0), torch.cuda.get_device_name(1)],
        "cases": cases,
        "profile": snapshot,
        "started_ns": started_ns,
        "finished_ns": time.time_ns(),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-mib", default="1,16,64")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    args.sizes_mib = tuple(int(value) for value in args.sizes_mib.split(",") if value.strip())
    if not args.sizes_mib or any(value <= 0 for value in args.sizes_mib):
        raise ValueError("--sizes-mib must contain positive integers")
    return args


def main() -> int:
    args = parse_args()
    result = run_probe(sizes_mib=args.sizes_mib, iterations=args.iterations)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
