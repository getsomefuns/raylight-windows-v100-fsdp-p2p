#!/usr/bin/env python3
"""Run a UI MiniMax H3 Director workflow through ComfyUI with full O9 telemetry."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import uuid

import websocket


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
BASE_SCRIPT = SCRIPT_ROOT / "benchmark_cold_warm.py"
CONVERTER_SCRIPT = SCRIPT_ROOT / "workflow_to_api.py"
P2P_PROFILE_PATTERN = re.compile(r"\[RAYLIGHT_P2P_PROFILE\] (\{.*\})")


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_script("raylight_minimax_benchmark_base", BASE_SCRIPT)
converter = _load_script("raylight_minimax_workflow_converter", CONVERTER_SCRIPT)


def _nodes(prompt: dict, class_type: str) -> list[tuple[str, dict]]:
    return [
        (node_id, node)
        for node_id, node in prompt.items()
        if node.get("class_type") == class_type
    ]


def validate_director_prompt(prompt: dict) -> None:
    directors = _nodes(prompt, "MiniMaxH3Director")
    initializers = _nodes(prompt, "RayInitializer")
    adapters = _nodes(prompt, "RayMMH3DirectorModelAdapter")
    if len(directors) != 1:
        raise ValueError(f"expected one MiniMaxH3Director, found {len(directors)}")
    if len(initializers) != 1:
        raise ValueError(f"expected one RayInitializer, found {len(initializers)}")
    if len(adapters) != 1:
        raise ValueError(f"expected one RayMMH3DirectorModelAdapter, found {len(adapters)}")
    if not _nodes(prompt, "SaveVideo"):
        raise ValueError("Director benchmark requires at least one SaveVideo")
    inputs = initializers[0][1].get("inputs", {})
    if int(inputs.get("GPU", 0)) != 2:
        raise ValueError("Director benchmark requires GPU=2")
    if not bool(inputs.get("FSDP")):
        raise ValueError("Director benchmark requires FSDP=true")


def summarize_director_phases(
    events: list[dict],
    *,
    execution_start_ms: int,
    execution_end_ms: int,
) -> dict:
    required = {
        "sampler_entry",
        "load_models_gpu_begin",
        "load_models_gpu_end",
        "sample_begin",
        "sample_returned",
        "sampler_return",
    }
    start_ns = execution_start_ms * 1_000_000
    end_ns = execution_end_ms * 1_000_000
    groups: dict[tuple[int, int, int], dict[str, dict]] = {}
    for event in events:
        name = event.get("event")
        event_time_ns = int(event.get("time_ns", -1))
        if name not in required or not start_ns <= event_time_ns <= end_ns:
            continue
        key = (int(event["invocation"]), int(event["rank"]), int(event["pid"]))
        target = groups.setdefault(key, {})
        if name in target:
            raise RuntimeError(f"duplicate {name} diagnostic for {key}")
        target[name] = event

    complete: dict[int, dict[int, tuple[tuple[int, int, int], dict[str, dict]]]] = {}
    for key, values in groups.items():
        if required <= values.keys():
            invocation, rank, _ = key
            if rank in complete.setdefault(invocation, {}):
                raise RuntimeError(f"duplicate complete rank {rank} for invocation {invocation}")
            complete[invocation][rank] = (key, values)
    if not complete:
        raise RuntimeError("no complete Director rank diagnostics found")

    rows = []
    worker_pids = set()
    for invocation, ranks in sorted(complete.items()):
        if set(ranks) != {0, 1}:
            raise RuntimeError(
                f"Director invocation {invocation} requires ranks 0 and 1, found {sorted(ranks)}"
            )

        def seconds(rank: int, end: str, begin: str) -> float:
            values = ranks[rank][1]
            return (values[end]["perf_ns"] - values[begin]["perf_ns"]) / 1e9

        per_rank = {}
        for rank in (0, 1):
            key, _ = ranks[rank]
            worker_pids.add(key[2])
            per_rank[str(rank)] = {
                "pid": key[2],
                "model_to_gpu_seconds": seconds(rank, "load_models_gpu_end", "load_models_gpu_begin"),
                "sampling_seconds": seconds(rank, "sample_returned", "sample_begin"),
                "sampler_total_seconds": seconds(rank, "sampler_return", "sampler_entry"),
            }
        rows.append(
            {
                "invocation": invocation,
                "sampling_seconds": max(value["sampling_seconds"] for value in per_rank.values()),
                "sampler_total_seconds": max(value["sampler_total_seconds"] for value in per_rank.values()),
                "model_to_gpu_seconds": max(value["model_to_gpu_seconds"] for value in per_rank.values()),
                "per_rank": per_rank,
            }
        )

    return {
        "invocation_count": len(rows),
        "worker_pids": sorted(worker_pids),
        "worker_pids_by_rank": {
            rank: rows[0]["per_rank"][rank]["pid"] for rank in ("0", "1")
        },
        "sampling_seconds": sum(row["sampling_seconds"] for row in rows),
        "sampler_total_seconds": sum(row["sampler_total_seconds"] for row in rows),
        "model_to_gpu_seconds": sum(row["model_to_gpu_seconds"] for row in rows),
        "invocations": rows,
    }


def wait_for_director_phases(read_segment, *, execution_start_ms: int, execution_end_ms: int):
    deadline = time.monotonic() + 60
    last_error = None
    while True:
        text, end_offset = read_segment()
        try:
            summary = summarize_director_phases(
                base.extract_rank_diag(text),
                execution_start_ms=execution_start_ms,
                execution_end_ms=execution_end_ms,
            )
            return summary, text, end_offset
        except RuntimeError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise RuntimeError("Director rank diagnostics remained incomplete") from last_error
        time.sleep(0.25)


def extract_p2p_profiles(text: str) -> list[dict]:
    profiles = []
    for line in text.splitlines():
        match = P2P_PROFILE_PATTERN.search(line)
        if not match:
            continue
        try:
            profiles.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return profiles


def run_workflow(
    workflow_path: Path,
    *,
    result_root: Path,
    port: int,
    timeout: int,
    p2p_capacity_mib: int,
    runs: int,
) -> Path:
    if p2p_capacity_mib <= 0:
        raise ValueError("p2p_capacity_mib must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")
    workflow_path = workflow_path.resolve()
    result_root = result_root.resolve()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = result_root / f"{timestamp}-director"
    result_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = result_dir / "comfyui.out.log"
    stderr_path = result_dir / "comfyui.err.log"
    report_path = result_dir / "benchmark.json"
    base_url = f"http://127.0.0.1:{port}"
    identity = base.collect_runtime_identity()
    base.validate_runtime_identity(identity)
    try:
        base.request_json(f"{base_url}/system_stats", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError(f"port {port} already has a running ComfyUI server")

    p2p_capacity_bytes = int(p2p_capacity_mib) * 2**20
    command = [
        str(base.PYTHON),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-cuda-malloc",
        "--reserve-vram",
        "3",
    ]
    report = {
        "workflow": str(workflow_path),
        "workflow_sha256": base.sha256_file(workflow_path),
        "runtime_identity": identity,
        "p2p_capacity_bytes": p2p_capacity_bytes,
        "started_ns": time.time_ns(),
        "runs": [],
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=base.COMFY_ROOT,
            env=base.server_environment(p2p_capacity_bytes=p2p_capacity_bytes),
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        report["server_pid"] = process.pid
        job = None
        try:
            job = base.attach_kill_on_close_job(process)
            base.wait_server(base_url, process)
            object_info = base.request_json(f"{base_url}/object_info", timeout=120)
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            prompt = converter.workflow_to_prompt(workflow, object_info)
            base.normalize_prompt_file_selectors(prompt, object_info)
            validate_director_prompt(prompt)
            director_id = _nodes(prompt, "MiniMaxH3Director")[0][0]
            for _, node in _nodes(prompt, "RayInitializer"):
                node["inputs"]["skip_comm_test"] = True
            for _, node in _nodes(prompt, "SaveVideo"):
                prefix = str(node.get("inputs", {}).get("filename_prefix", "video/MMH3Director"))
                node["inputs"]["filename_prefix"] = f"{prefix}_O9Adaptive128"
            report["api_prompt_template_sha256"] = base.write_prompt_artifact(
                result_dir / "api-prompt-template.json", prompt
            )

            for run_index in range(runs):
                run_prompt = copy.deepcopy(prompt)
                for _, node in _nodes(run_prompt, "SaveVideo"):
                    prefix = str(node.get("inputs", {}).get("filename_prefix", "video/MMH3Director"))
                    node["inputs"]["filename_prefix"] = f"{prefix}_run{run_index}"
                prompt_sha256 = base.write_prompt_artifact(
                    result_dir / f"run{run_index}-prompt.json", run_prompt
                )
                stdout.flush()
                stderr.flush()
                log_start = stdout_path.stat().st_size
                stderr_start = stderr_path.stat().st_size
                samples: list[dict] = []
                monitor_errors: list[str] = []
                stop_event = threading.Event()
                monitor = threading.Thread(
                    target=base.monitor_resources,
                    args=(stop_event, samples, monitor_errors),
                    daemon=True,
                )
                client_id = str(uuid.uuid4())
                connection = websocket.create_connection(
                    f"ws://127.0.0.1:{port}/ws?clientId={client_id}",
                    timeout=10,
                    suppress_origin=True,
                )
                connection.settimeout(1.0)
                node_events: list[dict] = []
                node_event_errors: list[str] = []
                node_stop_event = threading.Event()
                receiver = threading.Thread(
                    target=base.receive_node_events,
                    args=(connection, node_stop_event, node_events, node_event_errors),
                    daemon=True,
                )
                receiver.start()
                monitor.start()
                started = time.perf_counter()
                try:
                    response = base.request_json(
                        f"{base_url}/prompt",
                        {"prompt": run_prompt, "client_id": client_id},
                        timeout=60,
                    )
                    prompt_id = response["prompt_id"]
                    history = base.wait_prompt(base_url, prompt_id, process, timeout)
                    base.wait_for_terminal_node_event(node_events, prompt_id=prompt_id)
                finally:
                    stop_event.set()
                    monitor.join(timeout=15)
                    node_stop_event.set()
                    connection.close()
                    receiver.join(timeout=15)
                if monitor.is_alive() or receiver.is_alive():
                    raise RuntimeError("Director benchmark monitor did not stop")
                if node_event_errors:
                    raise RuntimeError(f"node event receiver failed: {node_event_errors[0]}")
                base.validate_monitor_samples(samples, monitor_errors)
                elapsed = time.perf_counter() - started
                base.reject_cached_sampler(history, director_id)
                execution_start_ms = base._history_timestamp_ms(history, "execution_start")
                execution_end_ms = base._history_timestamp_ms(history, "execution_success")

                def read_log_segment():
                    stdout.flush()
                    current_end = stdout_path.stat().st_size
                    with stdout_path.open("rb") as handle:
                        handle.seek(log_start)
                        text = handle.read(current_end - log_start).decode("utf-8", errors="replace")
                    return text, current_end

                phases, log_segment, log_end = wait_for_director_phases(
                    read_log_segment,
                    execution_start_ms=execution_start_ms,
                    execution_end_ms=execution_end_ms,
                )
                stderr.flush()
                stderr_end = stderr_path.stat().st_size
                with stderr_path.open("rb") as handle:
                    handle.seek(stderr_start)
                    stderr_segment = handle.read(stderr_end - stderr_start).decode(
                        "utf-8", errors="replace"
                    )
                run = {
                    "run_index": run_index,
                    "temperature": "cold" if run_index == 0 else "warm",
                    "prompt_id": prompt_id,
                    "elapsed_seconds": elapsed,
                    "api_prompt_sha256": prompt_sha256,
                    "phases": phases,
                    "node_timings": base.summarize_node_timings(
                        node_events, run_prompt, prompt_id=prompt_id
                    ),
                    "resources": base.summarize_resources(samples),
                    "outputs": history.get("outputs", {}),
                    "p2p_profiles": extract_p2p_profiles(log_segment),
                    "stderr_progress": base.extract_sampler_progress(
                        stderr_segment,
                        phases["worker_pids_by_rank"],
                    ),
                    "log_start": log_start,
                    "log_end": log_end,
                }
                report["runs"].append(run)
                base.write_monitor_csv(result_dir / f"run{run_index}-monitor.csv", samples)
                report_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                print(json.dumps(run, ensure_ascii=False), flush=True)
                time.sleep(5)
        finally:
            base.terminate_process_tree(process, job)
            report["finished_ns"] = time.time_ns()
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--result-root", type=Path, default=REPO_ROOT / ".local" / "artifacts" / "o9" / "mmh3director")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--p2p-capacity-mib", type=int, default=128)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args(argv)
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    return args


def main() -> int:
    args = parse_args()
    print(
        run_workflow(
            args.workflow,
            result_root=args.result_root,
            port=args.port,
            timeout=args.timeout,
            p2p_capacity_mib=args.p2p_capacity_mib,
            runs=args.runs,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
