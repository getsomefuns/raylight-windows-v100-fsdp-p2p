#!/usr/bin/env python3
"""Run a UI MiniMax H3 Director workflow through ComfyUI with full telemetry."""

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


def prepare_director_prompt(
    template: dict,
    run_index: int,
    *,
    diagnostic_steps: int | None = None,
    diffusion_attention: str | None = None,
) -> dict:
    if run_index < 0:
        raise ValueError("run_index must be non-negative")
    prompt = copy.deepcopy(template)
    directors = _nodes(prompt, "MiniMaxH3Director")
    if len(directors) != 1:
        raise ValueError(f"expected one MiniMaxH3Director, found {len(directors)}")
    director_inputs = directors[0][1].setdefault("inputs", {})
    director_inputs["seed"] = (int(director_inputs["seed"]) + run_index) % (2**64)
    if diagnostic_steps is not None:
        if diagnostic_steps <= 0:
            raise ValueError("diagnostic_steps must be positive")
        director_inputs["steps"] = diagnostic_steps
    if diffusion_attention is not None:
        if not diffusion_attention.strip():
            raise ValueError("diffusion_attention must not be empty")
        initializers = _nodes(prompt, "RayInitializer")
        if len(initializers) != 1:
            raise ValueError(f"expected one RayInitializer, found {len(initializers)}")
        initializers[0][1].setdefault("inputs", {})["XFuser_attention"] = diffusion_attention
    for _, node in _nodes(prompt, "SaveVideo"):
        inputs = node.setdefault("inputs", {})
        prefix = str(inputs.get("filename_prefix", "video/MMH3Director"))
        inputs["filename_prefix"] = f"{prefix}_run{run_index}"
    return prompt


def summarize_director_phases(
    events: list[dict],
    profiles: list[dict],
    *,
    execution_start_ms: int,
    execution_end_ms: int,
) -> dict:
    required_order = (
        "prepare_sampling_begin",
        "load_models_gpu_begin",
        "load_models_gpu_end",
        "prepare_sampling_end",
    )
    required = set(required_order)
    start_ns = execution_start_ms * 1_000_000
    end_ns = execution_end_ms * 1_000_000
    events_by_rank: dict[int, list[dict]] = {0: [], 1: []}
    for event in events:
        name = event.get("event")
        event_time_ns = int(event.get("time_ns", -1))
        if name not in required or not start_ns <= event_time_ns <= end_ns:
            continue
        rank = int(event["rank"])
        if rank in events_by_rank:
            events_by_rank[rank].append(event)

    sequences_by_rank: dict[int, list[dict[str, dict]]] = {}
    for rank in (0, 1):
        rank_events = sorted(events_by_rank[rank], key=lambda event: int(event["perf_ns"]))
        sequences: list[dict[str, dict]] = []
        current: dict[str, dict] = {}
        for event in rank_events:
            name = event["event"]
            if name == required_order[0]:
                if current:
                    raise RuntimeError(f"incomplete Director rank {rank} preparation sequence")
                current = {name: event}
                continue
            if not current:
                continue
            expected = required_order[len(current)]
            if name != expected:
                raise RuntimeError(
                    f"out-of-order Director rank {rank} diagnostic: expected {expected}, got {name}"
                )
            current[name] = event
            if name == required_order[-1]:
                sequences.append(current)
                current = {}
        if current:
            raise RuntimeError(f"incomplete Director rank {rank} preparation sequence")
        sequences_by_rank[rank] = sequences

    counts = {rank: len(values) for rank, values in sequences_by_rank.items()}
    if not counts[0] or counts[0] != counts[1]:
        found = [rank for rank, count in counts.items() if count]
        raise RuntimeError(
            f"Director operations require ranks 0 and 1 with equal counts; found ranks {found}, counts={counts}"
        )

    timed_profiles = []
    for profile in profiles:
        profile_time_ns = int(profile.get("time_ns", -1))
        if (
            profile.get("status") == "success"
            and int(profile.get("rank", -1)) in (0, 1)
            and start_ns <= profile_time_ns <= end_ns
        ):
            timed_profiles.append(profile)
    timed_profiles.sort(key=lambda profile: int(profile["time_ns"]))
    profile_groups: list[dict[int, dict]] = []
    profile_group_last_ns: list[int] = []
    pair_window_ns = 2_000_000_000
    for profile in timed_profiles:
        rank = int(profile["rank"])
        profile_time_ns = int(profile["time_ns"])
        if (
            not profile_groups
            or rank in profile_groups[-1]
            or profile_time_ns - profile_group_last_ns[-1] > pair_window_ns
        ):
            profile_groups.append({rank: profile})
            profile_group_last_ns.append(profile_time_ns)
        else:
            profile_groups[-1][rank] = profile
            profile_group_last_ns[-1] = profile_time_ns
    if len(profile_groups) != counts[0]:
        raise RuntimeError(
            f"Director P2P profiles do not match preparation sequences: profiles={len(profile_groups)} preparations={counts[0]}"
        )

    rows = []
    worker_pids = set()
    for operation_index, profile_group in enumerate(profile_groups):
        per_rank = {}
        for rank in (0, 1):
            values = sequences_by_rank[rank][operation_index]
            pid = int(values["prepare_sampling_begin"]["pid"])
            worker_pids.add(pid)
            per_rank[str(rank)] = {
                "pid": pid,
                "model_to_gpu_seconds": (
                    int(values["load_models_gpu_end"]["perf_ns"])
                    - int(values["load_models_gpu_begin"]["perf_ns"])
                )
                / 1e9,
                "prepare_sampling_seconds": (
                    int(values["prepare_sampling_end"]["perf_ns"])
                    - int(values["prepare_sampling_begin"]["perf_ns"])
                )
                / 1e9,
            }
        sampling_seconds = max(int(profile["elapsed_ns"]) for profile in profile_group.values()) / 1e9
        calls = {
            int(profile.get("profile", {})
                .get("collectives", {})
                .get("all_to_all", {})
                .get("calls", 0))
            for profile in profile_group.values()
        }
        if len(calls) != 1:
            raise RuntimeError(f"Director operation {operation_index} rank profiles disagree: {calls}")
        rows.append(
            {
                "operation_index": operation_index,
                "reported_invocations": sorted(
                    {int(profile.get("invocation", -1)) for profile in profile_group.values()}
                ),
                "sampling_seconds": sampling_seconds,
                "model_to_gpu_seconds": max(
                    value["model_to_gpu_seconds"] for value in per_rank.values()
                ),
                "prepare_sampling_seconds": max(
                    value["prepare_sampling_seconds"] for value in per_rank.values()
                ),
                "all_to_all_calls": calls.pop(),
                "profiled_ranks": sorted(profile_group),
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
        "prepare_sampling_seconds": sum(row["prepare_sampling_seconds"] for row in rows),
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
                extract_p2p_profiles(text),
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
    reserve_vram_gib: float,
    runs: int,
    diagnostic_steps: int | None = None,
    diffusion_attention: str | None = None,
) -> Path:
    if p2p_capacity_mib <= 0:
        raise ValueError("p2p_capacity_mib must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if reserve_vram_gib <= 0:
        raise ValueError("reserve_vram_gib must be positive")
    if diagnostic_steps is not None and diagnostic_steps <= 0:
        raise ValueError("diagnostic_steps must be positive")
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
        str(reserve_vram_gib),
    ]
    report = {
        "workflow": str(workflow_path),
        "workflow_sha256": base.sha256_file(workflow_path),
        "runtime_identity": identity,
        "p2p_capacity_bytes": p2p_capacity_bytes,
        "reserve_vram_gib": reserve_vram_gib,
        "diagnostic_steps": diagnostic_steps,
        "diffusion_attention": diffusion_attention,
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
            report["api_prompt_template_sha256"] = base.write_prompt_artifact(
                result_dir / "api-prompt-template.json", prompt
            )

            for run_index in range(runs):
                run_prompt = prepare_director_prompt(
                    prompt,
                    run_index,
                    diagnostic_steps=diagnostic_steps,
                    diffusion_attention=diffusion_attention,
                )
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
                    "director_seed": int(
                        _nodes(run_prompt, "MiniMaxH3Director")[0][1]["inputs"]["seed"]
                    ),
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
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / ".local" / "artifacts" / "o10" / "mmh3director",
    )
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--p2p-capacity-mib", type=int, default=128)
    parser.add_argument("--reserve-vram-gib", type=float, default=4.0)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--diagnostic-steps", type=int)
    parser.add_argument("--diffusion-attention")
    args = parser.parse_args(argv)
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.reserve_vram_gib <= 0:
        raise ValueError("reserve-vram-gib must be positive")
    if args.diagnostic_steps is not None and args.diagnostic_steps <= 0:
        raise ValueError("diagnostic-steps must be positive")
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
            reserve_vram_gib=args.reserve_vram_gib,
            runs=args.runs,
            diagnostic_steps=args.diagnostic_steps,
            diffusion_attention=args.diffusion_attention,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
