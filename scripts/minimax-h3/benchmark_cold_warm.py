#!/usr/bin/env python3
"""Run repeatable MiniMax H3 cold/warm benchmarks through the ComfyUI API."""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

import psutil


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
COMFY_ROOT = Path(os.environ.get("RAYLIGHT_COMFY_ROOT", REPO_ROOT.parent / "ComfyUI"))
ENV_ROOT = COMFY_ROOT.parent
PYTHON = Path(os.environ.get("RAYLIGHT_PYTHON", ENV_ROOT / "Python310" / "python.exe"))
RESULT_ROOT = Path(
    os.environ.get("RAYLIGHT_MINIMAX_BENCHMARK_ROOT", ENV_ROOT / "logs" / "minimax-h3" / "o2")
)
SEED_STRIDE = 1_000_003
P2P_CAPACITY_BYTES = 128 * 1024 * 1024
RANK_DIAG_PATTERN = re.compile(r"\[RAYLIGHT_RANK_DIAG\] (\{.*\})")
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class WindowsKillOnCloseJob:
    """Own the benchmark process tree without relying on reusable parent PIDs."""

    def __init__(self):
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        kernel32.SetInformationJobObject.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        kernel32.SetInformationJobObject.restype = ctypes.c_int
        kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        kernel32.AssignProcessToJobObject.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        information = _ExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise ctypes.WinError(error)
        self._kernel32 = kernel32
        self._handle = handle

    def assign(self, process: subprocess.Popen) -> None:
        if not self._kernel32.AssignProcessToJobObject(self._handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def attach_kill_on_close_job(process: subprocess.Popen, factory=WindowsKillOnCloseJob):
    candidate = factory()
    try:
        candidate.assign(process)
    except BaseException:
        candidate.close()
        raise
    return candidate


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nodes(prompt: dict, class_type: str) -> list[tuple[str, dict]]:
    return [(node_id, node) for node_id, node in prompt.items() if node.get("class_type") == class_type]


def _one_node(prompt: dict, class_type: str) -> tuple[str, dict]:
    matches = _nodes(prompt, class_type)
    if len(matches) != 1:
        raise ValueError(f"expected one {class_type}, found {len(matches)}")
    return matches[0]


def validate_prompt(prompt: dict) -> None:
    _, initializer = _one_node(prompt, "RayInitializer")
    inputs = initializer["inputs"]
    if int(inputs.get("GPU", 0)) != 2:
        raise ValueError("MiniMax benchmark requires GPU=2")
    if not bool(inputs.get("FSDP")):
        raise ValueError("MiniMax benchmark requires FSDP=true")
    _one_node(prompt, "RayUNETLoader")
    _one_node(prompt, "XFuserSamplerCustomAdvanced")
    _one_node(prompt, "SaveVideo")


def prepare_prompt(template: dict, mode: str, run_index: int) -> dict:
    prompt = copy.deepcopy(template)
    _, sampler = _one_node(prompt, "XFuserSamplerCustomAdvanced")
    sampler["inputs"]["noise_seed"] = int(sampler["inputs"]["noise_seed"]) + run_index * SEED_STRIDE
    _, save_video = _one_node(prompt, "SaveVideo")
    save_video["inputs"]["filename_prefix"] = f"video/raylight_o2/minimax_h3_{mode}_run{run_index}"
    validate_prompt(prompt)
    return prompt


def request_json(url: str, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def server_environment(base: dict | None = None) -> dict:
    environment = dict(os.environ if base is None else base)
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "USE_LIBUV": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RAYLIGHT_GLOO_HOST": "127.0.0.1",
            "RAY_DEBUG_DISABLE_MEMORY_MONITOR": "1",
            "RAY_memory_usage_threshold": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "RAYLIGHT_WINDOWS_P2P": "1",
            "RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES": str(P2P_CAPACITY_BYTES),
            "RAYLIGHT_WINDOWS_P2P_MIN_GIB_S": "50",
            "RAYLIGHT_WINDOWS_P2P_TIMEOUT_SECONDS": "10",
            "RAYLIGHT_P2P_PROFILE": "1",
            "RAYLIGHT_RANK_DIAG": "1",
        }
    )
    return environment


def committed_memory_mib() -> float | None:
    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        return None
    return info.CommitTotal * info.PageSize / 2**20


def gpu_metrics() -> list[dict]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode:
        raise RuntimeError(f"nvidia-smi failed with exit code {completed.returncode}")
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            rows.append(
                {
                    "index": int(fields[0]),
                    "memory_mib": float(fields[1]),
                    "util_percent": float(fields[2]),
                    "power_w": float(fields[3]),
                }
            )
    if {row["index"] for row in rows} != {0, 1}:
        raise RuntimeError(f"resource sample must include GPU 0 and 1, found {rows}")
    return rows


def monitor_resources(stop_event: threading.Event, samples: list[dict], errors: list[str]) -> None:
    try:
        while not stop_event.is_set():
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            committed = committed_memory_mib()
            if committed is None:
                raise RuntimeError("GetPerformanceInfo did not return committed memory")
            samples.append(
                {
                    "time_ns": time.time_ns(),
                    "physical_used_mib": memory.used / 2**20,
                    "committed_mib": committed,
                    "pagefile_used_mib": swap.used / 2**20,
                    "gpus": gpu_metrics(),
                }
            )
            stop_event.wait(1.0)
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        stop_event.set()


def validate_monitor_samples(samples: list[dict], errors: list[str], *, minimum_samples: int = 5) -> None:
    if errors:
        raise RuntimeError(f"resource monitor failed: {errors[0]}")
    if len(samples) < minimum_samples:
        raise RuntimeError(f"resource monitor captured too few samples: {len(samples)} < {minimum_samples}")
    if any({gpu.get("index") for gpu in sample.get("gpus", [])} != {0, 1} for sample in samples):
        raise RuntimeError("every resource sample must include GPU 0 and 1")


def summarize_resources(samples: list[dict]) -> dict:
    summary = {
        "samples": len(samples),
        "start_physical_used_mib": samples[0]["physical_used_mib"] if samples else 0,
        "end_physical_used_mib": samples[-1]["physical_used_mib"] if samples else 0,
        "peak_physical_used_mib": max((row["physical_used_mib"] for row in samples), default=0),
        "start_committed_mib": samples[0]["committed_mib"] if samples else 0,
        "end_committed_mib": samples[-1]["committed_mib"] if samples else 0,
        "peak_committed_mib": max((row["committed_mib"] or 0 for row in samples), default=0),
        "start_pagefile_used_mib": samples[0]["pagefile_used_mib"] if samples else 0,
        "end_pagefile_used_mib": samples[-1]["pagefile_used_mib"] if samples else 0,
        "peak_pagefile_used_mib": max((row["pagefile_used_mib"] for row in samples), default=0),
        "gpus": {},
    }
    for row in samples:
        for gpu in row["gpus"]:
            target = summary["gpus"].setdefault(
                str(gpu["index"]),
                {"peak_memory_mib": 0, "peak_util_percent": 0, "peak_power_w": 0},
            )
            target["peak_memory_mib"] = max(target["peak_memory_mib"], gpu["memory_mib"])
            target["peak_util_percent"] = max(target["peak_util_percent"], gpu["util_percent"])
            target["peak_power_w"] = max(target["peak_power_w"], gpu["power_w"])
    return summary


def extract_rank_diag(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        match = RANK_DIAG_PATTERN.search(line)
        if not match:
            continue
        try:
            events.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            continue
    return events


def _history_timestamp_ms(history: dict, message_name: str) -> int:
    for name, payload in history.get("status", {}).get("messages", []):
        if name == message_name and "timestamp" in payload:
            return int(payload["timestamp"])
    raise RuntimeError(f"history is missing {message_name} timestamp")


def summarize_phases(events: list[dict], *, execution_start_ms: int, execution_end_ms: int) -> dict:
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
        try:
            key = (int(event["rank"]), int(event["pid"]), int(event["invocation"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"diagnostic event is missing rank/pid/invocation: {event}") from exc
        target = groups.setdefault(key, {})
        if name in target:
            raise RuntimeError(f"duplicate {name} diagnostic for {key}")
        target[name] = event

    complete_by_rank: dict[int, list[tuple[tuple[int, int, int], dict[str, dict]]]] = {}
    for key, values in groups.items():
        if required <= values.keys():
            complete_by_rank.setdefault(key[0], []).append((key, values))
    if set(complete_by_rank) != {0, 1}:
        raise RuntimeError(f"expected complete diagnostics from ranks 0 and 1, found {sorted(complete_by_rank)}")
    selected: dict[int, tuple[tuple[int, int, int], dict[str, dict]]] = {}
    for rank in (0, 1):
        matches = complete_by_rank[rank]
        if len(matches) != 1:
            raise RuntimeError(f"rank {rank} must have exactly one complete invocation, found {len(matches)}")
        selected[rank] = matches[0]

    def seconds(rank: int, end: str, begin: str) -> float:
        values = selected[rank][1]
        return (values[end]["perf_ns"] - values[begin]["perf_ns"]) / 1e9

    latest_entry_ns = max(values["sampler_entry"]["time_ns"] for _, values in selected.values())
    latest_return_ns = max(values["sampler_return"]["time_ns"] for _, values in selected.values())
    per_rank = {}
    for rank, (key, values) in sorted(selected.items()):
        per_rank[str(rank)] = {
            "pid": key[1],
            "invocation": key[2],
            "model_to_gpu_seconds": seconds(rank, "load_models_gpu_end", "load_models_gpu_begin"),
            "sampling_seconds": seconds(rank, "sample_returned", "sample_begin"),
            "sampler_total_seconds": seconds(rank, "sampler_return", "sampler_entry"),
        }
    return {
        "rank_count": len(selected),
        "worker_pids": sorted({row["pid"] for row in per_rank.values()}),
        "pre_sampler_seconds": max(0.0, latest_entry_ns / 1e6 - execution_start_ms) / 1e3,
        "model_to_gpu_seconds": max(row["model_to_gpu_seconds"] for row in per_rank.values()),
        "sampling_seconds": max(row["sampling_seconds"] for row in per_rank.values()),
        "sampler_total_seconds": max(row["sampler_total_seconds"] for row in per_rank.values()),
        "decode_and_write_tail_seconds": max(0.0, execution_end_ms - latest_return_ns / 1e6) / 1e3,
        "per_rank": per_rank,
    }


def wait_for_phase_diagnostics(
    read_segment,
    *,
    execution_start_ms: int,
    execution_end_ms: int,
    timeout: float = 30.0,
    sleep=time.sleep,
) -> tuple[dict, str, int]:
    deadline = time.monotonic() + timeout
    last_error = None
    while True:
        text, end_offset = read_segment()
        try:
            phases = summarize_phases(
                extract_rank_diag(text),
                execution_start_ms=execution_start_ms,
                execution_end_ms=execution_end_ms,
            )
            return phases, text, end_offset
        except RuntimeError as exc:
            last_error = exc
        if time.monotonic() >= deadline:
            raise RuntimeError(f"rank diagnostics remained incomplete after {timeout:.1f}s") from last_error
        sleep(0.25)


def reject_cached_sampler(history: dict, sampler_node_id: str) -> None:
    for name, payload in history.get("status", {}).get("messages", []):
        if name == "execution_cached" and sampler_node_id in {str(node) for node in payload.get("nodes", [])}:
            raise RuntimeError("sampler was served from ComfyUI cache; benchmark is invalid")


def wait_server(base_url: str, process: subprocess.Popen, timeout=300) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during startup with code {process.returncode}")
        try:
            request_json(f"{base_url}/system_stats", timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError("ComfyUI did not become ready")


def wait_prompt(base_url: str, prompt_id: str, process: subprocess.Popen, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ComfyUI exited during prompt {prompt_id} with code {process.returncode}")
        try:
            history = request_json(f"{base_url}/history/{prompt_id}", timeout=10)
        except (TimeoutError, OSError):
            time.sleep(2)
            continue
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                if status.get("status_str") != "success":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False))
                return entry
        time.sleep(2)
    raise TimeoutError(f"prompt {prompt_id} exceeded {timeout} seconds")


def terminate_process_tree(process: subprocess.Popen, job: WindowsKillOnCloseJob | None = None) -> None:
    if job is not None:
        job.close()
        try:
            process.wait(timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            pass
        return
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    processes = parent.children(recursive=True) + [parent]
    for target in processes:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=20)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)


def _git_output(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode:
        raise RuntimeError(f"git {' '.join(arguments)} failed for {repository}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _runtime_source_dirty(repository: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--", "src/raylight"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(f"cannot inspect runtime source state: {completed.stderr.strip()}")
    return completed.returncode == 1


def read_deployed_source_commit(marker_path: Path) -> str:
    try:
        value = marker_path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        raise RuntimeError("installed Raylight deployment marker is missing or invalid") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError("installed Raylight deployment marker is missing or invalid")
    return value


def hash_input_files(paths: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def write_prompt_artifact(path: Path, prompt: dict) -> str:
    canonical = json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return hashlib.sha256(canonical).hexdigest()


def benchmark_input_paths(mode: str) -> list[Path]:
    workflow_name = {
        "i2v": "Minimax_H3_I2V_Raylight.json",
        "ref2va": "Minimax_H3_REF2VA_Raylight.json",
    }[mode]
    return [
        Path(__file__).resolve(),
        SCRIPT_ROOT / "build_workflows.py",
        SCRIPT_ROOT / "workflow_to_api.py",
        REPO_ROOT / "example_workflows" / workflow_name,
    ]


def collect_runtime_identity() -> dict:
    installed = COMFY_ROOT / "custom_nodes" / "raylight"
    source_status = _git_output(REPO_ROOT, "status", "--short")
    installed_status = _git_output(installed, "status", "--short")
    marker_path = Path(_git_output(installed, "rev-parse", "--git-path", "raylight-deployed-commit"))
    if not marker_path.is_absolute():
        marker_path = installed / marker_path
    deployed_source_commit = read_deployed_source_commit(marker_path)
    return {
        "source": {
            "head": _git_output(REPO_ROOT, "rev-parse", "HEAD"),
            "status": source_status,
            "runtime_dirty": _runtime_source_dirty(REPO_ROOT),
        },
        "installed": {
            "head": _git_output(installed, "rev-parse", "HEAD"),
            "status": installed_status,
            "dirty": bool(installed_status),
        },
        "deployed_source_commit": deployed_source_commit,
    }


def validate_runtime_identity(identity: dict) -> None:
    source = identity["source"]
    installed = identity["installed"]
    if source["runtime_dirty"]:
        raise RuntimeError("runtime source files are dirty; commit and deploy them before benchmarking")
    if installed["dirty"]:
        raise RuntimeError("installed Raylight node is dirty; redeploy before benchmarking")
    if source["head"] != identity["deployed_source_commit"]:
        raise RuntimeError("deployed Raylight source commit does not match the benchmark worktree HEAD")


def build_api_prompt(base_url: str, mode: str, profile: str, cpu_offload: bool) -> dict:
    builder = _load_script("minimax_h3_build_workflows", SCRIPT_ROOT / "build_workflows.py")
    converter = _load_script("minimax_h3_workflow_to_api", SCRIPT_ROOT / "workflow_to_api.py")
    workflow = builder.build_workflow(
        builder.SOURCE_WORKFLOWS[mode],
        mode=mode,
        profile=profile,
    )
    prompt = converter.workflow_to_prompt(workflow, request_json(f"{base_url}/object_info", timeout=120))
    _, initializer = _one_node(prompt, "RayInitializer")
    initializer["inputs"]["FSDP_CPU_OFFLOAD"] = bool(cpu_offload)
    initializer["inputs"]["skip_comm_test"] = True
    validate_prompt(prompt)
    return prompt


def write_monitor_csv(path: Path, samples: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_ns", "physical_used_mib", "committed_mib", "pagefile_used_mib", "gpus_json"])
        for row in samples:
            writer.writerow(
                [
                    row["time_ns"],
                    row["physical_used_mib"],
                    row["committed_mib"],
                    row["pagefile_used_mib"],
                    json.dumps(row["gpus"], separators=(",", ":")),
                ]
            )


def run_benchmark(
    *,
    mode: str,
    profile: str,
    runs: int,
    port: int,
    timeout: int,
    cpu_offload: bool,
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = RESULT_ROOT / f"{timestamp}-{mode}-{profile}"
    result_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = result_dir / "comfyui.out.log"
    stderr_path = result_dir / "comfyui.err.log"
    result_path = result_dir / "benchmark.json"
    base_url = f"http://127.0.0.1:{port}"
    runtime_identity = collect_runtime_identity()
    validate_runtime_identity(runtime_identity)
    try:
        request_json(f"{base_url}/system_stats", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError(f"port {port} already has a running ComfyUI server")

    command = [
        str(PYTHON),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-cuda-malloc",
        "--reserve-vram",
        "2",
    ]
    report = {
        "mode": mode,
        "profile": profile,
        "cpu_offload": cpu_offload,
        "runtime_identity": runtime_identity,
        "benchmark_input_hashes": hash_input_files(benchmark_input_paths(mode)),
        "started_ns": time.time_ns(),
        "runs": [],
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=COMFY_ROOT,
            env=server_environment(),
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        report["server_pid"] = process.pid
        job = None
        try:
            job = attach_kill_on_close_job(process)
            wait_server(base_url, process)
            template = build_api_prompt(base_url, mode, profile, cpu_offload)
            report["api_prompt_template_sha256"] = write_prompt_artifact(
                result_dir / "api-prompt-template.json", template
            )
            sampler_id, _ = _one_node(template, "XFuserSamplerCustomAdvanced")
            for run_index in range(runs):
                prompt = prepare_prompt(template, mode, run_index)
                stdout.flush()
                stderr.flush()
                prompt_sha256 = write_prompt_artifact(result_dir / f"run{run_index}-prompt.json", prompt)
                log_start = stdout_path.stat().st_size
                samples: list[dict] = []
                monitor_errors: list[str] = []
                stop_event = threading.Event()
                monitor = threading.Thread(target=monitor_resources, args=(stop_event, samples, monitor_errors), daemon=True)
                monitor.start()
                started = time.perf_counter()
                try:
                    response = request_json(
                        f"{base_url}/prompt",
                        {"prompt": prompt, "client_id": str(uuid.uuid4())},
                        timeout=60,
                    )
                    prompt_id = response["prompt_id"]
                    history = wait_prompt(base_url, prompt_id, process, timeout)
                finally:
                    stop_event.set()
                    monitor.join(timeout=15)
                if monitor.is_alive():
                    raise RuntimeError("resource monitor did not stop")
                validate_monitor_samples(samples, monitor_errors)
                elapsed = time.perf_counter() - started
                reject_cached_sampler(history, sampler_id)
                execution_start_ms = _history_timestamp_ms(history, "execution_start")
                execution_end_ms = _history_timestamp_ms(history, "execution_success")

                def read_log_segment():
                    stdout.flush()
                    current_end = stdout_path.stat().st_size
                    with stdout_path.open("rb") as handle:
                        handle.seek(log_start)
                        segment = handle.read(current_end - log_start).decode("utf-8", errors="replace")
                    return segment, current_end

                diagnostic_wait_started = time.perf_counter()
                phases, log_segment, log_end = wait_for_phase_diagnostics(
                    read_log_segment,
                    execution_start_ms=execution_start_ms,
                    execution_end_ms=execution_end_ms,
                )
                diagnostic_wait_seconds = time.perf_counter() - diagnostic_wait_started
                run = {
                    "run_index": run_index,
                    "temperature": "cold" if run_index == 0 else "warm",
                    "prompt_id": prompt_id,
                    "elapsed_seconds": elapsed,
                    "diagnostic_log_wait_seconds": diagnostic_wait_seconds,
                    "noise_seed": prompt[sampler_id]["inputs"]["noise_seed"],
                    "api_prompt_sha256": prompt_sha256,
                    "phases": phases,
                    "resources": summarize_resources(samples),
                    "outputs": history.get("outputs", {}),
                    "log_start": log_start,
                    "log_end": log_end,
                    "reuse_markers": {
                        "fsdp_already_registered": log_segment.count("FSDP already registered, skip wrapping"),
                        "checkpoint_changed": log_segment.count("Diffusion checkpoint changed"),
                    },
                }
                report["runs"].append(run)
                write_monitor_csv(result_dir / f"run{run_index}-monitor.csv", samples)
                result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(json.dumps(run, ensure_ascii=False), flush=True)
                time.sleep(5)
        finally:
            terminate_process_tree(process, job)
            report["finished_ns"] = time.time_ns()
            result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("i2v", "ref2va"), required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    print(
        run_benchmark(
            mode=args.mode,
            profile=args.profile,
            runs=args.runs,
            port=args.port,
            timeout=args.timeout,
            cpu_offload=args.cpu_offload,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
