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
import websocket


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
MODEL_MANIFEST = SCRIPT_ROOT / "models.json"
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


def configure_prompt_variant(
    prompt: dict,
    *,
    lora_name: str | None = None,
    steps: int | None = None,
) -> dict:
    if steps is not None:
        if steps < 1:
            raise ValueError("steps must be positive")
        _, scheduler = _one_node(prompt, "RayBasicScheduler")
        scheduler["inputs"]["steps"] = int(steps)

    if lora_name is not None:
        if not lora_name or Path(lora_name).name != lora_name:
            raise ValueError("lora_name must be a filename inside the ComfyUI loras directory")
        existing = _nodes(prompt, "RayLoraLoader")
        if len(existing) > 1:
            raise ValueError(f"expected at most one RayLoraLoader, found {len(existing)}")
        if existing:
            lora_id, lora_node = existing[0]
        else:
            numeric_ids = [int(node_id) for node_id in prompt if str(node_id).isdigit()]
            lora_id = str(max(numeric_ids, default=0) + 1)
            lora_node = {"class_type": "RayLoraLoader", "inputs": {}}
            prompt[lora_id] = lora_node
        lora_node["inputs"] = {"lora_name": lora_name, "strength_model": 1.0}
        _, unet = _one_node(prompt, "RayUNETLoader")
        unet["inputs"]["lora"] = [lora_id, 0]
    return prompt


def prepare_prompt(
    template: dict,
    mode: str,
    run_index: int,
    *,
    lora_name: str | None = None,
    steps: int | None = None,
    output_tag: str | None = None,
) -> dict:
    prompt = copy.deepcopy(template)
    configure_prompt_variant(prompt, lora_name=lora_name, steps=steps)
    _, sampler = _one_node(prompt, "XFuserSamplerCustomAdvanced")
    sampler["inputs"]["noise_seed"] = int(sampler["inputs"]["noise_seed"]) + run_index * SEED_STRIDE
    _, save_video = _one_node(prompt, "SaveVideo")
    if output_tag and output_tag.startswith("o6-"):
        output_group = "raylight_o6"
    elif output_tag:
        output_group = "raylight_o3"
    else:
        output_group = "raylight_o2"
    variant = f"_{output_tag}" if output_tag else ""
    save_video["inputs"]["filename_prefix"] = (
        f"video/{output_group}/minimax_h3_{mode}{variant}_run{run_index}"
    )
    validate_prompt(prompt)
    return prompt


def configure_prompt_geometry(
    prompt: dict,
    *,
    width: int,
    height: int,
    duration: float,
    fps: int,
    expected_frames: int,
) -> dict:
    if width <= 0 or height <= 0 or duration <= 0 or fps <= 0 or expected_frames <= 0:
        raise ValueError("benchmark geometry values must be positive")

    resolution_id, _ = _one_node(prompt, "ResolutionSelector")
    width_link = [resolution_id, 0]
    height_link = [resolution_id, 1]
    width_replacements = 0
    height_replacements = 0
    for node in prompt.values():
        for name, value in list(node.get("inputs", {}).items()):
            if value == width_link:
                node["inputs"][name] = int(width)
                width_replacements += 1
            elif value == height_link:
                node["inputs"][name] = int(height)
                height_replacements += 1
    if not width_replacements or not height_replacements:
        raise ValueError("ResolutionSelector outputs are not connected to width and height inputs")
    del prompt[resolution_id]

    length_replacements = 0
    for node in prompt.values():
        class_type = str(node.get("class_type", ""))
        inputs = node.get("inputs", {})
        if class_type.startswith("MiniMaxH3") and "length" in inputs:
            inputs["length"] = int(expected_frames)
            length_replacements += 1
    if not length_replacements:
        raise ValueError("MiniMax H3 generation node has no length input to lock")

    _, duration_node = _one_node(prompt, "PrimitiveFloat")
    duration_node["inputs"]["value"] = float(duration)
    _, create_video = _one_node(prompt, "CreateVideo")
    create_video["inputs"]["fps"] = int(fps)
    return {
        "width": int(width),
        "height": int(height),
        "duration_seconds": float(duration),
        "fps": int(fps),
        "expected_frames": int(expected_frames),
        "playback_duration_seconds": int(expected_frames) / int(fps),
    }


def summarize_node_timings(events: list[dict], prompt: dict, *, prompt_id: str) -> dict:
    selected = [
        event
        for event in events
        if event.get("prompt_id") == prompt_id and "node" in event and "time_ns" in event
    ]
    nodes: dict[str, dict] = {}
    first_sampler_index = next(
        (
            index
            for index, event in enumerate(selected)
            if event.get("node") is not None
            and prompt.get(str(event["node"]), {}).get("class_type") == "XFuserSamplerCustomAdvanced"
        ),
        len(selected),
    )
    preprocessing_seconds = 0.0
    ray_initialization_seconds = 0.0
    model_load_seconds = 0.0
    sampler_node_seconds = 0.0
    vae_decode_seconds = 0.0
    video_create_seconds = 0.0
    video_save_seconds = 0.0
    loader_types = {"RayUNETLoader", "RayLoraLoader", "CLIPLoader", "VAELoader", "UNETLoader"}

    for index, (current, following) in enumerate(zip(selected, selected[1:])):
        node_id = current.get("node")
        if node_id is None:
            continue
        node_id = str(node_id)
        node = prompt.get(node_id)
        if node is None:
            raise RuntimeError(f"node timing references unknown prompt node {node_id}")
        seconds = max(0, int(following["time_ns"]) - int(current["time_ns"])) / 1e9
        class_type = node.get("class_type", "unknown")
        target = nodes.setdefault(node_id, {"class_type": class_type, "seconds": 0.0})
        target["seconds"] += seconds
        if class_type == "RayInitializer":
            ray_initialization_seconds += seconds
        elif class_type in loader_types:
            model_load_seconds += seconds
        elif index < first_sampler_index:
            preprocessing_seconds += seconds
        if class_type == "XFuserSamplerCustomAdvanced":
            sampler_node_seconds += seconds
        elif class_type.startswith("VAEDecode"):
            vae_decode_seconds += seconds
        elif class_type == "CreateVideo":
            video_create_seconds += seconds
        elif class_type == "SaveVideo":
            video_save_seconds += seconds

    return {
        "model_load_seconds": model_load_seconds,
        "preprocessing_seconds": preprocessing_seconds,
        "ray_initialization_seconds": ray_initialization_seconds,
        "sampler_node_seconds": sampler_node_seconds,
        "vae_decode_seconds": vae_decode_seconds,
        "video_create_seconds": video_create_seconds,
        "video_save_seconds": video_save_seconds,
        "nodes": nodes,
    }


def extract_sampler_progress(text: str, worker_pids_by_rank: dict[str, int]) -> dict[str, dict]:
    pid_to_rank = {int(pid): str(rank) for rank, pid in worker_pids_by_rank.items()}
    results: dict[str, dict] = {}
    pattern = re.compile(r"\(pid=(\d+)\).*?100%.*?([0-9]+(?:\.[0-9]+)?)s/it")
    for match in pattern.finditer(text):
        pid = int(match.group(1))
        rank = pid_to_rank.get(pid)
        if rank is not None:
            results[rank] = {
                "pid": pid,
                "seconds_per_iteration": float(match.group(2)),
            }
    if not results:
        raise RuntimeError("worker stderr did not contain a final s/it record for any known rank")
    return results


def decode_node_event(message, *, time_ns: int) -> dict | None:
    if not isinstance(message, str):
        return None
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return None
    event_type = payload.get("type")
    data = payload.get("data", {})
    if "prompt_id" not in data:
        return None
    if event_type == "execution_success":
        node = None
    elif event_type == "executing" and "node" in data:
        node = None if data["node"] is None else str(data["node"])
    else:
        return None
    return {
        "prompt_id": str(data["prompt_id"]),
        "node": node,
        "time_ns": int(time_ns),
    }


def wait_for_terminal_node_event(
    events: list[dict],
    *,
    prompt_id: str,
    timeout: float = 10.0,
    poll_interval: float = 0.01,
) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        for event in reversed(events):
            if event.get("prompt_id") == prompt_id and event.get("node") is None:
                return event
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"WebSocket did not report a terminal event for prompt {prompt_id} within {timeout}s"
            )
        time.sleep(poll_interval)


def receive_node_events(
    connection,
    stop_event: threading.Event,
    events: list[dict],
    errors: list[str],
    *,
    clock=time.time_ns,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                message = connection.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException:
                if stop_event.is_set():
                    return
                raise
            event = decode_node_event(message, time_ns=clock())
            if event is not None:
                events.append(event)
    except BaseException as exc:
        if not stop_event.is_set():
            errors.append(f"{type(exc).__name__}: {exc}")


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


def server_environment(
    base: dict | None = None,
    *,
    p2p_capacity_bytes: int = P2P_CAPACITY_BYTES,
) -> dict:
    if p2p_capacity_bytes <= 0:
        raise ValueError("p2p_capacity_bytes must be positive")
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
            "RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES": str(int(p2p_capacity_bytes)),
            "RAYLIGHT_WINDOWS_P2P_MIN_GIB_S": "50",
            "RAYLIGHT_WINDOWS_P2P_TIMEOUT_SECONDS": "10",
            "RAYLIGHT_P2P_PROFILE": "1",
            "RAYLIGHT_RANK_DIAG": "1",
            "RAY_DEDUP_LOGS_ALLOW_REGEX": r"\[RAYLIGHT_RANK_DIAG\]",
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


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def hash_input_files(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


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


def resolve_turbo_variant(variant_id: str, mode: str, requested_steps: int | None) -> dict:
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    matches = [
        item
        for item in manifest.get("models", [])
        if item.get("id") == variant_id and "turbo" in item.get("groups", [])
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown Turbo variant: {variant_id}")
    item = matches[0]
    if item.get("mode") != mode:
        raise ValueError(
            f"Turbo variant {variant_id} mode is {item.get('mode')}, not requested mode {mode}"
        )
    official_steps = int(item["steps"])
    if requested_steps is not None and int(requested_steps) != official_steps:
        raise ValueError(
            f"Turbo variant {variant_id} requires {official_steps} steps, not {requested_steps}"
        )
    relative_path = Path(item["relative_path"])
    if relative_path.parent.as_posix() != "loras":
        raise ValueError(f"Turbo variant {variant_id} has an invalid LoRA path")
    return {
        "id": variant_id,
        "lora_name": relative_path.name,
        "steps": official_steps,
        "expected_bytes": int(item["expected_bytes"]),
        "sha256": str(item["sha256"]).lower(),
        "repository": manifest["repository"],
        "revision": manifest["revision"],
    }


def lora_asset_identity(lora_name: str | None) -> dict | None:
    if lora_name is None:
        return None
    if not lora_name or Path(lora_name).name != lora_name:
        raise ValueError("lora_name must be a filename inside the ComfyUI loras directory")
    lora_root = (COMFY_ROOT / "models" / "loras").resolve()
    path = (lora_root / lora_name).resolve()
    if path.parent != lora_root:
        raise ValueError("lora_name escapes the ComfyUI loras directory")
    if not path.is_file():
        raise FileNotFoundError(f"Turbo LoRA not found: {path}")
    return {
        "name": lora_name,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


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


def build_api_prompt(
    base_url: str,
    mode: str,
    profile: str,
    cpu_offload: bool,
    *,
    compute_dtype: str = "default",
    turbo_variant: str | None = None,
    geometry: dict | None = None,
) -> dict:
    builder = _load_script("minimax_h3_build_workflows", SCRIPT_ROOT / "build_workflows.py")
    converter = _load_script("minimax_h3_workflow_to_api", SCRIPT_ROOT / "workflow_to_api.py")
    workflow = builder.build_workflow(
        builder.SOURCE_WORKFLOWS[mode],
        mode=mode,
        profile=profile,
        compute_dtype=compute_dtype,
        turbo_variant=turbo_variant,
    )
    prompt = converter.workflow_to_prompt(workflow, request_json(f"{base_url}/object_info", timeout=120))
    _, initializer = _one_node(prompt, "RayInitializer")
    initializer["inputs"]["FSDP_CPU_OFFLOAD"] = bool(cpu_offload)
    initializer["inputs"]["skip_comm_test"] = True
    if geometry is not None:
        configure_prompt_geometry(prompt, **geometry)
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
    compute_dtype: str = "default",
    turbo_variant: str | None = None,
    steps: int | None = None,
    output_tag: str | None = None,
    width: int | None = None,
    height: int | None = None,
    duration: float | None = None,
    fps: int | None = None,
    expected_frames: int | None = None,
    p2p_capacity_mib: int = P2P_CAPACITY_BYTES // 2**20,
) -> Path:
    if compute_dtype == "fp16_h3_safe" and turbo_variant is None:
        raise ValueError("fp16_h3_safe requires a pinned MiniMax H3 Turbo variant")
    if p2p_capacity_mib <= 0:
        raise ValueError("p2p_capacity_mib must be positive")
    p2p_capacity_bytes = int(p2p_capacity_mib) * 2**20
    geometry_values = (width, height, duration, fps, expected_frames)
    if any(value is not None for value in geometry_values) and not all(
        value is not None for value in geometry_values
    ):
        raise ValueError("width, height, duration, fps and expected_frames must be provided together")
    geometry = None
    if all(value is not None for value in geometry_values):
        geometry = {
            "width": int(width),
            "height": int(height),
            "duration": float(duration),
            "fps": int(fps),
            "expected_frames": int(expected_frames),
        }

    turbo_spec = resolve_turbo_variant(turbo_variant, mode, steps) if turbo_variant else None
    lora_name = turbo_spec["lora_name"] if turbo_spec else None
    if turbo_spec:
        steps = turbo_spec["steps"]
        output_tag = output_tag or turbo_spec["id"]
    if output_tag is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", output_tag):
        raise ValueError("output_tag must use lowercase letters, digits, underscores or hyphens")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tag_suffix = f"-{output_tag}" if output_tag else ""
    result_dir = RESULT_ROOT / f"{timestamp}-{mode}-{profile}{tag_suffix}"
    result_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = result_dir / "comfyui.out.log"
    stderr_path = result_dir / "comfyui.err.log"
    result_path = result_dir / "benchmark.json"
    base_url = f"http://127.0.0.1:{port}"
    runtime_identity = collect_runtime_identity()
    validate_runtime_identity(runtime_identity)
    lora_identity = lora_asset_identity(lora_name)
    if turbo_spec and (
        lora_identity["size_bytes"] != turbo_spec["expected_bytes"]
        or lora_identity["sha256"] != turbo_spec["sha256"]
    ):
        raise RuntimeError(
            f"Turbo LoRA does not match pinned manifest identity: {lora_identity['path']}"
        )
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
        "compute_dtype": compute_dtype,
        "variant": turbo_variant or output_tag or "base",
        "turbo": turbo_spec,
        "steps": steps,
        "lora": lora_identity,
        "runtime_identity": runtime_identity,
        "p2p_capacity_bytes": p2p_capacity_bytes,
        "geometry": (
            {
                "width": geometry["width"],
                "height": geometry["height"],
                "duration_seconds": geometry["duration"],
                "fps": geometry["fps"],
                "expected_frames": geometry["expected_frames"],
            }
            if geometry
            else None
        ),
        "benchmark_input_hashes": hash_input_files(benchmark_input_paths(mode)),
        "started_ns": time.time_ns(),
        "runs": [],
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command,
            cwd=COMFY_ROOT,
            env=server_environment(p2p_capacity_bytes=p2p_capacity_bytes),
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        report["server_pid"] = process.pid
        job = None
        try:
            job = attach_kill_on_close_job(process)
            wait_server(base_url, process)
            template = build_api_prompt(
                base_url,
                mode,
                profile,
                cpu_offload,
                compute_dtype=compute_dtype,
                turbo_variant=turbo_variant,
                geometry=geometry,
            )
            configure_prompt_variant(template, lora_name=lora_name, steps=steps)
            report["api_prompt_template_sha256"] = write_prompt_artifact(
                result_dir / "api-prompt-template.json", template
            )
            sampler_id, _ = _one_node(template, "XFuserSamplerCustomAdvanced")
            for run_index in range(runs):
                prompt = prepare_prompt(
                    template,
                    mode,
                    run_index,
                    lora_name=lora_name,
                    steps=steps,
                    output_tag=output_tag,
                )
                stdout.flush()
                stderr.flush()
                prompt_sha256 = write_prompt_artifact(result_dir / f"run{run_index}-prompt.json", prompt)
                log_start = stdout_path.stat().st_size
                stderr_start = stderr_path.stat().st_size
                samples: list[dict] = []
                monitor_errors: list[str] = []
                stop_event = threading.Event()
                monitor = threading.Thread(target=monitor_resources, args=(stop_event, samples, monitor_errors), daemon=True)

                client_id = str(uuid.uuid4())
                websocket_url = f"ws://127.0.0.1:{port}/ws?clientId={client_id}"
                connection = websocket.create_connection(websocket_url, timeout=10, suppress_origin=True)
                connection.settimeout(1.0)
                node_events: list[dict] = []
                node_event_errors: list[str] = []
                node_stop_event = threading.Event()
                node_receiver = threading.Thread(
                    target=receive_node_events,
                    args=(connection, node_stop_event, node_events, node_event_errors),
                    daemon=True,
                )
                node_receiver.start()
                monitor.start()
                started = time.perf_counter()
                try:
                    response = request_json(
                        f"{base_url}/prompt",
                        {"prompt": prompt, "client_id": client_id},
                        timeout=60,
                    )
                    prompt_id = response["prompt_id"]
                    history = wait_prompt(base_url, prompt_id, process, timeout)
                    wait_for_terminal_node_event(node_events, prompt_id=prompt_id)
                finally:
                    stop_event.set()
                    monitor.join(timeout=15)
                    node_stop_event.set()
                    connection.close()
                    node_receiver.join(timeout=15)
                if monitor.is_alive():
                    raise RuntimeError("resource monitor did not stop")
                if node_receiver.is_alive():
                    raise RuntimeError("node event receiver did not stop")
                if node_event_errors:
                    raise RuntimeError(f"node event receiver failed: {node_event_errors[0]}")
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
                node_timings = summarize_node_timings(node_events, prompt, prompt_id=prompt_id)
                if not node_timings["nodes"] or node_timings["video_save_seconds"] <= 0:
                    raise RuntimeError("WebSocket node timing did not capture a complete SaveVideo execution")
                stderr.flush()
                stderr_end = stderr_path.stat().st_size
                with stderr_path.open("rb") as handle:
                    handle.seek(stderr_start)
                    stderr_segment = handle.read(stderr_end - stderr_start).decode("utf-8", errors="replace")
                worker_pids_by_rank = {
                    rank: int(values["pid"]) for rank, values in phases["per_rank"].items()
                }
                sampler_progress = extract_sampler_progress(stderr_segment, worker_pids_by_rank)
                sampling_seconds_per_iteration = phases["sampling_seconds"] / int(steps) if steps else None
                run = {
                    "run_index": run_index,
                    "temperature": "cold" if run_index == 0 else "warm",
                    "prompt_id": prompt_id,
                    "elapsed_seconds": elapsed,
                    "diagnostic_log_wait_seconds": diagnostic_wait_seconds,
                    "noise_seed": prompt[sampler_id]["inputs"]["noise_seed"],
                    "api_prompt_sha256": prompt_sha256,
                    "phases": phases,
                    "node_timings": node_timings,
                    "sampler_progress": sampler_progress,
                    "sampling_seconds_per_iteration": sampling_seconds_per_iteration,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("i2v", "ref2va"), required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compute-dtype",
        choices=("default", "fp16_h3_safe"),
        default="default",
    )
    parser.add_argument("--turbo-variant")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output-tag")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--expected-frames", type=int)
    parser.add_argument(
        "--p2p-capacity-mib",
        type=int,
        default=P2P_CAPACITY_BYTES // 2**20,
        help="per-rank Windows CUDA P2P IPC buffer capacity in MiB",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        raise ValueError("runs must be positive")
    if args.steps is not None and args.steps < 1:
        raise ValueError("steps must be positive")
    if args.p2p_capacity_mib < 1:
        raise ValueError("--p2p-capacity-mib must be positive")
    geometry_values = (args.width, args.height, args.duration, args.fps, args.expected_frames)
    if any(value is not None for value in geometry_values) and not all(
        value is not None for value in geometry_values
    ):
        raise ValueError(
            "--width, --height, --duration, --fps and --expected-frames must be provided together"
        )
    if any(value is not None and value <= 0 for value in geometry_values):
        raise ValueError("geometry values must be positive")
    if args.compute_dtype == "fp16_h3_safe" and args.turbo_variant is None:
        raise ValueError("fp16_h3_safe requires a pinned MiniMax H3 Turbo variant")
    return args


def main() -> int:
    args = parse_args()
    print(
        run_benchmark(
            mode=args.mode,
            profile=args.profile,
            runs=args.runs,
            port=args.port,
            timeout=args.timeout,
            cpu_offload=args.cpu_offload,
            compute_dtype=args.compute_dtype,
            turbo_variant=args.turbo_variant,
            steps=args.steps,
            output_tag=args.output_tag,
            width=args.width,
            height=args.height,
            duration=args.duration,
            fps=args.fps,
            expected_frames=args.expected_frames,
            p2p_capacity_mib=args.p2p_capacity_mib,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
