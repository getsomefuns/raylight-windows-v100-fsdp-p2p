"""Run matched cold/warm LTX 5-second benchmarks through the ComfyUI API."""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid

import psutil


REPO_ROOT = Path(__file__).parents[1]
COMFY_ROOT = Path(os.environ.get("RAYLIGHT_COMFY_ROOT", REPO_ROOT.parents[1]))
ENV_ROOT = COMFY_ROOT.parent
PYTHON = Path(os.environ.get("RAYLIGHT_PYTHON", ENV_ROOT / "Python310/python.exe"))
RESULT_ROOT = Path(
    os.environ.get("RAYLIGHT_BENCHMARK_RESULT_ROOT", ENV_ROOT / "logs/f4")
)
PAYLOAD_ROOT = Path(
    os.environ.get("RAYLIGHT_BENCHMARK_PAYLOAD_ROOT", REPO_ROOT / "benchmark_payloads")
)
BASE_SEEDS = (426_531_804_528_251, 513_480_587_783_135)
SEED_STRIDE = 1_000_003
WORKFLOW_P2P_CAPACITY_BYTES = 128 * 1024 * 1024


def seed_node_ids(mode: str) -> tuple[str, str]:
    return ("396", "397") if mode == "single" else ("383", "389")


def prepare_prompt(template: dict, mode: str, run_index: int, label: str | None = None) -> dict:
    prompt = copy.deepcopy(template)
    first_seed_node, second_seed_node = seed_node_ids(mode)
    prompt[first_seed_node]["inputs"]["noise_seed"] = BASE_SEEDS[0] + run_index * SEED_STRIDE
    prompt[second_seed_node]["inputs"]["noise_seed"] = BASE_SEEDS[1] + run_index * SEED_STRIDE
    output_label = label or mode
    prompt["75"]["inputs"]["filename_prefix"] = f"raylight/f4_{output_label}_run{run_index}"
    return prompt


def validate_mode_prompt(mode: str, prompt: dict) -> None:
    if mode not in ("single", "ray-single", "ulysses", "fsdp"):
        raise ValueError(f"unsupported benchmark mode: {mode}")
    if int(prompt["369"]["inputs"]["tile_size"]) != 384:
        raise ValueError("matched benchmark requires VAE tile_size=384")
    if prompt["374"]["inputs"]["expression"] != "a * b + 1":
        raise ValueError("matched benchmark requires the original frame expression")

    initializer = prompt.get("381")
    if mode == "single":
        if initializer is not None:
            raise ValueError("single benchmark must not contain RayInitializer")
        for node_id in ("383", "389"):
            if prompt[node_id]["class_type"] != "SamplerCustomAdvanced":
                raise ValueError("single benchmark requires SamplerCustomAdvanced")
        for sampler_id, noise_id in (("383", "396"), ("389", "397")):
            if prompt.get(noise_id, {}).get("class_type") != "RandomNoise":
                raise ValueError("single benchmark requires two RandomNoise nodes")
            if prompt[sampler_id]["inputs"].get("noise") != [noise_id, 0]:
                raise ValueError("single benchmark sampler noise link is missing")
        return

    if initializer is None or initializer["class_type"] != "RayInitializer":
        raise ValueError(f"{mode} benchmark requires RayInitializer")
    inputs = initializer["inputs"]
    expected_gpus = 1 if mode == "ray-single" else 2
    if int(inputs["GPU"]) != expected_gpus:
        raise ValueError(f"{mode} benchmark requires GPU={expected_gpus}")
    if int(inputs["dp_degree"]) != 1:
        raise ValueError(f"{mode} benchmark requires dp_degree=1")
    if mode == "ulysses":
        expected = (2, 1, 1, False)
    elif mode == "fsdp":
        expected = (0, 0, 0, True)
    else:
        expected = (0, 0, 0, False)
    observed = (
        int(inputs["ulysses_degree"]),
        int(inputs["ring_degree"]),
        int(inputs["cfg_degree"]),
        bool(inputs["FSDP"]),
    )
    if observed != expected:
        raise ValueError(f"{mode} topology mismatch: expected {expected}, found {observed}")
    for node_id in ("383", "389"):
        if prompt[node_id]["class_type"] != "XFuserSamplerCustomAdvanced":
            raise ValueError(f"{mode} benchmark requires XFuserSamplerCustomAdvanced")


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
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
    if completed.returncode:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "memory_mib": float(fields[1]),
                "util_percent": float(fields[2]),
                "power_w": float(fields[3]),
            }
        )
    return rows


def monitor_resources(stop_event: threading.Event, samples: list[dict]):
    while not stop_event.is_set():
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        samples.append(
            {
                "time_ns": time.time_ns(),
                "physical_used_mib": memory.used / 2**20,
                "committed_mib": committed_memory_mib(),
                "pagefile_used_mib": swap.used / 2**20,
                "gpus": gpu_metrics(),
            }
        )
        stop_event.wait(1.0)


def summarize_resources(samples: list[dict]) -> dict:
    summary = {
        "samples": len(samples),
        "peak_physical_used_mib": max((row["physical_used_mib"] for row in samples), default=0),
        "peak_committed_mib": max((row["committed_mib"] or 0 for row in samples), default=0),
        "peak_pagefile_used_mib": max((row["pagefile_used_mib"] for row in samples), default=0),
        "start_pagefile_used_mib": samples[0]["pagefile_used_mib"] if samples else 0,
        "end_pagefile_used_mib": samples[-1]["pagefile_used_mib"] if samples else 0,
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


def wait_server(base_url: str, process: subprocess.Popen, timeout=300):
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


def wait_prompt(base_url: str, prompt_id: str, process: subprocess.Popen, timeout=1800):
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


def terminate_process_tree(process: subprocess.Popen):
    if process.poll() is not None:
        return
    try:
        parent = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs(children + [parent], timeout=20)
    for remaining in alive:
        try:
            remaining.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=10)


def server_environment(mode: str) -> dict:
    environment = os.environ.copy()
    gloo_host = environment.get("RAYLIGHT_GLOO_HOST", "127.0.0.1")
    environment.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "USE_LIBUV": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "RAYLIGHT_GLOO_HOST": gloo_host,
            "RAY_DEBUG_DISABLE_MEMORY_MONITOR": "1",
            "RAY_memory_usage_threshold": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
            "RAYLIGHT_WINDOWS_P2P_TIMEOUT_SECONDS": "10",
        }
    )
    if mode in ("single", "ray-single"):
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "RAYLIGHT_WINDOWS_P2P": "0",
                "RAYLIGHT_P2P_PROFILE": "0",
            }
        )
    else:
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "RAYLIGHT_WINDOWS_P2P": "1",
                "RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES": str(WORKFLOW_P2P_CAPACITY_BYTES),
                "RAYLIGHT_WINDOWS_P2P_MIN_GIB_S": "50",
                "RAYLIGHT_P2P_PROFILE": "1",
                "RAYLIGHT_RANK_DIAG": "1",
            }
        )
    return environment


def submit_prompt_and_wait(
    base_url: str,
    prompt: dict,
    process: subprocess.Popen,
    samples: list[dict],
) -> tuple[str, dict, float]:
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_resources,
        args=(stop_event, samples),
        daemon=True,
    )
    monitor.start()
    started = time.perf_counter()
    try:
        response = request_json(
            f"{base_url}/prompt",
            {"prompt": prompt, "client_id": str(uuid.uuid4())},
            timeout=60,
        )
        prompt_id = response["prompt_id"]
        history = wait_prompt(base_url, prompt_id, process)
    finally:
        stop_event.set()
        monitor.join(timeout=15)
    return prompt_id, history, time.perf_counter() - started


def run_mode(mode: str, runs: int, port: int, variant: str | None = None) -> Path:
    artifact_label = variant or mode
    payload_path = PAYLOAD_ROOT / f"{artifact_label}-5s.json"
    template = json.loads(payload_path.read_text(encoding="utf-8"))
    validate_mode_prompt(mode, template)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    try:
        request_json(f"{base_url}/system_stats", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError(f"port {port} already has a running ComfyUI server")

    stdout_path = RESULT_ROOT / f"comfyui-f4-{artifact_label}.out.log"
    stderr_path = RESULT_ROOT / f"comfyui-f4-{artifact_label}.err.log"
    result_path = RESULT_ROOT / f"f4-{artifact_label}-benchmark.json"
    command = [
        str(PYTHON),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-cuda-malloc",
    ]
    report = {
        "mode": mode,
        "variant": variant,
        "payload": str(payload_path),
        "runs": [],
        "started_ns": time.time_ns(),
    }
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=COMFY_ROOT,
            env=server_environment(mode),
            stdout=stdout,
            stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        report["server_pid"] = process.pid
        try:
            wait_server(base_url, process)
            for run_index in range(runs):
                prompt = prepare_prompt(template, mode, run_index, artifact_label)
                validate_mode_prompt(mode, prompt)
                samples = []
                log_start = {
                    "stdout": stdout_path.stat().st_size,
                    "stderr": stderr_path.stat().st_size,
                }
                prompt_id, history, elapsed = submit_prompt_and_wait(
                    base_url,
                    prompt,
                    process,
                    samples,
                )
                status = "success"
                run = {
                    "run_index": run_index,
                    "temperature": "cold" if run_index == 0 else "warm",
                    "prompt_id": prompt_id,
                    "status": status,
                    "elapsed_seconds": elapsed,
                    "seeds": [
                        prompt[seed_node_ids(mode)[0]]["inputs"]["noise_seed"],
                        prompt[seed_node_ids(mode)[1]]["inputs"]["noise_seed"],
                    ],
                    "outputs": history.get("outputs", {}),
                    "resources": summarize_resources(samples),
                    "log_start": log_start,
                    "log_end": {
                        "stdout": stdout_path.stat().st_size,
                        "stderr": stderr_path.stat().st_size,
                    },
                }
                report["runs"].append(run)
                result_path.write_text(
                    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                monitor_path = RESULT_ROOT / f"f4-{artifact_label}-run{run_index}-monitor.csv"
                with monitor_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        [
                            "time_ns",
                            "physical_used_mib",
                            "committed_mib",
                            "pagefile_used_mib",
                            "gpus_json",
                        ]
                    )
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
                print(json.dumps(run, ensure_ascii=False), flush=True)
                time.sleep(5)
        finally:
            terminate_process_tree(process)
            report["finished_ns"] = time.time_ns()
            result_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    return result_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("single", "ray-single", "ulysses", "fsdp"), required=True)
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--variant", help="payload/result label while retaining the selected execution mode")
    parser.add_argument("--port", type=int, default=8188)
    args = parser.parse_args()
    if args.runs < 1:
        raise ValueError("runs must be positive")
    print(run_mode(args.mode, args.runs, args.port, args.variant))


if __name__ == "__main__":
    main()
