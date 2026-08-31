#!/usr/bin/env python3
"""Record three paired cold/warm MiniMax H3 Text Encoder workflow baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import NamedTuple

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from raylight.text_encoder_parallel.baseline import BaselineRecord


COMFY_ROOT = Path(os.environ.get("RAYLIGHT_COMFY_ROOT", REPO_ROOT.parent / "ComfyUI"))
ENV_ROOT = COMFY_ROOT.parent
PYTHON = Path(os.environ.get("RAYLIGHT_PYTHON", ENV_ROOT / "Python310" / "python.exe"))
DEFAULT_WORKFLOW = COMFY_ROOT / "user" / "default" / "workflows" / "MMH3Director.json"
DEFAULT_RESULT_ROOT = REPO_ROOT / ".local" / "artifacts" / "o10" / "text-encoder-baseline"
HELPERS_PATH = SCRIPT_ROOT / "benchmark_cold_warm.py"
TP2_STAGE_NAMES = (
    "preprocessor_load",
    "preprocess_and_vision",
    "preprocessor_cleanup",
    "backbone_load",
    "language_backbone",
    "backbone_cleanup",
    "conditioning_finalize",
)


class BaselineSpec(NamedTuple):
    checkpoint: str
    video_vae: str
    audio_vae: str
    prompt: str
    reference_image: str
    width: int
    height: int
    length: int
    workflow_sha256: str


def _one_node(workflow: dict, node_type: str, *, title: str | None = None) -> dict:
    matches = [
        node
        for node in workflow.get("nodes", [])
        if node.get("type") == node_type and (title is None or node.get("title") == title)
    ]
    if len(matches) != 1:
        suffix = f" titled {title!r}" if title else ""
        raise ValueError(f"expected one {node_type}{suffix}, found {len(matches)}")
    return matches[0]


def _timeline_from_director(node: dict) -> dict:
    candidates = []
    for value in node.get("widgets_values", []):
        if not isinstance(value, str) or not value.lstrip().startswith("{"):
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and isinstance(decoded.get("segments"), list):
            candidates.append(decoded)
    if len(candidates) != 1:
        raise ValueError(f"expected one Director timeline_data object, found {len(candidates)}")
    return candidates[0]


def load_baseline_spec(path: Path) -> BaselineSpec:
    raw = path.read_bytes()
    workflow = json.loads(raw.decode("utf-8-sig"))
    clip = _one_node(workflow, "CLIPLoader")
    video_vae = _one_node(workflow, "VAELoader", title="Video VAE")
    audio_vae = _one_node(workflow, "VAELoader", title="Audio VAE")
    timeline = _timeline_from_director(_one_node(workflow, "MiniMaxH3Director"))
    segments = timeline["segments"]
    if not segments:
        raise ValueError("Director timeline must contain at least one segment")
    segment = segments[0]
    refs = segment.get("refs") or []
    if not refs or not refs[0].get("imageFile"):
        raise ValueError("first Director segment must contain an input reference image")
    reference = refs[0]
    subfolder = str(reference.get("subfolder") or "").strip("/\\")
    reference_image = str(reference["imageFile"])
    if subfolder:
        reference_image = str(Path(subfolder) / reference_image)
    output = timeline.get("output") or {}
    width = int(timeline.get("width") or output.get("width") or 1280)
    height = int(timeline.get("height") or output.get("height") or 736)
    length = int(segment.get("frameCount") or segment.get("length") or 73)
    if width <= 0 or height <= 0 or length <= 0:
        raise ValueError("Director width, height and first-segment frame count must be positive")
    prompt = str(segment.get("prompt") or timeline.get("global", {}).get("prompt") or "").strip()
    if not prompt:
        raise ValueError("first Director segment prompt is empty")
    return BaselineSpec(
        checkpoint=str(clip["widgets_values"][0]),
        video_vae=str(video_vae["widgets_values"][0]),
        audio_vae=str(audio_vae["widgets_values"][0]),
        prompt=prompt,
        reference_image=reference_image,
        width=width,
        height=height,
        length=length,
        workflow_sha256=hashlib.sha256(raw).hexdigest(),
    )


def build_conditioning_prompt(
    spec: BaselineSpec,
    *,
    length: int,
    engine: str = "single",
) -> dict:
    if length <= 0:
        raise ValueError("length must be positive")
    if engine not in {"single", "tp2"}:
        raise ValueError("engine must be single or tp2")
    prompt = {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": spec.checkpoint, "type": "minimax", "device": "default"},
        },
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": spec.video_vae}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": spec.audio_vae}},
        "4": {"class_type": "LoadImage", "inputs": {"image": spec.reference_image}},
        "5": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["1", 0],
                "vae": ["2", 0],
                "audio_vae": ["3", 0],
                "prompt": spec.prompt,
                "width": spec.width,
                "height": spec.height,
                "length": int(length),
                "ref_image_size": "match",
                "ref_images.ref_image_0": ["4", 0],
            },
        },
        "6": {"class_type": "PreviewAny", "inputs": {"source": ["5", 0]}},
    }
    if engine == "single":
        return prompt
    prompt["0"] = {
        "class_type": "RayInitializer",
        "inputs": {
            "ray_cluster_address": "local",
            "ray_cluster_namespace": "default",
            "GPU": 2,
            "ulysses_degree": 2,
            "ring_degree": 1,
            "cfg_degree": 1,
            "dp_degree": 1,
            "sync_ulysses": True,
            "clear_vram_after_sampling": False,
            "FSDP": True,
            "FSDP_CPU_OFFLOAD": True,
            "XFuser_attention": "TORCH_EFFICIENT",
            "skip_comm_test": True,
            "use_mmap": True,
        },
    }
    prompt["1"] = {
        "class_type": "RayMiniMaxH3DualGPUCLIPLoader",
        "inputs": {
            "clip_name": spec.checkpoint,
            "execution_mode": "tp2",
            "ray_actors_init": ["0", 0],
        },
    }
    return prompt


def parse_event_pair(events: list[dict]) -> dict:
    begins = [event for event in events if event.get("event") == "text_encoder_begin"]
    completes = [event for event in events if event.get("event") == "text_encoder_complete"]
    if len(begins) != 1:
        raise ValueError(f"expected one text_encoder_begin event, found {len(begins)}")
    if len(completes) != 1:
        raise ValueError(f"expected one text_encoder_complete event, found {len(completes)}")
    begin, complete = begins[0], completes[0]
    if int(complete["time_ns"]) < int(begin["time_ns"]):
        raise ValueError("text_encoder_complete precedes text_encoder_begin")
    if int(complete["token_count"]) != int(begin["token_count"]):
        raise ValueError("Text Encoder token count changed during execution")
    return {**complete, "begin_time_ns": int(begin["time_ns"])}


def summarize_stage_events(events: list[dict]) -> dict[str, float]:
    rows = [event for event in events if event.get("event") == "text_encoder_stage"]
    by_name = {str(event.get("stage")): event for event in rows}
    if set(by_name) != set(TP2_STAGE_NAMES) or len(rows) != len(TP2_STAGE_NAMES):
        raise ValueError(
            "TP2 stage events must contain each expected stage exactly once; "
            f"found={sorted(by_name)}"
        )
    if any(bool(event.get("failed")) for event in rows):
        failed = [event["stage"] for event in rows if event.get("failed")]
        raise ValueError(f"TP2 stage timing reported failed stages: {failed}")
    return {name: float(by_name[name]["elapsed_s"]) for name in TP2_STAGE_NAMES}


def summarize_tp2_profile_events(events: list[dict]) -> dict:
    rows = [event for event in events if event.get("event") == "text_encoder_tp2_profile"]
    if len(rows) != 1:
        raise ValueError(f"expected exactly one TP2 runtime profile, found {len(rows)}")
    profile = rows[0].get("profile")
    if not isinstance(profile, dict):
        raise ValueError("TP2 runtime profile must contain an object")
    return profile


def summarize_records(records: list[BaselineRecord]) -> dict:
    summary = {}
    for mode in ("cold", "warm"):
        rows = [record for record in records if record.mode == mode]
        summary[mode] = {
            "runs": len(rows),
            "median_te_total_s": statistics.median(record.te_total_s for record in rows) if rows else None,
            "peak_gpu0_mib": max((record.gpu0_peak_mib for record in rows), default=None),
            "peak_ram_mib": max((record.ram_peak_mib for record in rows), default=None),
            "peak_pagefile_mib": max((record.pagefile_peak_mib for record in rows), default=None),
            "token_counts": sorted({record.token_count for record in rows}),
        }
    return summary


def _load_helpers():
    spec = importlib.util.spec_from_file_location("raylight_minimax_benchmark_helpers", HELPERS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark helpers: {HELPERS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _samples_for_interval(samples: list[dict], begin_ns: int, end_ns: int) -> list[dict]:
    selected = [row for row in samples if begin_ns <= int(row["time_ns"]) <= end_ns]
    if selected:
        return selected
    if not samples:
        raise RuntimeError("resource monitor captured no samples")
    nearest_begin = min(samples, key=lambda row: abs(int(row["time_ns"]) - begin_ns))
    nearest_end = min(samples, key=lambda row: abs(int(row["time_ns"]) - end_ns))
    return [nearest_begin] if nearest_begin is nearest_end else [nearest_begin, nearest_end]


def _write_records_csv(path: Path, records: list[BaselineRecord]) -> None:
    fields = list(records[0].to_mapping()) if records else list(BaselineRecord.__dataclass_fields__)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(record.to_mapping() for record in records)


def run_benchmark(
    workflow: Path,
    *,
    engine: str = "single",
    sessions: int = 3,
    port: int = 8193,
    timeout: int = 1800,
    result_root: Path = DEFAULT_RESULT_ROOT,
) -> Path:
    if sessions < 1:
        raise ValueError("sessions must be positive")
    if engine not in {"single", "tp2"}:
        raise ValueError("engine must be single or tp2")
    helpers = _load_helpers()
    spec = load_baseline_spec(workflow)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = result_root / f"{timestamp}-{engine}-v100-cold-warm"
    result_dir.mkdir(parents=True, exist_ok=False)
    report_path = result_dir / "benchmark.json"
    records: list[BaselineRecord] = []
    report = {
        "schema_version": 1,
        "workflow": str(workflow),
        "workflow_sha256": spec.workflow_sha256,
        "checkpoint": spec.checkpoint,
        "reference_image": spec.reference_image,
        "engine": engine,
        "sessions": sessions,
        "records": [],
        "started_ns": time.time_ns(),
    }
    base_url = f"http://127.0.0.1:{port}"
    try:
        helpers.request_json(f"{base_url}/system_stats", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError(f"port {port} already has a running ComfyUI server")

    for session_index in range(1, sessions + 1):
        session_dir = result_dir / f"session{session_index}"
        session_dir.mkdir()
        event_path = session_dir / "text-encoder-events.jsonl"
        stdout_path = session_dir / "comfyui.out.log"
        stderr_path = session_dir / "comfyui.err.log"
        environment = helpers.server_environment()
        environment.update(
            {
                "RAYLIGHT_TEXT_ENCODER_BENCHMARK": "1",
                "RAYLIGHT_TEXT_ENCODER_BENCHMARK_EVENT_PATH": str(event_path),
                "RAYLIGHT_TEXT_ENCODER_CHECKPOINT": spec.checkpoint,
                "RAYLIGHT_TEXT_ENCODER_WORKFLOW_SHA256": spec.workflow_sha256,
                "RAYLIGHT_P2P_PROFILE": "0",
                "RAYLIGHT_RANK_DIAG": "0",
            }
        )
        command = [
            str(PYTHON),
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-cuda-malloc",
            "--reserve-vram",
            "4",
        ]
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                command,
                cwd=COMFY_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            job = None
            try:
                job = helpers.attach_kill_on_close_job(process)
                helpers.wait_server(base_url, process)
                for mode, length in (("cold", spec.length), ("warm", spec.length + 17)):
                    prompt = build_conditioning_prompt(spec, length=length, engine=engine)
                    prompt_path = session_dir / f"{mode}-prompt.json"
                    prompt_text = json.dumps(prompt, indent=2, ensure_ascii=False) + "\n"
                    prompt_path.write_text(prompt_text, encoding="utf-8")
                    before_events = len(_read_events(event_path))
                    samples: list[dict] = []
                    monitor_errors: list[str] = []
                    stop_event = threading.Event()
                    monitor = threading.Thread(
                        target=helpers.monitor_resources,
                        args=(stop_event, samples, monitor_errors),
                        daemon=True,
                    )
                    monitor.start()
                    try:
                        response = helpers.request_json(
                            f"{base_url}/prompt", {"prompt": prompt}, timeout=60
                        )
                        history = helpers.wait_prompt(base_url, response["prompt_id"], process, timeout)
                    finally:
                        stop_event.set()
                        monitor.join(timeout=15)
                    if monitor.is_alive():
                        raise RuntimeError("resource monitor did not stop")
                    if monitor_errors:
                        raise RuntimeError(f"resource monitor failed: {monitor_errors[0]}")
                    new_events = _read_events(event_path)[before_events:]
                    event = parse_event_pair(new_events)
                    stages = summarize_stage_events(new_events) if engine == "tp2" else {}
                    tp2_profile = (
                        summarize_tp2_profile_events(new_events) if engine == "tp2" else {}
                    )
                    interval_samples = _samples_for_interval(
                        samples, int(event["begin_time_ns"]), int(event["time_ns"])
                    )
                    resources = helpers.summarize_resources(interval_samples)
                    record = BaselineRecord.from_mapping(
                        {
                            "checkpoint": spec.checkpoint,
                            "workflow_sha256": spec.workflow_sha256,
                            "mode": mode,
                            "run_index": session_index,
                            "token_count": int(event["token_count"]),
                            "te_total_s": float(event["te_total_s"]),
                            "gpu0_peak_mib": resources["gpus"]["0"]["peak_memory_mib"],
                            "ram_peak_mib": resources["peak_physical_used_mib"],
                            "pagefile_peak_mib": resources["peak_pagefile_used_mib"],
                        }
                    )
                    records.append(record)
                    time.sleep(1)
                    post_encode_gpus = helpers.gpu_metrics()
                    report["records"].append(
                        {
                            **record.to_mapping(),
                            "session": session_index,
                            "prompt_id": response["prompt_id"],
                            "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
                            "resources": resources,
                            "stages": stages,
                            "tp2_profile": tp2_profile,
                            "post_encode_gpus": post_encode_gpus,
                            "history_status": history.get("status", {}),
                        }
                    )
                    helpers.write_monitor_csv(session_dir / f"{mode}-monitor.csv", samples)
                    report["summary"] = summarize_records(records)
                    report_path.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                    )
                    print(record.to_json(), flush=True)
            finally:
                helpers.terminate_process_tree(process, job)
                time.sleep(3)

    report["summary"] = summarize_records(records)
    report["finished_ns"] = time.time_ns()
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_records_csv(result_dir / "records.csv", records)
    return report_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--engine", choices=("single", "tp2"), default="single")
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    print(
        run_benchmark(
            args.workflow,
            engine=args.engine,
            sessions=args.sessions,
            port=args.port,
            timeout=args.timeout,
            result_root=args.result_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
