#!/usr/bin/env python3
"""Run the frozen O11 maximum-context workflow through real ComfyUI nodes."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
from typing import Any, Iterator, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
COMFY_ROOT = Path(os.environ.get("RAYLIGHT_COMFY_ROOT", REPO_ROOT.parent / "ComfyUI"))
PYTHON = Path(os.environ.get("RAYLIGHT_PYTHON", REPO_ROOT.parent / "Python310" / "python.exe"))
DEFAULT_API_PROMPT = REPO_ROOT / ".local" / "artifacts" / "o11" / "max-context-api.json"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / ".local"
    / "artifacts"
    / "o11"
    / "o11-boundary-input-manifest.local.json"
)
DEFAULT_RESULT_ROOT = REPO_ROOT / ".local" / "artifacts" / "o11" / "max-context-runs"
HELPERS_PATH = SCRIPT_ROOT / "benchmark_cold_warm.py"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _one_node(prompt: dict[str, dict[str, Any]], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [(node_id, node) for node_id, node in prompt.items() if node.get("class_type") == class_type]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {class_type}, found {len(matches)}")
    return matches[0]


def select_backend(prompt: dict[str, dict[str, Any]], backend: str) -> dict[str, dict[str, Any]]:
    if backend not in {"AUTO", "XFORMERS_GQA", "TORCH_EFFICIENT"}:
        raise ValueError(f"unsupported O11 backend: {backend}")
    selected = copy.deepcopy(prompt)
    _, loader = _one_node(selected, "RayMiniMaxH3DualGPUCLIPLoader")
    loader.setdefault("inputs", {})["text_encoder_attention"] = backend
    return selected


def validate_frozen_fixture(
    prompt: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    input_root: Path,
) -> dict[str, Any]:
    _, source = _one_node(prompt, "MiniMaxH3ReferenceToVideo")
    _, loader = _one_node(prompt, "RayMiniMaxH3DualGPUCLIPLoader")
    source_inputs = source.get("inputs", {})
    prompt_text = str(source_inputs.get("prompt", ""))
    expected_prompt = manifest["prompt"]
    if len(prompt_text) != int(expected_prompt["normalized_characters"]):
        raise ValueError("prompt character count does not match the frozen O11 manifest")
    if sha256_text(prompt_text) != expected_prompt["sha256"]:
        raise ValueError("prompt SHA-256 does not match the frozen O11 manifest")

    image_nodes = {
        str(node["inputs"]["image"]): node_id
        for node_id, node in prompt.items()
        if node.get("class_type") == "LoadImage"
    }
    image_rows = manifest["media"]["images"]
    if len(image_rows) != 7 or len(image_nodes) != 7:
        raise ValueError("frozen O11 fixture requires exactly seven image files")
    if len({row["sha256"] for row in image_rows}) != 7:
        raise ValueError("frozen O11 image hashes are not unique")
    for row in image_rows:
        relative = str(row["file"])
        if relative not in image_nodes:
            raise ValueError(f"workflow does not load frozen image: {relative}")
        if sha256_file(input_root / relative) != row["sha256"]:
            raise ValueError(f"image SHA-256 mismatch: {relative}")

    _, video_node = _one_node(prompt, "LoadVideo")
    video_row = manifest["media"]["video"]
    video_relative = str(video_node["inputs"]["file"])
    if video_relative != str(video_row["file"]):
        raise ValueError("workflow video does not match the frozen O11 manifest")
    if sha256_file(input_root / video_relative) != video_row["sha256"]:
        raise ValueError(f"video SHA-256 mismatch: {video_relative}")

    ref_image_keys = [key for key in source_inputs if key.startswith("ref_images.ref_image_")]
    if len(ref_image_keys) != 7 or "ref_videos.ref_video_0" not in source_inputs:
        raise ValueError("conditioning node must connect seven images and one video")
    if int(source_inputs.get("width", 0)) != 1056 or int(source_inputs.get("height", 0)) != 608:
        raise ValueError("O11 canvas must be 1056x608")
    if int(source_inputs.get("length", 0)) != 362:
        raise ValueError("O11 length must be 362 frames")
    return {
        "prompt_characters": len(prompt_text),
        "image_count": len(image_rows),
        "video_duration_seconds": float(video_row.get("duration_seconds", 0)),
        "video_frames": int(video_row.get("frames", 0)),
        "backend": str(loader.get("inputs", {}).get("text_encoder_attention", "AUTO")),
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextlib.contextmanager
def failure_report(path: Path, report: dict[str, Any]) -> Iterator[None]:
    try:
        yield
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["finished_ns"] = time.time_ns()
        _write_report(path, report)
        raise


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _event_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    begins = [event for event in events if event.get("event") == "text_encoder_begin"]
    completes = [event for event in events if event.get("event") == "text_encoder_complete"]
    if len(begins) != 1 or len(completes) != 1:
        raise ValueError(
            "expected one Text Encoder begin/complete pair, "
            f"found begin={len(begins)} complete={len(completes)}"
        )
    stages = {
        str(event["stage"]): float(event["elapsed_s"])
        for event in events
        if event.get("event") == "text_encoder_stage" and not event.get("failed")
    }
    profiles = [event.get("profile") for event in events if event.get("event") == "text_encoder_tp2_profile"]
    begin, complete = begins[0], completes[0]
    return {
        "token_count": int(complete["token_count"]),
        "te_total_s": float(complete["te_total_s"]),
        "begin_time_ns": int(begin["time_ns"]),
        "complete_time_ns": int(complete["time_ns"]),
        "stages": stages,
        "tp2_profile": profiles[0] if len(profiles) == 1 else None,
    }


def run_benchmark(
    *,
    api_prompt_path: Path,
    manifest_path: Path,
    backend: str,
    runs: int,
    port: int,
    timeout: int,
    reserve_vram_gib: float,
    result_root: Path,
) -> Path:
    if runs < 1:
        raise ValueError("runs must be positive")
    helpers = _load_script("o11_benchmark_helpers", HELPERS_PATH)
    prompt = select_backend(json.loads(api_prompt_path.read_text(encoding="utf-8")), backend)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture = validate_frozen_fixture(prompt, manifest, COMFY_ROOT / "input")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    result_dir = result_root / f"{stamp}-{backend.lower()}"
    result_dir.mkdir(parents=True, exist_ok=False)
    report_path = result_dir / "report.json"
    event_path = result_dir / "text-encoder-events.jsonl"
    stdout_path = result_dir / "comfyui.out.log"
    stderr_path = result_dir / "comfyui.err.log"
    prompt_path = result_dir / "api-prompt.json"
    prompt_path.write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "backend_requested": backend,
        "fixture": fixture,
        "manifest_sha256": sha256_file(manifest_path),
        "api_prompt_sha256": sha256_file(prompt_path),
        "deployed_commit": (
            COMFY_ROOT / "custom_nodes" / "raylight" / ".raylight-deployed-commit"
        ).read_text(encoding="utf-8").strip(),
        "started_ns": time.time_ns(),
        "idle_gpus": helpers.gpu_metrics(),
        "idle_committed_mib": helpers.committed_memory_mib(),
        "records": [],
    }
    _write_report(report_path, report)

    base_url = f"http://127.0.0.1:{port}"
    try:
        helpers.request_json(f"{base_url}/system_stats", timeout=2)
    except Exception:
        pass
    else:
        raise RuntimeError(f"port {port} already has a running ComfyUI server")

    environment = helpers.server_environment(p2p_capacity_bytes=512 * 1024 * 1024)
    environment.update(
        {
            "RAYLIGHT_TEXT_ENCODER_BENCHMARK": "1",
            "RAYLIGHT_TEXT_ENCODER_BENCHMARK_EVENT_PATH": str(event_path),
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
        str(reserve_vram_gib),
    ]
    with failure_report(report_path, report), stdout_path.open(
        "w", encoding="utf-8"
    ) as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
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
            object_info = helpers.request_json(f"{base_url}/object_info", timeout=60)
            prompt = helpers.normalize_prompt_file_selectors(prompt, object_info)
            for run_index in range(1, runs + 1):
                before_events = len(_read_events(event_path))
                samples: list[dict[str, Any]] = []
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
                        f"{base_url}/prompt",
                        {"prompt": prompt},
                        timeout=60,
                    )
                    history = helpers.wait_prompt(
                        base_url,
                        response["prompt_id"],
                        process,
                        timeout,
                    )
                finally:
                    stop_event.set()
                    monitor.join(timeout=15)
                if monitor.is_alive():
                    raise RuntimeError("resource monitor did not stop")
                if monitor_errors:
                    raise RuntimeError(f"resource monitor failed: {monitor_errors[0]}")
                event_summary = _event_summary(_read_events(event_path)[before_events:])
                resources = helpers.summarize_resources(samples)
                record = {
                    "run": run_index,
                    "cache_mode": "cold" if run_index == 1 else "warm-file-cache",
                    "prompt_id": response["prompt_id"],
                    "history_status": history.get("status", {}),
                    **event_summary,
                    "resources": resources,
                    "post_encode_gpus": helpers.gpu_metrics(),
                    "post_encode_committed_mib": helpers.committed_memory_mib(),
                }
                report["records"].append(record)
                helpers.write_monitor_csv(result_dir / f"run-{run_index}-monitor.csv", samples)
                _write_report(report_path, report)
                print(json.dumps(record, ensure_ascii=False), flush=True)
                time.sleep(2)
        finally:
            helpers.terminate_process_tree(process, job)
            time.sleep(3)
    report["status"] = "passed"
    report["finished_ns"] = time.time_ns()
    report["final_gpus"] = helpers.gpu_metrics()
    report["final_committed_mib"] = helpers.committed_memory_mib()
    _write_report(report_path, report)
    return report_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-prompt", type=Path, default=DEFAULT_API_PROMPT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--backend",
        choices=("AUTO", "XFORMERS_GQA", "TORCH_EFFICIENT"),
        default="AUTO",
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--port", type=int, default=8194)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--reserve-vram-gib", type=float, default=4.0)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(
        api_prompt_path=args.api_prompt,
        manifest_path=args.manifest,
        backend=args.backend,
        runs=args.runs,
        port=args.port,
        timeout=args.timeout,
        reserve_vram_gib=args.reserve_vram_gib,
        result_root=args.result_root,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
