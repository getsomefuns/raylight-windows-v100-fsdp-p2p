"""Benchmark Raylight's MiniMax H3 dual-V100 temporal VideoVAE decoder."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import psutil
import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--latent-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-psnr-db", type=float, default=58.0)
    parser.add_argument("--maximum-abs-error", type=float, default=0.003)
    parser.add_argument("--maximum-warm-seconds", type=float, default=42.0)
    parser.add_argument("--minimum-improvement-percent", type=float, default=35.0)
    parser.add_argument("--maximum-gpu-peak-mib", type=float, default=7 * 1024)
    parser.add_argument("--maximum-dual-ram-growth-mib", type=float, default=4 * 1024)
    parser.add_argument("--dual-runs", type=int, default=5)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    return parser.parse_args(argv)


def extract_video_latent(payload) -> torch.Tensor:
    samples = payload.get("samples") if isinstance(payload, dict) else payload
    if getattr(samples, "is_nested", False):
        streams = samples.unbind()
        if not streams:
            raise ValueError("cached AV latent contains no streams")
        samples = streams[0]
    if not isinstance(samples, torch.Tensor):
        raise TypeError("cached latent must contain a torch.Tensor")
    if samples.ndim == 4:
        samples = samples.unsqueeze(0)
    if samples.ndim != 5:
        raise ValueError(f"video latent must be B,C,T,H,W; received {tuple(samples.shape)}")
    return samples.detach().to(device="cpu", dtype=torch.float32).contiguous()


def quality_metrics(baseline: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    if tuple(baseline.shape) != tuple(candidate.shape):
        raise ValueError(f"decoded shapes differ: {tuple(baseline.shape)} != {tuple(candidate.shape)}")
    squared_sum = 0.0
    element_count = 0
    maximum = 0.0
    finite = True
    for index in range(int(baseline.shape[2])):
        left = baseline[:, :, index].float()
        right = candidate[:, :, index].float()
        finite = finite and bool(torch.isfinite(left).all()) and bool(torch.isfinite(right).all())
        difference = left - right
        squared_sum += float(torch.sum(difference.double().square()).item())
        element_count += difference.numel()
        maximum = max(maximum, float(difference.abs().max().item()))
    mse = squared_sum / max(1, element_count)
    psnr = math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)
    return {
        "mse": mse,
        "psnr_db": psnr,
        "max_abs_error": maximum,
        "finite": finite,
    }


def normalize_native_output(decoded: torch.Tensor, *, batch: int, frames: int) -> torch.Tensor:
    if not isinstance(decoded, torch.Tensor):
        raise TypeError("ComfyUI VAE.decode must return a tensor")
    if decoded.ndim == 4 and decoded.shape[0] == batch * frames and decoded.shape[-1] == 3:
        return decoded.reshape(batch, frames, *decoded.shape[1:]).movedim(-1, 1)
    if decoded.ndim == 5 and decoded.shape[-1] == 3:
        return decoded.movedim(-1, 1)
    if decoded.ndim == 5 and decoded.shape[1] == 3:
        return decoded
    raise ValueError(f"unexpected native VAE output shape: {tuple(decoded.shape)}")


def build_report(*, native: dict, dual_runs: list[dict], quality: dict, thresholds: dict) -> dict:
    if not dual_runs:
        raise ValueError("at least one dual-GPU run is required")
    native_seconds = float(native["seconds"])
    warm_seconds = float(dual_runs[-1]["seconds"])
    speedup = native_seconds / warm_seconds
    improvement = (native_seconds - warm_seconds) / native_seconds * 100.0
    first_end_ram = float(dual_runs[0].get("resources", {}).get("end_physical_used_mib", 0.0))
    last_end_ram = float(dual_runs[-1].get("resources", {}).get("end_physical_used_mib", 0.0))
    dual_ram_growth = max(0.0, last_end_ram - first_end_ram)
    dual_gpu_peaks = []
    for run in dual_runs:
        dual_gpu_peaks.extend(
            float(gpu["peak_memory_mib"])
            for gpu in run.get("resources", {}).get("gpus", {}).values()
        )
    gates = {
        "finite": bool(quality["finite"]),
        "psnr": float(quality["psnr_db"]) >= float(thresholds["minimum_psnr_db"]),
        "max_abs_error": float(quality["max_abs_error"]) <= float(thresholds["maximum_abs_error"]),
        "warm_seconds": warm_seconds <= float(thresholds["maximum_warm_seconds"]),
        "improvement": improvement >= float(thresholds["minimum_improvement_percent"]),
        "gpu_peak": bool(dual_gpu_peaks)
        and max(dual_gpu_peaks) <= float(thresholds["maximum_gpu_peak_mib"]),
        "dual_ram_growth": dual_ram_growth
        <= float(thresholds.get("maximum_dual_ram_growth_mib", math.inf)),
    }
    return {
        "schema_version": 1,
        "native": native,
        "dual_runs": dual_runs,
        "quality": quality,
        "thresholds": thresholds,
        "comparison": {
            "warm_speedup": speedup,
            "warm_improvement_percent": improvement,
            "dual_ram_growth_mib": dual_ram_growth,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _committed_memory_mib() -> float:
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
        raise RuntimeError("GetPerformanceInfo failed")
    return info.CommitTotal * info.PageSize / 2**20


def _gpu_metrics() -> list[dict]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    rows = []
    for line in result.stdout.splitlines():
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
        raise RuntimeError(f"expected GPU 0 and 1, received {rows}")
    return rows


def _sample_resources(stop: threading.Event, samples: list[dict], interval: float) -> None:
    while not stop.is_set():
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        samples.append(
            {
                "time_ns": time.time_ns(),
                "physical_used_mib": memory.used / 2**20,
                "committed_mib": _committed_memory_mib(),
                "pagefile_used_mib": swap.used / 2**20,
                "gpus": _gpu_metrics(),
            }
        )
        stop.wait(interval)


def _summarize_resources(samples: list[dict]) -> dict:
    if not samples:
        raise RuntimeError("resource monitor captured no samples")
    summary = {
        "samples": len(samples),
        "start_physical_used_mib": samples[0]["physical_used_mib"],
        "end_physical_used_mib": samples[-1]["physical_used_mib"],
        "peak_physical_used_mib": max(row["physical_used_mib"] for row in samples),
        "start_committed_mib": samples[0]["committed_mib"],
        "end_committed_mib": samples[-1]["committed_mib"],
        "peak_committed_mib": max(row["committed_mib"] for row in samples),
        "start_pagefile_used_mib": samples[0]["pagefile_used_mib"],
        "end_pagefile_used_mib": samples[-1]["pagefile_used_mib"],
        "peak_pagefile_used_mib": max(row["pagefile_used_mib"] for row in samples),
        "gpus": {},
    }
    for row in samples:
        for gpu in row["gpus"]:
            target = summary["gpus"].setdefault(
                str(gpu["index"]),
                {"peak_memory_mib": 0.0, "peak_util_percent": 0.0, "peak_power_w": 0.0},
            )
            target["peak_memory_mib"] = max(target["peak_memory_mib"], gpu["memory_mib"])
            target["peak_util_percent"] = max(target["peak_util_percent"], gpu["util_percent"])
            target["peak_power_w"] = max(target["peak_power_w"], gpu["power_w"])
    return summary


def _run_measured(callable_, interval: float):
    samples: list[dict] = []
    stop = threading.Event()
    monitor = threading.Thread(target=_sample_resources, args=(stop, samples, interval), daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            value = callable_()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        stop.set()
        monitor.join()
    return value, {"seconds": seconds, "resources": _summarize_resources(samples)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class _GpuReservation:
    def gpu_ids(self):
        import ray

        return ray.get_gpu_ids()


class _VaeDecodeActor:
    def __init__(self, comfy_root: str, source_root: str):
        sys.path.insert(0, str(Path(source_root) / "src"))
        sys.path.insert(0, comfy_root)
        from raylight.distributed_worker.ray_worker_vae import (
            BackgroundVaePreloader,
            load_vae_model,
        )

        self._preloader = BackgroundVaePreloader(load_vae_model)
        self._vae = None
        self._path = None

    def gpu_ids(self):
        import ray

        return ray.get_gpu_ids()

    def preload(self, path: str):
        started = time.perf_counter()
        self._vae = self._preloader.wait(path)
        self._path = path
        return {"seconds": time.perf_counter() - started, "gpu_ids": self.gpu_ids()}

    def clear_sampling_vram(self):
        import comfy.model_management as model_management

        model_management.unload_all_models()
        torch.cuda.empty_cache()
        gc.collect()
        return True

    def ray_mmh3_vae_decode_chunks(self, samples, chunk_indices, vae_path):
        from raylight.distributed_worker.ray_worker_vae import decode_mmh3_temporal_chunks

        if self._vae is None or self._path != vae_path:
            self._vae = self._preloader.wait(vae_path)
            self._path = vae_path
        return decode_mmh3_temporal_chunks(self._vae, samples, chunk_indices)


def _save_frame(tensor: torch.Tensor, path: Path) -> None:
    from PIL import Image

    image = tensor.detach().float().clamp(0, 1).mul(255).round().to(torch.uint8)
    Image.fromarray(image.permute(1, 2, 0).numpy()).save(path)


def _main(args: argparse.Namespace) -> int:
    source_root = Path(__file__).resolve().parents[2]
    comfy_root = args.comfy_root.resolve()
    vae_path = args.vae_path.resolve()
    latent_path = args.latent_path.resolve()
    output_dir = args.output_dir.resolve()
    for path, label in ((comfy_root, "ComfyUI root"), (vae_path, "VAE"), (latent_path, "latent")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(comfy_root))

    import ray
    from raylight.distributed_worker.ray_worker_vae import (
        build_mmh3_temporal_plan,
        load_vae_model,
    )
    from raylight.integrations.mmh3director.vae_decode import (
        DirectorVaeDecodeContext,
        hybrid_decode_mmh3_video,
    )

    payload = torch.load(latent_path, map_location="cpu", weights_only=False)
    video = extract_video_latent(payload)
    vae = load_vae_model(str(vae_path))
    plan = build_mmh3_temporal_plan(vae.first_stage_model, video.shape[2])
    if plan["num_chunks"] < 2:
        raise RuntimeError("benchmark latent does not produce at least two temporal chunks")

    ray_temp = output_dir / "ray-runtime"
    ray.init(num_gpus=2, include_dashboard=False, _temp_dir=str(ray_temp))
    reservation = None
    actor = None
    try:
        reservation_type = ray.remote(num_gpus=1)(_GpuReservation)
        reservation = reservation_type.remote()
        reservation_ids = ray.get(reservation.gpu_ids.remote())
        actor_type = ray.remote(num_gpus=1)(_VaeDecodeActor)
        actor = actor_type.remote(str(comfy_root), str(source_root))
        actor_ids = ray.get(actor.gpu_ids.remote())
        if reservation_ids == actor_ids:
            raise RuntimeError(f"Ray assigned both actors to the same GPU: {actor_ids}")
        preload = ray.get(actor.preload.remote(str(vae_path)))

        def native_decode():
            return normalize_native_output(
                vae.decode(video),
                batch=int(video.shape[0]),
                frames=int(plan["output_frames"]),
            )

        baseline, native_record = _run_measured(native_decode, args.sample_interval)
        native_record["kind"] = "comfyui_native_vae_decode"

        context = DirectorVaeDecodeContext(
            remote_worker=actor,
            vae_path=str(vae_path),
            vae_identity=id(vae),
            workers=(actor,),
        )
        dual_runs = []
        candidate = None
        if args.dual_runs < 1:
            raise ValueError("--dual-runs must be at least 1")
        for run_index in range(args.dual_runs):
            decoded, record = _run_measured(
                lambda: hybrid_decode_mmh3_video(vae, video, context).movedim(-1, 1),
                args.sample_interval,
            )
            record.update({"kind": "dual_gpu_temporal", "run_index": run_index})
            dual_runs.append(record)
            candidate = decoded
            gc.collect()
        assert candidate is not None
        quality = quality_metrics(baseline, candidate)
        thresholds = {
            "minimum_psnr_db": args.minimum_psnr_db,
            "maximum_abs_error": args.maximum_abs_error,
            "maximum_warm_seconds": args.maximum_warm_seconds,
            "minimum_improvement_percent": args.minimum_improvement_percent,
            "maximum_gpu_peak_mib": args.maximum_gpu_peak_mib,
            "maximum_dual_ram_growth_mib": args.maximum_dual_ram_growth_mib,
        }
        report = build_report(
            native=native_record,
            dual_runs=dual_runs,
            quality=quality,
            thresholds=thresholds,
        )
        report.update(
            {
                "environment": {
                    "platform": platform.platform(),
                    "python": sys.version,
                    "torch": torch.__version__,
                    "ray": ray.__version__,
                    "reservation_gpu_ids": reservation_ids,
                    "decode_actor_gpu_ids": actor_ids,
                },
                "input": {
                    "vae_path": str(vae_path),
                    "vae_sha256": _sha256(vae_path),
                    "latent_path": str(latent_path),
                    "latent_sha256": _sha256(latent_path),
                    "latent_shape": list(video.shape),
                    "output_shape": list(baseline.shape),
                    "temporal_chunks": int(plan["num_chunks"]),
                },
                "remote_preload": preload,
            }
        )
        for frame_index in (0, int(baseline.shape[2]) // 2, int(baseline.shape[2]) - 1):
            _save_frame(baseline[0, :, frame_index], output_dir / f"native_frame_{frame_index:03d}.png")
            _save_frame(candidate[0, :, frame_index], output_dir / f"dual_frame_{frame_index:03d}.png")
        (output_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0 if report["passed"] else 2
    finally:
        if actor is not None:
            ray.kill(actor, no_restart=True)
        if reservation is not None:
            ray.kill(reservation, no_restart=True)
        ray.shutdown()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("USE_LIBUV", "0")
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
