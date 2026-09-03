from __future__ import annotations

import argparse
import importlib.metadata
import json
import queue
from pathlib import Path
import subprocess
import sys
from typing import Any


EXPECTED = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "torchaudio": "2.8.0+cu128",
    "xformers": "0.0.32.post2",
    "cuda": "12.8",
}


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _probe_xformers_forward(torch_module) -> tuple[bool, str | None]:
    try:
        from xformers.ops import memory_efficient_attention

        for device_index in range(torch_module.cuda.device_count()):
            device = torch_module.device(f"cuda:{device_index}")
            query = torch_module.randn(1, 128, 8, 64, device=device, dtype=torch_module.float16)
            output = memory_efficient_attention(query, query, query)
            if output.shape != query.shape or output.dtype != torch_module.float16:
                return False, f"cuda:{device_index} returned shape={tuple(output.shape)} dtype={output.dtype}"
            if not bool(torch_module.isfinite(output).all().item()):
                return False, f"cuda:{device_index} returned NaN/Inf"
        return True, None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _cuda_ipc_child(source, result_queue) -> None:
    import torch

    try:
        result_queue.put(
            {
                "equal": bool(torch.equal(source, torch.arange(4096, device=source.device, dtype=torch.float16))),
                "device": str(source.device),
                "finite": bool(torch.isfinite(source).all().item()),
            }
        )
    except Exception as error:
        result_queue.put({"error": f"{type(error).__name__}: {error}"})


def _probe_cuda_ipc_round_trip(torch_module) -> tuple[bool, str | None]:
    try:
        context = torch_module.multiprocessing.get_context("spawn")
        result_queue = context.Queue()
        source = torch_module.arange(4096, device="cuda:0", dtype=torch_module.float16)
        process = context.Process(target=_cuda_ipc_child, args=(source, result_queue))
        process.start()
        try:
            payload = result_queue.get(timeout=60)
        except queue.Empty:
            process.terminate()
            process.join(timeout=10)
            return False, "CUDA IPC child timed out"
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            return False, "CUDA IPC child did not exit"
        if process.exitcode != 0:
            return False, f"CUDA IPC child exited with code {process.exitcode}"
        if payload.get("error"):
            return False, str(payload["error"])
        passed = payload == {"equal": True, "device": "cuda:0", "finite": True}
        return passed, None if passed else f"unexpected CUDA IPC payload: {payload}"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def _probe_aimdo_subprocess() -> tuple[list[int], bool, str | None]:
    child = r'''
import importlib
import json
import comfy_aimdo.control

try:
    try:
        comfy_aimdo.control.init(simple_vram_headroom=None)
    except TypeError:
        comfy_aimdo.control.init()
    import torch
    devices = [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())]
    try:
        initialized = comfy_aimdo.control.init_devices((device.index, 0) for device in devices)
    except TypeError:
        initialized = comfy_aimdo.control.init_devices(device.index for device in devices)
    host_buffer = importlib.reload(importlib.import_module("comfy_aimdo.host_buffer"))
    print(json.dumps({
        "initialized": bool(initialized),
        "devices": [int(device.index) for device in devices] if initialized else [],
        "host_buffer_lib": getattr(host_buffer, "lib", None) is not None,
    }))
except Exception as error:
    print(json.dumps({
        "initialized": False,
        "devices": [],
        "host_buffer_lib": False,
        "error": f"{type(error).__name__}: {error}",
    }))
'''
    completed = subprocess.run(
        [sys.executable, "-c", child],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        detail = completed.stderr.strip() or completed.stdout.strip() or str(error)
        return [], False, f"AIMDO probe produced no JSON: {detail}"
    return (
        [int(index) for index in payload.get("devices", [])],
        bool(payload.get("host_buffer_lib", False)),
        payload.get("error"),
    )


def _dynamic_vram_state(comfyui_root: Path | None) -> tuple[bool, str | None]:
    if comfyui_root is None:
        return False, "ComfyUI root was not supplied"
    root = comfyui_root.resolve()
    if not (root / "comfy" / "model_patcher.py").is_file():
        return False, f"invalid ComfyUI root: {root}"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import comfy.memory_management
        import comfy.model_patcher

        active = bool(getattr(comfy.memory_management, "aimdo_enabled", False)) and (
            comfy.model_patcher.CoreModelPatcher is comfy.model_patcher.ModelPatcherDynamic
        )
        return active, None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def collect_runtime_facts(comfyui_root: Path | None = None, *, run_gpu_probes: bool = True) -> dict[str, Any]:
    import torch

    capabilities = [list(torch.cuda.get_device_capability(index)) for index in range(torch.cuda.device_count())]
    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    xformers_forward, xformers_error = (False, "GPU probes skipped")
    cuda_ipc_round_trip, cuda_ipc_error = (False, "GPU probes skipped")
    aimdo_devices: list[int] = []
    aimdo_host_buffer_lib = False
    aimdo_error = "GPU probes skipped"
    if run_gpu_probes and torch.cuda.is_available():
        xformers_forward, xformers_error = _probe_xformers_forward(torch)
        cuda_ipc_round_trip, cuda_ipc_error = _probe_cuda_ipc_round_trip(torch)
        aimdo_devices, aimdo_host_buffer_lib, aimdo_error = _probe_aimdo_subprocess()
    dynamic_vram, dynamic_vram_error = _dynamic_vram_state(comfyui_root)
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torchvision": _distribution_version("torchvision"),
        "torchaudio": _distribution_version("torchaudio"),
        "xformers": _distribution_version("xformers"),
        "comfy_aimdo": _distribution_version("comfy-aimdo"),
        "cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
        "devices": devices,
        "device_capabilities": capabilities,
        "arch_list": list(torch.cuda.get_arch_list()),
        "xformers_forward": xformers_forward,
        "xformers_error": xformers_error,
        "aimdo_devices": aimdo_devices,
        "aimdo_host_buffer_lib": aimdo_host_buffer_lib,
        "aimdo_error": aimdo_error,
        "cuda_ipc_round_trip": cuda_ipc_round_trip,
        "cuda_ipc_error": cuda_ipc_error,
        "dynamic_vram": dynamic_vram,
        "dynamic_vram_error": dynamic_vram_error,
    }


def validate_runtime_facts(facts: dict[str, Any], require_dynamic_vram: bool) -> list[str]:
    errors: list[str] = []
    for name, expected in EXPECTED.items():
        if facts.get(name) != expected:
            errors.append(f"{name} must be {expected}")
    if "sm_70" not in facts.get("arch_list", []):
        errors.append("torch wheel does not contain sm_70")
    if facts.get("device_capabilities") != [[7, 0], [7, 0]]:
        errors.append("both devices must be compute capability 7.0")
    if facts.get("xformers_forward") is not True:
        errors.append("xFormers V100 forward failed")
    if facts.get("aimdo_devices") != [0, 1]:
        errors.append("AIMDO did not register both devices")
    if facts.get("aimdo_host_buffer_lib") is not True:
        errors.append("AIMDO host buffer library is unavailable")
    if facts.get("cuda_ipc_round_trip") is not True:
        errors.append("CUDA IPC round-trip failed")
    if require_dynamic_vram and facts.get("dynamic_vram") is not True:
        errors.append("DynamicVRAM is not active")
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Raylight Windows V100 DynamicVRAM runtime")
    parser.add_argument("--comfyui-root", type=Path)
    parser.add_argument("--require-dynamic-vram", action="store_true")
    parser.add_argument("--skip-gpu-probes", action="store_true")
    parser.add_argument("--json", dest="json_path", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    facts = collect_runtime_facts(args.comfyui_root, run_gpu_probes=not args.skip_gpu_probes)
    errors = validate_runtime_facts(facts, require_dynamic_vram=args.require_dynamic_vram)
    payload = {"facts": facts, "errors": errors, "passed": not errors}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
