"""Minimal two-V100 FSDP2 forward probe over the Windows CUDA P2P backend."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import socket
import time

os.environ["RAY_DEBUG_DISABLE_MEMORY_MONITOR"] = "1"
os.environ["RAY_memory_usage_threshold"] = "1"
os.environ["USE_LIBUV"] = "0"

import ray


REPO_ROOT = Path(__file__).parents[1]
P2P_MODULE_PATH = REPO_ROOT / "src/raylight/distributed_worker/windows_p2p.py"
GLOO_MODULE_PATH = REPO_ROOT / "src/raylight/distributed_worker/windows_gloo.py"


def load_file_module(name, path):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@ray.remote(num_gpus=1)
class FSDPProbeActor:
    def __init__(self, rank, group_name, capacity_bytes):
        import torch

        torch.cuda.set_device(0)
        self.rank = rank
        self.p2p = load_file_module(f"raylight_windows_p2p_fsdp_{rank}", P2P_MODULE_PATH)
        self.gloo = load_file_module(f"raylight_windows_gloo_fsdp_{rank}", GLOO_MODULE_PATH)
        self.control = self.p2p.WindowsSpinControl(group_name, rank)
        self.endpoint = self.p2p.CudaP2PAllToAll(
            rank,
            capacity_bytes,
            self.control,
            timeout_seconds=10,
        )
        self.originals = None
        self.mesh = None

    def init_dist(self, port):
        import torch.distributed as dist

        host = self.gloo.init_windows_gloo_process_group(
            rank=self.rank,
            world_size=2,
            master_addr="127.0.0.1",
            port=port,
        )
        self.mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(2,))
        return {"rank": self.rank, "gloo_host": host}

    def metadata(self):
        return self.endpoint.local_ipc_metadata()

    def connect_and_enable(self, peer_metadata):
        import torch.distributed as dist

        self.endpoint.connect_ipc_metadata(peer_metadata)
        self.originals = self.p2p.install_collective_routers(self.endpoint, dist)
        return True

    def _memory_snapshot(self):
        import ctypes
        from ctypes import wintypes

        import psutil
        import torch

        class PerformanceInformation(ctypes.Structure):
            _fields_ = (
                ("cb", wintypes.DWORD),
                ("commit_total", ctypes.c_size_t),
                ("commit_limit", ctypes.c_size_t),
                ("commit_peak", ctypes.c_size_t),
                ("physical_total", ctypes.c_size_t),
                ("physical_available", ctypes.c_size_t),
                ("system_cache", ctypes.c_size_t),
                ("kernel_total", ctypes.c_size_t),
                ("kernel_paged", ctypes.c_size_t),
                ("kernel_nonpaged", ctypes.c_size_t),
                ("page_size", ctypes.c_size_t),
                ("handle_count", wintypes.DWORD),
                ("process_count", wintypes.DWORD),
                ("thread_count", wintypes.DWORD),
            )

        performance = PerformanceInformation()
        performance.cb = ctypes.sizeof(performance)
        if not ctypes.windll.psapi.GetPerformanceInfo(
            ctypes.byref(performance), performance.cb
        ):
            raise ctypes.WinError()
        mib = 1024 * 1024
        process = psutil.Process()
        swap = psutil.swap_memory()
        return {
            "vram_allocated_mib": torch.cuda.memory_allocated(0) / mib,
            "vram_reserved_mib": torch.cuda.memory_reserved(0) / mib,
            "vram_peak_allocated_mib": torch.cuda.max_memory_allocated(0) / mib,
            "rss_mib": process.memory_info().rss / mib,
            "commit_used_mib": performance.commit_total * performance.page_size / mib,
            "commit_limit_mib": performance.commit_limit * performance.page_size / mib,
            "pagefile_used_mib": swap.used / mib,
        }


    def run_forward(self):
        import torch
        import torch.nn as nn
        from torch.distributed.fsdp import FSDPModule, fully_shard

        torch.manual_seed(20260815)
        model = nn.Sequential(
            nn.Linear(8, 16, bias=False),
            nn.GELU(),
            nn.Linear(16, 4, bias=False),
        ).cuda().eval()
        sample = torch.arange(16, dtype=torch.float32, device="cuda:0").reshape(2, 8) / 16
        with torch.no_grad():
            baseline = model(sample).detach().clone()

        full_numel = sum(parameter.numel() for parameter in model.parameters())
        fully_shard(model, mesh=self.mesh, reshard_after_forward=True)
        local_numel = sum(
            parameter.to_local().numel() if hasattr(parameter, "to_local") else parameter.numel()
            for parameter in model.parameters()
        )

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            output = model(sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        max_error = float((output - baseline).abs().max().item())
        return {
            "rank": self.rank,
            "is_fsdp_module": isinstance(model, FSDPModule),
            "full_numel": full_numel,
            "local_numel": local_numel,
            "max_error": max_error,
            "elapsed_seconds": elapsed,
        }

    def run_large_forward(self, dimension=16384, warm_iterations=5):
        import gc
        import torch
        import torch.nn as nn
        from torch.distributed.fsdp import FSDPModule, fully_shard

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        snapshots = {"before_model": self._memory_snapshot()}
        model = nn.Linear(
            dimension,
            dimension,
            bias=False,
            device="cuda:0",
            dtype=torch.float16,
        ).eval()
        with torch.no_grad():
            model.weight.fill_(2.0 ** -14)
        model = nn.Sequential(model)
        sample = torch.ones(
            1, dimension, device="cuda:0", dtype=torch.float16
        )
        torch.cuda.synchronize()
        snapshots["full_model"] = self._memory_snapshot()

        full_parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        fully_shard(model[0], mesh=self.mesh, reshard_after_forward=True)
        fully_shard(model, mesh=self.mesh, reshard_after_forward=True)
        local_parameter_bytes = sum(
            parameter.to_local().numel() * parameter.element_size()
            if hasattr(parameter, "to_local")
            else parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        torch.cuda.synchronize()
        snapshots["sharded"] = self._memory_snapshot()
        torch.cuda.reset_peak_memory_stats(0)

        iteration_seconds = []
        max_error = 0.0
        finite = True
        output = None
        for _ in range(warm_iterations + 1):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                output = model(sample)
            torch.cuda.synchronize()
            iteration_seconds.append(time.perf_counter() - started)
            finite = finite and bool(torch.isfinite(output).all().item())
            max_error = max(
                max_error, float((output.float() - 1.0).abs().max().item())
            )
        snapshots["after_forwards"] = self._memory_snapshot()

        is_fsdp_module = isinstance(model, FSDPModule)
        del output, sample, model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        snapshots["after_teardown"] = self._memory_snapshot()
        return {
            "rank": self.rank,
            "is_fsdp_module": is_fsdp_module,
            "dimension": dimension,
            "full_parameter_bytes": full_parameter_bytes,
            "local_parameter_bytes": local_parameter_bytes,
            "cold_seconds": iteration_seconds[0],
            "warm_seconds": iteration_seconds[1:],
            "max_error": max_error,
            "finite": finite,
            "snapshots": snapshots,
        }


    def run_quantized_forward(self):
        import json
        import sys

        import torch
        import torch.nn as nn

        source_root = str(REPO_ROOT / "src")
        comfy_root = str(REPO_ROOT.parents[1])
        for candidate in (source_root, comfy_root):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)

        from comfy.quant_ops import get_layout_class
        from raylight.comfy_dist.fsdp_utils import (
            fully_shard_bottom_up,
            load_from_full_model_state_dict,
        )
        from raylight.comfy_dist.kitchen_patches.fp8 import (
            install_fp8_patches,
            restore_fp8_patches,
        )

        torch.cuda.empty_cache()
        torch.manual_seed(20260816)
        layout_name = "TensorCoreFP8E4M3Layout"
        layout = get_layout_class(layout_name)
        scale = torch.tensor(0.01, dtype=torch.float32)
        full_weight = torch.linspace(
            -1.0, 1.0, 16 * 8, dtype=torch.float32
        ).reshape(16, 8).to(torch.bfloat16)
        qdata, params = layout.quantize(full_weight, scale=scale)
        bias = torch.linspace(-0.1, 0.1, 16, dtype=torch.bfloat16)
        sample = torch.linspace(
            -0.5, 0.5, 2 * 8, device="cuda:0", dtype=torch.float32
        ).reshape(2, 8)
        expected = torch.nn.functional.linear(
            sample,
            qdata.to(device="cuda:0", dtype=torch.float32) * scale.to(device="cuda:0"),
            bias.to(device="cuda:0", dtype=torch.float32),
        )

        model = nn.Sequential(
            nn.Linear(8, 16, bias=True, device="meta", dtype=torch.bfloat16)
        ).eval()
        payload = {
            "0.weight": qdata,
            "0.weight_scale": scale,
            "0.comfy_quant": torch.tensor(
                list(json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")),
                dtype=torch.uint8,
            ),
            "0.bias": bias,
        }
        install_fp8_patches()
        try:
            fully_shard_bottom_up(
                model,
                fsdp_kwargs={
                    "mesh": self.mesh,
                    "reshard_after_forward": True,
                },
                native_ignore_scale=False,
            )
            load_from_full_model_state_dict(
                model,
                payload,
                device=torch.device("cuda:0"),
                strict=False,
                cpu_offload=False,
                release_sd=True,
            )
            local = model[0].weight.to_local()
            local_qdata_shape = list(local._qdata.shape)
            local_logical_shape = list(local._params.orig_shape)
            with torch.no_grad():
                actual = model(sample)
            torch.cuda.synchronize()
            max_error = float((actual.float() - expected).abs().max().item())
            return {
                "rank": self.rank,
                "output_shape": list(actual.shape),
                "local_qdata_shape": local_qdata_shape,
                "local_logical_shape": local_logical_shape,
                "max_error": max_error,
                "actual_first": actual[0].float().detach().cpu().tolist(),
                "expected_first": expected[0].float().detach().cpu().tolist(),
                "finite": bool(torch.isfinite(actual).all().item()),
            }
        finally:
            restore_fp8_patches()
    def close(self):
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
        if self.originals is not None:
            self.p2p.restore_collective_routers(dist, self.originals)
            self.originals = None
        self.endpoint.close()
        self.control.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--quantized", action="store_true")
    args = parser.parse_args()
    capacity_bytes = (128 if args.large else 16) * 1024 * 1024
    ray.init(num_gpus=2, include_dashboard=False, ignore_reinit_error=True)
    actors = []
    try:
        port = find_free_port()
        group_name = f"fsdp_{os.getpid()}_{time.time_ns()}"
        actors = [
            FSDPProbeActor.remote(rank, group_name, capacity_bytes)
            for rank in range(2)
        ]
        print(ray.get([actor.init_dist.remote(port) for actor in actors], timeout=120))
        metadata = ray.get([actor.metadata.remote() for actor in actors])
        ray.get(
            [
                actors[rank].connect_and_enable.remote(metadata[1 - rank])
                for rank in range(2)
            ],
            timeout=120,
        )
        if args.quantized:
            results = ray.get(
                [actor.run_quantized_forward.remote() for actor in actors], timeout=300
            )
            for result in results:
                print("[FSDP_QUANT] " + repr(result))
                if result["output_shape"] != [2, 16]:
                    raise RuntimeError(result)
                if not result["finite"] or result["max_error"] > 1e-5:
                    raise RuntimeError(result)
                if result["local_qdata_shape"] != [8, 8]:
                    raise RuntimeError(result)
            ray.get([actor.close.remote() for actor in actors], timeout=120)
            raise SystemExit(0)

        if args.large:
            results = ray.get(
                [actor.run_large_forward.remote() for actor in actors], timeout=300
            )
            for result in results:
                print(result)
                print(
                    "[F2_LARGE] "
                    + repr(
                        {
                            "rank": result["rank"],
                            "cold_seconds": result["cold_seconds"],
                            "warm_seconds": result["warm_seconds"],
                            "vram_sharded_mib": result["snapshots"]["sharded"]["vram_allocated_mib"],
                            "vram_after_forwards_mib": result["snapshots"]["after_forwards"]["vram_allocated_mib"],
                            "vram_after_teardown_mib": result["snapshots"]["after_teardown"]["vram_allocated_mib"],
                            "commit_delta_mib": result["snapshots"]["after_teardown"]["commit_used_mib"]
                            - result["snapshots"]["before_model"]["commit_used_mib"],
                            "pagefile_used_mib": result["snapshots"]["after_teardown"]["pagefile_used_mib"],
                        }
                    )
                )
                if not result["is_fsdp_module"]:
                    raise RuntimeError(result)
                if result["local_parameter_bytes"] * 2 != result["full_parameter_bytes"]:
                    raise RuntimeError(result)
                if not result["finite"] or result["max_error"] > 1e-3:
                    raise RuntimeError(result)
                if len(result["warm_seconds"]) != 5:
                    raise RuntimeError(result)
                snapshots = result["snapshots"]
                if snapshots["after_forwards"]["vram_allocated_mib"] > snapshots["sharded"]["vram_allocated_mib"] + 32:
                    raise RuntimeError(result)
                if snapshots["after_teardown"]["vram_allocated_mib"] > snapshots["before_model"]["vram_allocated_mib"] + 32:
                    raise RuntimeError(result)

            ray.get([actor.close.remote() for actor in actors], timeout=120)
            raise SystemExit(0)

        results = ray.get([actor.run_forward.remote() for actor in actors], timeout=120)
        for result in results:
            print(result)
            if not result["is_fsdp_module"]:
                raise RuntimeError(result)
            if result["local_numel"] * 2 != result["full_numel"]:
                raise RuntimeError(result)
            if result["max_error"] > 1e-5:
                raise RuntimeError(result)
        ray.get([actor.close.remote() for actor in actors], timeout=120)
    finally:
        ray.shutdown()
