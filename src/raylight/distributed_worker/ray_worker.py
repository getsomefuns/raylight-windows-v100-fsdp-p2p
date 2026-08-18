import os
import sys
import gc
import json
import logging
import functools
import time
import statistics
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
import ray

import comfy.patcher_extension as pe

from raylight.distributed_modules.pipefusion import (
    PipeFusionInjectRegistry,
    pipefusion_diffusion_model_wrapper,
    pipefusion_outer_sample_wrapper,
    pipefusion_predict_noise_wrapper,
)
from raylight.distributed_modules.usp import USPInjectRegistry
from raylight.distributed_modules.cfg import CFGParallelInjectRegistry
from raylight.distributed_worker.pipefusion_schema import (
    PipeFusionConfig,
    build_stage_plan,
)
from raylight.distributed_worker.pipefusion_state import (
    PIPEFUSION_RUNTIME_ATTACHMENT,
    PIPEFUSION_WRAPPER_KEY,
    PipeFusionRuntime,
)
from raylight.distributed_worker.parallel_group_manager import (
    initialize_xfuser_parallel,
    requires_xfuser_parallel,
    validate_hybrid_topology,
)
from raylight.distributed_worker.a2a_trace import create_a2a_tracer, trace_a2a_capture
from raylight.distributed_worker.ray_worker_controlnet import (
    _prepare_control_models,
    _remap_conditioning_devices,
    _restore_controlnet_refs,
)
from raylight.distributed_worker.ray_worker_vae import (
    load_vae_model,
    ray_vae_decode_finalize_impl,
    ray_vae_decode_partial_impl,
    ray_seedvr2_vae_decode_partial_impl,
)
from raylight.distributed_worker.utils import Noise_EmptyNoise, Noise_RandomNoise, patch_ray_tqdm
from raylight.distributed_worker.windows_gloo import (
    init_windows_gloo_process_group,
    is_windows,
)
from raylight.distributed_worker.windows_p2p import (
    CudaP2PAllToAll,
    DEFAULT_WINDOWS_P2P_CAPACITY_BYTES,
    WindowsSpinControl,
    install_collective_routers,
    restore_collective_routers,
    should_use_safetensors_mmap,
    synchronized_model_load,
)
from raylight.comfy_dist.quant_ops import patch_temp_fix_ck_ops
from raylight.comfy_dist.fsdp_utils import summarize_fsdp_parameters
from raylight.comfy_dist.minimax_h3_fp16 import (
    minimax_h3_safe_fp16_construction,
    prepare_minimax_h3_safe_fp16_worker,
)
from raylight.distributed_worker.sampling_profiler import run_with_optional_profile
from ray.exceptions import RayActorError


_WORKER_AIMDO_INIT_ATTEMPTED = False

def windows_p2p_health_iterations(
    size_bytes,
    target_remote_bytes=100 * 1024**3,
    minimum_iterations=100,
):
    if size_bytes <= 0 or size_bytes % 2:
        raise ValueError(f"health-check size must be a positive even byte count, got {size_bytes}")
    if target_remote_bytes < 0:
        raise ValueError("target_remote_bytes cannot be negative")
    if minimum_iterations < 1:
        raise ValueError("minimum_iterations must be positive")
    remote_bytes_per_iteration = size_bytes // 2
    target_iterations = (
        target_remote_bytes + remote_bytes_per_iteration - 1
    ) // remote_bytes_per_iteration
    return max(minimum_iterations, target_iterations)


def windows_p2p_warmup_ready(checks, minimum_gib_s):
    if minimum_gib_s < 0:
        raise ValueError("minimum_gib_s cannot be negative")
    ranks = []
    values = []
    for check in checks:
        ranks.append(int(check["rank"]))
        values.append(float(check["remote_gib_s"]))
    if len(ranks) != 2 or set(ranks) != {0, 1}:
        raise ValueError(
            f"Windows P2P warmup requires exactly ranks 0 and 1, found {sorted(ranks)}"
        )
    return all(value >= minimum_gib_s for value in values)


def summarize_windows_p2p_health(check_trials):
    if not check_trials:
        raise ValueError("at least one Windows P2P health trial is required")

    samples_by_rank = {}
    expected_ranks = None
    for trial_index, checks in enumerate(check_trials):
        trial_ranks = set()
        for check in checks:
            rank = int(check["rank"])
            if rank in trial_ranks:
                raise ValueError(
                    f"duplicate Windows P2P health rank {rank} in trial {trial_index}"
                )
            trial_ranks.add(rank)
            samples_by_rank.setdefault(rank, []).append(float(check["remote_gib_s"]))
        if expected_ranks is None:
            expected_ranks = trial_ranks
        elif trial_ranks != expected_ranks:
            raise ValueError(
                f"Windows P2P health rank mismatch in trial {trial_index}: "
                f"expected {sorted(expected_ranks)}, found {sorted(trial_ranks)}"
            )

    return [
        {
            "rank": rank,
            "samples_gib_s": samples,
            "median_gib_s": statistics.median(samples),
            "min_gib_s": min(samples),
            "max_gib_s": max(samples),
        }
        for rank, samples in sorted(samples_by_rank.items())
    ]


def trace_windows_p2p_profile(function):
    """Emit one aggregate P2P profile line around a sampler invocation."""

    @functools.wraps(function)
    def wrapped(worker, *args, **kwargs):
        endpoint = getattr(worker, "_windows_p2p", None)
        if endpoint is None:
            return function(worker, *args, **kwargs)
        initial = endpoint.profile_snapshot(reset=True)
        if not initial.get("enabled", False):
            return function(worker, *args, **kwargs)

        started_ns = time.perf_counter_ns()
        status = "success"
        error_type = None
        try:
            return function(worker, *args, **kwargs)
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            profile = endpoint.profile_snapshot(reset=True)
            payload = {
                "elapsed_ns": time.perf_counter_ns() - started_ns,
                "invocation": int(getattr(worker, "_sampler_invocation", 0)),
                "pid": os.getpid(),
                "profile": profile,
                "rank": int(worker.local_rank),
                "sampler": function.__name__,
                "status": status,
                "time_ns": time.time_ns(),
            }
            if error_type is not None:
                payload["error_type"] = error_type
            print(
                f"[RAYLIGHT_P2P_PROFILE] {json.dumps(payload, sort_keys=True)}",
                flush=True,
            )

    return wrapped


def _raylight_rank_diag_enabled():
    return os.environ.get("RAYLIGHT_RANK_DIAG", "0") == "1"


def _raylight_rank_diag(worker, invocation, event, **fields):
    """Emit low-frequency phase timing without synchronizing CUDA."""
    if not _raylight_rank_diag_enabled():
        return
    payload = {
        "event": event,
        "invocation": invocation,
        "perf_ns": time.perf_counter_ns(),
        "pid": os.getpid(),
        "rank": worker.local_rank,
        "time_ns": time.time_ns(),
    }
    try:
        import psutil

        memory = psutil.Process().memory_info()
        for name in ("rss", "vms", "pagefile", "peak_pagefile", "private", "num_page_faults"):
            value = getattr(memory, name, None)
            if value is not None:
                payload[f"process_{name}"] = value
    except Exception as exc:
        payload["process_memory_error"] = type(exc).__name__
    if torch.cuda.is_available():
        payload["cuda_allocated"] = torch.cuda.memory_allocated()
        payload["cuda_reserved"] = torch.cuda.memory_reserved()
    payload.update(fields)
    print(f"[RAYLIGHT_RANK_DIAG] {json.dumps(payload, ensure_ascii=False, sort_keys=True)}", flush=True)


def prepare_sampling_with_cuda_trim(fn, *args, **kwargs):
    """Run Comfy preparation, then return cached blocks to CUDA before forward."""
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


# Developer reminder, Checking model parameter outside ray actor is very expensive (e.g Comfy main thread)
# the model need to be serialized, send to object store and can cause OOM !, so setter and getter is the pattern !


# If ray actor function being called from outside, ray.get([task in actor task]) will become sync between rank
# If called from ray actor within. dist.barrier() become the sync.

# PIPEFUSION STUFF IS ONLY TESTING, IT IS BREAKING DOWN ALLLL THE TIME


# Comfy cli args, does not get pass through into ray actor
def patch_enable_comfy_kitchen_fsdp(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        from raylight.comfy_dist.kitchen_distributed import patch_enable_comfy_kitchen_fsdp as patcher

        return patcher(fn)(self, *args, **kwargs)

    return wrapper


# To pass comfyui cli basically
def _apply_worker_comfy_cli_args_from_env():
    raw_args = os.environ.get("RAYLIGHT_COMFY_CLI_ARGS_JSON")
    if not raw_args:
        return {}

    try:
        worker_cli_args = json.loads(raw_args)
    except Exception as exc:
        logging.warning(f"Failed to parse RAYLIGHT_COMFY_CLI_ARGS_JSON: {exc}")
        return {}

    try:
        import comfy.cli_args
        from comfy.cli_args import PerformanceFeature
    except Exception as exc:
        logging.warning(f"Failed to import Comfy modules for worker CLI arg sync: {exc}")
        return worker_cli_args

    if "fast" in worker_cli_args:
        worker_cli_args["fast"] = {PerformanceFeature(feature) for feature in worker_cli_args["fast"]}

    comfy_args = comfy.cli_args.args
    for key, value in worker_cli_args.items():
        if hasattr(comfy_args, key):
            setattr(comfy_args, key, value)

    return worker_cli_args


def _apply_worker_late_comfy_cli_args(worker_cli_args):
    if not worker_cli_args:
        return

    try:
        import comfy.model_management as comfy_model_management
        import comfy.utils as comfy_utils
    except Exception as exc:
        logging.warning(f"Failed to import Comfy modules for worker late CLI arg sync: {exc}")
        return

    if "disable_mmap" in worker_cli_args:
        comfy_utils.DISABLE_MMAP = bool(worker_cli_args["disable_mmap"])
    if "mmap_torch_files" in worker_cli_args:
        comfy_utils.MMAP_TORCH_FILES = bool(worker_cli_args["mmap_torch_files"])

    if "disable_smart_memory" in worker_cli_args:
        comfy_model_management.DISABLE_SMART_MEMORY = bool(worker_cli_args["disable_smart_memory"])

    reserve_vram = worker_cli_args.get("reserve_vram")
    if reserve_vram is not None:
        comfy_model_management.EXTRA_RESERVED_VRAM = reserve_vram * 1024 * 1024 * 1024

    if worker_cli_args.get("disable_pinned_memory", False):
        comfy_model_management.MAX_PINNED_MEMORY = -1

        def _pin_memory_disabled(tensor):
            del tensor
            return False

        def _unpin_memory_disabled(tensor):
            del tensor
            return False

        comfy_model_management.pin_memory = _pin_memory_disabled
        comfy_model_management.unpin_memory = _unpin_memory_disabled


def _reload_worker_aimdo_modules():
    import importlib

    modules = []
    for module_name in ("comfy_aimdo.host_buffer", "comfy_aimdo.vram_buffer", "comfy_aimdo.model_vbar"):
        try:
            module = importlib.import_module(module_name)
            module = importlib.reload(module)
            modules.append(module)
        except Exception as exc:
            logging.warning(f"[Raylight][AIMDO] Failed to reload {module_name}: {exc}")
    return modules


def _enable_worker_dynamic_vram(worker_cli_args=None):
    global _WORKER_AIMDO_INIT_ATTEMPTED
    if _WORKER_AIMDO_INIT_ATTEMPTED:
        return
    _WORKER_AIMDO_INIT_ATTEMPTED = True
    worker_cli_args = worker_cli_args or {}

    try:
        import comfy.cli_args
        from comfy.cli_args import enables_dynamic_vram
        import comfy_aimdo.control
    except Exception as exc:
        logging.warning(f"[Raylight][AIMDO] DynamicVRAM unavailable in worker: {exc}")
        return

    comfy_args = comfy.cli_args.args
    headroom_bytes = int(float(getattr(comfy_args, "vram_headroom", 0) or 0) * 1024 ** 3)
    reserve_vram = getattr(comfy_args, "reserve_vram", None)
    try:
        try:
            comfy_aimdo.control.init(
                simple_vram_headroom=None if reserve_vram is None else int(float(reserve_vram) * 1024 ** 3)
            )
        except TypeError:
            # comfy-aimdo 0.4.9 protocol.
            comfy_aimdo.control.init()

        aimdo_modules = _reload_worker_aimdo_modules()

        import comfy.memory_management as comfy_memory_management
        import comfy.model_management as comfy_model_management
        import comfy.model_patcher as comfy_model_patcher

        should_enable = bool(getattr(comfy_args, "enable_dynamic_vram", False)) or (
            enables_dynamic_vram()
            and comfy_model_management.is_nvidia()
            and not comfy_model_management.is_wsl()
        )
        if not should_enable:
            return

        if (
            not getattr(comfy_args, "enable_dynamic_vram", False)
            and getattr(comfy_model_management, "torch_version_numeric", (0, 0)) < (2, 8)
        ):
            logging.warning(
                "[Raylight][AIMDO] DynamicVRAM requires PyTorch 2.8 or later; "
                "worker is falling back to legacy ModelPatcher."
            )
            return

        devices = comfy_model_management.get_all_torch_devices()
        try:
            aimdo_initialized = comfy_aimdo.control.init_devices((d.index, headroom_bytes) for d in devices)
        except TypeError:
            # comfy-aimdo 0.4.9 protocol.
            aimdo_initialized = comfy_aimdo.control.init_devices(d.index for d in devices)
    except Exception as exc:
        logging.warning(f"[Raylight][AIMDO] DynamicVRAM init failed in worker: {exc}")
        return

    if not aimdo_initialized:
        logging.warning(
            "[Raylight][AIMDO] No working comfy-aimdo install detected in worker. "
            "DynamicVRAM disabled for Raylight worker."
        )
        return

    host_buffer_lib_ok = True
    for module in aimdo_modules:
        if module.__name__.endswith(".host_buffer"):
            host_buffer_lib_ok = getattr(module, "lib", None) is not None
            break
    if not host_buffer_lib_ok:
        logging.warning(
            "[Raylight][AIMDO] comfy_aimdo.host_buffer.lib is None after init; "
            "DynamicVRAM disabled for Raylight worker."
        )
        return

    verbose = getattr(comfy_args, "verbose", "INFO")
    if verbose == "DEBUG":
        comfy_aimdo.control.set_log_debug()
    elif verbose == "CRITICAL":
        comfy_aimdo.control.set_log_critical()
    elif verbose == "ERROR":
        comfy_aimdo.control.set_log_error()
    elif verbose == "WARNING":
        comfy_aimdo.control.set_log_warning()
    else:
        comfy_aimdo.control.set_log_info()

    comfy_model_patcher.CoreModelPatcher = comfy_model_patcher.ModelPatcherDynamic
    comfy_memory_management.aimdo_enabled = True
    print(
        "[Raylight][AIMDO] DynamicVRAM enabled in worker "
        f"devices={[str(d) for d in devices]} "
        f"vram_headroom_gb={float(getattr(comfy_args, 'vram_headroom', 0) or 0)} "
        f"python={sys.executable} "
        f"control={getattr(comfy_aimdo.control, '__file__', None)} "
        f"host_buffer_lib={host_buffer_lib_ok}"
    )


def _get_guider_conditionings(guider_spec):
    guider_type = guider_spec.get("type")
    if guider_type == "basic":
        return [guider_spec["positive"]]
    if guider_type == "cfg":
        return [guider_spec["positive"], guider_spec["negative"]]
    if guider_type == "dual_cfg":
        return [guider_spec["positive"], guider_spec["middle"], guider_spec["negative"]]
    if "positive" in guider_spec and "negative" in guider_spec:
        return [guider_spec["positive"], guider_spec["negative"]]
    raise ValueError(f"Unsupported RAY_GUIDER type: {guider_type!r}")


# Helper funcFor sampler custom
def _generate_advanced_noise(add_noise, noise_or_seed, latent):
    if not add_noise:
        noise_source = Noise_EmptyNoise()
        return noise_source.generate_noise(latent), noise_source.seed

    if hasattr(noise_or_seed, "generate_noise"):
        return noise_or_seed.generate_noise(latent), getattr(noise_or_seed, "seed", None)

    noise_source = Noise_RandomNoise(noise_or_seed)
    return noise_source.generate_noise(latent), noise_source.seed


def _build_ray_guider(model, guider_spec):
    import math

    import comfy.samplers
    import node_helpers

    from raylight.expansion.comfyui_ltxv.guiders import build_ltxv_ray_guider

    guider_type = guider_spec.get("type")

    if guider_type == "basic":
        class RayGuiderBasic(comfy.samplers.CFGGuider):
            def set_conds(self, positive):
                self.inner_set_conds({"positive": positive})

        guider = RayGuiderBasic(model)
        guider.set_conds(guider_spec["positive"])
        return guider

    if guider_type == "cfg":
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(guider_spec["positive"], guider_spec["negative"])
        guider.set_cfg(guider_spec["cfg"])
        return guider

    ltxv_guider = build_ltxv_ray_guider(model, guider_spec)
    if ltxv_guider is not None:
        return ltxv_guider

    if guider_type != "dual_cfg":
        raise ValueError(f"Unsupported RAY_GUIDER type: {guider_type!r}")

    class RayGuiderDualCFG(comfy.samplers.CFGGuider):
        def set_cfg(self, cfg1, cfg2, nested=False):
            self.cfg1 = cfg1
            self.cfg2 = cfg2
            self.nested = nested

        def set_conds(self, positive, middle, negative):
            middle = node_helpers.conditioning_set_values(middle, {"prompt_type": "negative"})
            self.inner_set_conds({"positive": positive, "middle": middle, "negative": negative})

        def predict_noise(self, x, timestep, model_options={}, seed=None):
            negative_cond = self.conds.get("negative", None)
            middle_cond = self.conds.get("middle", None)
            positive_cond = self.conds.get("positive", None)

            if self.nested:
                out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
                pred_text = comfy.samplers.cfg_function(
                    self.inner_model,
                    out[2],
                    out[1],
                    self.cfg1,
                    x,
                    timestep,
                    model_options=model_options,
                    cond=positive_cond,
                    uncond=middle_cond,
                )
                return out[0] + self.cfg2 * (pred_text - out[0])

            if not model_options.get("disable_cfg1_optimization", False):
                if math.isclose(self.cfg2, 1.0):
                    negative_cond = None
                    if math.isclose(self.cfg1, 1.0):
                        middle_cond = None

            out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
            return comfy.samplers.cfg_function(
                self.inner_model,
                out[1],
                out[0],
                self.cfg2,
                x,
                timestep,
                model_options=model_options,
                cond=middle_cond,
                uncond=negative_cond,
            ) + (out[2] - out[1]) * self.cfg1

    guider = RayGuiderDualCFG(model)
    guider.set_conds(guider_spec["positive"], guider_spec["middle"], guider_spec["negative"])
    guider.set_cfg(guider_spec["cfg1"], guider_spec["cfg2"], guider_spec.get("nested", False))
    return guider


def validate_windows_p2p_launch(world_size, shard_size, parallel_dict):
    """Validate the fixed two-rank Windows P2P topology for USP or FSDP inference."""
    if world_size != 2 or shard_size != 2:
        raise ValueError(
            "RAYLIGHT_WINDOWS_P2P=1 requires two ranks and shard_size=2"
        )

    validate_hybrid_topology(world_size, shard_size, parallel_dict)

class RayWorker:
    def __init__(self, local_rank, device_id, parallel_dict):
        worker_cli_args = _apply_worker_comfy_cli_args_from_env()
        self.model = None
        self.vae_model = None
        self.model_type = None
        self.state_dict = None
        self.cached_controlnet = None  # (path, controlnet_object) cache
        self.lora_list = None
        self.parallel_dict = parallel_dict
        self.overwrite_cast_dtype = None
        self.cached_base_model = None
        self.cached_base_key = None
        self.active_request_key = None

        self.local_rank = local_rank
        self.global_world_size = self.parallel_dict["global_world_size"]
        self.shard_size = self.parallel_dict["shard_size"]
        self.group_id = self.parallel_dict.get("group_id", 0)

        self.device_id = device_id
        self.parallel_dict = parallel_dict
        self.device = torch.device(f"cuda:{self.device_id}")
        self.device_mesh = None
        self.compute_capability = int("{}{}".format(*torch.cuda.get_device_capability()))
        self.device_name = torch.cuda.get_device_name(self.device)
        self.pipefusion_config = PipeFusionConfig.from_parallel_dict(self.parallel_dict)
        self.pipefusion_stage = None
        self.xfuser_parallel = None
        self._windows_p2p = None
        self._windows_p2p_original_collectives = None
        self._sampler_invocation = 0

        self.is_model_loaded = False
        self.is_cpu_offload = self.parallel_dict.get("fsdp_cpu_offload", False)

        os.environ["XDIT_LOGGING_LEVEL"] = "WARN"
        os.environ["NCCL_DEBUG"] = "WARN"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device_id)
        # torch.cuda was already initialized above under Ray's assigned GPU visibility.
        _enable_worker_dynamic_vram(worker_cli_args)
        _apply_worker_late_comfy_cli_args(worker_cli_args)

        from raylight.comfy_dist import patch_base_getattr

        patch_base_getattr()

        if self.parallel_dict.get("use_group_process_group") and self.parallel_dict.get("shard_size") is not None:
            nccl_world_size = self.parallel_dict["shard_size"]
        else:
            nccl_world_size = self.parallel_dict["global_world_size"]
        nccl_rank = local_rank

        # Each group gets its own port for NCCL isolation
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        base_port = int(os.environ.get("MASTER_PORT", "29500"))
        group_port = base_port + self.group_id

        if is_windows():
            gloo_host = init_windows_gloo_process_group(
                rank=nccl_rank,
                world_size=nccl_world_size,
                master_addr=master_addr,
                port=group_port,
            )
            print(
                f"[Raylight] Windows Gloo init OK rank={nccl_rank}/{nccl_world_size} "
                f"store={master_addr}:{group_port} device={gloo_host}"
            )
        else:
            dist.init_process_group(
                "nccl",
                rank=nccl_rank,
                world_size=nccl_world_size,
                timeout=timedelta(minutes=1),
                init_method=f"tcp://{master_addr}:{group_port}",
            )

        if self.parallel_dict["is_xdit"] or self.parallel_dict["is_fsdp"]:
            self.device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(nccl_world_size,))

        # Just experimenting, user can't trigger this
        elif not self.parallel_dict.get("pipefusion_enabled"):
            print(f"Running Ray in normal seperate sampler with: {self.global_world_size} number of workers")

        if requires_xfuser_parallel(self.parallel_dict):
            self.xfuser_parallel = initialize_xfuser_parallel(
                local_rank=self.local_rank,
                world_size=nccl_world_size,
                parallel_dict=self.parallel_dict,
            )
            if self.parallel_dict["is_xdit"]:
                print("XDiT is enable")
            if self.parallel_dict.get("pipefusion_enabled"):
                print("PipeFusion xFuser topology is enable")
            print(
                "Parallel Degree: "
                f"Ulysses={self.xfuser_parallel.config.ulysses_degree}, "
                f"Ring={self.xfuser_parallel.config.ring_degree}, "
                f"CFG={self.xfuser_parallel.config.cfg_degree}, "
                f"PP={self.xfuser_parallel.config.pp_degree}, "
                f"DP={self.xfuser_parallel.config.data_parallel_degree}"
            )
        self._a2a_tracer = create_a2a_tracer(local_rank)
        self._install_rank_phase_diagnostics()

    def _install_rank_phase_diagnostics(self):
        """Synchronize model loads and optionally trace the pre-collective path."""
        import comfy.model_management as comfy_model_management

        if is_windows() and os.environ.get("RAYLIGHT_WINDOWS_P2P", "0") == "1":
            original_model_load = comfy_model_management.LoadedModel.model_load
            if not getattr(original_model_load, "_raylight_model_load_sync_wrapped", False):

                @functools.wraps(original_model_load)
                def synchronized_load(loaded_model, lowvram_model_memory=0, force_patch_weights=False):
                    if not comfy_model_management.is_device_cuda(loaded_model.device):
                        return original_model_load(
                            loaded_model,
                            lowvram_model_memory,
                            force_patch_weights=force_patch_weights,
                        )

                    local_budget = float(lowvram_model_memory)

                    def reduce_min(value):
                        budget = torch.tensor([value], dtype=torch.float64, device="cpu")
                        dist.all_reduce(budget, op=dist.ReduceOp.MIN)
                        return float(budget.item())

                    def load_with_budget(synchronized_budget):
                        free_bytes, total_bytes = torch.cuda.mem_get_info()
                        _raylight_rank_diag(
                            self,
                            self._sampler_invocation,
                            "model_load_budget_synchronized",
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            cuda_free_bytes=free_bytes,
                            cuda_total_bytes=total_bytes,
                            local_budget_bytes=local_budget,
                            synchronized_budget_bytes=synchronized_budget,
                        )
                        return original_model_load(
                            loaded_model,
                            synchronized_budget,
                            force_patch_weights=force_patch_weights,
                        )

                    return synchronized_model_load(
                        local_budget,
                        load_with_budget,
                        reduce_min,
                        dist.barrier,
                    )

                synchronized_load._raylight_model_load_sync_wrapped = True
                comfy_model_management.LoadedModel.model_load = synchronized_load
            if self.parallel_dict.get("is_fsdp", False):
                import comfy.sampler_helpers as comfy_sampler_helpers

                original_prepare = comfy_sampler_helpers.prepare_sampling
                if not getattr(original_prepare, "_raylight_cuda_trim_wrapped", False):

                    @functools.wraps(original_prepare)
                    def prepare_with_cuda_trim(*args, **kwargs):
                        return prepare_sampling_with_cuda_trim(
                            original_prepare, *args, **kwargs
                        )

                    prepare_with_cuda_trim._raylight_cuda_trim_wrapped = True
                    comfy_sampler_helpers.prepare_sampling = prepare_with_cuda_trim
                    print("[FSDP] CUDA cache trim enabled after prepare_sampling")


        if not _raylight_rank_diag_enabled():
            return

        import comfy.sampler_helpers as comfy_sampler_helpers

        def wrap(target, name, label):
            original = getattr(target, name)
            if getattr(original, "_raylight_rank_diag_wrapped", False):
                return

            @functools.wraps(original)
            def traced(*args, **kwargs):
                invocation = self._sampler_invocation
                started = time.perf_counter()
                _raylight_rank_diag(self, invocation, f"{label}_begin")
                try:
                    return original(*args, **kwargs)
                finally:
                    _raylight_rank_diag(
                        self,
                        invocation,
                        f"{label}_end",
                        elapsed_seconds=time.perf_counter() - started,
                    )

            traced._raylight_rank_diag_wrapped = True
            setattr(target, name, traced)

        wrap(comfy_sampler_helpers, "prepare_sampling", "prepare_sampling")
        wrap(comfy_model_management, "load_models_gpu", "load_models_gpu")
        wrap(comfy_model_management, "free_memory", "free_memory")
        wrap(comfy_model_management.LoadedModel, "model_load", "loaded_model_model_load")
    def prepare_windows_p2p(self, group_name, capacity_bytes):
        if not is_windows():
            raise RuntimeError("Windows CUDA P2P backend is only available on Windows")
        if self.global_world_size != 2 or self.shard_size != 2:
            raise ValueError("Windows CUDA P2P backend currently requires exactly two ranks")
        self._windows_p2p_control = WindowsSpinControl(group_name, self.local_rank)
        self._windows_p2p = CudaP2PAllToAll(
            self.local_rank,
            capacity_bytes,
            self._windows_p2p_control,
            timeout_seconds=float(os.environ.get("RAYLIGHT_WINDOWS_P2P_TIMEOUT_SECONDS", "10")),
        )
        return self._windows_p2p.local_ipc_metadata()

    def connect_windows_p2p(self, peer_metadata):
        if self._windows_p2p is None:
            raise RuntimeError("prepare_windows_p2p must run before connect_windows_p2p")
        self._windows_p2p.connect_ipc_metadata(peer_metadata)
        return True

    def check_windows_p2p(self, size_bytes, iterations=100):
        if self._windows_p2p is None:
            raise RuntimeError("Windows CUDA P2P backend is not connected")
        elements = size_bytes // 4
        half = elements // 2
        source = torch.empty(elements, dtype=torch.float32, device=self.device)
        source[:half].fill_(self.local_rank * 100 + 1)
        source[half:].fill_(self.local_rank * 100 + 2)
        output = torch.empty_like(source)
        for _ in range(3):
            self._windows_p2p.all_to_all_single(output, source)
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for _ in range(iterations):
            self._windows_p2p.all_to_all_single(output, source)
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        expected = (1, 101) if self.local_rank == 0 else (2, 102)
        observed = (int(output[0].item()), int(output[half].item()))
        if observed != expected:
            raise RuntimeError(
                f"Windows P2P health check mismatch rank={self.local_rank}: "
                f"observed={observed}, expected={expected}"
            )
        return {
            "rank": self.local_rank,
            "elapsed_seconds": elapsed,
            "remote_gib_s": (size_bytes / 2 / 2**30) * iterations / elapsed,
        }

    def enable_windows_p2p(self):
        if self._windows_p2p is None:
            raise RuntimeError("Windows CUDA P2P backend is not connected")
        if getattr(self, "_windows_p2p_original_collectives", None) is None:
            self._windows_p2p_original_collectives = install_collective_routers(
                self._windows_p2p,
                dist,
            )
        return True

    def disable_windows_p2p(self):
        originals = getattr(self, "_windows_p2p_original_collectives", None)
        if originals is not None:
            restore_collective_routers(dist, originals)
            self._windows_p2p_original_collectives = None
        return True

    def get_meta_model(self):
        base_model = getattr(self.model, "model", self.model)
        first_param_device = next(base_model.parameters()).device
        if first_param_device == torch.device("meta"):
            return self.model
        else:
            raise ValueError("Model recieved is not meta, can cause OOM in large model")

    def set_meta_model(self, model):
        base_model = getattr(model, "model", model)
        first_param_device = next(base_model.parameters()).device
        if first_param_device == torch.device("meta"):
            # Free old model VRAM before replacing — without this, switching
            # workflows leaves the previous FSDP model's DTensor shards on GPU
            # and the new model's patch_fsdp() OOMs during set_model_state_dict.
            self._free_current_model()
            self.state_dict = None
            self.model = model
            self.model.config_fsdp(self.local_rank, self.device_mesh)
        else:
            raise ValueError("Model being set is not meta, can cause OOM in large model")

    def _free_current_model(self):
        """Eagerly free the current model's GPU storage.

        Handles both FSDP models (DTensor shards) and non-FSDP models.
        Must be called before dropping the model reference so VRAM is
        released deterministically without relying on __del__ / gc.
        """
        import comfy.model_management as comfy_model_management

        if self.model is None:
            return

        if hasattr(self.model, "free_fsdp_vram"):
            try:
                self.model.free_fsdp_vram()
            except Exception as e:
                print(f"[Rank {self.local_rank}] free_fsdp_vram failed in _free_current_model: {e}")
        else:
            try:
                self.model.unpatch_model(device_to=None)
            except Exception as e:
                print(f"[Rank {self.local_rank}] model.unpatch_model() failed in _free_current_model: {e}")

        try:
            self.model.cleanup()
        except Exception as e:
            print(f"[Rank {self.local_rank}] model.cleanup() failed in _free_current_model: {e}")

        self.model = None
        self.overwrite_cast_dtype = None
        self.active_request_key = None
        gc.collect()
        comfy_model_management.soft_empty_cache()

    def _free_cached_aux_models(self):
        """Free cached ControlNet and VAE GPU memory."""
        if self.cached_controlnet is not None:
            _, old_cnet = self.cached_controlnet
            old_model = getattr(old_cnet, "control_model", None)
            if old_model is not None:
                del old_model
            self.cached_controlnet = None

        if self.vae_model is not None:
            del self.vae_model
            self.vae_model = None
            self._cached_vae_path = None

        torch.cuda.empty_cache()
        gc.collect()

    def clear_sampling_vram(self):
        """Release worker-side CUDA memory after a Ray sampling node finishes.

        Keep the ModelPatcher object alive so cached ComfyUI workflows can run
        the sampler again without requiring the Ray UNet loader node to rerun.
        """
        import comfy.model_management as comfy_model_management

        try:
            comfy_model_management.unload_all_models()
        except Exception as e:
            print(f"[Rank {self.local_rank}] unload_all_models failed in clear_sampling_vram: {e}")

        if self.model is not None:
            try:
                self.model.unpatch_model(device_to=getattr(self.model, "offload_device", None))
            except Exception as e:
                print(f"[Rank {self.local_rank}] model.unpatch_model() failed in clear_sampling_vram: {e}")
            try:
                self.model.cleanup()
            except Exception as e:
                print(f"[Rank {self.local_rank}] model.cleanup() failed in clear_sampling_vram: {e}")

        if self.cached_base_model is not None and self.cached_base_model is not self.model:
            try:
                self.cached_base_model.cleanup()
            except Exception as e:
                print(f"[Rank {self.local_rank}] cached model cleanup failed in clear_sampling_vram: {e}")

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        comfy_model_management.soft_empty_cache()
        return True

    def check_model_loaded(self, unet_path, model_options):
        """Check if the currently loaded model matches the given parameters.

        Used for reuse detection: when MCP or a different workflow runs with the
        same model + lora + options, we can skip the full reload.
        """
        active_key = self._active_model_key(unet_path, model_options)
        return self.model is not None and self.active_request_key == active_key and not self._fsdp_init_failed()

    def _fsdp_init_failed(self):
        if self.model is None or getattr(self.model, "fsdp_state_dict", None) is None:
            return False
        base_model = getattr(self.model, "model", self.model)
        return isinstance(base_model.diffusion_model, FSDPModule)

    def _patch_fsdp_for_sampling(self):
        try:
            self.model.patch_fsdp()
        except Exception:
            self.active_request_key = None
            self.is_model_loaded = False
            raise

    def fsdp_preflight(self):
        if not self.parallel_dict.get("is_fsdp", False):
            raise ValueError("FSDP preflight requires RayInitializer FSDP=true")
        if self.model is None:
            raise ValueError("FSDP preflight requires RayUNETLoader to load a model first")

        self._patch_fsdp_for_sampling()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        base_model = getattr(self.model, "model", self.model)
        diffusion_model = base_model.diffusion_model
        diagnostics = summarize_fsdp_parameters(diffusion_model)
        endpoint = getattr(self, "_windows_p2p", None)
        result = {
            "rank": int(self.local_rank),
            "world_size": int(self.global_world_size),
            "is_fsdp_module": isinstance(diffusion_model, FSDPModule),
            "p2p_next_operation_id": int(getattr(endpoint, "_operation_id", 0)),
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
            **diagnostics,
        }
        print(f"[Rank {self.local_rank}][FSDP_PREFLIGHT] {json.dumps(result, sort_keys=True)}", flush=True)
        return result

    def _normalize_model_options(self, model_options):
        if not model_options:
            return ()

        normalized = []
        for key in sorted(model_options.keys()):
            value = model_options[key]
            if isinstance(value, (list, tuple)):
                value = tuple(str(v) for v in value)
            else:
                value = str(value)
            normalized.append((key, value))
        return tuple(normalized)

    def _lora_signature(self, lora_list=None):
        if lora_list is None:
            lora_list = self.lora_list

        if not lora_list:
            return ()

        return tuple((lora["path"], float(lora["strength_model"])) for lora in lora_list)

    def _base_model_key(self, unet_path, model_options):
        return (unet_path, self._normalize_model_options(model_options))

    def _active_model_key(self, unet_path, model_options, lora_list=None):
        return self._base_model_key(unet_path, model_options) + (self._lora_signature(lora_list),)

    def _reset_active_model(self):
        import comfy.model_management as comfy_model_management

        if self.model is not None:
            if hasattr(self.model, "free_fsdp_vram"):
                try:
                    self.model.free_fsdp_vram()
                except Exception as e:
                    print(f"[Rank {self.local_rank}] free_fsdp_vram failed in _reset_active_model: {e}")
                try:
                    self.model.detach()
                except Exception as e:
                    print(f"[Rank {self.local_rank}] model.detach() failed in _reset_active_model: {e}")
            else:
                # Non-FSDP: unpatch LoRA weights but do NOT move model to CPU.
                # clone() shares the underlying model with cached_base_model,
                # so free_model_vram() would corrupt the cache. Instead, restore
                # original weights from backup (device_to=None skips .to(CPU)).
                try:
                    self.model.unpatch_model(device_to=None)
                except Exception as e:
                    print(f"[Rank {self.local_rank}] model.unpatch_model() failed in _reset_active_model: {e}")
            try:
                self.model.cleanup()
            except Exception as e:
                print(f"[Rank {self.local_rank}] model.cleanup() failed in _reset_active_model: {e}")

        self.model = None
        self.overwrite_cast_dtype = None
        self.active_request_key = None
        gc.collect()
        comfy_model_management.soft_empty_cache()

    def _invalidate_non_fsdp_cache(self):
        self.cached_base_model = None
        self.cached_base_key = None
        self.active_request_key = None

    def _activate_cached_base_model(self, active_key):
        self._reset_active_model()
        self.model = self.cached_base_model.clone()

        if self.lora_list is not None:
            self.load_lora()

        base_model = getattr(self.model, "model", self.model)
        self.overwrite_cast_dtype = getattr(base_model, "manual_cast_dtype", None)
        self.active_request_key = active_key
        self.is_model_loaded = True

    def set_state_dict(self):
        if self.state_dict is None:
            if self.parallel_dict.get("is_fsdp") is True and self.parallel_dict.get("is_quant") is False:
                self.model.set_fsdp_state_dict({})
                return
            raise ValueError("Worker state_dict is None before set_state_dict")
        self.model.set_fsdp_state_dict(self.state_dict)
        self.state_dict = None

    def get_compute_capability(self):
        return self.compute_capability

    def get_parallel_dict(self):
        return self.parallel_dict

    def get_exec_group_info(self):
        if self.xfuser_parallel is None:
            if self.parallel_dict.get("use_group_process_group"):
                dp_rank = self.group_id
                dp_degree = self.parallel_dict.get("dp_degree", 1)
            else:
                dp_rank = self.local_rank
                dp_degree = self.global_world_size

            return {
                "global_rank": self.local_rank,
                "dp_rank": dp_rank,
                "dp_degree": dp_degree,
                "is_group_leader": self.local_rank == 0,
            }

        is_group_leader = (
            self.xfuser_parallel.sequence_rank == 0 and self.xfuser_parallel.pipeline_rank == 0 and self.xfuser_parallel.cfg_rank == 0
        )
        return {
            "global_rank": self.xfuser_parallel.global_rank,
            "dp_rank": self.xfuser_parallel.data_parallel_rank,
            "dp_degree": self.xfuser_parallel.data_parallel_world_size,
            "is_group_leader": is_group_leader,
        }

    def _grouped_sampling_result(self, result):
        group_info = self.get_exec_group_info()
        if not group_info["is_group_leader"]:
            return None

        return {
            "dp_rank": group_info["dp_rank"],
            "result": result,
        }

    def set_parallel_dict(self, parallel_dict):
        self.parallel_dict = parallel_dict
        self.pipefusion_config = PipeFusionConfig.from_parallel_dict(self.parallel_dict)

    def model_function_runner(self, fn, *args, **kwargs):
        self.model = fn(self.model, *args, **kwargs)

    def model_function_runner_get_values(self, fn, *args, **kwargs):
        return fn(self.model, *args, **kwargs)

    def get_local_rank(self):
        return self.local_rank

    def get_is_model_loaded(self):
        return self.is_model_loaded

    def patch_cfg(self):
        if CFGParallelInjectRegistry.has_direct_handler(self.model):
            self.model.add_callback(pe.CallbacksMP.ON_LOAD, CFGParallelInjectRegistry.inject_direct)
        else:
            self.model.add_wrapper(pe.WrappersMP.DIFFUSION_MODEL, CFGParallelInjectRegistry.inject(self.model))

    def patch_usp(self):
        self.model.add_callback(
            pe.CallbacksMP.ON_LOAD,
            USPInjectRegistry.inject,
        )

    def patch_pipefusion(self):
        if not self.pipefusion_config.enabled:
            return
        if self.parallel_dict.get("is_fsdp"):
            raise ValueError("PipeFusion v1 cannot be enabled together with FSDP")
        if self.xfuser_parallel is None:
            raise RuntimeError("PipeFusion requires xFuser model parallel state to be initialized")
        if self.xfuser_parallel.config.cfg_degree != 1:
            raise NotImplementedError("PipeFusion currently ignores CFG parallel execution; keep cfg_degree at 1")
        if self.xfuser_parallel.sequence_world_size != 1:
            raise NotImplementedError(
                "PipeFusion topology is now initialized through xFuser, but the Wan execution path does not yet combine PP with USP"
            )

        base_model = getattr(self.model, "model", self.model)
        if not hasattr(base_model, "diffusion_model") or not hasattr(base_model.diffusion_model, "blocks"):
            raise ValueError(f"PipeFusion requires a Wan diffusion model with blocks, got {type(base_model).__name__}")

        self.pipefusion_stage = getattr(self.model, "pipefusion_stage", None)
        if self.pipefusion_stage is None:
            self.pipefusion_stage = build_stage_plan(
                total_blocks=getattr(
                    base_model.diffusion_model, "_raylight_pipefusion_total_blocks", len(base_model.diffusion_model.blocks)
                ),
                rank=self.xfuser_parallel.pipeline_rank,
                world_size=self.xfuser_parallel.pipeline_world_size,
                config=self.pipefusion_config,
                group_ranks=tuple(self.xfuser_parallel.pp_group().ranks),
            )

        runtime = PipeFusionRuntime(
            config=self.pipefusion_config,
            stage=self.pipefusion_stage,
            model_name=type(base_model).__name__,
            parallel=self.xfuser_parallel,
        )
        if runtime.debug:
            print(
                "[PipeFusion] "
                f"global_rank={self.xfuser_parallel.global_rank} "
                f"pp_group={self.pipefusion_stage.group_ranks} "
                f"stage={self.pipefusion_stage.stage_start}:{self.pipefusion_stage.stage_end} "
                f"patches={self.pipefusion_stage.num_pipeline_patch}"
            )
        self.model.set_attachments(PIPEFUSION_RUNTIME_ATTACHMENT, runtime)

        self.model.remove_callbacks_with_key(pe.CallbacksMP.ON_LOAD, PIPEFUSION_WRAPPER_KEY)
        self.model.remove_wrappers_with_key(pe.WrappersMP.OUTER_SAMPLE, PIPEFUSION_WRAPPER_KEY)
        self.model.remove_wrappers_with_key(pe.WrappersMP.PREDICT_NOISE, PIPEFUSION_WRAPPER_KEY)
        self.model.remove_wrappers_with_key(pe.WrappersMP.DIFFUSION_MODEL, PIPEFUSION_WRAPPER_KEY)

        self.model.add_callback_with_key(
            pe.CallbacksMP.ON_LOAD,
            PIPEFUSION_WRAPPER_KEY,
            PipeFusionInjectRegistry.inject,
        )
        self.model.add_wrapper_with_key(
            pe.WrappersMP.OUTER_SAMPLE,
            PIPEFUSION_WRAPPER_KEY,
            pipefusion_outer_sample_wrapper,
        )
        self.model.add_wrapper_with_key(
            pe.WrappersMP.PREDICT_NOISE,
            PIPEFUSION_WRAPPER_KEY,
            pipefusion_predict_noise_wrapper,
        )
        self.model.add_wrapper_with_key(
            pe.WrappersMP.DIFFUSION_MODEL,
            PIPEFUSION_WRAPPER_KEY,
            pipefusion_diffusion_model_wrapper,
        )

    def load_unet(self, unet_path, model_options):
        safe_fp16_active = prepare_minimax_h3_safe_fp16_worker(
            model_options,
            compute_capability=self.compute_capability,
            device_name=getattr(self, "device_name", str(self.device)),
            rank=self.local_rank,
            is_fsdp=self.parallel_dict.get("is_fsdp") is True,
            is_xdit=self.parallel_dict.get("is_xdit") is True,
        )
        if self.parallel_dict["is_fsdp"] is True:
            active_key = self._active_model_key(unet_path, model_options)

            # Fast path: same base model + same LoRA — reuse FSDP-wrapped model
            if self.model is not None and self.active_request_key == active_key and not self._fsdp_init_failed():
                base_model = getattr(self.model, "model", self.model)
                self.overwrite_cast_dtype = getattr(base_model, "manual_cast_dtype", None)
                self.is_model_loaded = True
                return

            # Model or LoRA changed — free old VRAM deterministically, then reload
            if self.model is not None:
                try:
                    self.model.free_fsdp_vram()
                except Exception as e:
                    print(f"[Rank {self.local_rank}] free_fsdp_vram failed: {e}")
                self._free_cached_aux_models()

            # Monkey patch
            import comfy.model_patcher as model_patcher
            import comfy.model_management as model_management

            from raylight.comfy_dist.model_management import cleanup_models_gc
            from raylight.comfy_dist.model_patcher import LowVramPatch

            from raylight.comfy_dist.sd import fsdp_load_diffusion_model

            fsdp_model_options = dict(model_options)
            fsdp_model_options["use_mmap"] = self.parallel_dict.get("use_mmap", True)

            model_patcher.LowVramPatch = LowVramPatch
            model_management.cleanup_models_gc = cleanup_models_gc

            del self.model
            del self.state_dict
            self.model = None
            self.state_dict = None
            torch.cuda.synchronize()
            gc.collect()
            model_management.soft_empty_cache()

            with minimax_h3_safe_fp16_construction(fsdp_model_options):
                self.model, self.state_dict = fsdp_load_diffusion_model(
                    unet_path,
                    self.local_rank,
                    self.device_mesh,
                    self.is_cpu_offload,
                    model_options=fsdp_model_options,
                )
            torch.cuda.synchronize()
            model_management.soft_empty_cache()
            gc.collect()

            if self.lora_list is not None:
                self.load_lora()

            base_model = getattr(self.model, "model", self.model)
            self.overwrite_cast_dtype = getattr(base_model, "manual_cast_dtype", None)
            if safe_fp16_active:
                print(
                    f"[Raylight] MiniMax H3 safe FP16 rank={self.local_rank} "
                    f"model_dtype={fsdp_model_options.get('dtype')} "
                    f"manual_cast_dtype={self.overwrite_cast_dtype}"
                )
            self.is_model_loaded = True
            self.active_request_key = active_key
            return
        else:
            import comfy.sd as comfy_sd

            base_key = self._base_model_key(unet_path, model_options)
            active_key = self._active_model_key(unet_path, model_options)
            use_mmap = should_use_safetensors_mmap(self.parallel_dict, unet_path)

            if self.model is not None and self.active_request_key == active_key:
                base_model = getattr(self.model, "model", self.model)
                self.overwrite_cast_dtype = getattr(base_model, "manual_cast_dtype", None)
                self.is_model_loaded = True
                return

            if self.cached_base_model is not None and self.cached_base_key == base_key:
                # If cached model has GPU weights (from a previous full-load run
                # without LoRA), free them now. Otherwise partially_load() sees
                # model_loaded_weight_memory=0 and does a full_load=True pass that
                # converts fp8 weights to bf16 during LoRA patching, spiking VRAM
                # from ~15Gb to ~23Gb on each GPU.
                _cached_has_gpu_weights = False
                try:
                    cached_base_model = getattr(self.cached_base_model, "model", self.cached_base_model)
                    _first_p = next(cached_base_model.parameters(), None)
                    if _first_p is not None and _first_p.device.type == "cuda":
                        _cached_has_gpu_weights = True
                except Exception:
                    pass

                if _cached_has_gpu_weights:
                    from raylight.comfy_dist.model_patcher import free_model_vram

                    free_model_vram(self.cached_base_model)
                    self._invalidate_non_fsdp_cache()
                    # Fall through to reload from disk (mmap, no RAM spike)
                else:
                    self._activate_cached_base_model(active_key)
                    return

            self._reset_active_model()
            self._invalidate_non_fsdp_cache()
            self._free_cached_aux_models()
            if self.parallel_dict.get("pipefusion_enabled"):
                from raylight.comfy_dist.sd import pipefusion_load_diffusion_model

                if self.xfuser_parallel is None:
                    raise RuntimeError("PipeFusion model loading requires xFuser parallel context")

                pipefusion_model_options = dict(model_options)
                pipefusion_model_options["use_mmap"] = use_mmap
                loaded_model = pipefusion_load_diffusion_model(
                    unet_path,
                    pipefusion_config=self.pipefusion_config,
                    parallel_context=self.xfuser_parallel,
                    model_options=pipefusion_model_options,
                )
            elif use_mmap:
                from raylight.comfy_dist.sd import lazy_load_diffusion_model

                try:
                    loaded_model = lazy_load_diffusion_model(
                        unet_path,
                        model_options=model_options,
                    )
                except Exception as exc:
                    print(f"[RayWorker {self.local_rank}] Lazy safetensor load failed, falling back to eager load: {exc}")
                    loaded_model = comfy_sd.load_diffusion_model(
                        unet_path,
                        model_options=model_options,
                    )
            else:
                loaded_model = comfy_sd.load_diffusion_model(
                    unet_path,
                    model_options=model_options,
                )
            self.cached_base_model = loaded_model
            self.cached_base_key = base_key
            self._activate_cached_base_model(active_key)
            return

    def load_gguf_unet(self, unet_path, dequant_dtype, patch_dtype, use_mmap=None):
        self._reset_active_model()
        self._invalidate_non_fsdp_cache()
        self._free_cached_aux_models()
        if use_mmap is None:
            use_mmap = self.parallel_dict.get("use_mmap", True)
        if self.parallel_dict["is_fsdp"] is True:
            # GGUF FSDP stays disabled for now.
            raise RuntimeError("FSDP on GGUF is not supported")
        else:
            from raylight.comfy_dist.sd import gguf_load_diffusion_model

            if self.model is not None:
                try:
                    self.model.free_fsdp_vram()
                except Exception as e:
                    print(f"[Rank {self.local_rank}] free_fsdp_vram failed (bnb): {e}")

            self.model = gguf_load_diffusion_model(
                unet_path,
                model_options={"use_mmap": use_mmap},
                dequant_dtype=dequant_dtype,
                patch_dtype=patch_dtype,
            )

        if self.lora_list is not None:
            self.load_lora()

        self.is_model_loaded = True

    def set_lora_list(self, lora):
        self.lora_list = lora

    def get_lora_list(
        self,
    ):
        return self.lora_list

    def load_lora(
        self,
    ):
        import comfy.memory_management as comfy_memory_management
        import comfy.sd as comfy_sd
        import comfy.utils as comfy_utils

        for lora in self.lora_list:
            lora_path = lora["path"]
            strength_model = lora["strength_model"]
            lora_model = comfy_utils.load_torch_file(lora_path, safe_load=True)

            if self.parallel_dict["is_fsdp"] is True:
                from raylight.comfy_dist.sd import (
                    load_lora_for_models as ray_load_lora_for_models,
                    load_lora_for_models_quantized as ray_load_lora_for_models_quantized,
                )

                dynamic_sidecar = comfy_memory_management.aimdo_enabled
                if self.parallel_dict["is_quant"] is True or dynamic_sidecar:
                    self.model = ray_load_lora_for_models_quantized(
                        self.model,
                        lora_model,
                        strength_model,
                        dynamic_sidecar=dynamic_sidecar,
                        defer_device_move=True,
                        fallback_to_patches=not self.parallel_dict["is_quant"],
                    )
                else:
                    self.model = ray_load_lora_for_models(
                        self.model,
                        lora_model,
                        strength_model,
                    )
            else:
                self.model = comfy_sd.load_lora_for_models(self.model, None, lora_model, strength_model, 0)[0]
            del lora_model

    def kill(self):
        self._free_cached_aux_models()
        self._invalidate_non_fsdp_cache()
        self.model = None
        self.disable_windows_p2p()
        if getattr(self, "_windows_p2p", None) is not None:
            self._windows_p2p.close()
            self._windows_p2p = None
        if hasattr(self, "_windows_p2p_control"):
            self._windows_p2p_control.close()
        tracer = getattr(self, "_a2a_tracer", None)
        if tracer is not None:
            tracer.close()
        dist.destroy_process_group()
        ray.actor.exit_actor()

    def ray_vae_loader(self, vae_path):
        if self.vae_model is not None and getattr(self, "_cached_vae_path", None) == vae_path:
            return

        # Free old VAE before loading new one
        if self.vae_model is not None:
            del self.vae_model
            self.vae_model = None
            self._cached_vae_path = None
            torch.cuda.empty_cache()

        vae_model = load_vae_model(vae_path)

        if self.local_rank == 0:
            print(f"VAE loaded in {self.global_world_size} GPUs")
        self.vae_model = vae_model
        self._cached_vae_path = vae_path

    @patch_ray_tqdm
    def ray_vae_decode_partial(self, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8, job_rank=0, job_world_size=1):
        return ray_vae_decode_partial_impl(self, samples, tile_size, overlap, temporal_size, temporal_overlap, job_rank, job_world_size)

    def ray_vae_decode_finalize(self, decoded):
        return ray_vae_decode_finalize_impl(self, decoded)

    @patch_ray_tqdm
    def ray_seedvr2_vae_decode_partial(self, samples, tile_size, overlap=64, job_rank=0, job_world_size=1):
        return ray_seedvr2_vae_decode_partial_impl(self, samples, tile_size, overlap, job_rank, job_world_size)

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    @patch_enable_comfy_kitchen_fsdp
    @trace_a2a_capture
    @trace_windows_p2p_profile
    def custom_sampler_advanced(
        self,
        add_noise,
        noise_seed,
        guider_spec,
        sampler,
        sigmas,
        latent_image,
        grouped_output=False,
    ):
        import comfy.model_management as comfy_model_management
        import comfy.nested_tensor as comfy_nested_tensor
        import comfy.sample as comfy_sample
        import comfy.utils as comfy_utils
        import latent_preview

        self._sampler_invocation += 1
        invocation = self._sampler_invocation
        latent_samples = latent_image.get("samples") if isinstance(latent_image, dict) else None
        _raylight_rank_diag(
            self,
            invocation,
            "sampler_entry",
            latent_shape=list(latent_samples.shape) if hasattr(latent_samples, "shape") else None,
            sigmas_shape=list(sigmas.shape) if hasattr(sigmas, "shape") else None,
        )

        for cond_list in _get_guider_conditionings(guider_spec):
            _restore_controlnet_refs(cond_list, self.cached_controlnet, self.vae_model)
            _remap_conditioning_devices(cond_list, None)
            _prepare_control_models(cond_list, None)
        _raylight_rank_diag(self, invocation, "conditions_prepared")

        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy_sample.fix_empty_latent_channels(
            self.model,
            latent_image,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = latent_image
        _raylight_rank_diag(self, invocation, "latent_prepared")

        noise, sampling_seed = _generate_advanced_noise(add_noise, noise_seed, latent)
        _raylight_rank_diag(self, invocation, "noise_prepared")

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        if self.parallel_dict["is_fsdp"] is True:
            self._patch_fsdp_for_sampling()
            del self.state_dict
            self.state_dict = None
            torch.cuda.synchronize()
            comfy_model_management.soft_empty_cache()
            gc.collect()

        guider = _build_ray_guider(self.model, guider_spec)
        _raylight_rank_diag(self, invocation, "guider_built")
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = comfy_utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED

        with torch.no_grad():
            _raylight_rank_diag(self, invocation, "sample_begin")
            samples = run_with_optional_profile(
                lambda: guider.sample(
                    noise,
                    latent_image,
                    sampler,
                    sigmas,
                    denoise_mask=noise_mask,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=sampling_seed,
                ),
                rank=self.local_rank,
                invocation=invocation,
            )
            _raylight_rank_diag(self, invocation, "sample_returned")
            samples = samples.to(comfy_model_management.intermediate_device())

            out = latent.copy()
            out.pop("downscale_ratio_spacial", None)
            out.pop("downscale_ratio_temporal", None)
            out["samples"] = samples

            if "x0" in x0_output:
                x0 = x0_output["x0"]
                if samples.is_nested and not x0.is_nested:
                    latent_shapes = [x.shape for x in samples.unbind()]
                    x0 = comfy_nested_tensor.NestedTensor(comfy_utils.unpack_latents(x0, latent_shapes))
                x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
                out_denoised = latent.copy()
                out_denoised["samples"] = x0_out
            else:
                out_denoised = out

        try:
            self.model.cleanup()
        except Exception:
            pass
        comfy_model_management.soft_empty_cache()
        gc.collect()
        _raylight_rank_diag(self, invocation, "sampler_return")
        result = (out, out_denoised)
        if grouped_output:
            return self._grouped_sampling_result(result)
        return result

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    @patch_enable_comfy_kitchen_fsdp
    @trace_a2a_capture
    @trace_windows_p2p_profile
    def custom_sampler(
        self,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
        grouped_output=False,
    ):
        import comfy.model_management as comfy_model_management
        import comfy.sample as comfy_sample
        import comfy.utils as comfy_utils

        # Restore ControlNet refs from local cache (loaded by load_controlnet)
        _restore_controlnet_refs(positive, self.cached_controlnet, self.vae_model)
        _restore_controlnet_refs(negative, self.cached_controlnet, self.vae_model)

        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy_sample.fix_empty_latent_channels(self.model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        _remap_conditioning_devices(positive, negative)
        _prepare_control_models(positive, negative)

        if self.parallel_dict["is_fsdp"] is True:
            self._patch_fsdp_for_sampling()
            del self.state_dict
            self.state_dict = None
            torch.cuda.synchronize()
            comfy_model_management.soft_empty_cache()
            gc.collect()

        disable_pbar = comfy_utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED

        with torch.no_grad():
            samples = comfy_sample.sample_custom(
                self.model,
                noise,
                cfg,
                sampler,
                sigmas,
                positive,
                negative,
                latent_image,
                noise_mask=noise_mask,
                callback=None,
                disable_pbar=disable_pbar,
                seed=noise_seed,
            )
            out = latent.copy()
            out["samples"] = samples

        try:
            self.model.cleanup()
        except Exception:
            pass
        comfy_model_management.soft_empty_cache()
        gc.collect()
        if grouped_output:
            return self._grouped_sampling_result(out)
        return out

    def load_controlnet(self, controlnet_path):
        """Load a ControlNet model from disk into the worker.

        Caches the result so subsequent calls with the same path are free.
        Frees old ControlNet VRAM when the path changes.
        Returns True on success.
        """
        if self.cached_controlnet is not None and self.cached_controlnet[0] == controlnet_path:
            return True

        # Free old ControlNet VRAM if model changed
        if self.cached_controlnet is not None:
            _, old_cnet = self.cached_controlnet
            old_model = getattr(old_cnet, "control_model", None)
            if old_model is not None:
                del old_model
            self.cached_controlnet = None
            torch.cuda.empty_cache()
            gc.collect()

        import comfy.controlnet as comfy_cnet

        cnet = comfy_cnet.load_controlnet(controlnet_path)
        if cnet is None:
            print(f"[Rank {self.local_rank}] Failed to load ControlNet: {controlnet_path}")
            return False

        self.cached_controlnet = (controlnet_path, cnet)
        if self.local_rank == 0:
            print(f"[Rank {self.local_rank}] ControlNet loaded and cached from {controlnet_path}")
        return True

    def free_cached_controlnet(self):
        """Explicitly free the cached ControlNet (e.g. when switching workflows)."""
        if self.cached_controlnet is not None:
            _, old_cnet = self.cached_controlnet
            old_model = getattr(old_cnet, "control_model", None)
            if old_model is not None:
                del old_model
            self.cached_controlnet = None
            torch.cuda.empty_cache()
            gc.collect()
            if self.local_rank == 0:
                print(f"[Rank {self.local_rank}] ControlNet cache freed")

    def free_cached_vae(self):
        """Explicitly free the cached VAE (e.g. when switching workflows)."""
        if self.vae_model is not None:
            del self.vae_model
            self.vae_model = None
            self._cached_vae_path = None
            torch.cuda.empty_cache()
            gc.collect()
            if self.local_rank == 0:
                print(f"[Rank {self.local_rank}] VAE cache freed")

    @patch_temp_fix_ck_ops
    @patch_ray_tqdm
    @patch_enable_comfy_kitchen_fsdp
    @trace_a2a_capture
    @trace_windows_p2p_profile
    def common_ksampler(
        self,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        grouped_output=False,
    ):
        import comfy.model_management as comfy_model_management
        import comfy.sample as comfy_sample
        import comfy.utils as comfy_utils

        # Restore ControlNet refs from local cache (loaded by load_controlnet)
        _restore_controlnet_refs(positive, self.cached_controlnet, self.vae_model)
        _restore_controlnet_refs(negative, self.cached_controlnet, self.vae_model)

        latent_image = latent["samples"]
        latent_image = comfy_sample.fix_empty_latent_channels(self.model, latent_image)

        if self.parallel_dict["is_fsdp"] is True:
            self._patch_fsdp_for_sampling()

        if disable_noise:
            noise = torch.zeros(
                latent_image.size(),
                dtype=latent_image.dtype,
                layout=latent_image.layout,
                device="cpu",
            )
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy_sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        _remap_conditioning_devices(positive, negative)
        _prepare_control_models(positive, negative)

        disable_pbar = comfy_utils.PROGRESS_BAR_ENABLED
        if self.local_rank == 0:
            disable_pbar = not comfy_utils.PROGRESS_BAR_ENABLED

        with torch.no_grad():
            samples = comfy_sample.sample(
                self.model,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
                disable_noise=disable_noise,
                start_step=start_step,
                last_step=last_step,
                force_full_denoise=force_full_denoise,
                noise_mask=noise_mask,
                callback=None,
                disable_pbar=disable_pbar,
                seed=seed,
            )
            out = latent.copy()
            out["samples"] = samples

        try:
            self.model.cleanup()
        except Exception:
            pass
        comfy_model_management.soft_empty_cache()
        gc.collect()
        if grouped_output:
            return self._grouped_sampling_result(out)
        return (out,)


class RayCOMMTester:
    def __init__(self, local_rank, world_size, device_id):
        device = torch.device(f"cuda:{device_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

        if is_windows():
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = int(os.environ.get("MASTER_PORT", "29500")) + 100
            init_windows_gloo_process_group(
                rank=local_rank,
                world_size=world_size,
                master_addr=master_addr,
                port=master_port,
            )
        else:
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=world_size,
                timeout=timedelta(minutes=1),
                # device_id=self.device
            )
        print("Running COMM pre-run")

        # Each rank contributes rank+1
        x = torch.ones(1, device=device) * (local_rank + 1)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        result = x.item()

        # Expected sum = N(N+1)/2
        expected = world_size * (world_size + 1) // 2

        if abs(result - expected) > 1e-3:
            raise RuntimeError(f"[Rank {local_rank}] COMM test failed: got {result}, expected {expected}. world_size may be mismatched!")
        else:
            print(f"[Rank {local_rank}] COMM test passed ✅ (result={result})")

    def kill(self):
        dist.destroy_process_group()
        ray.actor.exit_actor()


def ray_nccl_tester(world_size):
    gpu_actor = ray.remote(RayCOMMTester)
    gpu_actors = []

    for local_rank in range(world_size):
        gpu_actors.append(
            gpu_actor.options(num_gpus=1, name=f"RayTest:{local_rank}").remote(
                local_rank=local_rank,
                world_size=world_size,
                device_id=0,
            )
        )
    for actor in gpu_actors:
        ray.get(actor.__ray_ready__.remote())

    for actor in gpu_actors:
        actor.kill.remote()


def make_ray_actor_fn(world_size, parallel_dict):
    num_replicas = parallel_dict.get("dp_degree", 1)
    shard_size = parallel_dict.get("shard_size", world_size)
    use_group_process_group = bool(parallel_dict.get("use_group_process_group"))

    def _init_ray_actor(world_size=world_size, parallel_dict=parallel_dict):
        ray_actors = dict()
        gpu_actor = ray.remote(RayWorker)
        gpu_actors = []

        if num_replicas <= 1 or not use_group_process_group:
            # XDiT DP stays in one global group; xFuser derives DP ranks internally.
            for local_rank in range(world_size):
                gpu_actors.append(
                    gpu_actor.options(num_gpus=1, name=f"RayWorker:{local_rank}").remote(
                        local_rank=local_rank,
                        device_id=0,
                        parallel_dict=parallel_dict,
                    )
                )
        else:
            # FSDP multi-replica DP uses one NCCL group per replica.
            for group_id in range(num_replicas):
                group_parallel_dict = dict(parallel_dict)
                group_parallel_dict["group_id"] = group_id
                group_parallel_dict["use_group_process_group"] = True

                for local_rank in range(shard_size):
                    gpu_actors.append(
                        gpu_actor.options(
                            num_gpus=1,
                            name=f"RayWorker:{group_id}_{local_rank}"
                        ).remote(
                            local_rank=local_rank,
                            device_id=0,
                            parallel_dict=group_parallel_dict,
                        )
                    )

        ray_actors["workers"] = gpu_actors

        for actor in ray_actors["workers"]:
            ray.get(actor.__ray_ready__.remote())
        if is_windows() and os.environ.get("RAYLIGHT_WINDOWS_P2P") == "1":
            validate_windows_p2p_launch(world_size, shard_size, parallel_dict)
            capacity_bytes = int(
                os.environ.get("RAYLIGHT_WINDOWS_P2P_CAPACITY_BYTES", str(DEFAULT_WINDOWS_P2P_CAPACITY_BYTES))
            )
            health_check_bytes = min(capacity_bytes * 2, 115_343_360)
            health_target_gib = float(
                os.environ.get("RAYLIGHT_WINDOWS_P2P_HEALTH_REMOTE_GIB", "100")
            )
            health_iterations = windows_p2p_health_iterations(
                health_check_bytes,
                target_remote_bytes=int(health_target_gib * 2**30),
                minimum_iterations=max(1, int(os.environ.get("RAYLIGHT_WINDOWS_P2P_HEALTH_MIN_ITERATIONS", "100"))),
            )
            warmup_target_gib = float(
                os.environ.get("RAYLIGHT_WINDOWS_P2P_WARMUP_REMOTE_GIB", "20")
            )
            warmup_iterations = windows_p2p_health_iterations(
                health_check_bytes,
                target_remote_bytes=int(warmup_target_gib * 2**30),
                minimum_iterations=max(1, int(os.environ.get("RAYLIGHT_WINDOWS_P2P_HEALTH_MIN_ITERATIONS", "100"))),
            )
            group_name = f"raylight_{os.getpid()}_{time.time_ns()}"
            metadata = ray.get(
                [
                    actor.prepare_windows_p2p.remote(group_name, capacity_bytes)
                    for actor in ray_actors["workers"]
                ]
            )
            ray.get(
                [
                    ray_actors["workers"][rank].connect_windows_p2p.remote(metadata[1 - rank])
                    for rank in range(2)
                ]
            )
            minimum_gib_s = float(os.environ.get("RAYLIGHT_WINDOWS_P2P_MIN_GIB_S", "50"))
            warmup_trials = max(
                1, int(os.environ.get("RAYLIGHT_WINDOWS_P2P_WARMUP_TRIALS", "5"))
            )
            warmup_checks = []
            for _ in range(warmup_trials):
                checks = ray.get(
                    [
                        actor.check_windows_p2p.remote(health_check_bytes, warmup_iterations)
                        for actor in ray_actors["workers"]
                    ]
                )
                warmup_checks.append(checks)
                if windows_p2p_warmup_ready(checks, minimum_gib_s):
                    break
            health_trials = max(
                1, int(os.environ.get("RAYLIGHT_WINDOWS_P2P_HEALTH_TRIALS", "5"))
            )
            check_trials = [
                ray.get(
                    [
                        actor.check_windows_p2p.remote(health_check_bytes, health_iterations)
                        for actor in ray_actors["workers"]
                    ]
                )
                for _ in range(health_trials)
            ]
            health = summarize_windows_p2p_health(check_trials)
            slow = [check for check in health if check["median_gib_s"] < minimum_gib_s]
            if slow:
                raise RuntimeError(
                    f"Windows CUDA P2P median health check below {minimum_gib_s} "
                    f"GiB/s: health={health}, trials={check_trials}"
                )
            ray.get([actor.enable_windows_p2p.remote() for actor in ray_actors["workers"]])
            print(
                f"[Raylight] Windows CUDA P2P enabled: health={health}, "
                f"iterations_per_trial={health_iterations}, "
                f"target_remote_gib={health_target_gib}, "
                f"warmup_iterations_per_trial={warmup_iterations}, "
                f"warmup_target_remote_gib={warmup_target_gib}, warmup={warmup_checks}, "
                f"trials={check_trials}"
            )
        return ray_actors

    _init_ray_actor.raylight_world_size = world_size
    _init_ray_actor.raylight_parallel_dict = dict(parallel_dict)
    return _init_ray_actor


def ray_worker_actor_names(world_size, parallel_dict):
    num_replicas = parallel_dict.get("dp_degree", 1)
    shard_size = parallel_dict.get("shard_size", world_size)
    use_group_process_group = bool(parallel_dict.get("use_group_process_group"))
    if num_replicas <= 1 or not use_group_process_group:
        return [f"RayWorker:{rank}" for rank in range(world_size)]
    return [
        f"RayWorker:{group_id}_{local_rank}"
        for group_id in range(num_replicas)
        for local_rank in range(shard_size)
    ]


def cleanup_named_ray_workers(world_size, parallel_dict):
    killed = []
    for name in ray_worker_actor_names(world_size, parallel_dict):
        try:
            actor = ray.get_actor(name)
        except Exception:
            continue
        try:
            ray.kill(actor, no_restart=True)
            killed.append(actor)
        except Exception:
            pass
    return killed


def wait_for_ray_workers_exit(workers, timeout_seconds=10.0):
    pending = list(workers)
    deadline = time.monotonic() + timeout_seconds
    while pending:
        still_running = []
        for actor in pending:
            try:
                remaining = max(0.05, min(0.25, deadline - time.monotonic()))
                ray.get(actor.get_parallel_dict.remote(), timeout=remaining)
            except ray.exceptions.GetTimeoutError:
                still_running.append(actor)
                continue
            except RayActorError:
                continue
            except Exception as error:
                raise RuntimeError("Unable to confirm Ray worker termination") from error
            still_running.append(actor)
        if not still_running:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for {len(still_running)} Ray workers to exit"
            )
        pending = still_running
        time.sleep(0.05)


def _actor_factory_topology(ray_actor_fn, fallback_world_size):
    configured_world_size = getattr(
        ray_actor_fn, "raylight_world_size", fallback_world_size
    )
    if not isinstance(configured_world_size, int):
        configured_world_size = fallback_world_size
    configured_parallel_dict = getattr(
        ray_actor_fn, "raylight_parallel_dict", {}
    )
    if not isinstance(configured_parallel_dict, dict):
        configured_parallel_dict = {}
    world_size = int(configured_world_size)
    parallel_dict = dict(configured_parallel_dict)
    return world_size, parallel_dict


# (TODO-Komikndr) Should be removed since FSDP can be unloaded properly
def ensure_fresh_actors(ray_actors_init):
    if not isinstance(ray_actors_init, list):
        raise TypeError("RAY_ACTORS_INIT payload must be mutable for worker recovery")
    ray_actors, ray_actor_fn = ray_actors_init
    gpu_actors = list(ray_actors.get("workers", []))
    expected_world_size, expected_parallel_dict = _actor_factory_topology(
        ray_actor_fn, len(gpu_actors)
    )
    try:
        if not gpu_actors:
            raise RuntimeError("Ray actor payload contains no workers")
        parallel_dicts = ray.get(
            [actor.get_parallel_dict.remote() for actor in gpu_actors],
            timeout=5,
        )
        if len(gpu_actors) != expected_world_size or len(parallel_dicts) != expected_world_size:
            raise RuntimeError("Ray actor probe returned an incomplete result set")
    except Exception:
        for actor in gpu_actors:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
        ray_actors_init[0] = {
            "workers": [],
            "world_size": expected_world_size,
        }
        try:
            if gpu_actors:
                wait_for_ray_workers_exit(gpu_actors)
            ray_actors = ray_actor_fn()
            gpu_actors = list(ray_actors["workers"])
            if len(gpu_actors) != expected_world_size:
                raise RuntimeError(
                    "Ray actor factory returned an incomplete actor set: "
                    f"expected {expected_world_size}, got {len(gpu_actors)}"
                )
            parallel_dicts = ray.get(
                [actor.get_parallel_dict.remote() for actor in gpu_actors],
                timeout=5,
            )
            if len(parallel_dicts) != expected_world_size:
                raise RuntimeError("Recovered Ray actor probe was incomplete")
        except Exception:
            leaked_workers = cleanup_named_ray_workers(
                expected_world_size,
                expected_parallel_dict,
            )
            if leaked_workers:
                wait_for_ray_workers_exit(leaked_workers)
            ray_actors_init[0] = {
                "workers": [],
                "world_size": expected_world_size,
            }
            raise
        ray_actors_init[0] = ray_actors

    return ray_actors, gpu_actors, parallel_dicts[0]
# Modified by the windows-v100-p2p fork for Windows Gloo, CUDA P2P routing, and worker reuse support.
