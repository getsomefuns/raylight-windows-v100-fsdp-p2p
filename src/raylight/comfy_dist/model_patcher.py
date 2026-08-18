from __future__ import annotations

import collections
import logging
import gc
import os
import sys
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.fsdp import CPUOffloadPolicy, FSDPModule, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.utils import _free_storage
from torch.distributed.tensor import DTensor

import comfy
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import get_key_weight, string_to_seed, move_weight_functions

from raylight import comfy_dist
from .fsdp_utils import (
    format_fsdp_diagnostics,
    align_fp8_logical_dtype,
    configure_minimax_h3_forward_prefetch,
    freeze_and_detect_qt,
    fully_shard_bottom_up,
    load_from_full_model_state_dict,
    materialize_excluded_params,
    summarize_fsdp_parameters,
    summarize_all_gather_inputs,
)
from .fsdp_host_register import (
    collect_fsdp_shard_cpu_tensors,
    register_cpu_storages,
)

if TYPE_CHECKING:
    from raylight.distributed_worker.parallel_group_manager import XFuserParallelContext
    from raylight.distributed_worker.pipefusion_schema import PipeFusionConfig, StagePlan

try:
    from comfy_kitchen.tensor.base import QuantizedTensor as _CKQuantizedTensor
except Exception:
    _CKQuantizedTensor = ()


_RETAINED_FAILED_HOST_REGISTRATIONS = []


class LowVramPatch:
    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches
        self.convert_func = convert_func
        self.set_func = set_func

    def __call__(self, weight):
        return comfy_dist.lora.calculate_weight(
            self.patches[self.key],
            weight,
            self.key,
            intermediate_dtype=weight.dtype,
        )


def wipe_lowvram_weight(m):
    if hasattr(m, "prev_comfy_cast_weights"):
        m.comfy_cast_weights = m.prev_comfy_cast_weights
        del m.prev_comfy_cast_weights

    if hasattr(m, "weight_function"):
        m.weight_function = []

    if hasattr(m, "bias_function"):
        m.bias_function = []


def _safe_free_storage(tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        return

    try:
        if tensor.device.type == "meta":
            return
        if _is_quantized_tensor_like(tensor):
            return
        if tensor.storage_offset() != 0:
            return
    except Exception:
        return

    try:
        _free_storage(tensor)
    except (RuntimeError, AssertionError) as e:
        msg = str(e)
        if "invalid python storage" in msg:
            return
        if "out of bounds for storage" in msg:
            return
        if "Freeing a tensor's storage is unsafe" in msg:
            return
        raise


def free_model_vram(model_patcher) -> None:
    """Eagerly free GPU storage from a non-FSDP ModelPatcher.

    Unlike model.detach(), this does NOT copy weights to CPU RAM.
    Call before dropping the model reference so VRAM is released
    deterministically without relying on __del__ / gc.collect().

    WARNING: ModelPatcher.clone() shares the underlying model object.
    Calling this on a clone will corrupt all other clones sharing the
    same model. Only use when the model is not shared (e.g. FSDP path
    where there is no cached_base_model).
    """
    model = getattr(model_patcher, "model", None)
    if model is None:
        return
    # Break reference cycle (model.current_patcher -> model_patcher -> model)
    # so __del__ -> detach(unpatch_all=False) won't re-copy to CPU.
    if hasattr(model, "current_patcher"):
        model.current_patcher = None

    for m in model.modules():
        for p in m.parameters(recurse=False):
            try:
                tensor = p.data
                if not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.device.type == "meta":
                    continue
                if _is_quantized_tensor_like(tensor):
                    continue
                _safe_free_storage(tensor)
            except Exception:
                continue
        for b in m.buffers(recurse=False):
            try:
                tensor = b.data
                if not isinstance(tensor, torch.Tensor):
                    continue
                if tensor.device.type == "meta":
                    continue
                _safe_free_storage(tensor)
            except Exception:
                continue


def _is_quantized_tensor_like(tensor: torch.Tensor) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if _CKQuantizedTensor and isinstance(tensor, _CKQuantizedTensor):
        return True
    return hasattr(tensor, "_qdata") and hasattr(tensor, "_layout_cls") and hasattr(tensor, "_params")


def _state_dict_has_quant_payload(state_dict) -> bool:
    if not isinstance(state_dict, dict):
        return False

    for key, value in state_dict.items():
        if key.endswith(".comfy_quant") or key.endswith(".scale_weight") or key.endswith(".weight_scale"):
            return True
        if key == "scaled_fp8" or key.endswith(".scaled_fp8"):
            return True
        if _is_quantized_tensor_like(value):
            return True

    return False


def _get_parent_module_and_name(model: torch.nn.Module, param_name: str) -> tuple[torch.nn.Module, str]:
    if "." not in param_name:
        return model, param_name
    parent_name, leaf_name = param_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf_name


def _unwrap_base_model(model):
    return getattr(model, "model", model)


def select_fsdp_mixed_precision_policy(model):
    """Use BF16 for FSDP unshard/forward only when the loader requested it."""
    base_model = _unwrap_base_model(model)
    dtype = getattr(model, "fsdp_param_dtype", None)
    dtype = dtype if dtype is not None else getattr(base_model, "manual_cast_dtype", None)
    if dtype is None:
        get_dtype = getattr(base_model, "get_dtype", None)
        if callable(get_dtype):
            dtype = get_dtype()

    if dtype is not torch.bfloat16:
        return None
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        output_dtype=None,
        cast_forward_inputs=True,
    )


def select_fsdp_cpu_offload_policy(model, *, platform: str | None = None):
    """Return the FSDP2 runtime offload policy requested by the node."""
    if not getattr(model, "is_cpu_offload", False):
        return None

    platform = sys.platform if platform is None else platform
    configured = os.environ.get("RAYLIGHT_FSDP_CPU_OFFLOAD_PIN_MEMORY")
    if configured is None:
        pin_memory = platform != "win32"
    else:
        normalized = configured.strip().lower()
        if normalized not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
            raise ValueError("RAYLIGHT_FSDP_CPU_OFFLOAD_PIN_MEMORY must be a boolean value")
        pin_memory = normalized in {"1", "true", "yes", "on"}

    return CPUOffloadPolicy(pin_memory=pin_memory)


def close_fsdp_cpu_offload_host_memory(model) -> bool:
    diffusion_model = getattr(model, "diffusion_model", model)
    registration = getattr(diffusion_model, "_raylight_fsdp_host_registration", None)
    if registration is None:
        return True
    if not registration.close():
        if not any(item is registration for item in _RETAINED_FAILED_HOST_REGISTRATIONS):
            _RETAINED_FAILED_HOST_REGISTRATIONS.append(registration)
        print(
            "[FSDP_HOST_REGISTER] unregister failed; retaining backing storage for retry",
            flush=True,
        )
        return False
    _RETAINED_FAILED_HOST_REGISTRATIONS[:] = [
        item for item in _RETAINED_FAILED_HOST_REGISTRATIONS if item is not registration
    ]
    delattr(diffusion_model, "_raylight_fsdp_host_registration")
    return True


def maybe_register_fsdp_cpu_offload_host_memory(
    model,
    *,
    is_cpu_offload: bool,
    platform: str | None = None,
):
    """Optionally page-lock loaded FSDP shards in place on native Windows."""
    if not is_cpu_offload:
        return None
    platform = sys.platform if platform is None else platform
    if platform != "win32":
        return None

    configured = os.environ.get("RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER", "0")
    normalized = configured.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError("RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER must be a boolean value")
    if normalized not in {"1", "true", "yes", "on"}:
        return None

    pin_configured = os.environ.get("RAYLIGHT_FSDP_CPU_OFFLOAD_PIN_MEMORY", "0")
    if pin_configured.strip().lower() in {"1", "true", "yes", "on"}:
        return None

    try:
        capacity_mib = int(
            os.environ.get("RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER_MIB", "12288")
        )
    except ValueError as exc:
        raise ValueError(
            "RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER_MIB must be an integer"
        ) from exc
    if capacity_mib <= 0:
        raise ValueError("RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER_MIB must be positive")

    if not close_fsdp_cpu_offload_host_memory(model):
        return getattr(model, "_raylight_fsdp_host_registration")
    tensors = list(
        collect_fsdp_shard_cpu_tensors(
            model,
            fsdp_module_type=FSDPModule,
        )
    )
    registration = register_cpu_storages(
        tensors,
        capacity_bytes=capacity_mib * 1024 * 1024,
    )
    model._raylight_fsdp_host_registration = registration
    print(
        "[FSDP_HOST_REGISTER] "
        f"registered={registration.registered_bytes / (1024 * 1024):.2f}MiB "
        f"storages={registration.registered_storages} "
        f"skipped_capacity={registration.skipped_capacity_bytes / (1024 * 1024):.2f}MiB "
        f"failures={registration.failed_storages}",
        flush=True,
    )
    return registration


def _can_release_fsdp_state_dict_during_load(excluded_modules) -> bool:
    return not excluded_modules


def _align_dense_meta_dtypes_from_state_dict(
    model: torch.nn.Module,
    full_sd: dict,
) -> int:
    """Preserve dense checkpoint storage dtype before FSDP builds its shards."""
    changed = 0
    supported = {torch.float16, torch.bfloat16, torch.float32, torch.float64}

    for name, param in list(model.named_parameters()):
        if not getattr(param, "is_meta", False):
            continue
        full_tensor = full_sd.get(name)
        if not isinstance(full_tensor, torch.Tensor):
            continue
        if _is_quantized_tensor_like(full_tensor):
            continue

        if name.endswith("weight"):
            prefix = name[: -len("weight")]
            if any(
                key in full_sd
                for key in (
                    f"{prefix}comfy_quant",
                    f"{prefix}weight_scale",
                    f"{prefix}scale_weight",
                )
            ):
                continue

        target_dtype = full_tensor.dtype
        if target_dtype not in supported or target_dtype == param.dtype:
            continue

        parent_module, leaf_name = _get_parent_module_and_name(model, name)
        parent_module.register_parameter(
            leaf_name,
            torch.nn.Parameter(
                torch.empty(tuple(param.shape), device="meta", dtype=target_dtype),
                requires_grad=param.requires_grad,
            ),
        )
        changed += 1
    return changed


def _promote_nonfloating_params_to_meta(model: torch.nn.Module, target_dtype: torch.dtype) -> int:
    replaced = 0
    for name, param in list(model.named_parameters()):
        if torch.is_floating_point(param) or torch.is_complex(param):
            continue
        parent_module, leaf_name = _get_parent_module_and_name(model, name)
        meta_param = torch.nn.Parameter(
            torch.empty(tuple(param.shape), device="meta", dtype=target_dtype),
            requires_grad=param.requires_grad,
        )
        parent_module.register_parameter(leaf_name, meta_param)
        replaced += 1
    return replaced


def _pre_init_fsdp(diffusion_model: torch.nn.Module) -> None:
    """Pre-initialize FSDP param groups so the root is always initialized first.

    Without this, a ControlNet calling ``base_model.time_text_embed()`` directly
    would trigger ``_lazy_init()`` on a *nested* FSDP unit instead of the root,
    leaving some param groups with stale meta-device references.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_state import (
        _get_module_fsdp_state,
        FSDPState,
    )

    root_state = _get_module_fsdp_state(diffusion_model)
    if root_state is None or not isinstance(root_state, FSDPState):
        return
    if root_state._is_root is not None:
        return  # already initialized

    root_state._lazy_init()


def _collect_controlnet_shared_modules(diffusion_model: torch.nn.Module) -> set[torch.nn.Module]:
    """Collect base model sub-modules that ControlNet calls directly.

    These modules must NOT be FSDP-wrapped because ControlNet invokes them
    outside the base model's normal forward path.  If they were FSDP-wrapped,
    each call would trigger an all_gather that only some ranks participate in,
    causing an NCCL collective timeout.
    """
    # Modules called by ControlNet.forward() via base_model.* (getattr skips
    # names that don't exist on a given architecture, so the union is safe).
    #   QwenImage FunControlNet: process_img, pe_embedder, img_in, txt_norm,
    #                             txt_in, time_text_embed
    #   Flux ControlNet:          img_in, time_in, vector_in, guidance_in,
    #                             txt_in, pe_embedder
    _CONTROLNET_SHARED_NAMES = (
        "process_img",
        "pe_embedder",
        "img_in",
        "txt_norm",
        "txt_in",
        "time_text_embed",
        "time_in",
        "vector_in",
        "guidance_in",
    )
    excluded: set[torch.nn.Module] = set()
    for name in _CONTROLNET_SHARED_NAMES:
        mod = getattr(diffusion_model, name, None)
        if mod is not None and isinstance(mod, torch.nn.Module) and any(True for _ in mod.parameters()):
            excluded.add(mod)
    return excluded


def patch_fsdp(self):
    print(f"[Rank {self.rank}] Applying FSDP to {type(self.model.diffusion_model).__name__}")

    if isinstance(self.model.diffusion_model, FSDPModule):
        if self.fsdp_state_dict is not None:
            raise RuntimeError("FSDP initialization previously failed; reload the Raylight model before sampling again")
        maybe_register_fsdp_cpu_offload_host_memory(
            self.model.diffusion_model,
            is_cpu_offload=self.is_cpu_offload,
        )
        print("FSDP already registered, skip wrapping...")
        return self.model

    if self.fsdp_state_dict is None:
        raise ValueError("FSDP state_dict is None. Call set_fsdp_state_dict before patch_fsdp.")

    diffusion_model = self.model.diffusion_model
    dense_dtype_aligned = _align_dense_meta_dtypes_from_state_dict(
        diffusion_model, self.fsdp_state_dict
    )
    if dense_dtype_aligned:
        print(
            f"[Rank {self.rank}] Preserved checkpoint dtype for {dense_dtype_aligned} dense FSDP parameters"
        )

    fsdp_kwargs = {"reshard_after_forward": True}
    mp_policy = select_fsdp_mixed_precision_policy(self)
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
        print("[FSDP] BF16 mixed-precision policy enabled for parameter all-gather and forward")
    offload_policy = select_fsdp_cpu_offload_policy(self)
    if offload_policy is not None:
        fsdp_kwargs["offload_policy"] = offload_policy
        print(
            f"[FSDP] CPU offload enabled for parameter shards (pin_memory={offload_policy.pin_memory})"
        )

    has_qt_runtime = freeze_and_detect_qt(diffusion_model)
    has_quant_sd = _state_dict_has_quant_payload(self.fsdp_state_dict)
    use_quant_loader = has_qt_runtime or has_quant_sd

    if use_quant_loader and mp_policy is not None:
        aligned = align_fp8_logical_dtype(diffusion_model, torch.bfloat16)
        if aligned > 0:
            print(
                f"[Rank {self.rank}] Aligned {aligned} FP8 logical dtypes to "
                "BF16 for FSDP bias sharding"
            )

    if use_quant_loader:
        base_model = _unwrap_base_model(self.model)
        placeholder_dtype = getattr(base_model, "manual_cast_dtype", None) or torch.bfloat16
        replaced = _promote_nonfloating_params_to_meta(diffusion_model, placeholder_dtype)
        if replaced > 0:
            print(f"[Rank {self.rank}] Promoted {replaced} non-floating quant params to meta placeholders before FSDP wrapping")

    excluded_modules = _collect_controlnet_shared_modules(diffusion_model)
    release_state_dict = _can_release_fsdp_state_dict_during_load(excluded_modules)
    if excluded_modules:
        print(f"[Rank {self.rank}] Excluding {len(excluded_modules)} ControlNet-shared modules from FSDP: "
              f"{[n for n, m in diffusion_model.named_modules() if m in excluded_modules]}")

    num_wrappers = fully_shard_bottom_up(
        diffusion_model,
        fsdp_kwargs=fsdp_kwargs,
        native_ignore_scale=not use_quant_loader,
        ignored_modules=excluded_modules,
    )

    target_device = (
        self.load_device if isinstance(self.load_device, torch.device) else torch.device("cuda", torch.cuda.current_device())
    )

    if use_quant_loader:
        load_from_full_model_state_dict(
            model=self.model,
            full_sd=self.fsdp_state_dict,
            device=target_device,
            strict=False,
            cpu_offload=self.is_cpu_offload,
            release_sd=release_state_dict,
        )
    else:
        options = StateDictOptions(
            full_state_dict=True,
            strict=False,
            cpu_offload=self.is_cpu_offload,
            broadcast_from_rank0=True,
        )
        set_model_state_dict(self.model, self.fsdp_state_dict, options=options)

    # Materialize excluded params AFTER state dict loading so that
    # set_model_state_dict only sees meta-device params (single device).
    if excluded_modules:
        count = materialize_excluded_params(
            model=self.model,
            excluded_modules=excluded_modules,
            full_sd=self.fsdp_state_dict,
            device=target_device,
            cpu_offload=self.is_cpu_offload,
        )
        if count > 0:
            print(f"[Rank {self.rank}] Materialized {count} excluded ControlNet-shared params on {target_device}")

    diagnostics = format_fsdp_diagnostics(summarize_fsdp_parameters(diffusion_model))
    cuda_memory = ""
    if target_device.type == "cuda":
        allocated_mib = torch.cuda.memory_allocated(target_device) / (1024 * 1024)
        reserved_mib = torch.cuda.memory_reserved(target_device) / (1024 * 1024)
        cuda_memory = f" cuda_allocated={allocated_mib:.2f}MiB cuda_reserved={reserved_mib:.2f}MiB"
    print(
        f"[Rank {self.rank}][FSDP_DIAG] wrappers={num_wrappers} "
        f"non_root_wrappers={num_wrappers - 1} {diagnostics}{cuda_memory}"
    )

    _pre_init_fsdp(diffusion_model)
    if self.is_cpu_offload:
        try:
            prefetch_mib = int(os.environ.get("RAYLIGHT_FSDP_FORWARD_PREFETCH_MIB", "0"))
        except ValueError as exc:
            raise ValueError("RAYLIGHT_FSDP_FORWARD_PREFETCH_MIB must be an integer") from exc
        if prefetch_mib < 0:
            raise ValueError("RAYLIGHT_FSDP_FORWARD_PREFETCH_MIB must be non-negative")
        if prefetch_mib:
            prefetch = configure_minimax_h3_forward_prefetch(
                diffusion_model,
                max_prefetch_bytes=prefetch_mib * 1024 * 1024,
            )
            print(
                "[FSDP_FORWARD_PREFETCH] "
                f"budget={prefetch_mib}MiB configured={prefetch['configured']} "
                f"skipped={prefetch['skipped']} "
                f"max_target={prefetch['max_configured_bytes'] / 1024 / 1024:.2f}MiB"
            )
    if os.environ.get("RAYLIGHT_RANK_DIAG", "0") == "1":
        adaln = getattr(diffusion_model, "adaln_single", None)
        linear = getattr(adaln, "linear", None)
        if isinstance(linear, FSDPModule):
            state = linear._get_fsdp_state()
            group = state._fsdp_param_group
            rows = []
            if group is not None:
                world_size = int(group.mesh_info.mesh.size())
                for fsdp_param in group.fsdp_params:
                    inner = fsdp_param._sharded_local_tensor
                    inputs = fsdp_param.all_gather_inputs
                    rows.append(
                        {
                            "param": fsdp_param._module_info.param_name,
                            "inner_type": type(inner).__name__,
                            "inner_dtype": str(inner.dtype),
                            "has_pre_all_gather": hasattr(inner, "fsdp_pre_all_gather"),
                            "orig_dtype": str(fsdp_param.orig_dtype),
                            "param_dtype": str(fsdp_param.param_dtype),
                            "gather": summarize_all_gather_inputs(inputs, world_size),
                        }
                    )
            print(
                f"[Rank {self.rank}][FSDP_GATHER_PLAN] module=adaln_single.linear rows={rows}",
                flush=True,
            )

    self.fsdp_state_dict = None
    maybe_register_fsdp_cpu_offload_host_memory(
        diffusion_model,
        is_cpu_offload=self.is_cpu_offload,
    )

    print("FSDP registered successfully.")
    return self.model


class FSDPModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        weight_inplace_update=False,
        rank: int = 0,
        fsdp_state_dict: dict | None = None,
        device_mesh=None,
        is_cpu_offload: bool = False,
    ):
        super().__init__(
            model=model,
            load_device=load_device,
            offload_device=offload_device,
            size=size,
            weight_inplace_update=weight_inplace_update,
        )
        self.rank = rank
        self.fsdp_state_dict = fsdp_state_dict
        self.device_mesh = device_mesh
        self.is_cpu_offload = is_cpu_offload
        self._has_quantized_dtensor_shards: bool | None = None
        self.fsdp_param_dtype: torch.dtype | None = None
        self._fsdp_full_load_ready = False
        self.patch_fsdp = patch_fsdp.__get__(self, FSDPModelPatcher)

    def is_dynamic(self):
        return True

    def model_size(self):
        if self.size > 0:
            return self.size
        total = comfy.model_management.module_size(self.model)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            self.size = total // world_size
        else:
            self.size = total
        return self.size

    def config_fsdp(self, rank, device_mesh):
        self.rank = rank
        self.device_mesh = device_mesh
        self.model.diffusion_model.to("meta")

    def set_fsdp_state_dict(self, sd):
        self.fsdp_state_dict = sd

    def free_fsdp_vram(self):
        """Eagerly free DTensor shard storage from GPU.

        Call before dropping the FSDPModelPatcher reference so VRAM is
        released deterministically without relying on __del__ / gc.collect().
        """
        model = getattr(self, "model", None)
        self._fsdp_full_load_ready = False
        if model is None:
            return
        if not close_fsdp_cpu_offload_host_memory(model):
            return
        # Break reference cycle (model.current_patcher -> self -> model)
        if hasattr(model, "current_patcher"):
            model.current_patcher = None

        has_qt_hint = self._has_quantized_dtensor_shards
        for m in model.modules():
            for p in m.parameters(recurse=False):
                try:
                    tensor = p.data if isinstance(p.data, torch.Tensor) else None
                    if tensor is None:
                        continue
                    if isinstance(tensor, DTensor):
                        if has_qt_hint is True:
                            continue
                        try:
                            local = getattr(tensor, "_local_tensor", None)
                            if local is None:
                                local = tensor.to_local()
                        except Exception:
                            continue
                        if has_qt_hint is None:
                            has_qt_hint = _is_quantized_tensor_like(local)
                            self._has_quantized_dtensor_shards = has_qt_hint
                        if has_qt_hint is True:
                            continue
                        _safe_free_storage(local.data)
                        continue
                    if _is_quantized_tensor_like(tensor):
                        continue
                    _safe_free_storage(tensor.data)
                except Exception:
                    continue

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False, convert_dtensor=False):
        weight, set_func, convert_func = get_key_weight(self.model, key)
        if key not in self.patches:
            return weight

        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup and not return_weight:
            self.backup[key] = collections.namedtuple("Dimension", ["weight", "inplace_update"])(
                weight.to(device=self.offload_device, copy=inplace_update), inplace_update
            )

        temp_dtype = comfy.model_management.lora_compute_dtype(device_to)
        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, temp_dtype, copy=True)
        else:
            temp_weight = weight.to(temp_dtype, copy=True)
        if convert_func is not None:
            temp_weight = convert_func(temp_weight, inplace=True)

        out_weight = comfy_dist.lora.calculate_weight(self.patches[key], temp_weight, key, device_mesh=self.device_mesh)
        if set_func is None:
            out_weight = comfy_dist.float.stochastic_rounding(
                out_weight, weight.dtype, seed=string_to_seed(key), device_mesh=self.device_mesh
            )

            if return_weight:
                return out_weight
            if inplace_update:
                comfy.utils.copy_to_param(self.model, key, out_weight)
            else:
                comfy.utils.set_attr_param(self.model, key, out_weight)

        else:
            return set_func(
                out_weight,
                inplace_update=inplace_update,
                seed=string_to_seed(key),
                return_weight=return_weight,
            )

    def clone(self, *args, **kwargs):
        # Call parent clone normally (keeps init signature correct)
        n = super(FSDPModelPatcher, self).clone(*args, **kwargs)

        n.__class__ = FSDPModelPatcher
        n.rank = self.rank
        n.fsdp_state_dict = self.fsdp_state_dict
        n.device_mesh = self.device_mesh
        n.is_cpu_offload = self.is_cpu_offload
        n._has_quantized_dtensor_shards = self._has_quantized_dtensor_shards
        n.fsdp_param_dtype = self.fsdp_param_dtype
        n._fsdp_full_load_ready = getattr(self, "_fsdp_full_load_ready", False)

        return n

    def _load_list(self, for_dynamic=False, default_device=None):
        loading = []
        for n, m in self.model.named_modules():
            params = []
            skip = False
            for name, param in m.named_parameters(recurse=False):
                params.append(name)
            for name, param in m.named_parameters(recurse=True):
                if name not in params:
                    skip = True  # skip random weights in non leaf modules
                    break
            if not skip and (hasattr(m, "comfy_cast_weights") or len(params) > 0):
                module_mem = comfy.model_management.module_size(m)
                loading.append((module_mem, module_mem, n, m, params))
        return loading

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        with self.use_ejected():
            if not isinstance(self.model.diffusion_model, FSDPModule):
                self.patch_fsdp()
            self.unpatch_hooks()
            mem_counter = 0
            patch_counter = 0
            lowvram_counter = 0
            loading = self._load_list()

            loading.sort(reverse=True)
            for x in loading:
                _, module_mem, n, m, params = x

                weight_key = "{}.weight".format(n)
                bias_key = "{}.bias".format(n)

                if not full_load and hasattr(m, "comfy_cast_weights"):
                    if mem_counter + module_mem >= lowvram_model_memory:
                        lowvram_counter += 1
                        if hasattr(m, "prev_comfy_cast_weights"):  # Already lowvramed
                            continue

                cast_weight = self.force_cast_weights
                m.comfy_force_cast_weights = self.force_cast_weights

                if hasattr(m, "comfy_cast_weights"):
                    m.weight_function = []
                    m.bias_function = []

                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        _, set_func, convert_func = get_key_weight(self.model, weight_key)
                        m.weight_function = [LowVramPatch(weight_key, self.patches, convert_func, set_func)]
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        _, set_func, convert_func = get_key_weight(self.model, bias_key)
                        m.bias_function = [LowVramPatch(bias_key, self.patches, convert_func, set_func)]
                        patch_counter += 1

                cast_weight = True

                if cast_weight and hasattr(m, "comfy_cast_weights"):
                    m.prev_comfy_cast_weights = m.comfy_cast_weights
                    m.comfy_cast_weights = True

                if weight_key in self.weight_wrapper_patches:
                    m.weight_function.extend(self.weight_wrapper_patches[weight_key])

                if bias_key in self.weight_wrapper_patches:
                    m.bias_function.extend(self.weight_wrapper_patches[bias_key])

                mem_counter += move_weight_functions(m, device_to)

            if lowvram_counter > 0:
                logging.info(
                    "loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter)
                )
                self.model.model_lowvram = True
                self._fsdp_full_load_ready = False
            else:
                logging.info(
                    "loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load)
                )
                self.model.model_lowvram = False
                self._fsdp_full_load_ready = bool(full_load)
                if full_load:
                    # FSDP owns shard placement and materializes its meta placeholders
                    # during unshard. Moving the whole module like a regular ComfyUI
                    # model is both unnecessary and invalid for those placeholders.
                    logging.info("FSDP full-load budget accepted without whole-model device move")
                    if not self.is_cpu_offload:
                        mem_counter = self.model_size()

            self.model.lowvram_patch_counter += patch_counter
            self.model.device = device_to
            self.model.model_loaded_weight_memory = mem_counter
            self.model.model_offload_buffer_memory = 0
            self.model.current_weight_patches_uuid = self.patches_uuid
            self.model.current_patcher = self

            for callback in self.get_all_callbacks(CallbacksMP.ON_LOAD):
                callback(self, device_to, lowvram_model_memory, force_patch_weights, full_load)

            self.apply_hooks(self.forced_hooks, force_apply=True)

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        if (
            self._fsdp_full_load_ready
            and not force_patch_weights
            and self.model.current_weight_patches_uuid == self.patches_uuid
        ):
            with self.use_ejected(skip_and_inject_on_exit_only=True):
                # CPU-offloaded shards are ready without being GPU-resident.
                # Restore object patches and injections without reloading weights.
                self.patch_model(load_weights=False)
                self.model.current_patcher = self
                self.apply_hooks(self.forced_hooks, force_apply=True)
            return 0
        return super().partially_load(device_to, extra_memory, force_patch_weights)

    def cleanup(self):
        self.clean_hooks()
        if hasattr(self.model, "current_patcher"):
            self.model.current_patcher = None
        for callback in self.get_all_callbacks(CallbacksMP.ON_CLEANUP):
            callback(self)

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()

            if self.model.model_lowvram:
                for m in self.model.modules():
                    move_weight_functions(m, device_to)
                    wipe_lowvram_weight(m)

                self.model.model_lowvram = False
                self.model.lowvram_patch_counter = 0

            keys = list(self.backup.keys())

            for k in keys:
                bk = self.backup[k]
                if bk.inplace_update:
                    comfy.utils.copy_to_param(self.model, k, bk.weight)
                else:
                    comfy.utils.set_attr_param(self.model, k, bk.weight)

            self.model.current_weight_patches_uuid = None
            self._fsdp_full_load_ready = False
            self.backup.clear()

            if device_to is not None:
                if next(self.model.parameters()).device == torch.device("meta"):
                    pass
                elif isinstance(self.model.diffusion_model, FSDPModule):
                    # FSDP-wrapped model: self.model.to(device_to) would call
                    # DTensor.to(device) on each parameter, which materializes
                    # full tensors per worker (breaking sharding).
                    # Instead, break the reference cycle
                    # (model.current_patcher -> FSDPModelPatcher -> self.model)
                    # so __del__ runs deterministically on del and frees
                    # DTensor shard storage via _safe_free_storage.
                    close_fsdp_cpu_offload_host_memory(self.model)
                    if hasattr(self.model, "current_patcher"):
                        self.model.current_patcher = None
                else:
                    self.model.to(device_to)
                    self.model.device = device_to
            self.model.model_loaded_weight_memory = 0
            self.model.model_offload_buffer_memory = 0

            for m in self.model.modules():
                if hasattr(m, "comfy_patched_weights"):
                    del m.comfy_patched_weights

        keys = list(self.object_patches_backup.keys())
        for k in keys:
            comfy.utils.set_attr(self.model, k, self.object_patches_backup[k])

        self.object_patches_backup.clear()

    def __del__(self):
        try:
            self.detach(unpatch_all=False)
        except Exception:
            pass

        model = getattr(self, "model", None)
        if model is not None:
            close_fsdp_cpu_offload_host_memory(model)
            # Break reference cycle so the inner model can be collected
            if hasattr(model, "current_patcher"):
                model.current_patcher = None

        self.model = None
        try:
            comfy.model_management.soft_empty_cache()
            gc.collect()
        except Exception:
            pass


class PipefusionModelPatcher(comfy.model_patcher.ModelPatcher):
    def __init__(
        self,
        model,
        load_device,
        offload_device,
        size=0,
        weight_inplace_update=False,
        pipefusion_config: "PipeFusionConfig | None" = None,
        stage_plan: "StagePlan | None" = None,
        parallel_context: "XFuserParallelContext | None" = None,
    ):
        super().__init__(
            model=model,
            load_device=load_device,
            offload_device=offload_device,
            size=size,
            weight_inplace_update=weight_inplace_update,
        )
        self.pipefusion_config = pipefusion_config
        self.pipefusion_stage = stage_plan
        self.parallel_context = parallel_context

    def clone(self, *args, **kwargs):
        n = super(PipefusionModelPatcher, self).clone(*args, **kwargs)

        n.__class__ = PipefusionModelPatcher
        n.pipefusion_config = self.pipefusion_config
        n.pipefusion_stage = self.pipefusion_stage
        n.parallel_context = self.parallel_context

        return n
