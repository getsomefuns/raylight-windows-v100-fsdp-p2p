from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import json
from dataclasses import replace
from types import MethodType
from typing import Any, cast

import torch
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor

try:
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
except Exception:  # pragma: no cover
    from comfy_kitchen.tensor import QuantizedTensor, get_layout_class  # type: ignore

    QUANT_ALGOS = {}


@contextmanager
def quantized_state_dict_zero_copy():
    """Avoid redundant QuantizedTensor clones during assign=True FSDP loading."""
    try:
        import comfy_kitchen.tensor.base as kitchen_base
    except Exception:
        yield
        return

    op = torch.ops.aten._to_copy.default
    dispatch_table = getattr(kitchen_base, "_DISPATCH_TABLE", None)
    original = dispatch_table.get(op) if dispatch_table is not None else None
    if original is None:
        yield
        return

    def reuse_identical(qt, args, kwargs):
        target_device, target_dtype = kitchen_base._parse_to_args(args, kwargs)
        if isinstance(target_device, str):
            target_device = torch.device(target_device)
        same_device = target_device is None or target_device == qt._qdata.device
        same_dtype = target_dtype is None or target_dtype == qt._params.orig_dtype
        if same_device and same_dtype:
            return qt
        return original(qt, args, kwargs)

    dispatch_table[op] = reuse_identical
    try:
        yield
    finally:
        if dispatch_table.get(op) is reuse_identical:
            dispatch_table[op] = original


def _state_value_is_quantized(value: Any) -> bool:
    local = value
    if isinstance(local, DTensor):
        local = local._local_tensor
    return isinstance(local, QuantizedTensor)


@contextmanager
def plain_bf16_state_dict_assign(model: torch.nn.Module, state_dict: dict[str, Any]):
    """Assign plain BF16 shards without Comfy recasting them to FP32.

    MixedPrecisionOps.Linear uses a custom state-dict loader that casts every
    descriptor-free weight to its compute dtype. FSDP has already created the
    correctly typed shard at this point, so that second cast only expands the
    shard and can OOM. Patch only the individual plain-BF16 module instances;
    quantized modules keep their custom loader.
    """
    patched = []
    seen = set()

    def direct_assign(
        module,
        incoming,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        return torch.nn.Module._load_from_state_dict(
            module,
            incoming,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    for key, value in state_dict.items():
        if not isinstance(key, str) or not key.endswith(".weight"):
            continue
        if not isinstance(value, torch.Tensor) or value.dtype is not torch.bfloat16:
            continue
        if _state_value_is_quantized(value):
            continue
        key_prefix = key[: -len("weight")]
        if f"{key_prefix}comfy_quant" in state_dict:
            continue
        parent, _ = _get_parent_module_and_name(model, key)
        if parent in seen:
            continue
        class_loader = getattr(type(parent), "_load_from_state_dict", None)
        if class_loader is torch.nn.Module._load_from_state_dict:
            continue
        seen.add(parent)
        had_instance_loader = "_load_from_state_dict" in parent.__dict__
        instance_loader = parent.__dict__.get("_load_from_state_dict")
        parent._load_from_state_dict = MethodType(direct_assign, parent)
        patched.append((parent, had_instance_loader, instance_loader))

    try:
        yield len(patched)
    finally:
        for parent, had_instance_loader, instance_loader in reversed(patched):
            if had_instance_loader:
                parent._load_from_state_dict = instance_loader
            else:
                del parent.__dict__["_load_from_state_dict"]



def enable_low_peak_fsdp_unshard(model: torch.nn.Module) -> int:
    """Avoid implicit forward all-gather double buffering during inference."""
    if not isinstance(model, FSDPModule):
        return 0

    wrappers = sum(1 for module in model.modules() if isinstance(module, FSDPModule))
    # Use the default stream and free each all-gather result immediately.
    model._set_unshard_async_op(True)
    return wrappers



"""
 This stuff is for systematically fully_shard from bottom up,
 with "*-1 parents are sharded, then continue up to root*", the def collect_bottom_up_shard_order is the function
 the tree looks like this:

model                          [FSDP]
├── block0                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   │   ├── q.weight           [SHARDED]
│   │   ├── q.scale            [IGNORED]
│   │   ├── k.weight           [SHARDED]
│   │   ├── k.scale            [IGNORED]
│   │   ├── v.weight           [SHARDED]
│   │   └── v.scale            [IGNORED]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   │   ├── weight             [SHARDED]
│   │   └── scale              [IGNORED]
│   └── conv                   [FSDP, ignored_params={} ]
│       ├── weight             [SHARDED]
│       └── bias               [SHARDED]
│
├── block1                     [FSDP]
│   ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
│   ├── ffn                    [FSDP, ignored_params={scale}]
│   └── conv                   [FSDP]
│
└── block2                     [FSDP]
    ├── qkv                    [FSDP, ignored_params={q.scale,k.scale,v.scale}]
    ├── ffn                    [FSDP, ignored_params={scale}]
    └── conv                   [FSDP]
"""


def freeze_and_detect_qt(model: torch.nn.Module) -> bool:
    has_qt = False
    for param in model.parameters():
        param.requires_grad = False
        local = getattr(param, "_local_tensor", None)
        if isinstance(param, QuantizedTensor) or isinstance(local, QuantizedTensor):
            has_qt = True
    return has_qt


def _mod_name(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _module_has_subtree_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=True))


def _module_has_direct_params(module: torch.nn.Module) -> bool:
    return any(True for _ in module.named_parameters(recurse=False))


def _children_with_params(name: str, module: torch.nn.Module, named_map: dict[str, torch.nn.Module]) -> list[str]:
    out: list[str] = []
    for child_name, _child in module.named_children():
        full = _mod_name(name, child_name)
        if _module_has_subtree_params(named_map[full]):
            out.append(full)
    return out


def _is_descendant(path: str, ancestor: str) -> bool:
    if ancestor == "":
        return path != ""
    return path == ancestor or path.startswith(ancestor + ".")


def _collect_leaf_parent_targets(model: torch.nn.Module) -> set[str]:
    named = dict(model.named_modules())
    return {
        name
        for name, module in named.items()
        if name and _module_has_direct_params(module)
    }


def _add_ancestors_to_root(targets: set[str]) -> set[str]:
    out = set(targets)
    for target in list(targets):
        cur = target
        while "." in cur:
            cur = cur.rsplit(".", 1)[0]
            out.add(cur)
    out.add("")
    return out


def _depth(name: str) -> int:
    return 0 if name == "" else name.count(".") + 1


def _supports_fully_shard(module: torch.nn.Module) -> bool:
    return type(module).forward is not torch.nn.Module.forward


def collect_bottom_up_shard_order(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    named = dict(model.named_modules())
    leaf_parents = _collect_leaf_parent_targets(model)
    all_targets = _add_ancestors_to_root(leaf_parents)
    ordered_names = sorted(all_targets, key=_depth, reverse=True)

    out: list[tuple[str, torch.nn.Module]] = []
    for name in ordered_names:
        module = model if name == "" else named[name]
        if name and type(module) is torch.nn.Sequential:
            continue
        if _supports_fully_shard(module):
            out.append((name, module))
    return out


def validate_inference_shard_order(shard_order: list[tuple[str, torch.nn.Module]]) -> int:
    """Reject a root-only FSDP topology that would retain full params after forward.

    Returns the number of non-root parameter-bearing wrappers so callers can
    include the value in compact diagnostics.
    """
    root_positions = [index for index, (name, _module) in enumerate(shard_order) if name == ""]
    if root_positions != [len(shard_order) - 1]:
        raise ValueError("FSDP inference topology must contain exactly one root wrapper, placed last")

    non_root_count = sum(
        1
        for name, module in shard_order
        if name and _module_has_subtree_params(module)
    )
    if non_root_count == 0:
        raise ValueError(
            "FSDP inference root-only topology is unsafe because root FSDP may retain full parameters after forward"
        )
    return non_root_count


def _dtype_name(dtype: Any) -> str:
    value = str(dtype)
    return value.removeprefix("torch.")


def _layout_name(tensor: Any) -> str | None:
    layout = getattr(tensor, "_layout_cls", None)
    if layout is None:
        return None
    if isinstance(layout, str):
        return layout
    return getattr(layout, "__name__", str(layout))


def _storage_tensor(tensor: Any) -> Any:
    qdata = getattr(tensor, "_qdata", None)
    return qdata if qdata is not None else tensor


def _tensor_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())

def summarize_all_gather_inputs(
    inputs: list[torch.Tensor],
    world_size: int,
) -> list[dict[str, Any]]:
    """Describe the real storage buffers FSDP will allocate for all-gather."""
    return [
        {
            "shape": list(tensor.shape),
            "dtype": _dtype_name(tensor.dtype),
            "input_bytes": _tensor_bytes(tensor),
            "output_bytes": _tensor_bytes(tensor) * int(world_size),
        }
        for tensor in inputs
    ]



def summarize_fsdp_parameters(model: torch.nn.Module) -> dict[str, Any]:
    """Summarize FSDP parameter placement without copying tensor payloads."""
    parameter_count = 0
    dtensor_count = 0
    logical_parameter_bytes = 0
    local_payload_bytes = 0
    unsharded_parameter_bytes = 0
    layouts: Counter[str] = Counter()
    logical_dtypes: Counter[str] = Counter()
    storage_dtypes: Counter[str] = Counter()

    for param in model.parameters():
        parameter_count += 1
        logical_parameter_bytes += _tensor_bytes(param)
        logical_dtypes[_dtype_name(getattr(param, "dtype", "unknown"))] += 1

        is_dtensor = isinstance(param, DTensor)
        local = param.to_local() if is_dtensor else param
        if is_dtensor:
            dtensor_count += 1

        layout_name = _layout_name(local)
        if layout_name is not None:
            layouts[layout_name] += 1

        storage = _storage_tensor(local)
        payload_bytes = _tensor_bytes(storage)
        local_payload_bytes += payload_bytes
        storage_dtypes[_dtype_name(getattr(storage, "dtype", "unknown"))] += 1
        if not is_dtensor:
            unsharded_parameter_bytes += payload_bytes

    return {
        "parameter_count": parameter_count,
        "dtensor_count": dtensor_count,
        "logical_parameter_bytes": logical_parameter_bytes,
        "local_payload_bytes": local_payload_bytes,
        "unsharded_parameter_bytes": unsharded_parameter_bytes,
        "layouts": dict(sorted(layouts.items())),
        "logical_dtypes": dict(sorted(logical_dtypes.items())),
        "storage_dtypes": dict(sorted(storage_dtypes.items())),
    }


def _format_counts(values: dict[str, int]) -> str:
    if not values:
        return "none"
    return ",".join(f"{name}:{count}" for name, count in sorted(values.items()))


def format_fsdp_diagnostics(diagnostics: dict[str, Any]) -> str:
    mib = 1024 * 1024
    return (
        f"params={diagnostics['parameter_count']} dtensors={diagnostics['dtensor_count']} "
        f"logical={diagnostics['logical_parameter_bytes'] / mib:.2f}MiB "
        f"local_payload={diagnostics['local_payload_bytes'] / mib:.2f}MiB "
        f"unsharded={diagnostics['unsharded_parameter_bytes'] / mib:.2f}MiB "
        f"layouts={_format_counts(diagnostics['layouts'])} "
        f"logical_dtypes={_format_counts(diagnostics['logical_dtypes'])} "
        f"storage_dtypes={_format_counts(diagnostics['storage_dtypes'])}"
    )


def select_mixed_dtype_ignored_params(
    params: list[torch.nn.Parameter],
    ignored_params: set[torch.nn.Parameter],
) -> set[torch.nn.Parameter]:
    """Keep each FSDP unit dtype-uniform without casting auxiliary parameters."""
    dtype_groups: dict[torch.dtype, list[torch.nn.Parameter]] = {}
    for param in params:
        if param in ignored_params or isinstance(param, DTensor):
            continue
        dtype_groups.setdefault(param.dtype, []).append(param)

    if len(dtype_groups) <= 1:
        return set()

    primary_dtype = max(
        dtype_groups,
        key=lambda dtype: (
            sum(_tensor_bytes(param) for param in dtype_groups[dtype]),
            str(dtype),
        ),
    )
    return {
        param for dtype, dtype_params in dtype_groups.items() if dtype != primary_dtype for param in dtype_params
    }

def summarize_selected_parameters(
    model: torch.nn.Module,
    params: set[torch.nn.Parameter],
    largest_limit: int = 12,
) -> dict[str, Any]:
    names_by_id = {id(param): name for name, param in model.named_parameters()}
    dtype_stats: dict[str, dict[str, int]] = {}
    suffix_counts: Counter[str] = Counter()
    rows = []

    for param in params:
        name = names_by_id.get(id(param), "<unnamed>")
        param_bytes = _tensor_bytes(param)
        dtype_name = str(param.dtype).removeprefix("torch.")
        stats = dtype_stats.setdefault(dtype_name, {"count": 0, "bytes": 0})
        stats["count"] += 1
        stats["bytes"] += param_bytes
        suffix_counts[name.rsplit(".", 1)[-1]] += 1
        rows.append(
            {
                "name": name,
                "dtype": dtype_name,
                "shape": list(param.shape),
                "bytes": param_bytes,
            }
        )

    rows.sort(key=lambda row: (-row["bytes"], row["name"]))
    return {
        "parameter_count": len(params),
        "parameter_bytes": sum(row["bytes"] for row in rows),
        "dtypes": {name: dtype_stats[name] for name in sorted(dtype_stats)},
        "suffixes": dict(sorted(suffix_counts.items())),
        "largest": rows[: max(0, largest_limit)],
    }



_FP8_LAYOUT_NAMES = {
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
}


def align_fp8_logical_dtype(
    model: torch.nn.Module,
    target_dtype: torch.dtype,
) -> int:
    """Align FP8 wrapper metadata for an explicit FSDP inference dtype.

    The FP8 payload is reused without copying. Only the wrapper and immutable
    layout params are replaced so FSDP may group the quantized weight with its
    same-dtype bias instead of replicating every bias.
    """
    aligned_count = 0
    for name, param in list(model.named_parameters()):
        if not isinstance(param, QuantizedTensor):
            continue
        if getattr(param, "_layout_cls", None) not in _FP8_LAYOUT_NAMES:
            continue
        if param.dtype is target_dtype:
            continue

        aligned_params = replace(param._params, orig_dtype=target_dtype)
        aligned_tensor = param._copy_with(params=aligned_params, clone_params=False)
        parent, leaf_name = _get_parent_module_and_name(model, name)
        parent.register_parameter(
            leaf_name,
            torch.nn.Parameter(aligned_tensor, requires_grad=False),
        )
        aligned_count += 1
    return aligned_count
def collect_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "scale" in param_name:
            ignored.add(param)
    return ignored


def collect_input_scale_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for param_name, param in module.named_parameters(recurse=True):
        if "input_scale" in param_name:
            ignored.add(param)
    return ignored


def collect_scalar_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if param.ndim == 0:
            ignored.add(param)
    return ignored


def _has_odd_shard_dim(param: torch.Tensor) -> bool:
    return param.ndim > 0 and (int(param.shape[0]) % 2) == 1


def collect_odd_dim0_ignored_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    ignored: set[torch.nn.Parameter] = set()
    for _param_name, param in module.named_parameters(recurse=True):
        if _has_odd_shard_dim(param):
            ignored.add(param)
    return ignored


def _should_materialize_unsharded_param(
    param_name: str,
    param: torch.Tensor,
    full_sd: dict[str, Any] | None = None,
) -> bool:
    if param_name.endswith("input_scale") or param.ndim == 0:
        return True
    if not _has_odd_shard_dim(param):
        return False
    if full_sd is not None and _is_quant_param(param_name, full_sd, param):
        return False
    return True


def _get_parent_module_and_name(model: torch.nn.Module, param_name: str) -> tuple[torch.nn.Module, str]:
    if "." not in param_name:
        return model, param_name
    parent_name, leaf_name = param_name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf_name


def _maybe_collapse_replicated_leading_dim(full_tensor: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    expected_shape = tuple(target_shape)
    if tuple(full_tensor.shape) == expected_shape:
        return full_tensor
    if full_tensor.ndim == 0 or full_tensor.ndim != len(expected_shape):
        return full_tensor
    if tuple(full_tensor.shape[1:]) != expected_shape[1:]:
        return full_tensor

    actual_leading = full_tensor.shape[0]
    expected_leading = expected_shape[0]
    if expected_leading <= 0 or actual_leading < expected_leading or actual_leading % expected_leading != 0:
        return full_tensor

    replicas = actual_leading // expected_leading
    if replicas <= 1:
        return full_tensor

    collapsed = full_tensor.reshape(expected_leading, replicas, *full_tensor.shape[1:])
    canonical = collapsed[:, 0, ...]
    if torch.equal(collapsed, canonical.unsqueeze(1).expand_as(collapsed)):
        return canonical
    return full_tensor


def _materialize_unsharded_param(
    model: torch.nn.Module,
    param_name: str,
    meta_param: torch.Tensor,
    full_tensor: torch.Tensor,
    device: torch.device,
    cpu_offload: bool,
) -> None:
    full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, meta_param.shape)
    full_tensor = full_tensor.to(dtype=meta_param.dtype, device=device)
    if cpu_offload:
        full_tensor = full_tensor.cpu()
    parent_module, leaf_name = _get_parent_module_and_name(model, param_name)
    parent_module.register_parameter(
        leaf_name,
        torch.nn.Parameter(full_tensor, requires_grad=meta_param.requires_grad),
    )


def _materialize_missing_ignored_params(
    model: torch.nn.Module,
    full_sd: dict[str, Any],
    device: torch.device,
    strict: bool,
    cpu_offload: bool,
    release_sd: bool,
) -> None:
    for param_name, param in list(model.named_parameters()):
        if not getattr(param, "is_meta", False):
            continue
        if not _should_materialize_unsharded_param(param_name, param, full_sd):
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        _materialize_unsharded_param(model, param_name, param, full_tensor, device, cpu_offload)
        if release_sd:
            full_sd[param_name] = None


def _collect_subtree_params(module: torch.nn.Module) -> set[torch.nn.Parameter]:
    return set(module.parameters())


def fully_shard_bottom_up(
    model: torch.nn.Module,
    fsdp_kwargs: dict[str, Any],
    native_ignore_scale: bool,
    ignored_modules: set[torch.nn.Module] | None = None,
) -> int:
    excluded_params: set[torch.nn.Parameter] = set()
    ignored_module_ids: set[int] = set()
    if ignored_modules:
        for mod in ignored_modules:
            excluded_params |= _collect_subtree_params(mod)
            ignored_module_ids.update(id(child) for child in mod.modules())

    shard_order = collect_bottom_up_shard_order(model)
    if ignored_modules:
        shard_order = [(n, m) for n, m in shard_order if id(m) not in ignored_module_ids]

    validate_inference_shard_order(shard_order)

    mixed_dtype_ignored_params: set[torch.nn.Parameter] = set()
    num_layers_sharded = 0
    for _name, module in shard_order:
        kwargs = dict(fsdp_kwargs)
        ignored_params: set[torch.nn.Parameter] = set()
        if native_ignore_scale:
            ignored_params |= collect_scale_ignored_params(module)

        ignored_params |= collect_input_scale_ignored_params(module)
        ignored_params |= collect_scalar_ignored_params(module)
        ignored_params |= collect_odd_dim0_ignored_params(module)

        subtree_params = set(module.parameters())
        ignored_params |= (mixed_dtype_ignored_params & subtree_params)
        unmanaged_params = [param for param in module.parameters() if not isinstance(param, DTensor)]
        newly_ignored = select_mixed_dtype_ignored_params(unmanaged_params, ignored_params)
        mixed_dtype_ignored_params |= newly_ignored
        ignored_params |= newly_ignored

        ignored_params |= (excluded_params & subtree_params)

        if ignored_params:
            kwargs["ignored_params"] = ignored_params

        fully_shard(module, **kwargs)
        num_layers_sharded += 1

    low_peak_wrappers = enable_low_peak_fsdp_unshard(model)
    if low_peak_wrappers:
        print(
            f"[FSDP] Enabled low-peak default-stream unshard for "
            f"{low_peak_wrappers} wrappers"
        )

    if mixed_dtype_ignored_params:
        replicated_bytes = sum(_tensor_bytes(param) for param in mixed_dtype_ignored_params)
        print(
            f"[FSDP] Replicating {len(mixed_dtype_ignored_params)} auxiliary mixed-dtype params ({replicated_bytes / (1024 * 1024):.2f}MiB)"
        )
        detail = summarize_selected_parameters(model, mixed_dtype_ignored_params)
        print(
            f"[FSDP] Mixed-dtype replicated detail "
            f"{json.dumps(detail, sort_keys=True)}"
        )

    if num_layers_sharded == 0:
        raise ValueError("No layer modules were sharded. Please check if shard conditions are working as expected.")
    return num_layers_sharded


def materialize_excluded_params(
    model: torch.nn.Module,
    excluded_modules: set[torch.nn.Module],
    full_sd: dict[str, Any],
    device: torch.device,
    cpu_offload: bool = False,
) -> int:
    """Load parameters of modules excluded from FSDP wrapping from a full state dict.

    After FSDP wrapping, excluded-module parameters remain on meta device.
    This function materializes them onto *device* (or CPU when cpu_offload).
    Returns the number of parameters materialized.
    """
    module_to_prefix: dict[int, str] = {}
    for name, mod in model.named_modules():
        module_to_prefix[id(mod)] = name

    count = 0
    for module in excluded_modules:
        prefix = module_to_prefix.get(id(module), "")
        for param_name, param in module.named_parameters(recurse=True):
            full_name = f"{prefix}.{param_name}" if prefix else param_name
            if not getattr(param, "is_meta", False):
                continue
            full_tensor = full_sd.get(full_name)
            if full_tensor is None:
                continue
            _materialize_unsharded_param(model, full_name, param, full_tensor, device, cpu_offload)
            count += 1
    return count


def _decode_comfy_quant(conf: Any) -> dict[str, Any] | None:
    if conf is None:
        return None
    if isinstance(conf, dict):
        return conf
    if isinstance(conf, (bytes, bytearray)):
        return json.loads(conf.decode("utf-8"))
    if isinstance(conf, torch.Tensor):
        raw = conf.detach().cpu().numpy().tobytes()
        if conf.dtype == torch.uint8:
            return json.loads(raw)
        return json.loads(raw.decode("utf-8"))
    if isinstance(conf, str):
        return json.loads(conf)
    raise TypeError(f"Unsupported comfy_quant type: {type(conf)}")


def _find_scaled_fp8_key(full_sd: dict[str, Any]) -> str | None:
    if "scaled_fp8" in full_sd:
        return "scaled_fp8"

    for key in full_sd.keys():
        if key.endswith(".scaled_fp8"):
            return key

    return None


def _legacy_scaled_fp8_conf(prefix: str, full_sd: dict[str, Any]) -> dict[str, Any] | None:
    has_legacy_scale = f"{prefix}scale_weight" in full_sd
    has_converted_scale = f"{prefix}weight_scale" in full_sd
    if not has_legacy_scale and not has_converted_scale:
        return None

    conf: dict[str, Any] = {"format": "float8_e4m3fn"}
    scaled_fp8_key = _find_scaled_fp8_key(full_sd)
    if scaled_fp8_key is not None:
        scaled_fp8_weight = full_sd.get(scaled_fp8_key)
        if isinstance(scaled_fp8_weight, torch.Tensor) and scaled_fp8_weight.nelement() == 2:
            conf["full_precision_matrix_mult"] = True

    return conf


def _quant_payload_debug_info(param_name: str, full_sd: dict[str, Any]) -> str:
    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else param_name
    debug_bits = {
        "weight": param_name in full_sd,
        "comfy_quant": f"{prefix}comfy_quant" in full_sd,
        "weight_scale": f"{prefix}weight_scale" in full_sd,
        "weight_scale_2": f"{prefix}weight_scale_2" in full_sd,
        "input_scale": f"{prefix}input_scale" in full_sd,
        "legacy_scale_weight": f"{prefix}scale_weight" in full_sd,
        "legacy_scale_input": f"{prefix}scale_input" in full_sd,
        "scaled_fp8": _find_scaled_fp8_key(full_sd) is not None,
    }
    prefix_keys = sorted(
        key
        for key in full_sd.keys()
        if key.startswith(prefix)
        and (
            key == param_name
            or key.endswith("comfy_quant")
            or key.endswith("weight_scale")
            or key.endswith("weight_scale_2")
            or key.endswith("input_scale")
            or key.endswith("scale_weight")
            or key.endswith("scale_input")
        )
    )
    return f"payload={debug_bits}, prefix_keys={prefix_keys}"


def _shard_tensor(
    full_tensor: torch.Tensor,
    sharded_meta_param: Any,
    device: torch.device,
    *,
    pad_to_local_meta: bool = True,
) -> torch.Tensor:
    if not hasattr(sharded_meta_param, "device_mesh"):
        return full_tensor.to(device=device)

    mesh = sharded_meta_param.device_mesh
    if mesh.ndim > 1:
        raise NotImplementedError(f"only support 1D FSDP but got {mesh.ndim}")

    shard_mesh_dim = 0
    shard_world_size = mesh.size(shard_mesh_dim)
    shard_rank = cast(torch.distributed.ProcessGroup, mesh.get_group(shard_mesh_dim)).rank()

    chunk = torch.tensor_split(full_tensor, shard_world_size, dim=0)[shard_rank].to(device=device)

    local_meta = getattr(sharded_meta_param, "_local_tensor", None)
    if not pad_to_local_meta or not isinstance(local_meta, torch.Tensor):
        return chunk

    local_shape = tuple(local_meta.shape)
    if tuple(chunk.shape) == local_shape:
        return chunk
    if len(local_shape) != chunk.ndim:
        return chunk
    if any(local_dim < chunk_dim for local_dim, chunk_dim in zip(local_shape, chunk.shape)):
        return chunk

    sharded_param = full_tensor.new_zeros(local_shape, device=device)
    if chunk.numel() > 0:
        sharded_param[tuple(slice(0, dim) for dim in chunk.shape)].copy_(chunk)
    return sharded_param


def _is_quant_param(param_name: str, full_sd: dict[str, Any], sharded_meta_param: Any) -> bool:
    if isinstance(full_sd.get(param_name), QuantizedTensor):
        return True

    prefix = param_name[: -len("weight")] if param_name.endswith("weight") else None
    if prefix is not None and (
        f"{prefix}comfy_quant" in full_sd
        or f"{prefix}weight_scale" in full_sd
        or f"{prefix}scale_weight" in full_sd
    ):
        return True

    if isinstance(sharded_meta_param, QuantizedTensor):
        return True
    if hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        return True

    return False


def _build_quantized_tensor(
    param_name: str,
    full_sd: dict[str, Any],
    sharded_meta_param: Any,
    device: torch.device,
):
    def _local_orig_shape(layout_name: str, local_qdata: torch.Tensor, logical_orig_shape: tuple[int, ...] | None) -> tuple[int, ...]:
        if logical_orig_shape is None:
            return tuple(local_qdata.shape)
        if layout_name == "TensorCoreNVFP4Layout" and len(logical_orig_shape) == 2 and local_qdata.dim() == 2:
            return (int(local_qdata.shape[0]), int(logical_orig_shape[1]))
        return tuple(local_qdata.shape)

    if not param_name.endswith("weight"):
        return None

    full_q = full_sd.get(param_name)
    if isinstance(full_q, QuantizedTensor):
        qt = cast(Any, full_q)
        if qt._layout_cls not in ("TensorCoreFP8Layout", "TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout"):
            raise NotImplementedError(
                f"Raylight FSDP direct QuantizedTensor loading only supports FP8 layouts, got {qt._layout_cls} for {param_name}. "
                "Use a comfy_quant state dict payload for supported formats or disable Raylight FSDP quant loading."
            )
        local_qdata = _shard_tensor(qt._qdata, sharded_meta_param, device, pad_to_local_meta=False)
        local_params = replace(
            qt._params, orig_shape=_local_orig_shape(qt._layout_cls, local_qdata, getattr(qt._params, "orig_shape", None))
        )
        return QuantizedTensor(local_qdata, qt._layout_cls, local_params)

    prefix = param_name[: -len("weight")]
    conf = _decode_comfy_quant(full_sd.get(f"{prefix}comfy_quant"))
    if conf is None:
        conf = _legacy_scaled_fp8_conf(prefix, full_sd)
    if conf is None:
        return None

    quant_format = conf.get("format", None)
    if quant_format is None or quant_format not in QUANT_ALGOS:
        raise ValueError(f"Unknown quantization format for {param_name}: {quant_format}")
    if quant_format == "mxfp8":
        raise NotImplementedError(
            "Raylight FSDP does not support MXFP8 quantized weights yet. "
            "Use FP8/NVFP4 weights or disable Raylight FSDP quant loading."
        )
    qconfig = QUANT_ALGOS[quant_format]
    layout_name = qconfig["comfy_tensor_layout"]
    layout_cls = get_layout_class(layout_name)
    if layout_cls is None:
        raise ValueError(f"Missing layout class for {layout_name}")

    full_qdata = full_sd.get(param_name)
    if full_qdata is None:
        raise ValueError(f"Missing quantized weight for {param_name}")

    qdata = full_qdata.to(dtype=qconfig["storage_t"])
    qdata = _shard_tensor(qdata, sharded_meta_param, device, pad_to_local_meta=False)

    params_kwargs: dict[str, Any] = {"orig_shape": tuple(qdata.shape)}

    local_meta = None
    if isinstance(sharded_meta_param, QuantizedTensor):
        local_meta = sharded_meta_param
    elif hasattr(sharded_meta_param, "_local_tensor") and isinstance(sharded_meta_param._local_tensor, QuantizedTensor):
        local_meta = sharded_meta_param._local_tensor
    orig_dtype = None
    if local_meta is not None and hasattr(local_meta, "_params"):
        orig_dtype = getattr(local_meta._params, "orig_dtype", None)
    if orig_dtype is None:
        orig_dtype = getattr(sharded_meta_param, "dtype", None)
    if orig_dtype is not None:
        params_kwargs["orig_dtype"] = orig_dtype

    logical_orig_shape = getattr(getattr(local_meta, "_params", None), "orig_shape", None)
    if logical_orig_shape is None and quant_format == "nvfp4" and full_qdata.dim() == 2:
        logical_orig_shape = (int(full_qdata.shape[0]), int(full_qdata.shape[1] * 2))
    params_kwargs["orig_shape"] = _local_orig_shape(layout_name, qdata, logical_orig_shape)

    if quant_format in ("float8_e4m3fn", "float8_e5m2"):
        scale = full_sd.get(f"{prefix}weight_scale")
        if scale is None:
            scale = full_sd.get(f"{prefix}scale_weight")
        if scale is not None:
            scale = scale.to(device=device)
        params_kwargs["scale"] = scale
    elif quant_format == "nvfp4":
        tensor_scale = full_sd.get(f"{prefix}weight_scale_2")
        block_scale = full_sd.get(f"{prefix}weight_scale")
        if tensor_scale is None or block_scale is None:
            raise ValueError(f"Missing NVFP4 scales for {param_name}")
        tensor_scale = tensor_scale.to(device=device)
        block_scale = block_scale.view(dtype=torch.float8_e4m3fn)
        block_scale = _shard_tensor(block_scale, sharded_meta_param, device, pad_to_local_meta=False)
        params_kwargs["scale"] = tensor_scale
        params_kwargs["block_scale"] = block_scale
    elif quant_format == "int8_tensorwise":
        scale = full_sd.get(f"{prefix}weight_scale")
        if scale is None:
            raise ValueError(f"Missing INT8 weight scale for {param_name}")
        if isinstance(scale, torch.Tensor) and scale.dim() > 0 and scale.numel() > 1:
            scale = _shard_tensor(scale, sharded_meta_param, device, pad_to_local_meta=False)
        elif isinstance(scale, torch.Tensor):
            scale = scale.to(device=device)
        params_kwargs["scale"] = scale
        params_conf = conf.get("params", {})
        if not isinstance(params_conf, dict):
            params_conf = {}
        if conf.get("convrot", params_conf.get("convrot", False)):
            params_kwargs["convrot"] = True
            params_kwargs["convrot_groupsize"] = int(conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
    elif quant_format == "gguf":
        n_blocks_per_superblock = conf.get("n_blocks_per_superblock", 8)
        super_block_scale_scale = full_sd.get(f"{prefix}super_block_scale_scale")
        super_block_min_scale = full_sd.get(f"{prefix}super_block_min_scale")
        quantized_block_scale = full_sd.get(f"{prefix}quantized_block_scale")
        quantized_block_min = full_sd.get(f"{prefix}quantized_block_min")
        if (
            super_block_scale_scale is None
            or super_block_min_scale is None
            or quantized_block_scale is None
            or quantized_block_min is None
        ):
            raise ValueError(f"Missing GGUF scales for {param_name}")
        super_block_scale_scale = super_block_scale_scale.to(device=device)
        super_block_min_scale = super_block_min_scale.to(device=device)
        quantized_block_scale = quantized_block_scale.to(device=device)
        quantized_block_min = quantized_block_min.to(device=device)
        super_block_scale_scale = _shard_tensor(super_block_scale_scale, sharded_meta_param, device, pad_to_local_meta=False)
        super_block_min_scale = _shard_tensor(super_block_min_scale, sharded_meta_param, device, pad_to_local_meta=False)
        quantized_block_scale = _shard_tensor(quantized_block_scale, sharded_meta_param, device, pad_to_local_meta=False)
        quantized_block_min = _shard_tensor(quantized_block_min, sharded_meta_param, device, pad_to_local_meta=False)
        params_kwargs["n_blocks_per_superblock"] = n_blocks_per_superblock
        params_kwargs["super_block_scale_scale"] = super_block_scale_scale
        params_kwargs["super_block_min_scale"] = super_block_min_scale
        params_kwargs["quantized_block_scale"] = quantized_block_scale
        params_kwargs["quantized_block_min"] = quantized_block_min
        if f"{prefix}scale" in full_sd:
            scale = full_sd.get(f"{prefix}scale")
            if scale is not None:
                params_kwargs["scale"] = scale.to(device=device)
    else:
        raise ValueError(f"Unsupported quantization format: {quant_format}")

    params = layout_cls.Params(**params_kwargs)
    return QuantizedTensor(qdata, layout_name, params)


def _release_quant_keys(full_sd: dict[str, Any], param_name: str) -> None:
    prefix = param_name[: -len("weight")]
    for key in (
        param_name,
        f"{prefix}weight_scale",
        f"{prefix}weight_scale_2",
        f"{prefix}input_scale",
        f"{prefix}scale_weight",
        f"{prefix}scale_input",
        f"{prefix}comfy_quant",
        f"{prefix}super_block_scale_scale",
        f"{prefix}super_block_min_scale",
        f"{prefix}quantized_block_scale",
        f"{prefix}quantized_block_min",
    ):
        if key in full_sd:
            full_sd[key] = None


# Heavily modified from
# https://github.com/meta-pytorch/torchtune/blob/d0f63bb33d00b8bd3905a010b71d8c6324c2e980/torchtune/training/_distributed.py#L336
# Need to be done since dcp loader cause wrong dtype among rank when broadcasting.
def load_from_full_model_state_dict(
    model,
    full_sd,
    device,
    strict=False,
    cpu_offload=False,
    release_sd=True,
):
    meta_sharded_sd = model.state_dict()
    sharded_sd: dict[str, torch.Tensor] = {}
    for param_name, sharded_meta_param in meta_sharded_sd.items():
        parent_module, leaf_name = _get_parent_module_and_name(model, param_name)
        is_buffer = leaf_name in parent_module._buffers
        if not is_buffer and _should_materialize_unsharded_param(param_name, sharded_meta_param, full_sd):
            full_tensor = full_sd.get(param_name)
            if full_tensor is None:
                if strict:
                    raise ValueError(f"Missing parameter {param_name} in state_dict")
                continue
            _materialize_unsharded_param(model, param_name, sharded_meta_param, full_tensor, device, cpu_offload)
            if release_sd:
                full_sd[param_name] = None
            continue

        if not is_buffer and _is_quant_param(param_name, full_sd, sharded_meta_param):
            quant_tensor = _build_quantized_tensor(param_name, full_sd, sharded_meta_param, device)
            if quant_tensor is None:
                raise ValueError(
                    f"Expected quantized tensor for {param_name}, but could not build it ({_quant_payload_debug_info(param_name, full_sd)})"
                )
            if hasattr(sharded_meta_param, "device_mesh"):
                sharded_tensor = DTensor.from_local(
                    quant_tensor,
                    device_mesh=sharded_meta_param.device_mesh,
                    placements=sharded_meta_param.placements,
                )
            else:
                sharded_tensor = quant_tensor
            if cpu_offload:
                sharded_tensor = sharded_tensor.cpu()
            sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
            if release_sd:
                _release_quant_keys(full_sd, param_name)
            continue
        full_tensor = full_sd.get(param_name)
        if full_tensor is None:
            if strict:
                raise ValueError(f"Missing parameter {param_name} in state_dict")
            continue
        if not hasattr(sharded_meta_param, "device_mesh"):
            full_tensor = _maybe_collapse_replicated_leading_dim(full_tensor, sharded_meta_param.shape)
            full_tensor = full_tensor.to(sharded_meta_param.dtype).to(device)
            sharded_tensor = full_tensor
        else:
            full_tensor = full_tensor.to(sharded_meta_param.dtype)
            local_dense = _shard_tensor(full_tensor, sharded_meta_param, device)
            sharded_tensor = DTensor.from_local(
                local_dense,
                device_mesh=sharded_meta_param.device_mesh,
                placements=sharded_meta_param.placements,
            )
        if cpu_offload:
            sharded_tensor = sharded_tensor.cpu()
        sharded_sd[param_name] = sharded_tensor if is_buffer else torch.nn.Parameter(sharded_tensor)
        if release_sd:
            full_sd[param_name] = None
    with quantized_state_dict_zero_copy(), plain_bf16_state_dict_assign(model, sharded_sd):
        out = model.load_state_dict(sharded_sd, strict=strict, assign=True)
    _materialize_missing_ignored_params(model, full_sd, device, strict, cpu_offload, release_sd)
    return out
