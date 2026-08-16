import comfy_kitchen as ck
import os
from contextlib import contextmanager
from functools import wraps
from comfy_kitchen.tensor import TensorCoreFP8Layout, TensorCoreNVFP4Layout
import torch

_PATCH_INSTALL_LOGGED = False
_FP8_FALLBACK_LOGGED = False
_FP8_OOM_RETRY_LOGGED = False
_NVFP4_FALLBACK_LOGGED = False
_ACTIVATION_PATCH_LOGGED = False
_V100_BF16_ATTENTION_LOGGED = False
_V100_RMS_NORM_LOGGED = False
_V100_BF16_LONG_ATTN_LOGGED = False

_V100_BF16_DENSE_LOGGED = False

def _tensor_desc(tensor):
    shape = getattr(tensor, "shape", None)
    device = getattr(tensor, "device", None)
    dtype = getattr(tensor, "dtype", None)
    return f"shape={tuple(shape) if shape is not None else None} device={device} dtype={dtype}"


def dequantize_ray_temp_fix_fp8(qdata, params, dtype):
    global _FP8_FALLBACK_LOGGED, _FP8_OOM_RETRY_LOGGED
    if not _FP8_FALLBACK_LOGGED:
        print(f"[Raylight][comfy_kitchen][fp8] fallback dequantize dtype={dtype} qdata={_tensor_desc(qdata)}")
        _FP8_FALLBACK_LOGGED = True
    try:
        return ck.dequantize_per_tensor_fp8(qdata, params.scale, dtype)
    except torch.OutOfMemoryError:
        if not _FP8_OOM_RETRY_LOGGED:
            print("[Raylight][comfy_kitchen][fp8] dequantize OOM; releasing cached CUDA blocks and retrying once")
            _FP8_OOM_RETRY_LOGGED = True
        torch.cuda.empty_cache()
        return ck.dequantize_per_tensor_fp8(qdata, params.scale, dtype)


def dequantize_ray_temp_fix_nvfp4(qdata, params, dtype):
    global _NVFP4_FALLBACK_LOGGED
    if not _NVFP4_FALLBACK_LOGGED:
        print(f"[Raylight][comfy_kitchen][nvfp4] fallback dequantize dtype={dtype} qdata={_tensor_desc(qdata)}")
        _NVFP4_FALLBACK_LOGGED = True
    return ck.dequantize_nvfp4(qdata, params.scale, params.block_scale, dtype)

def apply_activation_inplace_chunked(value, activation, max_temp_bytes):
    """Apply an inference activation in chunks, reusing storage when shapes match."""
    if max_temp_bytes <= 0:
        raise ValueError("activation temporary budget must be positive")
    if value.requires_grad:
        raise RuntimeError("in-place chunked activation is inference-only")
    if value.ndim == 0:
        activated = activation(value)
        if activated.shape == value.shape:
            value.copy_(activated)
            return value
        return activated
    try:
        rows = value.view(-1, value.shape[-1])
    except RuntimeError as exc:
        raise ValueError("chunked activation requires contiguous row storage") from exc
    bytes_per_row = max(1, rows.shape[-1] * rows.element_size())
    rows_per_chunk = max(1, max_temp_bytes // bytes_per_row)

    first_end = min(rows_per_chunk, rows.shape[0])
    first_activated = activation(rows[:first_end])
    if first_activated.shape == rows[:first_end].shape:
        rows[:first_end].copy_(first_activated)
        for start in range(first_end, rows.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, rows.shape[0])
            activated = activation(rows[start:end])
            if activated.shape != rows[start:end].shape:
                raise ValueError("activation output shape changed between chunks")
            rows[start:end].copy_(activated)
        return value

    if first_activated.ndim != 2 or first_activated.shape[0] != first_end:
        raise ValueError("chunked activation must preserve the flattened row dimension")
    output_width = first_activated.shape[1]
    output_rows = torch.empty(
        (rows.shape[0], output_width),
        dtype=first_activated.dtype,
        device=first_activated.device,
    )
    output_rows[:first_end].copy_(first_activated)
    for start in range(first_end, rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, rows.shape[0])
        activated = activation(rows[start:end])
        if activated.shape != (end - start, output_width):
            raise ValueError("activation output shape changed between chunks")
        output_rows[start:end].copy_(activated)
    return output_rows.view(*value.shape[:-1], output_width)

def bf16_linear_fp32_chunked(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    max_temp_bytes: int,
    linear_fn=torch.nn.functional.linear,
) -> torch.Tensor:
    """Cast dense BF16 weight rows to FP32 without a full-size copy."""
    if max_temp_bytes <= 0:
        raise ValueError("dense linear temporary budget must be positive")
    if input_tensor.requires_grad or weight.requires_grad:
        raise RuntimeError("chunked dense linear is inference-only")
    if input_tensor.ndim < 1 or weight.ndim != 2:
        raise ValueError("chunked dense linear requires a 2D weight")
    if input_tensor.shape[-1] != weight.shape[1]:
        raise ValueError("dense linear input dimension does not match weight")

    output_features = weight.shape[0]
    output = torch.empty(
        (*input_tensor.shape[:-1], output_features),
        dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    input_rows = input_tensor.numel() // input_tensor.shape[-1]
    bytes_per_output_row = max(
        1,
        (weight.shape[1] + input_rows) * input_tensor.element_size(),
    )
    rows_per_chunk = max(1, max_temp_bytes // bytes_per_output_row)

    for start in range(0, output_features, rows_per_chunk):
        end = min(start + rows_per_chunk, output_features)
        weight_chunk = weight[start:end].to(
            device=input_tensor.device,
            dtype=input_tensor.dtype,
        )
        bias_chunk = None
        if bias is not None:
            bias_chunk = bias[start:end].to(
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )
        output_chunk = linear_fn(input_tensor, weight_chunk, bias_chunk)
        output[..., start:end].copy_(output_chunk)

    return output


@contextmanager
def temporary_v100_bf16_dense_linear(worker):
    parallel_dict = getattr(worker, "parallel_dict", {}) or {}
    if (
        not parallel_dict.get("is_fsdp", False)
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] != 7
    ):
        yield False
        return

    import comfy.ops

    global _V100_BF16_DENSE_LOGGED
    original = comfy.ops.disable_weight_init.Linear.forward_comfy_cast_weights
    try:
        chunk_bytes = max(
            1,
            int(os.environ.get("RAYLIGHT_DENSE_CAST_CHUNK_BYTES", str(32 * 1024 * 1024))),
        )
        threshold_bytes = max(
            1,
            int(os.environ.get("RAYLIGHT_DENSE_CAST_THRESHOLD_BYTES", str(64 * 1024 * 1024))),
        )
    except ValueError:
        chunk_bytes = 32 * 1024 * 1024
        threshold_bytes = 64 * 1024 * 1024

    def forward_chunked(module, input_tensor):
        global _V100_BF16_DENSE_LOGGED
        weight = module.weight
        can_chunk = (
            isinstance(weight, torch.Tensor)
            and weight.ndim == 2
            and weight.device == input_tensor.device
            and weight.dtype is torch.bfloat16
            and input_tensor.dtype is torch.float32
            and weight.numel() * input_tensor.element_size() >= threshold_bytes
            and len(module.weight_function) == 0
            and len(module.bias_function) == 0
        )
        if not can_chunk:
            return original(module, input_tensor)
        if not _V100_BF16_DENSE_LOGGED:
            print(
                f"[Raylight][V100] chunked BF16-to-FP32 dense Linear enabled budget={chunk_bytes / 1024 / 1024:.0f}MiB"
            )
            _V100_BF16_DENSE_LOGGED = True
        return bf16_linear_fp32_chunked(
            input_tensor, weight, module.bias, chunk_bytes
        )

    comfy.ops.disable_weight_init.Linear.forward_comfy_cast_weights = forward_chunked
    try:
        yield True
    finally:
        comfy.ops.disable_weight_init.Linear.forward_comfy_cast_weights = original



def linear_input_act_v100_chunked(linear, value, input_act):
    import comfy.ops
    try:
        max_temp_bytes = max(
            1,
            int(os.environ.get("RAYLIGHT_ACTIVATION_CHUNK_BYTES", str(32 * 1024 * 1024))),
        )
    except ValueError:
        max_temp_bytes = 32 * 1024 * 1024
    activation = comfy.ops.INPUT_ACT_EAGER[input_act]
    value = apply_activation_inplace_chunked(value, activation, max_temp_bytes)
    return linear(value)


def should_use_v100_chunked_rms_norm(worker, cuda_capability=None):
    parallel_dict = getattr(worker, "parallel_dict", {}) or {}
    if not parallel_dict.get("is_fsdp", False):
        return False
    if cuda_capability is None:
        if not torch.cuda.is_available():
            return False
        cuda_capability = torch.cuda.get_device_capability()
    return tuple(cuda_capability) == (7, 0)


def apply_rms_norm_chunked(
    value,
    normalized_shape,
    weight,
    eps,
    rms_norm,
    max_temp_bytes,
):
    """Evaluate inference RMSNorm in bounded chunks without mutating residual input."""
    if max_temp_bytes <= 0:
        raise ValueError("RMSNorm temporary budget must be positive")
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    trailing_shape = tuple(value.shape[-len(normalized_shape):])
    if (
        value.requires_grad
        or not value.is_contiguous()
        or trailing_shape != normalized_shape
    ):
        return rms_norm(value, normalized_shape, weight, eps)

    normalized_elements = 1
    for dimension in normalized_shape:
        normalized_elements *= dimension
    rows = value.view(-1, *normalized_shape)
    try:
        output = torch.empty_like(value)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        output = torch.empty_like(value)
    output_rows = output.view(-1, *normalized_shape)
    bytes_per_row = max(1, normalized_elements * value.element_size())
    rows_per_chunk = max(1, max_temp_bytes // bytes_per_row)
    for start in range(0, rows.shape[0], rows_per_chunk):
        end = min(start + rows_per_chunk, rows.shape[0])
        normalized = rms_norm(rows[start:end], normalized_shape, weight, eps)
        output_rows[start:end].copy_(normalized)
    return output


def v100_rms_norm_chunk_bytes():
    default_bytes = 16 * 1024 * 1024
    try:
        return max(
            1,
            int(os.environ.get("RAYLIGHT_RMS_NORM_CHUNK_BYTES", str(default_bytes))),
        )
    except ValueError:
        return default_bytes


@contextmanager
def temporary_v100_chunked_rms_norm(worker):
    global _V100_RMS_NORM_LOGGED
    if not should_use_v100_chunked_rms_norm(worker):
        yield False
        return

    import torch.nn.functional as functional

    original_rms_norm = functional.rms_norm
    max_temp_bytes = v100_rms_norm_chunk_bytes()

    def chunked_rms_norm(value, normalized_shape, weight=None, eps=None):
        return apply_rms_norm_chunked(
            value,
            normalized_shape,
            weight,
            eps,
            original_rms_norm,
            max_temp_bytes,
        )

    functional.rms_norm = chunked_rms_norm
    if not _V100_RMS_NORM_LOGGED:
        print(
            f"[Raylight][V100] residual-safe chunked RMSNorm enabled "
            f"budget={max_temp_bytes / 1024 / 1024:.0f}MiB"
        )
        _V100_RMS_NORM_LOGGED = True
    try:
        yield True
    finally:
        functional.rms_norm = original_rms_norm


def should_use_v100_bf16_chunked_attention(worker, cuda_capability=None):
    """Select bounded attention only for the V100 BF16 FSDP path."""
    parallel_dict = getattr(worker, "parallel_dict", {}) or {}
    if not parallel_dict.get("is_fsdp", False):
        return False

    model = getattr(worker, "model", None)
    if getattr(model, "fsdp_param_dtype", None) is not torch.bfloat16:
        return False

    if cuda_capability is None:
        if not torch.cuda.is_available():
            return False
        cuda_capability = torch.cuda.get_device_capability()

    return tuple(cuda_capability) == (7, 0)


@contextmanager
def temporary_v100_bf16_chunked_attention(worker):
    """Use memory-bounded attention when xFormers has no V100 BF16 kernel."""
    global _V100_BF16_ATTENTION_LOGGED
    if not should_use_v100_bf16_chunked_attention(worker):
        yield False
        return

    import comfy.ldm.modules.attention as attention

    original_attention = attention.optimized_attention
    original_attention_masked = attention.optimized_attention_masked
    original_efficient_attention = attention.efficient_dot_product_attention
    try:
        split_threshold_bytes = max(
            1,
            int(os.environ.get("RAYLIGHT_ATTENTION_LONG_THRESHOLD_BYTES", str(64 * 1024 * 1024))),
        )
    except ValueError:
        split_threshold_bytes = 64 * 1024 * 1024

    try:
        long_query_chunk_tokens = max(
            1,
            int(os.environ.get("RAYLIGHT_ATTENTION_QUERY_CHUNK_TOKENS", "64")),
        )
    except ValueError:
        long_query_chunk_tokens = 64

    def bounded_efficient_attention(query, key_t, value, *args, **kwargs):
        input_bytes = query.numel() * query.element_size()
        if input_bytes >= split_threshold_bytes:
            requested_query_chunk = int(kwargs.get("query_chunk_size", 1024))
            kwargs["query_chunk_size"] = min(
                requested_query_chunk,
                long_query_chunk_tokens,
            )
            kwargs["kv_chunk_size"] = key_t.shape[-1]
        return original_efficient_attention(query, key_t, value, *args, **kwargs)

    def bounded_attention(q, *args, **kwargs):
        global _V100_BF16_LONG_ATTN_LOGGED
        input_bytes = q.numel() * q.element_size()
        if input_bytes >= split_threshold_bytes:
            if not _V100_BF16_LONG_ATTN_LOGGED:
                print(
                    f"[Raylight][V100][BF16] long-sequence full-KV query slicing enabled "
                    f"input={input_bytes / 1024 / 1024:.0f}MiB "
                    f"threshold={split_threshold_bytes / 1024 / 1024:.0f}MiB "
                    f"query_chunk={long_query_chunk_tokens}"
                )
                _V100_BF16_LONG_ATTN_LOGGED = True
        return attention.attention_sub_quad(q, *args, **kwargs)

    attention.efficient_dot_product_attention = bounded_efficient_attention
    attention.optimized_attention = bounded_attention
    attention.optimized_attention_masked = bounded_attention
    if not _V100_BF16_ATTENTION_LOGGED:
        print(
            "[Raylight][V100][BF16] memory-bounded attention router enabled; "
            "sub-quadratic for normal inputs, full-KV query slicing for long sequences"
        )
        _V100_BF16_ATTENTION_LOGGED = True
    try:
        yield True
    finally:
        attention.efficient_dot_product_attention = original_efficient_attention
        attention.optimized_attention = original_attention
        attention.optimized_attention_masked = original_attention_masked



def patch_temp_fix_ck_ops(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global _PATCH_INSTALL_LOGGED, _ACTIVATION_PATCH_LOGGED
        from comfy import ops as comfy_ops

        self = args[0]
        parallel_dict = getattr(self, "parallel_dict", {}) or {}
        overwrite_cast_dtype = getattr(self, "overwrite_cast_dtype", None)
        layouts = parallel_dict.get("comfy_kitchen_layouts", ("fp8", "nvfp4", "int8"))
        install_ck_patches = bool(parallel_dict.get("is_quant", False))
        ck_patched = False
        restore_sitepkg_ck_patches = None
        original_fp8 = None
        original_nvfp4 = None
        dense_linear_context = None
        original_linear_input_act = None
        attention_context = None
        rms_norm_context = None

        original_dense_linear_forward = None
        try:
            attention_context = temporary_v100_bf16_chunked_attention(self)
            dense_linear_context = temporary_v100_bf16_dense_linear(self)
            dense_linear_context.__enter__()
            attention_context.__enter__()
            rms_norm_context = temporary_v100_chunked_rms_norm(self)
            rms_norm_context.__enter__()
            if install_ck_patches:
                from raylight.comfy_dist.kitchen_distributed import install_sitepkg_ck_patches, restore_sitepkg_ck_patches

                install_sitepkg_ck_patches(layouts=layouts)
                ck_patched = True
                if not _PATCH_INSTALL_LOGGED:
                    print(
                        "[Raylight][comfy_kitchen] installing quant patches "
                        f"layouts={layouts} is_fsdp={parallel_dict.get('is_fsdp', False)} is_quant={parallel_dict.get('is_quant', False)}"
                    )
                    _PATCH_INSTALL_LOGGED = True

            if parallel_dict.get("is_fsdp", False) and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 7:
                original_linear_input_act = comfy_ops.linear_input_act
                comfy_ops.linear_input_act = linear_input_act_v100_chunked
                if not _ACTIVATION_PATCH_LOGGED:
                    print("[Raylight][V100] chunked in-place linear_input_act enabled")
                    _ACTIVATION_PATCH_LOGGED = True

            if overwrite_cast_dtype is not None:
                original_dense_linear_forward = (
                    comfy_ops.disable_weight_init.Linear.forward_comfy_cast_weights
                )
                try:
                    dense_chunk_bytes = max(
                        1,
                        int(os.environ.get("RAYLIGHT_DENSE_CAST_CHUNK_BYTES", str(32 * 1024 * 1024))),
                    )
                    dense_threshold_bytes = max(
                        1,
                        int(os.environ.get("RAYLIGHT_DENSE_CAST_THRESHOLD_BYTES", str(64 * 1024 * 1024))),
                    )
                except ValueError:
                    dense_chunk_bytes = 32 * 1024 * 1024
                    dense_threshold_bytes = 64 * 1024 * 1024

                def forward_dense_cast_chunked(module, input_tensor):
                    global _V100_BF16_DENSE_LOGGED
                    weight = module.weight
                    can_chunk = (
                        isinstance(weight, torch.Tensor)
                        and weight.ndim == 2
                        and weight.device == input_tensor.device
                        and weight.dtype is torch.bfloat16
                        and input_tensor.dtype is torch.float32
                        and weight.numel() * input_tensor.element_size() >= dense_threshold_bytes
                        and len(module.weight_function) == 0
                        and len(module.bias_function) == 0
                    )
                    if not can_chunk:
                        return original_dense_linear_forward(module, input_tensor)
                    if not _V100_BF16_DENSE_LOGGED:
                        print(
                            f"[Raylight][V100] chunked BF16-to-FP32 dense Linear enabled budget={dense_chunk_bytes / 1024 / 1024:.0f}MiB"
                        )
                        _V100_BF16_DENSE_LOGGED = True
                    return bf16_linear_fp32_chunked(
                        input_tensor, weight, module.bias, dense_chunk_bytes
                    )

                comfy_ops.disable_weight_init.Linear.forward_comfy_cast_weights = forward_dense_cast_chunked
                original_fp8 = TensorCoreFP8Layout.dequantize
                original_nvfp4 = TensorCoreNVFP4Layout.dequantize

                TensorCoreFP8Layout.dequantize = classmethod(
                    lambda cls, qdata, params:
                        dequantize_ray_temp_fix_fp8(
                            qdata,
                            params,
                            overwrite_cast_dtype
                        )
                )

                TensorCoreNVFP4Layout.dequantize = classmethod(
                    lambda cls, qdata, params:
                        dequantize_ray_temp_fix_nvfp4(
                            qdata,
                            params,
                            overwrite_cast_dtype
                        )
                )

            return func(*args, **kwargs)

        finally:
            if original_linear_input_act is not None:
                comfy_ops.linear_input_act = original_linear_input_act
            if original_dense_linear_forward is not None:
                comfy_ops.disable_weight_init.Linear.forward_comfy_cast_weights = original_dense_linear_forward
            if original_fp8 is not None:
                TensorCoreFP8Layout.dequantize = original_fp8
            if original_nvfp4 is not None:
                TensorCoreNVFP4Layout.dequantize = original_nvfp4
            if dense_linear_context is not None:
                dense_linear_context.__exit__(None, None, None)
            if ck_patched and restore_sitepkg_ck_patches is not None:
                restore_sitepkg_ck_patches(layouts=layouts)
            if rms_norm_context is not None:
                rms_norm_context.__exit__(None, None, None)
            if attention_context is not None:
                attention_context.__exit__(None, None, None)

    return wrapper
