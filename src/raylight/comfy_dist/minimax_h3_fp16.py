"""Safe FP16 execution islands for ComfyUI's MiniMax H3 model.

This adapts the exact power-of-two rescaling strategy from
``Amduraznak/minimax-h3-fp16-fix`` (Copyright 2026 Amduraznak, MIT) for
Raylight workers. The upstream notice is retained in
``docs/third-party/minimax-h3-fp16-fix.md``. The patch is
opt-in and only activates on MiniMax H3 instances constructed with
``torch.float16``.  FP32 and BF16 model behavior remains unchanged.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable

import torch


SAFE_FP16_OPTION = "minimax_h3_safe_fp16"
SAFE_FP16_LOADER_VALUE = "fp16_h3_safe"

K_OUT_PROJ = 64.0
K_FC2 = 256.0

_MODULE_PATCH_MARKER = "_raylight_h3_safe_fp16_installed"
_MODEL_PATCH_MARKER = "_raylight_h3_safe_fp16_active"
_PROJECTION_PATCH_MARKER = "_raylight_h3_safe_fp16_projection"
_BLOCK_PATCH_MARKER = "_raylight_h3_safe_fp16"
_MLP_PATCH_MARKER = "_raylight_h3_safe_fp16"
_SAFE_CONSTRUCTION_ACTIVE: ContextVar[bool] = ContextVar(
    "raylight_minimax_h3_safe_fp16_construction",
    default=False,
)


def safe_fp16_requested(model_options: dict[str, Any] | None) -> bool:
    """Return whether model options describe the explicit safe-FP16 mode."""

    if not model_options:
        return False
    return bool(model_options.get(SAFE_FP16_OPTION)) and model_options.get("dtype") is torch.float16


@contextmanager
def minimax_h3_safe_fp16_construction(model_options: dict[str, Any] | None):
    """Bind class-level adapters to one explicit safe-FP16 construction."""

    active = safe_fp16_requested(model_options)
    token = _SAFE_CONSTRUCTION_ACTIVE.set(active)
    try:
        yield active
    finally:
        _SAFE_CONSTRUCTION_ACTIVE.reset(token)


def ray_unet_model_options(weight_dtype: str) -> dict[str, Any]:
    """Translate the RayUNETLoader widget value into serializable model options."""

    if weight_dtype == "fp8_e4m3fn":
        return {"dtype": torch.float8_e4m3fn}
    if weight_dtype == "fp8_e4m3fn_fast":
        return {"dtype": torch.float8_e4m3fn, "fp8_optimizations": True}
    if weight_dtype == "fp8_e5m2":
        return {"dtype": torch.float8_e5m2}
    if weight_dtype == "bf16":
        return {"dtype": torch.bfloat16}
    if weight_dtype == "fp16":
        return {"dtype": torch.float16}
    if weight_dtype == SAFE_FP16_LOADER_VALUE:
        return {"dtype": torch.float16, SAFE_FP16_OPTION: True}
    if weight_dtype == "default":
        return {}
    raise ValueError(f"unsupported RayUNETLoader weight dtype: {weight_dtype}")


def resolve_minimax_h3_safe_fp16_manual_cast(
    model_options: dict[str, Any],
    model_config: Any,
    manual_cast_dtype: torch.dtype | None,
) -> torch.dtype | None:
    """Apply the opt-in FP16 cast override after ComfyUI model detection."""

    if not safe_fp16_requested(model_options):
        return manual_cast_dtype
    config_name = type(model_config).__name__
    if config_name != "MiniMaxH3":
        raise RuntimeError(
            "fp16_h3_safe can only load a ComfyUI MiniMaxH3 model config; "
            f"detected {config_name}."
        )
    return torch.float16


def prepare_minimax_h3_safe_fp16_worker(
    model_options: dict[str, Any],
    *,
    compute_capability: int,
    device_name: str,
    rank: int,
    is_fsdp: bool = True,
    is_xdit: bool = True,
    install_fn: Callable[[], bool] | None = None,
    emit_fn: Callable[[str], Any] = print,
    allow_unsupported: bool = False,
) -> bool:
    """Validate and install safe FP16 before a worker constructs MiniMax H3."""

    requested = bool(model_options.get(SAFE_FP16_OPTION))
    if not requested:
        return False
    if model_options.get("dtype") is not torch.float16:
        raise ValueError(
            f"{SAFE_FP16_OPTION}=True requires dtype=torch.float16; "
            f"received {model_options.get('dtype')!r}"
        )
    if not is_fsdp or not is_xdit:
        raise RuntimeError(
            "MiniMax H3 safe FP16 currently requires Raylight FSDP with USP/xDiT enabled"
        )
    is_v100 = compute_capability == 70 and "V100" in device_name.upper()
    if not is_v100 and not allow_unsupported:
        raise RuntimeError(
            "MiniMax H3 safe FP16 is currently validated only on Tesla V100 "
            f"(compute capability 7.0); received {device_name!r}, capability={compute_capability}."
        )
    if install_fn is None:
        install_fn = install_minimax_h3_safe_fp16_patch
    installed = install_fn()
    capability = f"{compute_capability // 10}.{compute_capability % 10}"
    emit_fn(
        "[Raylight] MiniMax H3 safe FP16 "
        f"rank={rank} active=True newly_installed={installed} "
        f"device={device_name} compute_capability={capability}"
    )
    return True


def safe_linear_projection(
    projection: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run a linear projection in FP16 without constraining its FP32 output range.

    Division and multiplication by a power of two only shift the exponent.  The
    linear operation remains mathematically equivalent, apart from normal FP16
    matrix-multiplication rounding, while outputs above 65,504 remain finite.
    """

    if scale <= 0 or not float(scale).is_integer() or int(scale) & (int(scale) - 1):
        raise ValueError("safe FP16 projection scale must be a positive power of two")
    result = projection((x / scale).to(torch.float16))
    return result.to(torch.float32).mul_(scale)


def safe_attention_output_projection(projection: Any, x: torch.Tensor) -> torch.Tensor:
    """Scale around the final projection call, including post-installed LoRA."""

    if not getattr(projection, _PROJECTION_PATCH_MARKER, False):
        return projection(x)
    return safe_linear_projection(projection, x, K_OUT_PROJ)


def activate_minimax_h3_safe_fp16_model(model: Any) -> bool:
    """Activate instance-level condition/out-projection protections once."""

    if getattr(model, "dtype", None) is not torch.float16:
        return False
    if getattr(model, _MODEL_PATCH_MARKER, False):
        return False

    condition_proj = model.condition_proj
    if not getattr(condition_proj, _PROJECTION_PATCH_MARKER, False):
        original_condition_forward = condition_proj.forward

        @functools.wraps(original_condition_forward)
        def safe_condition_forward(x: torch.Tensor):
            return original_condition_forward(x.to(torch.float32))

        condition_proj.forward = safe_condition_forward
        setattr(condition_proj, _PROJECTION_PATCH_MARKER, True)

    for block in model.blocks:
        setattr(block, _BLOCK_PATCH_MARKER, True)
        setattr(block.mlp, _MLP_PATCH_MARKER, True)
        out_proj = block.attn.out_proj
        setattr(out_proj, _PROJECTION_PATCH_MARKER, True)

    setattr(model, _MODEL_PATCH_MARKER, True)
    return True


def _signature_names(callable_obj: Callable[..., Any]) -> list[str]:
    return list(inspect.signature(callable_obj).parameters)


def validate_minimax_h3_api(mm_module: Any) -> None:
    """Fail clearly when ComfyUI changes an API the safe patch relies on."""

    required = ("MiniMaxH3Model", "MLP", "DiTBlock", "_mod_scale_shift", "_mod_gate")
    missing = [name for name in required if not hasattr(mm_module, name)]
    expected_block = ["self", "x", "t_emb", "mod_segments", "rope_freqs", "transformer_options"]
    block_names = _signature_names(mm_module.DiTBlock.forward) if not missing else []
    mlp_names = _signature_names(mm_module.MLP.forward) if not missing else []
    init_names = _signature_names(mm_module.MiniMaxH3Model.__init__) if not missing else []
    if missing or block_names != expected_block or mlp_names != ["self", "x"] or "dtype" not in init_names:
        detail = (
            f"missing={missing}, DiTBlock.forward={block_names}, "
            f"MLP.forward={mlp_names}, MiniMaxH3Model.__init__={init_names}"
        )
        raise RuntimeError(
            "Unsupported ComfyUI MiniMax H3 API for Raylight safe FP16; "
            "update the compatibility adapter before loading the model. " + detail
        )


def install_minimax_h3_safe_fp16_patch(mm_module: Any | None = None) -> bool:
    """Install the class-level adapter once in the current Ray worker process."""

    if mm_module is None:
        import comfy.ldm.minimax.model as mm_module

    if getattr(mm_module, _MODULE_PATCH_MARKER, False):
        return False
    validate_minimax_h3_api(mm_module)

    original_model_init = mm_module.MiniMaxH3Model.__init__
    original_mlp_forward = mm_module.MLP.forward
    original_block_forward = mm_module.DiTBlock.forward

    @functools.wraps(original_model_init)
    def safe_model_init(self, *args, **kwargs):
        original_model_init(self, *args, **kwargs)
        if _SAFE_CONSTRUCTION_ACTIVE.get():
            activate_minimax_h3_safe_fp16_model(self)

    @functools.wraps(original_mlp_forward)
    def safe_mlp_forward(self, x: torch.Tensor):
        if not getattr(self, _MLP_PATCH_MARKER, False):
            return original_mlp_forward(self, x)
        projected = self.fc1(x)
        gate, up = projected.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate.to(torch.float32)).mul_(up.to(torch.float32))
        return safe_linear_projection(self.fc2, activated, K_FC2)

    @functools.wraps(original_block_forward)
    def safe_block_forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        mod_segments,
        rope_freqs,
        transformer_options={},
    ):
        if not getattr(self, _BLOCK_PATCH_MARKER, False):
            return original_block_forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options)
        residual = x if x.dtype is torch.float32 else x.to(torch.float32)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(t_emb)
        branch = mm_module._mod_scale_shift(
            self.norm1(residual), shift_msa, scale_msa, mod_segments
        ).to(torch.float16)
        attention = self.attn(
            branch, rope_freqs=rope_freqs, transformer_options=transformer_options
        ).to(torch.float32)
        residual = mm_module._mod_gate(residual, gate_msa, attention, mod_segments)
        branch = mm_module._mod_scale_shift(
            self.norm2(residual), shift_mlp, scale_mlp, mod_segments
        ).to(torch.float16)
        mlp = self.mlp(branch).to(torch.float32)
        return mm_module._mod_gate(residual, gate_mlp, mlp, mod_segments)

    mm_module.MiniMaxH3Model.__init__ = safe_model_init
    mm_module.MLP.forward = safe_mlp_forward
    mm_module.DiTBlock.forward = safe_block_forward
    setattr(mm_module, _MODULE_PATCH_MARKER, True)
    return True


__all__ = [
    "K_FC2",
    "K_OUT_PROJ",
    "SAFE_FP16_LOADER_VALUE",
    "SAFE_FP16_OPTION",
    "activate_minimax_h3_safe_fp16_model",
    "install_minimax_h3_safe_fp16_patch",
    "minimax_h3_safe_fp16_construction",
    "prepare_minimax_h3_safe_fp16_worker",
    "ray_unet_model_options",
    "resolve_minimax_h3_safe_fp16_manual_cast",
    "safe_fp16_requested",
    "safe_attention_output_projection",
    "safe_linear_projection",
    "validate_minimax_h3_api",
]
