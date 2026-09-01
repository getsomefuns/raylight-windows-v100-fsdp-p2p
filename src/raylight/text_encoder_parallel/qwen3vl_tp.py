"""Qwen3-VL language-block tensor parallelism for native-Windows dual V100."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .linear import ColumnParallelLinear, RowParallelLinear


@dataclass(frozen=True)
class TP2Qwen3VLConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    rope_dims: tuple[int, int, int] | None
    interleaved_mrope: bool

    def __post_init__(self) -> None:
        if self.num_attention_heads % 2 or self.num_key_value_heads % 2:
            raise ValueError("TP2 requires attention and key/value head counts divisible by 2")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads for GQA")
        if self.intermediate_size % 2:
            raise ValueError("TP2 requires intermediate_size divisible by 2")
        if min(self.hidden_size, self.intermediate_size, self.head_dim) <= 0:
            raise ValueError("Qwen3-VL dimensions must be positive")

    @property
    def local_attention_heads(self) -> int:
        return self.num_attention_heads // 2

    @property
    def local_key_value_heads(self) -> int:
        return self.num_key_value_heads // 2

    @property
    def attention_inner_size(self) -> int:
        return self.num_attention_heads * self.head_dim


MINIMAX_H3_QWEN3VL_CONFIG = TP2Qwen3VLConfig(
    hidden_size=5120,
    intermediate_size=25600,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    rms_norm_eps=1e-6,
    rope_theta=5000000.0,
    rope_dims=(24, 20, 20),
    interleaved_mrope=True,
)


def build_qwen3vl_rope(
    config: TP2Qwen3VLConfig,
    position_ids: torch.Tensor,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce ComfyUI's Qwen3-VL interleaved MRoPE layout."""

    position_ids = position_ids.to(device=device)
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0)
    numerator = torch.arange(0, config.head_dim, 2, device=device).float()
    inv_freq = 1.0 / (config.rope_theta ** (numerator / config.head_dim))
    inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)

    if config.rope_dims is not None and position_ids.shape[0] > 1 and config.interleaved_mrope:
        freqs_interleaved = freqs[0].clone()
        for axis_idx, offset in ((1, 1), (2, 2)):
            length = config.rope_dims[axis_idx] * 3
            indices = slice(offset, length, 3)
            freqs_interleaved[..., indices] = freqs[axis_idx, ..., indices]
        embedding = torch.cat((freqs_interleaved, freqs_interleaved), dim=-1)
        cosine = embedding.cos().unsqueeze(0)
        sine = embedding.sin().unsqueeze(0)
    else:
        embedding = torch.cat((freqs, freqs), dim=-1)
        cosine = embedding.cos()
        sine = embedding.sin()
        if config.rope_dims is not None and position_ids.shape[0] > 1:
            sections = config.rope_dims * 2
            cosine = torch.cat([part[index % 3] for index, part in enumerate(cosine.split(sections, dim=-1))], dim=-1).unsqueeze(0)
            sine = torch.cat([part[index % 3] for index, part in enumerate(sine.split(sections, dim=-1))], dim=-1).unsqueeze(0)
        else:
            cosine = cosine.unsqueeze(1)
            sine = sine.unsqueeze(1)
    split = sine.shape[-1] // 2
    return cosine, sine[..., :split], -sine[..., split:]


def apply_qwen3vl_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    original_dtype = query.dtype
    cosine, sine, negative_sine = rope

    query_out = query * cosine
    query_split = query_out.shape[-1] // 2
    query_out[..., :query_split].addcmul_(query[..., query_split:], negative_sine)
    query_out[..., query_split:].addcmul_(query[..., :query_split], sine)

    key_out = key * cosine
    key_split = key_out.shape[-1] // 2
    key_out[..., :key_split].addcmul_(key[..., key_split:], negative_sine)
    key_out[..., key_split:].addcmul_(key[..., :key_split], sine)
    return query_out.to(original_dtype), key_out.to(original_dtype)


def _norm_pair(weights, *, name: str, expected: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(weights, (tuple, list)) or len(weights) != 2:
        raise ValueError(f"{name} must contain rank 0 and rank 1 weights")
    pair = (weights[0], weights[1])
    for rank, weight in enumerate(pair):
        if tuple(weight.shape) != (expected,):
            raise ValueError(f"rank {rank} {name} shape must be ({expected},)")
    return pair


class TP2Qwen3VLBlock:
    """One replicated-residual, head/channel-sharded Qwen3-VL decoder block."""

    def __init__(
        self,
        config: TP2Qwen3VLConfig,
        *,
        q_proj: ColumnParallelLinear,
        k_proj: ColumnParallelLinear,
        v_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
        gate_proj: ColumnParallelLinear,
        up_proj: ColumnParallelLinear,
        down_proj: RowParallelLinear,
        input_layernorm,
        post_attention_layernorm,
        q_norm,
        k_norm,
        attention_forward: Callable[..., torch.Tensor],
    ) -> None:
        self.config = config
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.input_layernorm = _norm_pair(input_layernorm, name="input_layernorm", expected=config.hidden_size)
        self.post_attention_layernorm = _norm_pair(
            post_attention_layernorm, name="post_attention_layernorm", expected=config.hidden_size
        )
        self.q_norm = _norm_pair(q_norm, name="q_norm", expected=config.head_dim)
        self.k_norm = _norm_pair(k_norm, name="k_norm", expected=config.head_dim)
        self.attention_forward = attention_forward
        self.executor = q_proj.executor
        all_linears = (k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
        if any(linear.executor is not self.executor for linear in all_linears):
            raise ValueError("all TP2 block linears must share one dual-device executor")
        if o_proj.collective is not down_proj.collective:
            raise ValueError("o_proj and down_proj must share one ordered CUDA peer collective")
        self.collective = o_proj.collective
        self.execution_profiles: list[tuple[str, Any]] = []
        self._validate_linear_shapes()

    def _validate_linear_shapes(self) -> None:
        config = self.config
        expected = {
            "q_proj": (config.local_attention_heads * config.head_dim, config.hidden_size),
            "k_proj": (config.local_key_value_heads * config.head_dim, config.hidden_size),
            "v_proj": (config.local_key_value_heads * config.head_dim, config.hidden_size),
            "o_proj": (config.hidden_size, config.attention_inner_size // 2),
            "gate_proj": (config.intermediate_size // 2, config.hidden_size),
            "up_proj": (config.intermediate_size // 2, config.hidden_size),
            "down_proj": (config.hidden_size, config.intermediate_size // 2),
        }
        for name, shape in expected.items():
            linear = getattr(self, name)
            if any(tuple(shard.logical_shape) != shape for shard in linear.shards):
                raise ValueError(f"{name} local shard shape must be {shape}")

    def _run_pair(self, name: str, functions):
        outputs, stats = self.executor.run_functions(functions)
        self.execution_profiles.append((name, stats))
        return outputs

    def forward(
        self,
        hidden_rank0: torch.Tensor,
        hidden_rank1: torch.Tensor,
        position_ids: torch.Tensor,
        *,
        operation_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(operation_id, bool) or not isinstance(operation_id, int) or operation_id <= 0:
            raise ValueError("operation_id must be a positive integer")
        hidden = (hidden_rank0, hidden_rank1)
        if tuple(hidden_rank0.shape) != tuple(hidden_rank1.shape):
            raise ValueError("rank hidden states must have identical shapes")
        if hidden_rank0.shape[-1] != self.config.hidden_size:
            raise ValueError(f"hidden width must be {self.config.hidden_size}")
        self.execution_profiles = []

        residual = hidden
        normed = self._run_pair(
            "input_layernorm",
            tuple(
                lambda rank=rank: F.rms_norm(
                    hidden[rank], (self.config.hidden_size,), self.input_layernorm[rank], self.config.rms_norm_eps
                )
                for rank in (0, 1)
            ),
        )
        q = self.q_proj.forward(normed)
        k = self.k_proj.forward(normed)
        v = self.v_proj.forward(normed)
        batch, sequence = hidden_rank0.shape[:2]

        def attend(rank: int):
            query = q[rank].view(batch, sequence, self.config.local_attention_heads, self.config.head_dim).transpose(1, 2)
            key = k[rank].view(batch, sequence, self.config.local_key_value_heads, self.config.head_dim).transpose(1, 2)
            value = v[rank].view(batch, sequence, self.config.local_key_value_heads, self.config.head_dim).transpose(1, 2)
            query = F.rms_norm(query, (self.config.head_dim,), self.q_norm[rank], self.config.rms_norm_eps)
            key = F.rms_norm(key, (self.config.head_dim,), self.k_norm[rank], self.config.rms_norm_eps)
            rope = build_qwen3vl_rope(self.config, position_ids, device=query.device)
            query, key = apply_qwen3vl_rope(query, key, rope)
            attended = self.attention_forward(
                query,
                key,
                value,
                is_causal=True,
            )
            return attended.transpose(1, 2).reshape(batch, sequence, -1).contiguous()

        attended = self._run_pair("attention", tuple(lambda rank=rank: attend(rank) for rank in (0, 1)))
        attention_output = self.o_proj.forward(attended, operation_id=operation_id)
        hidden = self._run_pair(
            "attention_residual",
            tuple(lambda rank=rank: residual[rank] + attention_output[rank] for rank in (0, 1)),
        )

        residual = hidden
        normed = self._run_pair(
            "post_attention_layernorm",
            tuple(
                lambda rank=rank: F.rms_norm(
                    hidden[rank],
                    (self.config.hidden_size,),
                    self.post_attention_layernorm[rank],
                    self.config.rms_norm_eps,
                )
                for rank in (0, 1)
            ),
        )
        gate = self.gate_proj.forward(normed)
        up = self.up_proj.forward(normed)
        activated = self._run_pair(
            "mlp_activation",
            tuple(lambda rank=rank: F.silu(gate[rank]) * up[rank] for rank in (0, 1)),
        )
        mlp_output = self.down_proj.forward(activated, operation_id=operation_id + 1)
        return self._run_pair(
            "mlp_residual",
            tuple(lambda rank=rank: residual[rank] + mlp_output[rank] for rank in (0, 1)),
        )

    __call__ = forward


__all__ = [
    "MINIMAX_H3_QWEN3VL_CONFIG",
    "TP2Qwen3VLBlock",
    "TP2Qwen3VLConfig",
    "apply_qwen3vl_rope",
    "build_qwen3vl_rope",
]
