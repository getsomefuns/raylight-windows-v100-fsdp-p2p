import os

from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

import torch
import torch.distributed as dist
import xformers.ops as xops
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import PROCESS_GROUP
from yunchang.kernels import AttnType
from .sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

_ATTN_TYPE = None
_SYNC_ULYSSES = None
XFORMERS_MEMORY_EFFICIENT = "XFORMERS_MEMORY_EFFICIENT"
XFORMERS_HEAD_CHUNK = 7
XFORMERS_QUERY_CHUNK_TOKENS = 16384


def _windows_p2p_all_to_all_4d_owned(
    owned,
    *,
    scatter_idx,
    gather_idx,
    group,
    use_sync,
):
    """Consume one 4D tensor and reuse its communication layout buffer.

    The Windows P2P endpoint copies every outbound chunk into its fixed staging
    buffer before it overwrites the corresponding receive range, so equal-size
    two-rank all-to-all is safe with the same contiguous tensor as input/output.
    """

    if len(owned) != 1:
        raise ValueError("owned all-to-all requires exactly one tensor")
    value = owned.pop()
    if value.dim() != 4:
        raise ValueError(f"owned all-to-all requires a 4D tensor, got {value.dim()}D")
    world_size = dist.get_world_size(group)
    if world_size != 2:
        raise ValueError(f"Windows P2P owned all-to-all requires two ranks, got {world_size}")

    if scatter_idx == 2 and gather_idx == 1:
        batch, local_sequence, heads, head_dim = value.shape
        if heads % world_size:
            raise ValueError("attention heads must be divisible by the Ulysses degree")
        local_heads = heads // world_size
        layout = (
            value.reshape(batch, local_sequence, world_size, local_heads, head_dim)
            .transpose(0, 2)
            .contiguous()
        )
        del value
        dist.all_to_all_single(layout, layout, group=group)
        if use_sync:
            torch.cuda.synchronize()
        return layout.reshape(local_sequence * world_size, batch, local_heads, head_dim).transpose(0, 1)

    if scatter_idx == 1 and gather_idx == 2:
        batch, sequence, local_heads, head_dim = value.shape
        if sequence % world_size:
            raise ValueError("attention sequence must be divisible by the Ulysses degree")
        local_sequence = sequence // world_size
        heads = local_heads * world_size
        layout = (
            value.reshape(batch, world_size, local_sequence, local_heads, head_dim)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(world_size, local_heads, local_sequence, batch, head_dim)
        )
        del value
        dist.all_to_all_single(layout, layout, group=group)
        if use_sync:
            torch.cuda.synchronize()
        return layout.reshape(heads, local_sequence, batch, head_dim).transpose(0, 2)

    raise ValueError("owned all-to-all supports only (2, 1) and (1, 2)")


def _ulysses_exchange_owned(owned, *, scatter_idx, gather_idx, group, use_sync):
    value = owned.pop(0).transpose(1, 2)
    if os.name == "nt" and os.environ.get("RAYLIGHT_WINDOWS_P2P", "0") == "1":
        return _windows_p2p_all_to_all_4d_owned(
            [value],
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            group=group,
            use_sync=use_sync,
        )
    return SeqAllToAll4D.apply(group, value, scatter_idx, gather_idx, use_sync)


def xfuser_attention_choices():
    return [member.name for member in AttnType] + [XFORMERS_MEMORY_EFFICIENT]


def set_attn_type(attn):
    global _ATTN_TYPE
    _ATTN_TYPE = attn


def get_attn_type():
    if _ATTN_TYPE is None:
        raise RuntimeError("_ATTN_TYPE is not initialized")
    else:
        return _ATTN_TYPE


def set_sync_ulysses(is_sync):
    global _SYNC_ULYSSES
    _SYNC_ULYSSES = is_sync


def get_sync_ulysses():
    if _SYNC_ULYSSES is None:
        raise RuntimeError("_SYNC_ULYSSES variable is not initialized")
    else:
        return _SYNC_ULYSSES


def xformers_ulysses_attention_owned(
    qkv,
    *,
    use_sync,
    scale=None,
    causal=False,
    chunk_heads=XFORMERS_HEAD_CHUNK,
    chunk_query_tokens=XFORMERS_QUERY_CHUNK_TOKENS,
    transformer_options=None,
):
    """Consume Q/K/V references and return the rank-local Ulysses output."""

    if len(qkv) != 3:
        raise ValueError("owned Ulysses attention requires exactly Q, K, and V")
    if chunk_heads <= 0:
        raise ValueError("XFormers attention chunk_heads must be positive")
    if chunk_query_tokens <= 0:
        raise ValueError("XFormers attention chunk_query_tokens must be positive")
    ring_group = PROCESS_GROUP.RING_PG
    if ring_group is None or dist.get_world_size(ring_group) != 1:
        raise ValueError("XFORMERS_MEMORY_EFFICIENT requires ring degree 1")
    ulysses_group = PROCESS_GROUP.ULYSSES_PG
    if ulysses_group is None:
        raise ValueError("XFORMERS_MEMORY_EFFICIENT requires an Ulysses process group")

    query_layer = _ulysses_exchange_owned(
        qkv, scatter_idx=2, gather_idx=1, group=ulysses_group, use_sync=use_sync
    )
    key_layer = _ulysses_exchange_owned(
        qkv, scatter_idx=2, gather_idx=1, group=ulysses_group, use_sync=use_sync
    )
    value_layer = _ulysses_exchange_owned(
        qkv, scatter_idx=2, gather_idx=1, group=ulysses_group, use_sync=use_sync
    )
    options = transformer_options or {}

    def dense_local(query_bhsd, key_bhsd, value_bhsd, _heads, **_kwargs):
        dense = _xformers_local_attention(
            query_bhsd.transpose(1, 2),
            key_bhsd.transpose(1, 2),
            value_bhsd.transpose(1, 2),
            scale=scale,
            causal=causal,
            chunk_heads=chunk_heads,
            chunk_query_tokens=chunk_query_tokens,
        )
        return dense.reshape(dense.shape[0], dense.shape[1], -1)

    override = options.get("optimized_attention_override")
    if override is not None and getattr(override, "_solattn_override", False):
        local_heads = query_layer.shape[2]
        flat_context = override(
            dense_local,
            query_layer.transpose(1, 2),
            key_layer.transpose(1, 2),
            value_layer.transpose(1, 2),
            local_heads,
            mask=None,
            skip_reshape=True,
            skip_output_reshape=False,
            scale=scale,
            causal=causal,
            transformer_options=options,
        )
        context_layer = flat_context.reshape(
            query_layer.shape[0],
            query_layer.shape[1],
            local_heads,
            query_layer.shape[3],
        )
        del flat_context
    else:
        context_layer = _xformers_local_attention(
            query_layer,
            key_layer,
            value_layer,
            scale=scale,
            causal=causal,
            chunk_heads=chunk_heads,
            chunk_query_tokens=chunk_query_tokens,
            output=query_layer,
        )
    del query_layer, key_layer, value_layer
    output = _ulysses_exchange_owned(
        [context_layer.transpose(1, 2)],
        scatter_idx=1,
        gather_idx=2,
        group=ulysses_group,
        use_sync=use_sync,
    )
    return output


def _xformers_local_attention(
    query_layer,
    key_layer,
    value_layer,
    *,
    scale,
    causal,
    chunk_heads,
    chunk_query_tokens,
    output=None,
):
    attention_bias = xops.LowerTriangularMask() if causal else None
    context_layer = torch.empty_like(query_layer) if output is None else output
    if causal and chunk_query_tokens < query_layer.shape[1]:
        raise ValueError("query chunking does not support causal attention offsets")
    for query_start in range(0, query_layer.shape[1], chunk_query_tokens):
        query_stop = min(query_start + chunk_query_tokens, query_layer.shape[1])
        for head_start in range(0, query_layer.shape[2], chunk_heads):
            head_stop = min(head_start + chunk_heads, query_layer.shape[2])
            context_chunk = xops.memory_efficient_attention(
                query_layer[:, query_start:query_stop, head_start:head_stop],
                key_layer[:, :, head_start:head_stop],
                value_layer[:, :, head_start:head_stop],
                attn_bias=attention_bias,
                p=0.0,
                scale=scale,
            )
            context_layer[
                :, query_start:query_stop, head_start:head_stop
            ].copy_(context_chunk)
            del context_chunk
    return context_layer


def make_xfuser_attention(attn_type, sync_ulysses):
    print(f"Using XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}")
    use_xformers_ulysses = attn_type == XFORMERS_MEMORY_EFFICIENT
    attn = AttnType.TORCH_EFFICIENT if use_xformers_ulysses else AttnType[attn_type]
    if attn_type == "SAGE_FP8_CUDA":
        ensure_hf_fp8_cuda_kernel()
    elif attn_type == "SAGE_FP8_SM90":
        ensure_hf_sm90_kernel

    xfuser_attn = None
    if not use_xformers_ulysses:
        xfuser_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn)

    def _attention_xfuser_unmask(
            q,
            k,
            v,
            heads,
            join_q=None,
            join_k=None,
            join_v=None,
            mask=None,
            attn_precision=None,
            skip_reshape=False,
            skip_output_reshape=False,
            *args,
            **kwargs):

        if skip_reshape:
            b, _, _, dim_head = q.shape
            if join_q is not None:
                j_b, _, _, j_dim_head = join_q.shape
        else:
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = map(
                lambda t: t.view(b, -1, heads, dim_head).transpose(1, 2),
                (q, k, v),
            )
            if join_q is not None:
                j_b, _, j_dim_head = join_q.shape
                j_dim_head //= heads
                join_q, join_k, join_v = map(
                    lambda t: t.view(j_b, -1, heads, j_dim_head).transpose(1, 2),
                    (join_q, join_k, join_v),
                )

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
        if use_xformers_ulysses:
            if join_q is not None or mask is not None:
                raise ValueError(
                    "XFORMERS_MEMORY_EFFICIENT supports unmasked non-joint attention only"
                )
            owned = [q, k, v]
            del q, k, v
            output = xformers_ulysses_attention_owned(
                owned,
                use_sync=sync_ulysses,
                scale=kwargs.get("scale", None),
                causal=kwargs.get("causal", False),
                transformer_options=kwargs.get("transformer_options"),
            )
            out = output.transpose(1, 2)

        # Check if using join attention, for MMDiT model
        elif join_q is not None:
            query = q.transpose(1, 2)
            key = k.transpose(1, 2)
            value = v.transpose(1, 2)
            out = xfuser_attn(
                None,
                query,
                key,
                value,
                joint_strategy="rear",
                joint_tensor_query=join_q.transpose(1, 2),
                joint_tensor_key=join_k.transpose(1, 2),
                joint_tensor_value=join_v.transpose(1, 2),
                softmax_scale=kwargs.get("scale", None),
            ).transpose(1, 2)
        else:
            query = q.transpose(1, 2)
            key = k.transpose(1, 2)
            value = v.transpose(1, 2)
            out = xfuser_attn(
                None,
                query,
                key,
                value,
                softmax_scale=kwargs.get("scale", None),
            ).transpose(1, 2)
        if not skip_output_reshape:
            out = (
                out.transpose(1, 2).reshape(b, -1, heads * dim_head)
            )
        return out

    return _attention_xfuser_unmask
