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
):
    """Consume Q/K/V references and return the rank-local Ulysses output."""

    if len(qkv) != 3:
        raise ValueError("owned Ulysses attention requires exactly Q, K, and V")
    if chunk_heads <= 0:
        raise ValueError("XFormers attention chunk_heads must be positive")
    ring_group = PROCESS_GROUP.RING_PG
    if ring_group is None or dist.get_world_size(ring_group) != 1:
        raise ValueError("XFORMERS_MEMORY_EFFICIENT requires ring degree 1")
    ulysses_group = PROCESS_GROUP.ULYSSES_PG
    if ulysses_group is None:
        raise ValueError("XFORMERS_MEMORY_EFFICIENT requires an Ulysses process group")

    query = qkv.pop(0).transpose(1, 2)
    query_layer = SeqAllToAll4D.apply(ulysses_group, query, 2, 1, use_sync)
    del query
    key = qkv.pop(0).transpose(1, 2)
    key_layer = SeqAllToAll4D.apply(ulysses_group, key, 2, 1, use_sync)
    del key
    value = qkv.pop(0).transpose(1, 2)
    value_layer = SeqAllToAll4D.apply(ulysses_group, value, 2, 1, use_sync)
    del value
    attention_bias = xops.LowerTriangularMask() if causal else None
    context_layer = torch.empty_like(query_layer)
    for start in range(0, query_layer.shape[2], chunk_heads):
        stop = min(start + chunk_heads, query_layer.shape[2])
        context_chunk = xops.memory_efficient_attention(
            query_layer[:, :, start:stop],
            key_layer[:, :, start:stop],
            value_layer[:, :, start:stop],
            attn_bias=attention_bias,
            p=0.0,
            scale=scale,
        )
        context_layer[:, :, start:stop].copy_(context_chunk)
        del context_chunk
    del query_layer, key_layer, value_layer
    output = SeqAllToAll4D.apply(ulysses_group, context_layer, 1, 2, use_sync)
    del context_layer
    return output


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
