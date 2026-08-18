from xfuser.core.long_ctx_attention import (
    xFuserLongContextAttention,
)

import torch.distributed as dist
from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.kernels import AttnType
from yunchang.kernels import select_flash_attn_impl
from .sageattention_hf_patch import ensure_hf_fp8_cuda_kernel, ensure_hf_sm90_kernel

_ATTN_TYPE = None
_SYNC_ULYSSES = None


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


def single_ring_ulysses_attention(
    query,
    key,
    value,
    *,
    ulysses_group,
    attention_kernel,
    all_to_all=SeqAllToAll4D,
    softmax_scale=None,
    use_sync=False,
):
    """Run Ulysses attention without xFuser's redundant world-size-one ring merge.

    xFuser always enters its ring implementation, even when the ring group has
    one member.  That path promotes the full attention result to FP32 solely to
    merge block LSE values, then casts it back to FP16.  A one-member ring has
    nothing to merge, so the promotion and LSE retention are mathematically
    unnecessary and particularly expensive for MiniMax H3's long sequences.
    """

    query_layer = all_to_all.apply(ulysses_group, query, 2, 1, use_sync)
    key_layer = all_to_all.apply(ulysses_group, key, 2, 1, use_sync)
    value_layer = all_to_all.apply(ulysses_group, value, 2, 1, use_sync)
    kernel_result = attention_kernel(
        query_layer,
        key_layer,
        value_layer,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        return_softmax=False,
    )
    context_layer = kernel_result[0] if isinstance(kernel_result, tuple) else kernel_result
    context_layer = context_layer.to(query_layer.dtype)
    return all_to_all.apply(ulysses_group, context_layer, 1, 2, use_sync)


def make_xfuser_attention(attn_type, sync_ulysses):
    print(f"Using XFuser {attn_type} attention, Sync Ulysses: {sync_ulysses}")
    attn = AttnType[attn_type]
    if attn_type == "SAGE_FP8_CUDA":
        ensure_hf_fp8_cuda_kernel()
    elif attn_type == "SAGE_FP8_SM90":
        ensure_hf_sm90_kernel

    xfuser_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn)
    direct_kernel = select_flash_attn_impl(attn, stage="fwd-only")
    single_ring = dist.get_world_size(xfuser_attn.ring_pg) == 1
    if single_ring:
        print("Using direct Ulysses attention for ring_degree=1 (skip redundant FP32 ring merge)")

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
        query = q.transpose(1, 2)
        key = k.transpose(1, 2)
        value = v.transpose(1, 2)

        # Check if using join attention, for MMDiT model
        if join_q is not None:
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
        elif single_ring:
            out = single_ring_ulysses_attention(
                query,
                key,
                value,
                ulysses_group=xfuser_attn.ulysses_pg,
                attention_kernel=direct_kernel,
                softmax_scale=kwargs.get("scale", None),
                use_sync=xfuser_attn.use_sync,
            ).transpose(1, 2)
        else:
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
