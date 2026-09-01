import json
import os

import torch

import comfy
from comfy.ldm.minimax.model import pack_audio, patchify_video, rope_rotation_table, time_shift_sigma, unpack_audio, unpatchify_video, PackedLayout, VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP, FRAME_RESCALE
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

import raylight.distributed_modules.attention as xfuser_attn
from raylight.comfy_dist.minimax_h3_fp16 import safe_attention_output_projection
from .director_continuity import prepare_director_continuity
from ..utils import pad_to_world_size


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)
FINAL_PROJECTION_CHUNK_TOKENS = 4096
QK_NORM_CHUNK_TOKENS = 8192
ROPE_CHUNK_TOKENS = 8192
_TRACE_HIGH_WATER = {}


def _trace_cuda_memory(event):
    if os.environ.get("RAYLIGHT_H3_MEMORY_TRACE", "0") != "1" or not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    previous_allocated, previous_reserved = _TRACE_HIGH_WATER.get(device, (-1, -1))
    if peak_allocated <= previous_allocated and peak_reserved <= previous_reserved:
        return
    _TRACE_HIGH_WATER[device] = (
        max(previous_allocated, peak_allocated),
        max(previous_reserved, peak_reserved),
    )
    payload = {
        "event": event,
        "pid": os.getpid(),
        "device": device,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "peak_allocated_bytes": peak_allocated,
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_reserved_bytes": peak_reserved,
    }
    print("[RAYLIGHT_H3_MEMORY] " + json.dumps(payload, sort_keys=True))


def _split_packed_sequence(h, rope_freqs, mod_segments):
    world_size = get_sequence_parallel_world_size()
    local_size = h.shape[0] // world_size
    start = get_sequence_parallel_rank() * local_size
    end = start + local_size
    local_segments = []
    for segment_start, segment_end, row in mod_segments:
        segment_start = max(segment_start, start)
        segment_end = min(segment_end, end)
        if segment_start < segment_end:
            local_segments.append((segment_start - start, segment_end - start, row))
    return h[start:end].clone(), rope_freqs[:, start:end].clone(), local_segments


def _segment_rank_counts(segment, local_size, world_size):
    start, stop, _ = segment
    return [
        max(0, min(stop, (rank + 1) * local_size) - max(start, rank * local_size))
        for rank in range(world_size)
    ]


def _project_local_final_segment(
    final_layer,
    h,
    shift,
    scale,
    segment,
    projector,
    *,
    rank=None,
    world_size=None,
    chunk_tokens=FINAL_PROJECTION_CHUNK_TOKENS,
):
    if rank is None:
        rank = get_sequence_parallel_rank()
    if world_size is None:
        world_size = get_sequence_parallel_world_size()
    if chunk_tokens <= 0:
        raise ValueError("final projection chunk_tokens must be positive")

    local_size = h.shape[0]
    counts = _segment_rank_counts(segment, local_size, world_size)
    count = counts[rank]
    max_count = max(counts)
    output = torch.empty(
        (max_count, projector.weight.shape[0]),
        dtype=torch.float32,
        device=h.device,
    )
    if max_count == 0:
        return output, counts

    segment_start, segment_stop, timestep_row = segment
    rank_start = rank * local_size
    overlap_start = max(segment_start, rank_start)
    overlap_stop = min(segment_stop, rank_start + local_size)
    local_start = overlap_start - rank_start
    local_stop = overlap_stop - rank_start
    local_rows = h[local_start:local_stop] if count else h[:0]
    if count == max_count:
        rows = local_rows
    else:
        rows = h.new_zeros((max_count, h.shape[1]))
        rows[:count].copy_(local_rows)
    segment_scale = 1.0 + scale[timestep_row]
    segment_shift = shift[timestep_row]
    for start in range(0, max_count, chunk_tokens):
        stop = min(start + chunk_tokens, max_count)
        normalized = final_layer.norm(rows[start:stop])
        normalized = (normalized * segment_scale + segment_shift).to(torch.float32)
        projected = projector(normalized)
        output[start:stop].copy_(projected)
        del normalized, projected
    if count == max_count:
        return output, counts
    return output[:count].clone(), counts


def _gather_projected_rows(local_rows, counts, group):
    if len(counts) == 1:
        return local_rows
    max_rows = max(counts)
    padded = local_rows.new_zeros((max_rows, *local_rows.shape[1:]))
    padded[: local_rows.shape[0]].copy_(local_rows)
    gathered = group.all_gather(padded.contiguous(), dim=0)
    gathered = gathered.view(len(counts), max_rows, *local_rows.shape[1:])
    pieces = [gathered[rank, :count] for rank, count in enumerate(counts) if count]
    return torch.cat(pieces, dim=0)


def _normalize_one_in_place(norm, value, *, chunk_tokens):
    manual_fsdp = all(
        callable(getattr(norm, method, None)) for method in ("unshard", "reshard")
    )
    if manual_fsdp:
        norm.unshard()
    try:
        forward = norm.forward if manual_fsdp else norm
        for start in range(0, value.shape[0], chunk_tokens):
            stop = min(start + chunk_tokens, value.shape[0])
            normalized = forward(value[start:stop])
            value[start:stop].copy_(normalized)
            del normalized
    finally:
        if manual_fsdp:
            norm.reshard()


def _normalize_qk_in_place(
    q_norm,
    k_norm,
    q,
    k,
    *,
    chunk_tokens=QK_NORM_CHUNK_TOKENS,
):
    if chunk_tokens <= 0:
        raise ValueError("Q/K norm chunk_tokens must be positive")
    _normalize_one_in_place(q_norm, q, chunk_tokens=chunk_tokens)
    _normalize_one_in_place(k_norm, k, chunk_tokens=chunk_tokens)
    return q, k


def _apply_rope_in_place(
    q,
    k,
    rope_freqs,
    *,
    rot,
    chunk_tokens=ROPE_CHUNK_TOKENS,
):
    if chunk_tokens <= 0:
        raise ValueError("RoPE chunk_tokens must be positive")
    for start in range(0, q.shape[0], chunk_tokens):
        stop = min(start + chunk_tokens, q.shape[0])
        rotated_q, rotated_k = comfy.quant_ops.ck.apply_rope_split_half(
            q[start:stop, ..., :rot],
            k[start:stop, ..., :rot],
            rope_freqs[:, start:stop],
        )
        q[start:stop, ..., :rot].copy_(rotated_q)
        k[start:stop, ..., :rot].copy_(rotated_k)
        del rotated_q, rotated_k
    return q, k


def usp_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    if isinstance(x, list):
        if len(x) != 1:
            raise ValueError("owned MiniMax H3 attention input requires one tensor")
        x = x.pop()
    sequence_length = x.shape[0]
    projected = self.qkv_proj(x)
    del x
    _trace_cuda_memory("attention_qkv_projected")
    q, k, v = projected.split(self.heads * self.head_dim, dim=-1)
    del projected
    q = q.view(sequence_length, self.heads, self.head_dim)
    k = k.view(sequence_length, self.heads, self.head_dim)
    q, k = _normalize_qk_in_place(self.q_norm, self.k_norm, q, k)
    _trace_cuda_memory("attention_qk_normalized_in_place")
    v = v.view(sequence_length, self.heads, self.head_dim)
    if rope_freqs is not None:
        rot = rope_freqs.shape[-3] * 2
        q, k = _apply_rope_in_place(q, k, rope_freqs, rot=rot)
        _trace_cuda_memory("attention_rope_applied")
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if attn_type == xfuser_attn.XFORMERS_MEMORY_EFFICIENT:
        owned = [q, k, v]
        del q, k, v
        out = xfuser_attn.xformers_ulysses_attention_owned(
            owned,
            use_sync=sync_ulysses,
        ).reshape(1, sequence_length, self.heads * self.head_dim)
    else:
        out = xfuser_optimized_attention(q, k, v, self.heads, skip_reshape=True)
    _trace_cuda_memory("attention_context_ready")
    result = safe_attention_output_projection(self.out_proj, out.squeeze(0))
    _trace_cuda_memory("attention_output_projected")
    return result


def usp_dit_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
    video_x, audio_x = x[0], x[1]
    video_dtype = video_x.dtype
    audio_dtype = audio_x.dtype
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype  # compute dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    text_len = context.shape[1]
    # extra_conds prebuilds the layout once per sampling run
    layout = payload.get("layout")
    if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                              keyframes=payload.get("keyframes"),
                              refs=payload.get("refs"),
                              frame_count=payload.get("frame_count"))
    payload = prepare_director_continuity(payload, layout, frame_rescale=FRAME_RESCALE)

    # model_base passes model_sampling.timestep(sigma) = sigma * 1000
    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    # distinct timesteps are known analytically: text/pad follow video, cond rows pin near 1
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    has_vis_cond = any(k in ("cond", "ref_img") for _, _, k in layout.segments)
    has_aud_cond = any(k == "ref_audio" for _, _, k in layout.segments)
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
             "ref_audio": max(t_a, aud_aug)}
    unique_t = sorted({t_v, t_a} | ({seg_t["cond"]} if has_vis_cond else set())
                      | ({seg_t["ref_audio"]} if has_aud_cond else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for a, b, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            # the presentation text span mixes tags (vision pads carry the video modality) split into tag runs
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, b - a + 1):
                if i == b - a or tags[i] != tags[run_start]:
                    mod_segments.append((a + run_start, a + i, row_base + int(tags[run_start])))
                    run_start = i
        else:
            mod_segments.append((a, b, row_base + seg_tag[kind]))

    # embed
    img_update = layout.img_update.to(device)
    audio_update = layout.audio_update.to(device)
    video_rows = patchify_video(video_x.to(torch.float32), self.patch_size)
    audio_rows = pack_audio(audio_x.to(torch.float32))
    cond_video_rows = self._cond_video_rows(payload, device)
    cond_audio_rows = self._cond_audio_rows(payload, device)

    all_video_rows = video_rows
    if cond_video_rows is not None:
        all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], dtype=torch.float32, device=device)
        all_video_rows[~img_update] = cond_video_rows
        all_video_rows[img_update] = video_rows
    all_audio_rows = audio_rows
    if cond_audio_rows is not None:
        all_audio_rows = torch.empty(audio_update.shape[0], audio_rows.shape[1], dtype=torch.float32, device=device)
        all_audio_rows[~audio_update] = cond_audio_rows
        all_audio_rows[audio_update] = audio_rows

    video_embed = self.video_patch_proj(all_video_rows).to(dtype)
    audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
    text_states = context[0]
    if text_states.shape[-1] != self.hidden_size:
        text_states = self.token_refiner(self.condition_proj(text_states),
                                         transformer_options=transformer_options)

    # segments are contiguous: assemble by slices, embed rows follow segment order
    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    voff = aoff = 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind == "text":
            h[a:b] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[a:b] = video_embed[voff:voff + n]
            voff += n
        else:  # ref_audio / audio
            h[a:b] = audio_embed[aoff:aoff + n]
            aoff += n
    del video_x, audio_x, context
    del video_rows, audio_rows, cond_video_rows, cond_audio_rows
    del all_video_rows, all_audio_rows, video_embed, audio_embed, text_states
    del img_update, audio_update
    _trace_cuda_memory("packed_hidden_built")

    t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
    if self.use_adaln_curves:
        # adaln projections consume interpolated coordinates of the time-embedding curve
        table = comfy.model_management.cast_to(self.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)     # t in [0,1] -> fractional grid index, out-of-range t clamps to the curve ends
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)   # lower grid row, max-clamp keeps t=1.0 on the last interval instead of reading past the table
        t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))  # blend the two rows by the fractional part
    else:
        t_emb = self.time_embedder(t_vals).to(dtype)

    # rotation table computed once per forward, consumed by the kitchen split-half rope
    rope_freqs = rope_rotation_table(self.rope_freqs(layout.position_ids, device), dtype)
    # ===================== SP SPLIT ====================== #
    h, _ = pad_to_world_size(h, dim=0)
    rope_freqs, _ = pad_to_world_size(rope_freqs, dim=1)
    h, rope_freqs, mod_segments = _split_packed_sequence(h, rope_freqs, mod_segments)
    _trace_cuda_memory("sequence_split")

    # blocks
    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                     transformer_options=args["transformer_options"])}
            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options},
                {"original_block": block_wrap})["img"]
        else:
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        _trace_cuda_memory(f"block_{i}_end")
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    video_seg = next((a, b, t_row[seg_t["video"]]) for a, b, k in layout.segments if k == "video")
    audio_seg = next((a, b, t_row[seg_t["audio"]]) for a, b, k in layout.segments if k == "audio")
    shift, scale = self.final_layer.adaln_proj(t_emb)
    v_local, v_counts = _project_local_final_segment(
        self.final_layer,
        h,
        shift,
        scale,
        video_seg,
        self.final_layer.video_out,
    )
    _trace_cuda_memory("final_video_local")
    a_local, a_counts = _project_local_final_segment(
        self.final_layer,
        h,
        shift,
        scale,
        audio_seg,
        self.final_layer.audio_out,
    )
    _trace_cuda_memory("final_audio_local")
    del h, rope_freqs, mod_segments, shift, scale
    _trace_cuda_memory("local_hidden_released")
    sp_group = get_sp_group()
    v = _gather_projected_rows(v_local, v_counts, sp_group)
    _trace_cuda_memory("final_video_gathered")
    a = _gather_projected_rows(a_local, a_counts, sp_group)
    _trace_cuda_memory("final_audio_gathered")

    video_out = unpatchify_video(v, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(a)

    return [-video_out.to(video_dtype), -audio_out.to(audio_dtype)]
