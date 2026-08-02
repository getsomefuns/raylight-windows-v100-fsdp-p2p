import torch

import comfy
from comfy.ldm.minimax.model import pack_audio, patchify_video, rope_rotation_table, time_shift_sigma, time_shift_slope, unpack_audio, unpatchify_video
from xfuser.core.distributed import get_sequence_parallel_rank, get_sequence_parallel_world_size, get_sp_group

import raylight.distributed_modules.attention as xfuser_attn


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def _split_packed_sequence(h, rope_freqs, mod_segments):
    world_size = get_sequence_parallel_world_size()
    if h.shape[0] % world_size:
        raise ValueError(
            f"MiniMax H3 packed sequence length {h.shape[0]} must be divisible by the USP degree {world_size}. "
            "H3 uses unmasked attention, so padding would change the result."
        )

    local_size = h.shape[0] // world_size
    start = get_sequence_parallel_rank() * local_size
    end = start + local_size
    local_segments = []
    for segment_start, segment_end, row in mod_segments:
        segment_start = max(segment_start, start)
        segment_end = min(segment_end, end)
        if segment_start < segment_end:
            local_segments.append((segment_start - start, segment_end - start, row))
    return h[start:end], rope_freqs[start:end], local_segments


def usp_attn_forward(self, x, rope_freqs=None, transformer_options={}):
    sequence_length = x.shape[0]
    q, k, v = self.qkv_proj(x).split(self.heads * self.head_dim, dim=-1)
    q = self.q_norm(q.view(sequence_length, self.heads, self.head_dim))
    k = self.k_norm(k.view(sequence_length, self.heads, self.head_dim))
    v = v.view(sequence_length, self.heads, self.head_dim)
    if rope_freqs is not None:
        rot = rope_freqs.shape[-3] * 2
        q[..., :rot], k[..., :rot] = comfy.quant_ops.ck.apply_rope_split_half(q[..., :rot], k[..., :rot], rope_freqs)
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    out = xfuser_optimized_attention(q, k, v, self.heads, skip_reshape=True)
    return self.out_proj(out.squeeze(0))


def usp_dit_forward(self, x, timestep, context, transformer_options={}, minimax_payload=None, **kwargs):
    video_x, audio_x = x[0], x[1]
    orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
    if video_x.shape[0] != 1:
        raise ValueError("MiniMax H3 supports batch size 1")
    payload = minimax_payload or {}
    device = video_x.device
    dtype = context.dtype

    latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
    audio_t = audio_x.shape[-1]
    layout = self._layout(context.shape[1], latent_t, lat_h, lat_w, audio_t, payload)

    shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
    shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
    sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))

    vis_aug = float(payload.get("visual_cond_noise_aug", 0.999))
    aud_aug = float(payload.get("audio_cond_noise_aug", 1.0))
    has_vis_cond = any(kind in ("cond", "ref_img") for _, _, kind in layout.segments)
    has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
    video_t = 0.999 if transformer_options.get("minimax_h3_clean_video", False) else t_v
    seg_t = {"text": t_v, "video": video_t, "audio": t_a, "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug), "ref_audio": max(t_a, aud_aug)}
    unique_t = sorted({t_v, t_a, video_t} | ({seg_t["cond"]} if has_vis_cond else set()) | ({seg_t["ref_audio"]} if has_aud_cond else set()))
    t_row = {t: i for i, t in enumerate(unique_t)}
    seg_tag = {"text": 1, "video": 0, "audio": 2, "cond": 0, "ref_img": 0, "ref_audio": 2}

    text_tags = payload.get("text_token_tags")
    mod_segments = []
    for start, end, kind in layout.segments:
        row_base = t_row[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            tags = text_tags.view(-1).tolist()
            run_start = 0
            for i in range(1, end - start + 1):
                if i == end - start or tags[i] != tags[run_start]:
                    mod_segments.append((start + run_start, start + i, row_base + int(tags[run_start])))
                    run_start = i
        else:
            mod_segments.append((start, end, row_base + seg_tag[kind]))

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
        text_states = self.token_refiner(self.condition_proj(text_states), transformer_options=transformer_options)

    h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
    video_offset = audio_offset = 0
    for start, end, kind in layout.segments:
        length = end - start
        if kind == "text":
            h[start:end] = text_states
        elif kind in ("cond", "ref_img", "video"):
            h[start:end] = video_embed[video_offset:video_offset + length]
            video_offset += length
        else:
            h[start:end] = audio_embed[audio_offset:audio_offset + length]
            audio_offset += length

    timestep_values = torch.tensor(unique_t, dtype=torch.float32, device=device)
    modulation_provider = transformer_options.get("minimax_h3_modulation")
    if modulation_provider is None:
        if self.split_modulation:
            raise RuntimeError("MiniMax H3 split transformer requires its modulation model")
        t_emb = self.time_embedder(timestep_values, frame_rate=transformer_options.get("minimax_h3_frame_rate")).to(dtype)
        block_modulation = final_modulation = None
    else:
        if transformer_options.get("patches_replace", {}).get("dit", {}):
            raise RuntimeError("MiniMax H3 split modulation does not support block replacement patches")
        t_emb = None
        block_modulation, final_modulation = modulation_provider(timestep_values)

    rope_frame_rate = transformer_options.get("minimax_h3_rope_frame_rate")
    rope_end_timestep = transformer_options.get("minimax_h3_rope_end_timestep", 1.0)
    rope_scale = 1.0
    if rope_frame_rate is not None and rope_frame_rate != 24.0 and t_v <= rope_end_timestep:
        rope_scale = 24.0 / rope_frame_rate
        sigma_profile = transformer_options.get("minimax_h3_rope_sigma_profile", "constant")
        if sigma_profile != "constant":
            sigma_end = transformer_options.get("minimax_h3_rope_sigma_end", 0.0)
            sigma_weight = max(0.0, min(1.0, (float(sigma_v) - sigma_end) / (1.0 - sigma_end)))
            if sigma_profile == "smoothstep":
                sigma_weight = sigma_weight * sigma_weight * (3.0 - 2.0 * sigma_weight)
            rope_scale = 1.0 + (rope_scale - 1.0) * sigma_weight
    rope_freqs = rope_rotation_table(
        self.rope_freqs(
            layout.position_ids,
            device,
            video_rows=(layout.token_tags == 0).to(device),
            temporal_scale=rope_scale,
            low_frequency_count=transformer_options.get("minimax_h3_rope_low_frequency_count", 16),
            frequency_profile=transformer_options.get("minimax_h3_rope_frequency_profile", "hard"),
        ),
        dtype,
    )
    h, rope_freqs, mod_segments = _split_packed_sequence(h, rope_freqs, mod_segments)

    patches_replace = transformer_options.get("patches_replace", {})
    blocks_replace = patches_replace.get("dit", {})
    prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks), device, transformer_options)
    for i, block in enumerate(self.blocks):
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
        if ("double_block", i) in blocks_replace:
            def block_wrap(args):
                return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=args["transformer_options"], modulation=None)}

            h = blocks_replace[("double_block", i)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs, "transformer_options": transformer_options},
                {"original_block": block_wrap},
            )["img"]
        else:
            modulation = None if block_modulation is None else block_modulation(i)
            h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options, modulation=modulation)
    if prefetch_queue is not None:
        comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

    h = get_sp_group().all_gather(h.contiguous(), dim=0)
    video_seg = next((start, end, t_row[seg_t["video"]]) for start, end, kind in layout.segments if kind == "video")
    audio_seg = next((start, end, t_row[seg_t["audio"]]) for start, end, kind in layout.segments if kind == "audio")
    video_rows, audio_rows = self.final_layer(h, t_emb, video_seg, audio_seg, modulation=final_modulation)
    video_out = unpatchify_video(video_rows, latent_t, lat_h // 2, lat_w // 2, self.latents_dim, self.patch_size)
    video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
    audio_out = unpack_audio(audio_rows)
    slope_a = time_shift_slope(sigma_v, shift_v, shift_a).to(audio_out.dtype)
    return [-video_out.to(video_x.dtype), (-slope_a) * audio_out.to(audio_x.dtype)]
