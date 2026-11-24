"""Native H3 layout and per-token modulation across supported core versions."""
from functools import lru_cache
import inspect

import torch


@lru_cache(maxsize=8)
def _supports_frame_count(layout_type):
    return "frame_count" in inspect.signature(layout_type).parameters


def build_packed_layout(layout_type, text_len, latent_t, lat_h, lat_w, audio_t, payload):
    options = {"keyframes": payload.get("keyframes"), "refs": payload.get("refs")}
    if _supports_frame_count(layout_type):
        options["frame_count"] = payload.get("frame_count")
    return layout_type(text_len, latent_t, lat_h, lat_w, audio_t, **options)


def _video_mask_rows(mask, latent_t, lat_h, lat_w):
    mask = torch.nn.functional.pad(
        mask, (0, lat_w - mask.shape[-1], 0, lat_h - mask.shape[-2]), mode="replicate",
    )
    values = mask.reshape(latent_t, lat_h // 2, 2, lat_w // 2, 2).amax(dim=(2, 4)).reshape(-1)
    return None if bool((values >= 1.0 - 1e-3).all()) else values


def build_modulation(layout, payload, t_v, t_a, sigma_v, latent_shape,
                     visual_cond_t, audio_cond_t, denoise_mask=None, audio_denoise_mask=None):
    seg_t = {"text": t_v, "video": t_v, "audio": t_a,
             "cond": max(t_v, float(payload.get("visual_cond_noise_aug", visual_cond_t))),
             "ref_img": max(t_v, float(payload.get("visual_cond_noise_aug", visual_cond_t))),
             "cond_audio": max(t_a, float(payload.get("audio_cond_noise_aug", audio_cond_t))),
             "ref_audio": max(t_a, float(payload.get("audio_cond_noise_aug", audio_cond_t)))}
    per_row = {}
    if denoise_mask is not None:
        m = _video_mask_rows(denoise_mask[0, 0].float(), *latent_shape)
        if m is not None:
            values = (1.0 - m * sigma_v.to(m.device)).clamp(max=max(t_v, visual_cond_t))
            if values.unique().numel() == 1:
                seg_t["video"] = float(values[0])
            else:
                per_row["video"] = values
    if audio_denoise_mask is not None:
        m = audio_denoise_mask[0, 0].float().reshape(-1)
        if not bool((m >= 1.0 - 1e-3).all()):
            values = (1.0 - m * (1.0 - t_a)).clamp(max=max(t_a, audio_cond_t))
            if values.unique().numel() == 1:
                seg_t["audio"] = float(values[0])
            else:
                per_row["audio"] = values
    unique = {t_v, t_a} | {seg_t[kind] for _, _, kind in layout.segments}
    for values in per_row.values():
        unique.update(values.unique().tolist())
    unique_t = sorted(unique)
    rows = {t: i for i, t in enumerate(unique_t)}
    tags = {"text": 1, "video": 0, "audio": 2, "cond": 0,
            "cond_audio": 2, "ref_img": 0, "ref_audio": 2}
    indices = {}
    for kind, values in per_row.items():
        levels = values.unique()
        base = torch.tensor([rows[t] for t in levels.tolist()], dtype=torch.long, device=values.device)
        indices[kind] = base[torch.searchsorted(levels, values)]
    segments = []
    text_tags = payload.get("text_token_tags")
    for a, b, kind in layout.segments:
        row_base = rows[seg_t[kind]] * 3
        if kind == "text" and text_tags is not None:
            token_tags = text_tags.reshape(-1).tolist()
            start = 0
            for stop in range(1, b - a + 1):
                if stop == b - a or token_tags[stop] != token_tags[start]:
                    segments.append((a + start, a + stop, row_base + int(token_tags[start])))
                    start = stop
        elif kind in indices:
            segments.append((a, b, indices[kind] * 3 + tags[kind]))
        else:
            segments.append((a, b, row_base + tags[kind]))
    final_segments = []
    for kind in ("video", "audio"):
        a, b, _ = next(segment for segment in layout.segments if segment[2] == kind)
        final_segments.append((a, b, indices.get(kind, rows[seg_t[kind]])))
    return unique_t, segments, *final_segments
