import torch
import torch.nn.functional as F
from einops import rearrange
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

import comfy.ldm.common_dit
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from ..utils import pad_to_world_size
import raylight.distributed_modules.attention as xfuser_attn


attn_type = xfuser_attn.get_attn_type()
sync_ulysses = xfuser_attn.get_sync_ulysses()
xfuser_optimized_attention = xfuser_attn.make_xfuser_attention(attn_type, sync_ulysses)


def usp_dit_forward(self, x, timesteps, context, attention_mask=None, transformer_options={}, **kwargs):
    temporal = x.ndim == 5
    if temporal:
        b5, c5, t5, h5, w5 = x.shape
        x = x.reshape(b5 * t5, c5, h5, w5)

    bs, c, H_orig, W_orig = x.shape
    patch = self.patch
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
    H, W = x.shape[-2], x.shape[-1]
    h_, w_ = H // patch, W // patch

    context = self._unpack_context(context)

    img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    img = self.first(img)

    t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
    tvec = self.tproj(t)

    txtpos = torch.zeros(bs, context.shape[1], 3, device=img.device, dtype=torch.float32)
    imgids = torch.zeros(h_, w_, 3, device=img.device, dtype=torch.float32)
    imgids[..., 1] = torch.arange(h_, device=img.device, dtype=torch.float32)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device, dtype=torch.float32)[None, :]
    imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)

    context, _ = pad_to_world_size(context, dim=1)
    txtpos, _ = pad_to_world_size(txtpos, dim=1)
    img, img_orig_size = pad_to_world_size(img, dim=1)
    imgpos, _ = pad_to_world_size(imgpos, dim=1)

    sp_rank = get_sequence_parallel_rank()
    sp_world_size = get_sequence_parallel_world_size()

    # ===================== SP SPLIT ====================== #
    context = torch.chunk(context, sp_world_size, dim=1)[sp_rank]
    txtpos = torch.chunk(txtpos, sp_world_size, dim=1)[sp_rank]
    img = torch.chunk(img, sp_world_size, dim=1)[sp_rank]
    imgpos = torch.chunk(imgpos, sp_world_size, dim=1)[sp_rank]

    context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
    context = self.txtmlp(context)

    txtlen, imglen = context.shape[1], img.shape[1]
    combined = torch.cat((context, img), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    freqs = self.pe_embedder(pos)

    for block in self.blocks:
        combined = block(combined, tvec, freqs, None, transformer_options=transformer_options)

    final = self.last(combined, t)
    out = final[:, txtlen:txtlen + imglen, :]

    # ===================== SP GATHER ===================== #
    out = get_sp_group().all_gather(out.contiguous(), dim=1)
    out = out[:, :img_orig_size, :]

    out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
    out = out[:, :, :H_orig, :W_orig]
    if temporal:
        out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
    return out


def usp_attention_forward(self, x, freqs=None, mask=None, transformer_options={}):
    q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
    q, k = self.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if self.kvheads != self.heads:
        rep = self.heads // self.kvheads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    out = xfuser_optimized_attention(q, k, v, self.heads, mask=mask, skip_reshape=True)
    return self.wo(out * F.sigmoid(gate))
