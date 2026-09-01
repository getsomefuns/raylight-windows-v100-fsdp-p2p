# 1.13.0 / O11: efficient MiniMax H3 Text Encoder attention

O11 replaces the dual-V100 MiniMax H3 TP2 Text Encoder path that could fall back to mathematical SDPA with explicit memory-efficient attention routing.

## Changes

- Adds independent Text Encoder choices: `AUTO`, `XFORMERS_GQA`, `TORCH_EFFICIENT`, and `MATH_DEBUG`.
- AUTO prefers xFormers GQA on the target V100 system. Production routes fail explicitly instead of falling back to Math.
- Removes the dense floating-point causal mask.
- Long-sequence xFormers executes independent KV groups in bounded batches and reuses dead Q storage. The 28k peak drops from 1753.38 MiB to 1314.13 MiB.
- A 4096-token MLP chunk and earlier Attention temporary cleanup fix the maximum-conditioning `down_proj` OOM.
- The optional widget remains compatible with old workflows through the AUTO default and UI-to-API schema defaults.
- O11 artifacts and videos are confined to the canonical worktree and `output/video/raylight_o11`.
- The MiniMax H3 safe-FP16 Diffusion follow-up releases packed inputs, gives each rank owned local sequence storage, shortens Attention/MLP temporary lifetimes, and adds bounded xFormers Ulysses head/query chunks.
- `fp16_h3_safe` workers enable the accepted block-boundary cache trim by default. An explicit `RAYLIGHT_H3_BLOCK_CACHE_TRIM=0` still disables it, with no launcher coupling.

## Verified boundary

- One 15-second reference video, seven distinct images, and a meaningful 7000-character prompt pass at an expanded sequence of 24,236.
- Cold/warm Text Encoder times are 112.94/112.55 seconds; language times are 76.08/76.29 seconds.
- The user Director case with seven images, one video, and one audio expands to 12,302 positions and completes Text Encoder in 62.88 seconds, peaking at 9244/9451 MiB of PyTorch allocations.
- A three-second 480p REF2VA Turbo4 sampling integration and media-quality smoke pass.
- In the historical baseline, 0.6 MP fails in RMSNorm and 0.4 MP advances to an MLP `fc2` OOM.
- The final fixed 0.4 MP case (365 frames at 24 fps, seven images + one video + one audio, REF2VA Turbo4) completes in 1456.12 seconds of sampling and 1823.77 seconds end to end. The matched control takes 1999.41/2332.86 seconds, so sampling is 27.17% faster and E2E is 21.82% faster.
- Sampling-window board peaks fall from 16215/16230 to 13005/12873 MiB. PyTorch active peaks at 7.42 GiB, about 6.57 GiB below the 13.99 GiB failure baseline. The active-memory 6 GiB gate passes; board memory falls by about 3.14/3.28 GiB and is not misreported as a 6 GiB physical reduction.
- The formal output is 864x480, 365 frames, and 15.209 seconds. H.264/AAC full decode passes with no black interval. Framewise SSIM against the prior accepted four-step output is 0.9416. One-step diagnostic videos are not quality evidence.
- The 0.6 MP boundary has not completed the same end-to-end gate and remains outside the supported claim.

See the [Chinese test record](../testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md) for full data.
