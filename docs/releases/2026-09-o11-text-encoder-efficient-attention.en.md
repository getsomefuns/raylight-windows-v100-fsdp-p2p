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

## Verified boundary

- One 15-second reference video, seven distinct images, and a meaningful 7000-character prompt pass at an expanded sequence of 24,236.
- Cold/warm Text Encoder times are 112.94/112.55 seconds; language times are 76.08/76.29 seconds.
- The user Director case with seven images, one video, and one audio expands to 12,302 positions and completes Text Encoder in 62.88 seconds, peaking at 9244/9451 MiB of PyTorch allocations.
- A three-second 480p REF2VA Turbo4 sampling integration and media-quality smoke pass.
- The 15-second Director run still OOMs in the later Diffusion first step. At 0.6 MP it fails in RMSNorm; 0.4 MP advances into MLP `fc2` but still OOMs. This release does not claim that maximum sampling boundary.

See the [Chinese test record](../testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md) for full data.
