# MiniMaxH3Director b8f721c compatibility update

## Why this update exists

Between MiniMaxH3Director commits `d655912` and `b8f721c`, the sampling path moved from a direct `comfy.sample.sample()` call to:

`sample_single_stage → BasicGuider/CFGGuider → SamplerCustomAdvanced`

The Raylight proxy is intentionally not a full ComfyUI `ModelPatcher`. Current Director therefore reached `is_dynamic()` before remote dispatch and failed. Adding only that method would expose the next missing ModelPatcher method and would not address the routing break.

## Implemented

- Retain the legacy `comfy.sample.sample()` bridge used by `d655912`.
- Add a `sample_single_stage()` bridge for `b8f721c`.
- Intercept only `RayMMH3DirectorModelProxy`; ordinary MODEL objects call Director unchanged.
- Refresh stage aliases when the adapter node executes, removing custom-node import-order sensitivity.
- Preserve seed, steps, CFG, sampler, scheduler, denoise, video/audio shifts, noise mask, and optional external SIGMAS.
- Honor Director's `apply_shift=False` in workers while defaulting old requests to shift enabled.
- Fail explicitly on remote errors and never fall back silently to main-process GPU0.

Director, ComfyUI core, and model files remain unmodified.

## Runtime

- Windows, two Tesla V100-SXM2-16GB GPUs, TCC
- Python 3.10.11
- PyTorch 2.7.0+cu126
- xFormers 0.0.30
- comfy-aimdo 0.4.13
- MiniMaxH3Director `b8f721c`

The PyTorch 2.8/AIMDO TCC DynamicVRAM experiment never replaced the formal runtime. Its candidate Python, build tree, logs, and dedicated repository files have been removed.

## Verification

- Legacy and current Director contracts are capability-checked.
- Current Director's `core_sampling`, first-pass executor, and Refine executor aliases install the same stage bridge and restore their originals.
- Regression coverage includes ordinary MODEL pass-through, no proxy fallback, equal dual-rank output, NaN/Inf rejection, external SIGMAS, shift control, multi-segment reuse, and cleanup.

The final dual-GPU workflow result will be appended after deployment acceptance.

## Boundaries

- Remote sampling still has no independent per-step TAE preview stream; the completed latent returns to Director.
- Future Director interfaces outside the listed coordinates require contract and hardware acceptance.
- This update changes only Director-to-worker routing. FSDP, P2P, Text Encoder, VAE, and model precision policies are unchanged.
