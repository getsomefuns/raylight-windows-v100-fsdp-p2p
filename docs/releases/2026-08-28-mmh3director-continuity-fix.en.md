# 1.10.1: MiniMaxH3Director multi-segment continuity fix

Date: 2026-08-28

## Fixed behavior

MiniMaxH3Director prepares motion-context keyframes in the main ComfyUI process, while Raylight performs diffusion sampling in independent Ray workers. When a continued segment also contained a current reference image, stock MiniMax conditioning preparation replaced the keyframe latents with the reference-image latent while the packed layout still retained every condition segment. This produced failures such as a `510`-row condition being assigned to a `4080`-row layout.

Version 1.10.1 recognizes Director-only continuity markers inside each worker and:

- restores video conditions in motion-keyframe-then-current-reference order;
- places continuity keyframes on the current target-video timeline;
- repositions marked continuous-audio references to their intended end time;
- validates keyframe counts, reference kinds, layout order, and stereo audio rows;
- returns unmarked MiniMax, LTX, I2V, FL2VA, and REF2VA requests unchanged by identity.

## Verification

- Four focused regressions cover latent merging, keyframe time, audio time, and unchanged ordinary payloads.
- The combined Director/Raylight regression selection reports `50 passed`.
- The full repository suite reports `423 passed, 30 subtests passed`.
- A dual Tesla V100-SXM2-16GB TCC/FSDP/CUDA-P2P run completed two Director main samples and two Refine samples on both ranks.
- End-to-end time was about 12 minutes 47 seconds; outputs were 1280×736, 24 FPS, 365 frames, and about 15.2 seconds long.
- Both saved results passed full FFmpeg decode checks.

## Compatibility boundary

The validated Director coordinate remains `AIMixer/ComfyUI_MiniMaxH3_Director` commit `d655912`. Upstream added further cache, memory, FL2V, and Refine changes through 2026-08-27; those changes are not bundled with this release and are not claimed as accepted yet.

Missing local start- or end-frame files in a workflow remain an input-asset configuration error before Ray worker sampling and are outside this fix.
