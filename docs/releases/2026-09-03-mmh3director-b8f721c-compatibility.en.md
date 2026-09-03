# 1.13.1: MiniMaxH3Director b8f721c compatibility update

[简体中文](2026-09-03-mmh3director-b8f721c-compatibility.zh-CN.md)

## Release positioning

- Version: `1.13.1`
- Baseline: `1.13.0` (O11)
- Type: backward-compatible patch release; it does not open an O12 stage
- Compatibility code commit: `47c5ee0`
- Acceptance documentation commit: `311c14f`

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
- The first hardware run exposed that official MiniMax AV latent `samples` use ComfyUI `NestedTensor`, not a plain `torch.Tensor`; a real-container regression test and a minimal type-gate fix were added.
- Automated acceptance: `701 passed` in the full suite, `91 passed` in the Director set, and `31 passed` in the hardware-finding-focused set.

## Dual-GPU hardware acceptance

- Compatibility code commit: `47c5ee0`; all 467 managed files in the acceptance deployment matched its source commit and SHA-256 manifest.
- Workflow: the active single segment in `MMH3Director.json`, 56 frames at 864×480 and 24 fps (about 2.33 seconds). Sampling was temporarily overridden to one step for this compatibility smoke test; the source workflow was not modified.
- The current stage bridge intercepted successfully and rank 0/1 workers completed the same operation with no main-process GPU0 fallback.
- Remote sampling: 48.55 seconds; model-to-GPU preparation: about 2.03 seconds; complete queued run: 157.26 seconds.
- Communication: 200 all-to-all calls and 479 all-gather calls; both GPUs reached 100% utilization.
- Peak GPU memory: 10819 MiB on GPU0 and 9701 MiB on GPU1.
- Output: H.264, 864×480, 24 fps, 56 frames, 2.333 seconds; Director completed VAE decode and save.
- After the job, the ComfyUI/Ray process tree exited and both GPUs returned to about 10 MiB at 0% utilization.
- The complete structured record is local at `.local/artifacts/director-b8f721c-smoke/20260903-150416-director/benchmark.json` and is not committed as source.

## Boundaries

- Remote sampling still has no independent per-step TAE preview stream; the completed latent returns to Director.
- Future Director interfaces outside the listed coordinates require contract and hardware acceptance.
- This update changes only Director-to-worker routing. FSDP, P2P, Text Encoder, VAE, and model precision policies are unchanged.
