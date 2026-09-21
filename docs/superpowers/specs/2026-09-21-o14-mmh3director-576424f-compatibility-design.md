# O14 MiniMaxH3 Director 576424f Compatibility Design

## Goal

Upgrade the deployed MiniMaxH3 Director from `52f8fb7` to upstream `576424f` and keep its new SelfLift, Semantic Bridge, FaceRefine, independent refine seed, pre-refine VRAM cleanup, cache fixes, and reference-video fixes compatible with Raylight's Windows dual-V100 FSDP/P2P path. ComfyUI, Director, SOL Attention, and TE-Speed source remain unmodified.

## Upstream Delta

`52f8fb7..576424f` contains 13 commits across 47 files: external-group cache witnesses and sizing, package prompt import fixes, true reference-video duration, SelfLift and high-resolution tiling, Semantic Bridge, FaceRefine, pre-refine cleanup, independent refine seeds, and first-pass aspect preservation.

## Architecture

Raylight continues to intercept Director's single `sample_single_stage` seam. First pass, Refine, FaceRefine, and both SelfLift stages share the same bridge and worker pool.

Director now passes `on_step_state`, `zero_noise`, and `noise_override`. Raylight serializes the selected noise policy. Rank 0 retains the final full `x0` and `x` callback state and returns it with the samples; the driver invokes `on_step_state` before returning to Director. Current SelfLift only consumes the latest state, so this avoids transferring every full latent on every step.

Semantic Bridge remains a driver-side conditioning transform. Face detection/cropping/stitching and VAE remain in Director, while FaceRefine diffusion uses the existing Ray MODEL proxy. Optional features without installed weights are reported as contract-tested, not quality-validated.

Before the update, Director `52f8fb7` is archived under Raylight `.local/backups/director/`. Raylight changes occur only in the unique source worktree and are deployed with the existing deployment script after commit and verification.

## Performance and Quality Baseline

Run the same O13 REF2VA Turbo4 + SOL + Refine2 workflow with identical input, seed, dimensions, frame count, P2P capacity, and VRAM reserve once before and once after the upgrade. Record end-to-end, Ray initialization, model nodes, first pass, Refine, VAE/save, per-GPU VRAM/utilization peaks, physical RAM, and commit peak. The compatibility update must not cause a stable regression above 5% and must produce finite, playable output without new artifacts.

SelfLift receives a separate fixed-input acceptance run. Both low- and high-resolution stages must use the same worker pair. Release notes claim only measured speed or memory gains, never community estimates.

## Acceptance

- Director is clean at `576424f`, with a verified rollback archive for `52f8fb7`.
- Current and legacy stage contracts pass; unsupported callbacks fail closed.
- SelfLift low/high stages, exact noise semantics, and final-state return run on both workers.
- Existing first pass, Director Refine, remask, tiling, previews, SOL, and TE paths do not regress.
- Semantic Bridge conditioning crosses the bridge; FaceRefine sampling is contract-tested and hardware-tested when optional assets exist.
- Reports contain pre/post phase timing, GPU memory/utilization, RAM/commit, and media checks.
- Source, deployment, READMEs, bilingual release notes, test report, commits, and GitHub remote agree.
