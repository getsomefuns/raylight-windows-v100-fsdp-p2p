# Latest Director, Pika and RelayVSR compatibility plan

## Scope

Keep the canonical Raylight worktree, retain existing dual-V100 workflows and local Director fixes, then validate new features before claiming support. Do not change the driver, overwrite workflows/caches, or silently upgrade ComfyUI core. Runtime changes must be made in this worktree, tested and deployed before GitHub synchronization.

## Findings

Raylight starts at `af4500d` (1.18.5). Director starts at `6e14f23`, containing PR231 over upstream `4c6245e`, plus five modified files for export-tail continuity and face-selection/tracking options. Upstream main fetched on 2026-11-24 is `aff2579`; its five newer commits add RelayVSR, batch deletion and large-pack export improvements. There is no Pika-specific sampler/node implementation in that range; Pikachu prompt examples are not evidence of support.

Installed ComfyUI `62b3c94b` lacks `RelayVSRConditioning`, `VSRDecode` and the RelayVSR model implementation. No Pika LoRA or RelayVSR UNET/Flash VAE was found in the shared model tree.

## Approach and acceptance gates

1. Back up the complete local Director and Git history, commit the dirty fixes locally, then rebase both local commits onto `aff2579`. Never reset/reinstall over the changes.
2. Re-run PR231 Python/browser-logic regressions, upstream audio tests, and Raylight's full suite. Store portable patches and bilingual maintenance instructions in Raylight; keep machine-specific backups/results ignored under `.local/`.
3. H3 sampling continues through Ray's MODEL proxy. RelayVSR must use its separate ordinary MODEL and matching Flash VAE on the upstream postprocessing route, not pretend to be an H3 FSDP model.
4. Confirm Pika's actual weights, task family and sigma schedule. Reuse existing LoRA/sigma transport if sufficient; only add a Raylight bridge for a demonstrated missing contract.
5. A ComfyUI update requires explicit authority beyond the current core-freeze constraint. Until then, VSR is blocked by a concrete missing backend, not claimed operational. Download weights only from verified sources after selecting the backend.
6. Run matched short workflows for existing Turbo, Pika, Turbo+VSR and Pika+VSR. Record cold/warm loading, TE, sampling, H3 decode, VSR sample/decode, saving, end-to-end time and resource peaks. Check full-frame decoding, audio duration, scale factor, visual quality and cache reuse. A six-step versus twenty-step claim cannot be substituted for comparison against our existing four-step baseline.

No speed or complete-generation acceptance is implied by structural/unit tests. An updated Director alone does not provide a missing core model backend. No new repository is needed.
