# MiniMax H3 Windows V100 Memory and Performance Phase

## Goal

Keep the accepted full I2V and REF2VA workflows correct while making repeated interactive use safe and measurable. Preserve fast reuse for an unchanged checkpoint, but prevent committed-memory and pagefile accumulation when the diffusion checkpoint changes.

## Baseline

- Full I2V and REF2VA both pass with FSDP CPU offload.
- Same-checkpoint quantized FSDP reuse passes and avoids reloading or resharding.
- Switching live workers from REF2VA to I2V passes, but peaks at about 148.4 GiB committed memory and 42.9 GiB pagefile use.
- A fresh REF2VA cold start peaks near 125.7 GiB committed memory and 6.8 GiB pagefile use.

## Phase O1: checkpoint-aware worker recycling

**Status: complete and accepted on 2026-08-17.** The REF2VA-to-I2V live switch replaced both worker PIDs, reduced committed memory to about 51.1 GiB before loading the new checkpoint, and limited the transition peak to 127.45 GiB. A changed-seed I2V warm run reused both replacement PIDs without reloading or rewrapping the checkpoint. Full evidence is recorded in `docs/testing/minimax-h3/CHECKPOINT_RECYCLING_2026-08-17.md`.

1. Add a worker query that returns the active model key without exposing checkpoint contents.
2. In `RayUNETLoader`, retain the current fast path when all workers already hold the requested checkpoint and LoRA/options signature.
3. When live workers hold a different checkpoint, retire the complete actor set and recreate it through the existing initializer factory before loading the new model.
4. Do not attempt in-process allocator cleanup as the primary solution; the acceptance target is operating-system reclamation of the old worker address spaces.
5. Add focused tests for unchanged-model reuse, changed-model recycling, dead-worker recovery and partial actor failure.

### O1 acceptance

- Same-checkpoint smoke run reuses the original worker PIDs and does not invoke checkpoint loading again.
- Changed-checkpoint run uses new worker PIDs before `RayUNETLoader` loads the checkpoint.
- Changed-checkpoint peak committed memory is at most 135 GiB on the reference machine.
- Changed-checkpoint peak pagefile use is at most 12 GiB after starting from an idle system.
- Both ranks register 684 wrappers, use CUDA P2P, reach high GPU utilization, and return exact finite outputs.
- Media has valid video/audio streams, no black interval and temporal variation.

## Phase O2: repeatable cold/warm benchmark

**Status: active.** O1 provides the stable worker lifecycle required for meaningful repeated-run measurements.

1. Record one cold and two warm runs for the selected I2V profile.
2. Record one cold and two warm runs for the selected REF2VA development profile; use one full REF2VA release run after optimization because the full profile is expensive.
3. Separate preprocessing, worker creation/checkpoint load, sampling, VAE decode and container-write time.
4. Report physical memory, committed memory, pagefile, per-GPU VRAM/utilization and P2P traffic with the same sampling method.

### O2 acceptance

- No progressive committed-memory, pagefile or VRAM growth across two warm runs.
- No rank skew timeout or output mismatch.
- Unchanged-checkpoint warm runs do not reload or reshard the diffusion model.
- Performance claims use sampler time and end-to-end time separately.

## Phase O3: speed/quality variants

1. Compare the FP8 base checkpoint against a compatible official Turbo LoRA at matched input, seed and output settings.
2. Evaluate an INT8 checkpoint only after its exact model artifact is available and validated.
3. Keep FP8 as the default unless another variant passes numerical/media checks and materially improves speed or memory.

## Documentation and release boundary

- Update the maintained MiniMax validation log after every accepted run.
- Keep raw logs, telemetry, generated media and model weights out of the repository.
- Do not push or alter the public repository until the user reviews the local changes.
