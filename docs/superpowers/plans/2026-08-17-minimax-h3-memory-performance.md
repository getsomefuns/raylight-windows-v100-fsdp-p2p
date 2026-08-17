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

**Status: complete and accepted on 2026-08-17.** I2V and REF2VA each passed one cold and two warm runs. Warm runs reused the same worker PIDs and existing FSDP registrations, both ranks completed, media passed black/frozen-frame checks, and committed memory, pagefile and VRAM did not grow progressively. Results are recorded in `docs/testing/minimax-h3/COLD_WARM_BENCHMARK_2026-08-17.md`.

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

**Status: complete and accepted on 2026-08-17.** The official FL2V Turbo 8-step and REF2V Turbo 4-step LoRAs pass quantized FSDP loading, two-rank CUDA P2P execution and media validation. Their warm end-to-end means improve by 54.9% and 58.9% respectively against matched 20-step baselines. Results and the FSDP LoRA compatibility fix are recorded in `docs/testing/minimax-h3/SPEED_QUALITY_VARIANTS_2026-08-17.md`.

1. Compare the FP8 base checkpoint against a compatible official Turbo LoRA at matched input, seed and output settings.
2. Evaluate an INT8 checkpoint only after its exact model artifact is available and validated.
3. Keep FP8 as the default unless another variant passes numerical/media checks and materially improves speed or memory.

### O3 acceptance

- Pin the exact Turbo LoRA artifact, size and SHA-256 before execution; do not infer compatibility from a filename alone.
- Use the same model, input image(s), dimensions, frame count, steps and seed for baseline and Turbo comparisons, changing only documented LoRA/sampling settings required by the official artifact.
- Record end-to-end time, sampler time, both-rank participation, peak host commit/pagefile and per-GPU VRAM/utilization.
- Require finite rank outputs, no rank mismatch, valid video/audio streams, no black interval and temporal variation.
- Adopt a variant only when its output is valid and its performance or memory benefit is repeatable; otherwise retain the accepted baseline.

## Phase O4: full-specification Turbo release validation

**Status: technically complete and accepted on 2026-08-17.** The full 640x640 I2V Turbo 8-step and 864x480 REF2VA Turbo 4-step workflows pass two-rank FSDP/P2P, resource and media validation. Cold end-to-end time improves by 39.6% and 68.0% against the accepted full 20-step baselines. Automated correctness is complete; visual preference remains a user decision. Results are recorded in `docs/testing/minimax-h3/FULL_TURBO_RELEASE_2026-08-17.md`.

1. Run one clean cold FL2V Turbo 8-step I2V job at the complete workflow resolution, duration and conditioning settings.
2. Run one clean cold REF2V Turbo 4-step REF2VA job at the complete workflow resolution, duration and reference-conditioning settings.
3. Compare each run with the accepted full 20-step result in `docs/testing/minimax-h3/FULL_WORKFLOWS_2026-08-17.md`.
4. Keep full logs, resource telemetry and media local; write only concise results and evidence paths into the maintained validation report.
5. Present base and Turbo media for user visual review before selecting a default quality/speed preset.

### O4 acceptance

- The API prompt preserves the original author's full workflow settings except for the pinned official LoRA and its required step count.
- Both ranks register 684 wrappers, finish sampling, use CUDA P2P and reach high GPU utilization.
- Host commit, pagefile and per-GPU VRAM remain within the already accepted full-workflow operating envelope.
- I2V and REF2VA outputs have the expected dimensions, frame count, video/audio streams, no black interval, temporal variation and finite non-silent audio.
- No CUDA OOM, rank mismatch, collective timeout, unsupported LoRA key or numerical failure occurs.
- Performance is reported separately for cold end-to-end, sampler and decode/write stages; Turbo is not described as quality-equivalent without user review.

## Phase O5: user-loadable Turbo release candidate

**Status: active.** O4 proves the runtime path. O5 packages the accepted settings as normal ComfyUI workflow JSON files and user-facing instructions instead of requiring the benchmark harness.

1. Generate dedicated I2V Turbo 8-step and REF2VA Turbo 4-step GUI workflows without modifying the upstream or 20-step FSDP examples.
2. Pin the intended LoRA filename, strength and sampler step count in each workflow while preserving the original prompts, inputs and full output settings.
3. Validate workflow JSON round-trip conversion, node availability and exact model references in the installed ComfyUI environment.
4. Document launch flags, expected model locations, capacity mode and the base-versus-Turbo choice.
5. Present the accepted full base and Turbo media paths for user visual review before selecting the default preset.

### O5 acceptance

- Both GUI workflow files load without missing nodes and convert to valid API prompts.
- Turbo workflow settings reproduce the pinned O4 LoRA/step contracts.
- Base workflow files remain byte-for-byte unchanged.
- Documentation gives a direct manual-use path and clearly distinguishes the quality baseline from the speed preset.
- Repository remains local until user review; no model, generated media or raw log is added to Git.

## Documentation and release boundary

- Update the maintained MiniMax validation log after every accepted run.
- Keep raw logs, telemetry, generated media and model weights out of the repository.
- Do not push or alter the public repository until the user reviews the local changes.
