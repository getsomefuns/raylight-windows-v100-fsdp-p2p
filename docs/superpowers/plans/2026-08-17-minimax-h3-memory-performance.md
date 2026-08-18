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

**Status: complete and accepted on 2026-08-18.** Dedicated I2V Turbo 8-step and REF2VA Turbo 4-step GUI workflows load against the installed ComfyUI node catalog and each convert to a 21-node API prompt with the pinned LoRA, strength and step contract. Generated artifacts are locked by byte-for-byte tests; upstream and 20-step workflows retain their prior SHA-256 values. Manual instructions are in `docs/testing/minimax-h3/TURBO_WORKFLOW_USAGE.md`.

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

## Phase O6: safe FP16 compute for MiniMax H3 on Raylight FSDP

**Status: functional and quality implementation complete on 2026-08-18; initial 4x performance gate remains open.** The model-specific `fp16_h3_safe` worker integration is implemented and validated with FP32 numerical islands, FP16 attention/MLP branches, FP8 FSDP storage, Turbo LoRA, exact two-rank output and valid media. Full I2V/REF2VA sampling reaches 3.299x/3.416x, and the best optional REF2VA host-registration experiment reaches 3.714x. Details and rejected experiments are recorded in the bilingual release notes under `docs/releases/`.

### O6 pre-development baseline

- Verified: both accepted Turbo workflows use FP8-storage checkpoints with `RayUNETLoader.weight_dtype=default`; existing worker logs show V100 fallback dequantization and rank outputs in FP32.
- Verified: the external patch imports against the current ComfyUI 0.31 model classes and its `DiTBlock`/`MLP` signatures still match.
- Verified: a file placed only in `ComfyUI/custom_nodes` is not shipped or imported by the Ray actor runtime. Copying the external file as documented by its author therefore does not make the patch active in either FSDP rank.
- Verified: Raylight replaces MiniMax attention and model `_forward` for USP, but still invokes every `DiTBlock` and `out_proj`; FSDP wrapping occurs after model construction. This leaves a feasible integration point before each worker builds and shards the model.
- Hypothesis to test: retain FP8 checkpoint storage/FSDP sharding, the condition projection/residual/final heads in FP32, and run the dominant attention/MLP branches plus their V100 chunked FP8 dequantization in FP16.
- Hypothesis to test: FP16 branch tensors reduce temporary dequantized-weight memory and Ulysses activation traffic, while host checkpoint storage and the FP32 residual stream remain largely unchanged.
- Compatibility risk: BF16 Turbo LoRA sidecars, instance-level forward wrappers, repeated model loads and FSDP2 hooks must be validated together; no quality-equivalence claim is accepted from the external single-machine report.
- Before any O6 code change, run the existing Turbo8 I2V and Turbo4 REF2VA paths at 1120x768, 124 frames and 24 FPS from a clean cold start.
- Record full video wall time, worker/model load, preprocessing, every sampler's total and per-iteration timing, VAE decode, media write, both-GPU utilization/VRAM, host physical/committed memory and pagefile use.
- The measured local FP32-compute runs are the only O6 performance denominators. External timing figures are not release gates.

### O6 acceptance boundary

- Existing `default` FP32-compute workflows remain byte-for-byte baselines; FP16 is opt-in through dedicated experimental workflows and an explicit Ray loader mode.
- Both ranks prove FP32 residual/condition/output islands and FP16 attention/MLP branches; V100 FP8 fallback logs the actual FP16 dequantization dtype.
- FSDP remains at 684 wrappers per rank, CPU offload remains available, CUDA P2P remains active, and no tensor collective falls back to host-staged Gloo.
- Both Turbo LoRAs load all 208 grouped sidecar adapters with zero unsupported entries and execute in the branch input dtype.
- Smoke and full runs have finite rank-matched outputs, valid video/audio, no black interval, temporal variation and no NaN/Inf.
- For each workflow, the initial gate is at least 4x faster stable sampling than its matched local pre-O6 baseline. The later 11x goal is tracked separately and is not substituted for the initial release gate.
- Worker/model load, preprocessing, VAE decode and media-write stages must be no slower than their matched local baselines. End-to-end time must improve; no faster sampler may hide a regression elsewhere.
- Peak VRAM, committed memory and pagefile remain within the O4 accepted operating envelope. A speed gain cannot be accepted by weakening numerical or media gates.

## Phase O7: model-specific safe FP16 research for LTX/LTXAV

**Status: preliminary feasibility plan only; scheduled for recalibration after O6.** The MiniMax H3 patch cannot be copied directly because LTX/LTXAV uses different transformer blocks, video/audio branches, normalization and output paths. Previous global-FP16 LTX tests produced black video and audio NaN/Inf, so O7 starts from identifying LTX-specific FP32 numerical islands instead of adding FP16 to the global inference allowlist.

### O7 preliminary direction

- Preserve the currently accepted LTX BF16/FP32 workflows and FP8 storage behavior unchanged.
- Instrument video and audio residual streams, AdaLN/RMSNorm, attention/MLP branches, cross-modal gates and final projections to find the first non-finite or destructive precision transition.
- Evaluate FP16 only for proven-safe high-cost matrix multiplications; retain sensitive reductions, normalization, residual accumulation, modulation and output heads in FP32 where required.
- Install any experimental policy inside both Ray actors before FSDP wrapping, with an explicit LTX-only loader mode and no global dtype allowlist change.
- Validate LTX and LTXAV separately because valid video does not prove valid audio.
- Reuse O6's dtype tracing, rank-consistency, P2P/FSDP and stage-timing infrastructure, then set O7 performance gates from a fresh matched local baseline after O6 closes.

The detailed preliminary O7 plan is in `docs/superpowers/plans/2026-08-18-ltx-safe-fp16-research.md`.

## Documentation and release boundary

- Update the maintained MiniMax validation log after every accepted run.
- Keep raw logs, telemetry, generated media and model weights out of the repository.
- Keep README, implementation plans, validation summaries and release notes synchronized with the actual O6 state before the next public GitHub update. O6 is opt-in while its 4x gate remains open.
