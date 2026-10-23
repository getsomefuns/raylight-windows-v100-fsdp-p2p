# Raylight 1.18.3: Long-reference Attention OOM and lifecycle memory records

## Failure and cause

A 15-second MiniMax H3 Director REF2VA job with high-resolution SelfLift failed inside Ulysses Attention on two V100s. The old path split a single QKV projection into three views that shared one backing allocation, so completing communication for one component could not release that storage. Yunchang `all_to_all` then allocated a second full-size `empty_like` output. With about 14.5 GiB of physical VRAM already in use per GPU, this additional roughly 1.56 GiB output was the immediate OOM trigger.

## Fix

1. Project MiniMax H3 QKV in 4,096-token chunks into three independently owned tensors while preserving one FSDP unshard/reshard pair per Attention call.
2. Reuse the communication-layout tensor as both input and output for two-rank Windows CUDA P2P `all_to_all_single`. The existing 128 MiB staging protocol preserves each outbound chunk before overwriting its destination, eliminating the second full-size output.
3. Write dense xFormers Attention output into consumed Query storage to remove another full context allocation.
4. Keep the original yunchang path on non-Windows-P2P backends. Sampling math, precision, LoRA, and Director controls are unchanged.

## Memory records

The launcher now sets `RAYLIGHT_MEMORY_LOG_DIR=<ComfyUI>/user/raylight_memory`. One-second records cover the complete Director lifecycle and every nested sampling pass, including:

- system RAM and Windows commit/commit limit;
- main-process and both-worker RSS, private, pagefile, and VMS;
- PyTorch CUDA allocated/reserved;
- whole-device used/free memory from `cudaMemGetInfo`.

Success and failure both append JSONL summaries below date-specific directories. Set `RAYLIGHT_MEMORY_RECORD=0` to disable recording. CUDA peak counters are reset before each stage, so a previous stage cannot contaminate the result.

## Validation

- Real two-V100 in-place and out-of-place P2P probes through 115,343,360 bytes report `mismatch=0` and `max_error=0` at every tested size.
- Full automated suite: `808 passed, 35 subtests passed`.
- Hardware acceptance: 362 frames at 24 fps and 768×1344, two subject images, an approximately 354-frame reference video, REF2VA, and two low-resolution plus two high-resolution SelfLift steps.
- Attention reaches about 152,332 positions at low resolution and 233,224 at high resolution. Both ranks return successfully and the final 362-frame MP4 completes with `success` status.
- Observed board peaks are about 16,083 MiB on GPU0 and 16,213 MiB on GPU1. The former extra roughly 1.56 GiB allocation is absent and no OOM occurs.
- Director timings: 238.2 seconds conditioning, 2,299.2 seconds sampling, 77.9 seconds decode, 16.7 seconds cache, and 46 minutes end to end.

This release improves capacity stability for long references; it is not a speed optimization. The successful 768×1344 boundary case is too slow to recommend as a production preset.
