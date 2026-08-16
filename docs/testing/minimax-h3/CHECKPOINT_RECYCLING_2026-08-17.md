# MiniMax H3 Checkpoint Recycling Validation (2026-08-17)

## Validation target

Verify that native-Windows Raylight keeps the fast FSDP path for an unchanged diffusion checkpoint, but replaces the complete Ray worker set when the checkpoint changes. The changed-checkpoint path must reclaim the old worker address spaces before loading the next model, preserve CUDA P2P/FSDP correctness, and avoid the committed-memory/pagefile accumulation seen in the previous live REF2VA-to-I2V switch.

## Fixed conditions and variables

- Reference system: Windows 23H2, driver 577.00, two Tesla V100-SXM2-16GB GPUs in TCC mode.
- Runtime: Python 3.10.11, PyTorch 2.7.0+cu126, Ray 2.57.0, ComfyUI 0.31.0.
- ComfyUI arguments: `--disable-cuda-malloc --reserve-vram 2`.
- Raylight: FSDP enabled, FSDP CPU offload enabled, Ulysses 2, ring 1, synchronized Ulysses, TORCH_EFFICIENT attention and CUDA P2P transport.
- Changed-checkpoint sequence: REF2VA smoke followed by I2V smoke in one live ComfyUI/Ray session.
- Same-checkpoint sequence: rerun I2V smoke with the seed incremented by one so ComfyUI cannot satisfy the request entirely from its node cache.
- Resource sampling: physical memory, committed memory, pagefile use, per-GPU VRAM/utilization and power sampled once per second.

## Focus information

- Old and new Ray worker PIDs around the checkpoint transition.
- Whether the changed-checkpoint path actually terminates the old processes.
- Whether the same-checkpoint path avoids checkpoint loading and FSDP rewrapping.
- OS committed-memory/pagefile behavior during process replacement.
- Both-rank FSDP registration, CUDA P2P health, GPU participation and numerical agreement.
- Output container, video/audio streams, black intervals and temporal variation.

## Results

| Run | Worker behavior | End-to-end | Sampler | Resource result |
|---|---|---:|---:|---|
| REF2VA smoke, cold | Created PIDs 30456 / 23392 | 374.39 s | 12 steps, about 15.3 s/it | 125.07 GiB peak commit; pagefile rose under physical-memory pressure |
| REF2VA -> I2V smoke | Recycled old workers; created PIDs 16292 / 19276 | 269.47 s | 12 steps, about 10.8 s/it | Commit fell to 51.06 GiB before reload; 127.45 GiB transition peak; post-recycle pagefile about 2.35 GiB |
| I2V smoke, changed seed | Reused PIDs 16292 / 19276; `FSDP already registered` on both ranks | 145.80 s | 12 steps, about 10.7 s/it | 89.07 GiB peak commit; 2.34 GiB peak pagefile; no progressive growth |

The changed-checkpoint CUDA P2P health probe measured median one-way throughput of about 59.09 GiB/s on both ranks. Both ranks registered 684 FSDP wrappers and reached sustained high GPU utilization. The changed-checkpoint and warm-run outputs were finite FP32 tensors and matched exactly across ranks.

The transition monitor's overall pagefile maximum includes the approximately 16.9 GiB already present before worker recycling. After the old workers exited, pagefile use dropped to approximately 2.35 GiB and stayed there while the new I2V model sampled. The transition committed-memory peak of 127.45 GiB is below the 135 GiB O1 limit and is substantially below the previous non-recycling switch peak of about 148.4 GiB.

## Media validation

- REF2VA output: 608x352, 56 H.264 frames, AAC audio, 2.333 s; 56/56 unique decoded frame hashes; no black interval.
- Changed-checkpoint I2V output: 448x448, 39 H.264 frames, AAC audio, 1.625 s; 39/39 unique decoded frame hashes; no black interval.
- Same-checkpoint warm I2V output: 448x448, 39 H.264 frames, AAC audio, 1.625 s; 39/39 unique decoded frame hashes; no black interval.

Generated media, raw ComfyUI logs and per-second CSV telemetry remain local artifacts and are not stored in the repository.

## Conclusion

Phase O1 is accepted. Raylight now has two distinct and verified behaviors:

1. An unchanged diffusion checkpoint keeps its existing Ray worker processes and FSDP shards.
2. A changed diffusion checkpoint replaces the full actor set, allowing Windows to reclaim the old model's process memory before loading the next checkpoint.

This fixes the main interactive-session memory accumulation without sacrificing CUDA P2P, dual-GPU participation or output correctness. Phase O2 can now measure repeatable cold and warm runs on top of this stable lifecycle.
