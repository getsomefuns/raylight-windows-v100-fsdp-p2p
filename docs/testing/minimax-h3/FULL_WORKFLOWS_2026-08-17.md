# MiniMax H3 Full Workflow Validation - 2026-08-17

## Validation goal

Validate the two upstream MiniMax H3 example workloads at their original settings on native Windows with two V100-SXM2-16GB GPUs. Acceptance requires real two-rank FSDP participation, CUDA P2P/NVLink collectives, finite and identical rank outputs, valid video and audio, temporal variation, and measured host/GPU resource use.

## Test conditions

- Windows 23H2, NVIDIA driver 577.00, two V100-SXM2-16GB GPUs in TCC mode.
- Python 3.10.11, PyTorch 2.7.0+cu126, Ray 2.57.0, yunchang 0.6.4, comfy-kitchen 0.2.30, ComfyUI 0.31.0.
- ComfyUI arguments: `--disable-cuda-malloc --reserve-vram 2`.
- Raylight: two workers, Ulysses 2, ring 1, synchronized Ulysses, FSDP enabled, FSDP CPU offload enabled, mmap enabled, TORCH_EFFICIENT attention.
- FP8-scaled MiniMax diffusion checkpoints, NVFP4 Qwen3-VL text encoder, FP16 video VAE and FP32 audio VAE.
- CUDA P2P health before the REF2VA run: about 59.36 GiB/s median per direction.

## Workloads

| Workflow | Upstream settings | Actual media |
|---|---|---|
| I2V | 0.4 MP, 1:1, 2 seconds, 20 steps | 640x640, 56 frames, 2.333 s |
| REF2VA | 0.4 MP, 16:9, 5 seconds, 20 steps | 864x480, 124 frames, 5.167 s |

The upstream spear/portal image was used for I2V. Both REF2VA reference inputs used the upstream green-robot image. Prompts, seeds, schedulers and sampling settings were not changed.

## Results

| Item | I2V | REF2VA |
|---|---:|---:|
| End-to-end status | Pass | Pass |
| End-to-end time | 642.01 s | 1324.98 s |
| Sampler | 20 steps; about 7.3 min; about 21.9 s/it | 20 steps; about 18 min 52 s; about 56.5 s/it |
| FSDP | 684 wrappers; about 10.04 GiB local payload/rank | 684 wrappers; about 10.04 GiB local payload/rank |
| Sampling VRAM | about 6.83/6.62 GiB | about 12.71/12.51 GiB |
| Recorded GPU peak | 15,659/6,619 MiB | 15,925/12,508 MiB |
| GPU utilization peak | 99%/98% | 100%/100% |
| Peak physical memory | 65,439.51 MiB | 65,436.75 MiB |
| Peak committed memory | 148,369.65 MiB | 125,711.54 MiB |
| Peak pagefile use | 42,916 MiB | 6,757 MiB |
| Rank comparison | Exact; finite FP32 video/audio latents | Exact; finite FP32 video/audio latents |
| Media validation | H.264 + stereo AAC 32 kHz; 56/56 distinct frame hashes; no black interval | H.264 + stereo AAC 32 kHz; 124/124 distinct frame hashes; no black interval |

GPU 0 peaks include the unsharded text-encoder/video-VAE preprocessing stage. During distributed sampling both GPUs participated continuously.

## Fixes validated by these runs

1. MiniMax preprocessing is ordered before Ray worker startup through the optional `wait_for` dependency, then ComfyUI preprocessing models are unloaded before Ray initialization.
2. FSDP2 CPU offload uses `CPUOffloadPolicy(pin_memory=False)` on Windows. It provides the VRAM headroom required by the full workloads.
3. Quantized full state-dict entries are released incrementally after local FSDP shards are created when no excluded ControlNet modules need them later.
4. An unchanged quantized FSDP model can be reused across prompts. A full REF2VA run followed by the reduced REF2VA workload completed without reloading or resharding the model; the regression run finished in 216.98 s.

The state-dict release change preserves numerical output: the reduced REF2VA output matched both the non-offload and prior CPU-offload baselines exactly for all 56 decoded video frames and for decoded audio.

## Capacity and performance findings

- CPU offload is a capacity mode, not a speed mode. On the reduced REF2VA workload it lowered sampling VRAM from roughly 16 GiB/card to roughly 6 GiB/card, but increased steady sampler time from about 10.7 to 15.2 s/it.
- Full REF2VA needs about 12.7/12.5 GiB during sampling even with the parameter shards offloaded, so the non-offload profile does not have a safe 16 GiB margin.
- I2V is substantially faster and lighter than REF2VA at the upstream settings because its temporal/reference conditioning workload is smaller.
- Reusing the same workers for a different 20 GB diffusion checkpoint is functionally correct but not memory-stable. The REF2VA-to-I2V switch raised committed memory to about 148.4 GiB and pagefile use to about 42.9 GiB. Different checkpoints should use fresh workers until cross-model worker recycling is implemented.
- Incremental state-dict release lowers the post-load steady committed set, but it does not remove the short full-checkpoint-to-shard conversion peak. That peak remains the main cold-start host-memory risk.

## Conclusion

Both upstream MiniMax H3 workflows pass the Windows dual-V100 FSDP/P2P acceptance gate with CPU offload enabled. The output is numerically finite, identical across ranks, temporally non-static and correctly encoded with audio. The next optimization gate is safe worker recycling when the diffusion checkpoint changes, followed by matched cold/warm performance and Turbo/INT8 comparisons.

Raw logs, one-second telemetry, contact sheets and generated media remain local test artifacts and are not stored in the repository.
