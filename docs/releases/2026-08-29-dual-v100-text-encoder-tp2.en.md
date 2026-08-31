# 1.12.0 / O10: MiniMax H3 dual-V100 Text Encoder TP2

Date: 2026-08-31

Status: capacity, numerical, dual-compute, and real-workflow quality gates pass; the performance gate fails; the feature remains explicit and default-off

## User-visible changes

- Adds `Load MiniMax H3 Text Encoder (Dual V100 TP2)` with explicit `tp2` and `single_gpu_fallback` modes.
- Supports only `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`; existing `CLIPLoader` nodes, workflows, and defaults remain unchanged.
- GPU 0 runs the vision tower. The q/k/v/o/gate/up/down Linear operations in all 50 language layers execute concurrently on both V100s, merging row/column partials through CUDA P2P/NVLink SUM.
- A synchronized two-worker lease releases main-process GPU models and worker-resident Diffusion weights for the encoding window, then returns both GPUs after success or failure.

## Shipped implementation

- Direct row/column sharding of packed NVFP4 weights without materializing the full 32B model as FP16/FP32.
- Dual-device CUDA executor with strict shape, dtype, device, operation-order, and failure checks.
- Two-GPU P2P SUM, CUDA-event overlap profiling, synchronized worker lease state, and exclusion against sampling-model load.
- A MiniMax H3 conditioning CLIP proxy plus ordinary I2V/REF2VA and MMH3Director integration.
- Three-run cold/warm medians, GPU/RAM/commit/pagefile monitoring, stage timing, real video validation, and a five-segment stress run.

## Formal measurements

| Metric | One V100 | TP2 | Result |
|---|---:|---:|---|
| Cold median | 20.76 s | 27.17 s | TP2 30.9% slower; fail |
| Warm median | 10.37 s | 26.91 s | TP2 159.6% slower; fail |
| GPU 0 peak | 13,906 MiB | 9,467 MiB | 31.9% lower; pass |
| GPU 1 peak | no compute path | 9,449 MiB | below 14 GiB; pass |
| Peak commit growth | baseline | +6.9% | below 20%; pass |
| Page-file used delta | baseline | 0 | below 1 GiB; pass |

The full 50-layer profile contains 350 dual-device Linear operations. Cold/warm weighted kernel overlap is 96.49%/96.81%, and all 50 layers exceed the layer-level 80% overlap gate. A warm encode performs 100 P2P SUM operations with 4,453.12 MiB aggregate bidirectional payload in 70.70 ms, about 1.96% of language compute. The failure is therefore not an idle GPU 1 or slow NVLink; approximately 16.4 seconds of NVFP4 backbone shard loading dominates each encode.

## Real-workflow acceptance

Ordinary I2V Turbo8, ordinary REF2VA Turbo4, Director 864x480 three-second, Director 960x544 five-second, and Director 864x480 three-second x5-segment runs all finish. Their 56, 124, 73, 124, and 350 frames fully decode; unique-frame count equals total frames, blackdetect is zero, and audio/video contain no NaN/Inf. The five-segment run reuses one worker pair; GPU peaks rise only from about 9,367/9,323 MiB after segment one to 9,447/9,403 MiB after segment five. Worker exit returns both cards to about 10 MiB.

## Boundaries and upgrade guidance

- This is separate Text Encoder tensor parallelism, not FSDP and not a transparent 32 GB pool.
- Raylight FSDP/Ulysses continues to handle the Diffusion Model. Video/Audio VAE, upscalers, mux, and save are unchanged.
- Other checkpoints, Text Encoder LoRA, batch>1, more than two GPUs, multi-node, and training are unsupported.
- Select `tp2` only when lowering the one-card Text Encoder peak is more important than speed. Use ordinary `CLIPLoader` or `single_gpu_fallback` when speed is the priority.
- Future work must first reduce the NVFP4 loader hot path. It must not optimize the roughly 2% P2P SUM or attribute Diffusion FSDP end-to-end gains to Text Encoder TP2.

See the [test summary](../testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md) for artifact indexes and Gate A-F evidence.
