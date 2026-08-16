# MiniMax H3 REF2VA Smoke Validation - 2026-08-17

## Validation goal

Prove that the Windows V100 branch can execute the complete MiniMax H3 REF2VA path after reference-image and text preprocessing, with the diffusion model sharded across both GPUs and CUDA collectives carried by the custom P2P/NVLink backend.

## Test conditions

- Native Windows 23H2; two Tesla V100-SXM2-16GB in TCC mode; driver 577.00.
- Python 3.10.11; PyTorch 2.7.0+cu126; Ray 2.57.0; yunchang 0.6.4; comfy-kitchen 0.2.30; ComfyUI 0.31.0.
- FP8-scaled REF2VA diffusion model, NVFP4 Qwen3-VL text encoder, FP16 video VAE and FP32 audio VAE.
- FSDP enabled, Ulysses 2, synchronized Ulysses, TORCH_EFFICIENT attention, mmap enabled, CPU offload disabled.
- Both reference inputs use the upstream green-robot sample image.
- Reduced smoke workload: 0.2 MP, requested 2 seconds, 12 sampler steps.
- ComfyUI arguments: `--disable-cuda-malloc --reserve-vram 2`.

## Root-cause validation

Two earlier attempts failed while creating rank 0:

1. MiniMax preprocessing and Ray worker creation were independent graph branches, so they could compete for GPU 0 concurrently.
2. After adding an execution dependency, ComfyUI still retained the video VAE and text encoder for reuse. GPU 0 remained near 13.7 GiB and the worker import still had no CUDA initialization headroom.

The accepted fix adds an optional, value-agnostic `wait_for` input to `RayInitializer`. MiniMax positive conditioning connects to it, enforcing preprocessing-before-worker order. When connected, the initializer unloads ComfyUI preprocessing models and empties the main-process CUDA cache before starting Ray. The conditioning value is not transformed or sent through Ray by this input.

Observed transition in the accepted run:

- preprocessing peak on GPU 0: about 15.95 GiB;
- after the cleanup barrier and during worker import: about 1.25 GiB on GPU 0 and 1.05 GiB on GPU 1;
- after FSDP load and during sampling: both GPUs near 16 GiB and 99% utilization.

## Results

| Item | Result |
|---|---|
| End-to-end status | Pass |
| Cold end-to-end time | 268.85 s |
| Sampler progress | 12 steps; about 118 s to step 12; reported steady-state about 10.7 s/it |
| FSDP | Registered on both ranks; 684 wrappers; about 10.04 GiB local payload per rank |
| CUDA P2P health | About 59.27 GiB/s median per direction |
| GPU 0 peak | 16,177 MiB; 99% sustained sampling utilization |
| GPU 1 peak | 16,089 MiB; 99% sustained sampling utilization |
| Peak physical memory | 65,174 MiB |
| Peak pagefile use | 289 MiB |
| Commit telemetry | Invalid for this run because the selected Windows performance-counter class was unavailable; the sampler must use the OS virtual-memory counters on subsequent runs |
| Rank output comparison | Exact match; video and audio latent streams are finite FP32 tensors |
| Media | H.264 608x352 at 24 fps plus stereo AAC 32 kHz; 56 frames; duration 2.333 s |
| Frame checks | No black interval; all 56 frame hashes are distinct |

## Conclusion

The reduced REF2VA acceptance gate passes. This run proves that reference preprocessing, two-rank FSDP, CUDA P2P/NVLink sampling, audio decode, video decode and MP4 output all complete correctly on native Windows.

The full upstream 0.4 MP, 5-second profile is not yet accepted. The smoke run left only about 207 MiB on GPU 0 and 295 MiB on GPU 1 at peak, while physical RAM was also almost fully occupied. Scaling must therefore follow a measured memory-reduction step rather than immediately increasing resolution and duration.

Raw logs, one-second telemetry, the contact sheet and generated media remain local test artifacts and are intentionally not stored in the repository.
