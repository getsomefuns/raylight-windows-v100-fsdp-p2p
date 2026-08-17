# MiniMax H3 Speed/Quality Variants - 2026-08-17

## Validation goal

Determine whether the official MiniMax H3 Turbo LoRAs are compatible with the Windows V100 FSDP/P2P branch, quantify their repeatable benefit against a 20-step baseline, and reject outputs that are black, frozen, silent, non-finite, or produced by only one rank.

## Fixed conditions

- Windows 23H2, NVIDIA driver 577.00
- 2x Tesla V100-SXM2-16GB in TCC mode
- Python 3.10.11, PyTorch 2.7.0+cu126, xformers 0.0.30
- Ray 2.57.0, yunchang 0.6.4, ComfyUI 0.31.0
- Raylight source/deployment identity: `719feb5fba3c3aadd05d5159b84ea9570472e1fb`
- ComfyUI flags: `--disable-cuda-malloc --reserve-vram 2`
- FSDP world size 2, CPU offload, Windows CUDA P2P transport
- One cold run followed by two changed-seed warm runs in one ComfyUI process
- I2V: 448x448, 39 frames; base 20 steps versus official FL2V Turbo 8-step LoRA
- REF2VA: 608x352, 56 frames; base 20 steps versus official REF2V Turbo 4-step LoRA

The baseline and Turbo run at the same model, inputs, dimensions, frame count and per-run seed. Only the official LoRA and its required step count change. This is a speed/quality preset comparison, not a claim that 8- or 4-step output is numerically equivalent to the 20-step baseline.

## Pinned artifacts

Repository revision: `Comfy-Org/MiniMax-H3@cec22ac7545ee166df6af79fda42bd41558f8558`.

| Variant | File | Bytes | SHA-256 |
|---|---|---:|---|
| FL2V Turbo 8-step | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 1,956,193,000 | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` |
| REF2V Turbo 4-step | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 1,956,193,000 | `5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c` |

The first Turbo load exposed an FSDP integration defect: ComfyUI's ordinary full-load path attempted to move the complete FSDP meta module to one GPU after LoRA attachment. Commit `719feb5` preserves ComfyUI hook injection while keeping FSDP shard ownership and CPU-offload accounting intact. The focused regression tests and full suite pass before the accepted runs.

## Results

### I2V

| Variant/run | End-to-end | Sampling | Sampler total | Before sampler | Decode/write tail | Peak commit | Peak pagefile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base 20 cold | 389.13 s | 255.77 s | 294.37 s | 86.93 s | 7.17 s | 124.14 GiB | 6.32 GiB |
| Base 20 warm 1 | 238.95 s | 226.80 s | 227.26 s | 4.89 s | 5.69 s | 87.19 GiB | 5.98 GiB |
| Base 20 warm 2 | 237.02 s | 225.80 s | 226.26 s | 2.97 s | 5.85 s | 87.13 GiB | 5.98 GiB |
| Turbo 8 cold | 257.60 s | 108.15 s | 139.69 s | 110.63 s | 6.67 s | 128.18 GiB | 1.27 GiB |
| Turbo 8 warm 1 | 107.32 s | 96.38 s | 96.86 s | 3.52 s | 5.68 s | 90.89 GiB | 1.27 GiB |
| Turbo 8 warm 2 | 107.37 s | 95.93 s | 96.40 s | 3.34 s | 6.04 s | 90.77 GiB | 1.27 GiB |

Warm means are 237.99 seconds for base and 107.35 seconds for Turbo end-to-end, a repeatable 54.9% reduction. Sampling falls from 226.30 to 96.15 seconds, a 57.5% reduction. Both GPUs reach at least 97% utilization; warm VRAM peaks near 7.0 GiB on GPU 0 and 5.5 GiB on GPU 1.

### REF2VA

| Variant/run | End-to-end | Sampling | Sampler total | Before sampler | Decode/write tail | Peak commit | Peak pagefile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base 20 cold | 342.90 s | 203.23 s | 242.12 s | 89.08 s | 10.86 s | 123.83 GiB | 13.74 GiB |
| Base 20 warm 1 | 200.88 s | 186.36 s | 186.82 s | 4.57 s | 8.45 s | 87.44 GiB | 13.68 GiB |
| Base 20 warm 2 | 198.39 s | 186.00 s | 186.42 s | 3.20 s | 8.36 s | 87.48 GiB | 13.63 GiB |
| Turbo 4 cold | 217.96 s | 84.99 s | 117.41 s | 88.49 s | 10.14 s | 127.86 GiB | 1.49 GiB |
| Turbo 4 warm 1 | 81.10 s | 67.17 s | 67.63 s | 4.15 s | 8.67 s | 91.40 GiB | 1.30 GiB |
| Turbo 4 warm 2 | 83.04 s | 67.10 s | 67.54 s | 4.17 s | 9.98 s | 91.28 GiB | 1.19 GiB |

Warm means are 199.64 seconds for base and 82.07 seconds for Turbo end-to-end, a repeatable 58.9% reduction. Sampling falls from 186.18 to 67.14 seconds, a 63.9% reduction. Both GPUs reach at least 97% utilization; warm VRAM peaks near 7.0 GiB on GPU 0 and 5.9 GiB on GPU 1.

## Correctness and media validation

- All 12 runs complete on both ranks with stable worker PIDs; both warm runs emit two `fsdp_already_registered` markers and no checkpoint-change marker.
- The Turbo LoRA mapping loads 208 grouped sidecar adapters with zero unsupported entries.
- CUDA P2P profiles complete successfully, and both GPUs reach high compute utilization.
- I2V outputs contain 448x448 H.264 video, 39/39 unique decoded frame hashes, and 32 kHz stereo AAC.
- REF2VA outputs contain 608x352 H.264 video, 56/56 unique decoded frame hashes, and 32 kHz stereo AAC.
- `blackdetect` reports no black interval in any output.
- Audio is non-silent and `astats` reports zero NaNs in all outputs.
- No rank mismatch, CUDA OOM, P2P failure, meta-tensor failure or numerical error occurs in the accepted runs.

## Conclusion

O3 is accepted. Official Turbo LoRAs are compatible with quantized FSDP CPU offload after the FSDP full-load/hook fix. The benefit is repeatable and far exceeds the project's 20% speed target for these smoke-sized workflows.

Adopt FL2V Turbo 8-step and REF2V Turbo 4-step as speed presets. Retain the 20-step base workflows as quality baselines because structural media checks cannot prove equivalent visual or semantic quality. Final default selection requires full-specification outputs and user visual review.

## Local evidence

Raw logs, telemetry and generated media remain outside Git:

- I2V base: `E:\ComfyUI-py310\logs\minimax-h3\o3\20260817-061936-i2v-smoke-base20\benchmark.json`
- I2V Turbo: `E:\ComfyUI-py310\logs\minimax-h3\o3\20260817-071315-i2v-smoke-fl2v-turbo-8step\benchmark.json`
- REF2VA base: `E:\ComfyUI-py310\logs\minimax-h3\o3\20260817-072433-ref2va-smoke-base20\benchmark.json`
- REF2VA Turbo: `E:\ComfyUI-py310\logs\minimax-h3\o3\20260817-073801-ref2va-smoke-ref2v-turbo-4step\benchmark.json`
- Media: `E:\ComfyUI-py310\ComfyUI\output\video\raylight_o3`
