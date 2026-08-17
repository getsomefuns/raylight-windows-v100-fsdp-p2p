# MiniMax H3 Full Turbo Release Validation - 2026-08-17

## Validation goal

Validate the official FL2V Turbo 8-step and REF2V Turbo 4-step LoRAs at the original author's complete workflow settings on native Windows with two V100-SXM2-16GB GPUs. This is the full-resolution release gate after the shortened O3 repeatability benchmarks.

## Fixed conditions

- Windows 23H2, NVIDIA driver 577.00, two V100-SXM2-16GB GPUs in TCC mode.
- Python 3.10.11, PyTorch 2.7.0+cu126, Ray 2.57.0, yunchang 0.6.4, comfy-kitchen 0.2.30, ComfyUI 0.31.0.
- Raylight source/deployment identity: `719feb5fba3c3aadd05d5159b84ea9570472e1fb`.
- ComfyUI arguments: `--disable-cuda-malloc --reserve-vram 2`.
- Two FSDP workers, CPU offload, synchronized Ulysses 2, ring 1, TORCH_EFFICIENT attention and Windows CUDA P2P transport.
- Original prompts, seeds, input images, dimensions, durations and conditioning settings.
- Only the pinned official LoRA and its required step count differ from the accepted 20-step full baseline.

## Pinned Turbo presets

| Workflow | LoRA | Steps | SHA-256 |
|---|---|---:|---|
| I2V | `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` | 8 | `2339acdf19bfe123f46b971ea35d367a84adb85de43627e1eceafa5a5b2b111e` |
| REF2VA | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` | 4 | `5b9ab5ade15d0775676d01a907268a69a1468dc6033b3b0d3ded5502f3ebb84c` |

## Results

| Item | I2V Turbo 8 | REF2VA Turbo 4 |
|---|---:|---:|
| Output settings | 640x640, 56 frames | 864x480, 124 frames |
| Cold end-to-end | 387.98 s | 424.59 s |
| Before sampler | 128.86 s | 88.50 s |
| Main sampling | 212.01 s | 265.35 s |
| Sampler total | 245.94 s | 298.96 s |
| Decode/write tail | 12.04 s | 36.25 s |
| FSDP | 684 wrappers/rank | 684 wrappers/rank |
| Sampling VRAM | about 6.8/6.6 GiB | about 12.7/12.5 GiB |
| Recorded GPU peak | 15,875/6,605 MiB | 15,925/12,508 MiB |
| GPU utilization peak | 100%/100% | 100%/100% |
| Peak physical memory | 63.82 GiB | 63.87 GiB |
| Peak committed memory | 129.13 GiB | 128.78 GiB |
| Peak pagefile use | 1.31 GiB | 7.66 GiB |

The accepted 20-step full baselines are 642.01 seconds for I2V and 1324.98 seconds for REF2VA end-to-end. Turbo reduces cold end-to-end time by 39.6% and 68.0% respectively. Approximate main sampling time falls from 438 to 212 seconds for I2V and from 1132 to 265 seconds for REF2VA, reductions of about 51.6% and 76.6%.

The comparison reflects the official reduction in sampling steps. It does not assert that Turbo is visually identical to the 20-step baseline.

## Distributed and numerical checks

- Both ranks register 684 FSDP wrappers and finish the complete sampling call.
- Both GPUs participate continuously during sampling and reach 100% utilization.
- CUDA P2P profiles complete successfully; no Gloo tensor staging is used for the project collectives.
- Each Turbo LoRA loads 208 grouped sidecar adapters with zero unsupported entries.
- No CUDA OOM, rank mismatch, collective timeout, meta-tensor error or numerical failure occurs.
- Peak committed memory and pagefile use remain within the accepted 135 GiB/12 GiB cold-workflow operating gate.

## Media validation

- I2V: H.264 640x640 at 24 FPS, 56/56 unique decoded frame hashes, 32 kHz stereo AAC, 2.333 seconds.
- REF2VA: H.264 864x480 at 24 FPS, 124/124 unique decoded frame hashes, 32 kHz stereo AAC, 5.167 seconds.
- `blackdetect` reports no black interval in either output.
- Audio is non-silent and `astats` reports zero NaNs.

These automated checks reject black, frozen, missing-audio and non-finite outputs. Final aesthetic and prompt-adherence comparison remains a user visual decision.

## Conclusion

O4 technical acceptance is complete. Both official Turbo presets run at the original author's full workflow settings on the Windows dual-V100 FSDP/P2P branch and exceed the project's 20% speed target. Keep the 20-step workflows as quality baselines and offer Turbo as the recommended speed preset until user visual review selects a default.

## Local evidence

- I2V report: `E:\ComfyUI-py310\logs\minimax-h3\o4\20260817-075033-i2v-full-fl2v-turbo-8step\benchmark.json`
- REF2VA report: `E:\ComfyUI-py310\logs\minimax-h3\o4\20260817-075829-ref2va-full-ref2v-turbo-4step\benchmark.json`
- I2V media: `E:\ComfyUI-py310\ComfyUI\output\video\raylight_o3\minimax_h3_i2v_fl2v-turbo-8step_run0_00002_.mp4`
- REF2VA media: `E:\ComfyUI-py310\ComfyUI\output\video\raylight_o3\minimax_h3_ref2va_ref2v-turbo-4step_run0_00002_.mp4`
