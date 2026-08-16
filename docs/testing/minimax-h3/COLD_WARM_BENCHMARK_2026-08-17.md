# MiniMax H3 Cold/Warm Benchmark — 2026-08-17

## Validation goal

Verify that checkpoint-aware worker reuse is both fast and stable for repeated MiniMax H3 use on native Windows. The benchmark must distinguish cold-start overhead from sampling time, prove that both Ray ranks participate, and reject outputs that are merely present but black or temporally frozen.

## Fixed conditions

- Windows 23H2, NVIDIA driver 577.00
- 2x Tesla V100-SXM2-16GB in TCC mode
- Python 3.10.11, PyTorch 2.7.0+cu126, xformers 0.0.30
- Ray 2.57.0, yunchang 0.6.4, ComfyUI 0.31.0
- Raylight source/deployment identity: `6b015c3d2c430476dc1178edaecd073899756750`
- ComfyUI flags: `--disable-cuda-malloc --reserve-vram 2`
- FSDP world size 2 with CPU offload and Windows CUDA P2P transport
- One cold run followed by two changed-seed warm runs in the same ComfyUI process
- I2V profile: 448x448, 39 frames, 12 steps
- REF2VA profile: 608x352, 56 frames, 12 steps

The harness stores the exact API prompt for every run, hashes the workflow and benchmark inputs, samples host/GPU resources, and waits for complete per-rank diagnostics without adding that wait to execution time.

## Results

### I2V

| Run | End-to-end | Sampling | Sampler total | Before sampler | Decode/write tail | Peak commit | Peak pagefile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 305.28 s | 162.99 s | 206.31 s | 86.50 s | 11.39 s | 127.22 GiB | 14.00 GiB |
| Warm 1 | 154.25 s | 131.22 s | 131.70 s | 16.52 s | 5.62 s | 87.05 GiB | 9.61 GiB |
| Warm 2 | 155.89 s | 131.92 s | 132.40 s | 17.01 s | 5.92 s | 87.09 GiB | 9.60 GiB |

Warm mean is 155.07 seconds end-to-end and 131.57 seconds sampling. Relative to cold, this removes about 49.2% of end-to-end time and 19.3% of measured sampling time. The two warm end-to-end results differ by about 1.1%.

Both warm runs reused worker PIDs `28348` and `32232`, emitted two `fsdp_already_registered` markers and emitted no checkpoint-change marker. Their peak committed memory differs by only 38 MiB; pagefile use slightly decreases rather than accumulating. GPU 0 and GPU 1 both reached 98–99% utilization during the runs.

### REF2VA

| Run | End-to-end | Sampling | Sampler total | Before sampler | Decode/write tail | Peak commit | Peak pagefile |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cold | 326.38 s | 193.36 s | 230.96 s | 84.75 s | 9.67 s | 122.97 GiB | 13.06 GiB |
| Warm 1 | 200.54 s | 185.63 s | 186.12 s | 4.03 s | 8.82 s | 87.23 GiB | 13.00 GiB |
| Warm 2 | 198.65 s | 184.60 s | 185.10 s | 3.59 s | 9.03 s | 87.17 GiB | 12.96 GiB |

Warm mean is 199.60 seconds end-to-end and 185.11 seconds sampling. Relative to cold, this removes about 38.8% of end-to-end time and 4.3% of measured sampling time. The two warm end-to-end results differ by under 1.0%.

Both warm runs reused worker PIDs `13116` and `32696`, emitted two `fsdp_already_registered` markers and emitted no checkpoint-change marker. Their peak committed memory differs by about 59 MiB; pagefile use again decreases slightly. Both GPUs reached at least 94% utilization during sampling. VAE decode remains a main-process, GPU-0-only stage, which is an existing Raylight capability boundary rather than failed FSDP participation.

## Output validation

All six accepted outputs contain H.264 video and AAC audio:

- I2V: three 448x448 files, each 39 frames and 1.625 seconds.
- REF2VA: three 608x352 files, each 56 frames and 2.333 seconds.
- `blackdetect` found no black interval.
- Every file has one unique decoded frame hash per frame (39/39 or 56/56), rejecting frozen-output false positives.
- Both ranks completed `sample_returned` and `sampler_return`; no rank timeout or result mismatch occurred.

## Conclusion

O2 is accepted. Same-checkpoint warm reuse is repeatable, substantially reduces end-to-end latency, and does not show progressive committed-memory, pagefile or VRAM growth across the two warm runs. The result also separates the actual dual-GPU sampling phase from the single-GPU VAE tail, so future speed claims can target the correct stage.

The next phase is O3: compare the current FP8-storage/FP32-compute baseline with a compatible official Turbo LoRA at matched workflow settings, seed and output checks. INT8 remains deferred until an exact compatible artifact is available.

## Local evidence

Raw logs, per-sample telemetry and generated media are intentionally kept outside Git:

- I2V report: `E:\ComfyUI-py310\logs\minimax-h3\o2\20260817-051049-i2v-smoke\benchmark.json`
- REF2VA report: `E:\ComfyUI-py310\logs\minimax-h3\o2\20260817-052247-ref2va-smoke\benchmark.json`
- Media: `E:\ComfyUI-py310\ComfyUI\output\video\raylight_o2`
