# O15.2 Dual-V100 MiniMax H3 VideoVAE Decode Acceptance

## Objective and conditions

Raylight 1.18.0 must parallelize standard Director `VAEDecode` temporal chunks between main-process GPU0 and Ray-worker GPU1 without modifying ComfyUI or MiniMaxH3Director. Acceptance requires output parity with official `vae.decode()`, at least 35% lower time for 107 frames at 1920x1088, a warm time no greater than 42 seconds, five-run lifecycle stability, both GPUs active below 7168 MiB in the isolated decode, and a real multi-segment Director pass.

The validated host uses native Windows 11 23H2, driver 577.00, two V100-SXM2-16GB GPUs in TCC with NVLink/P2P, Python 3.12.10, PyTorch 2.12.0+cu126, and Ray 2.57.0. The large latent shape is `[1,24,32,68,120]`; official output is `[1,3,107,1088,1920]` across six temporal chunks. Resource sampling is 0.25 seconds.

## Isolated decode results

| Path | Seconds | GPU0 peak | GPU1 peak | Peak utilization |
|---|---:|---:|---:|---:|
| Official ComfyUI native | 65.777 | 6993 MiB | 316 MiB | 99% / 0% |
| Dual run 1 | 42.069 | 6993 MiB | 6513 MiB | 99% / 100% |
| Dual run 2 | 39.601 | 6993 MiB | 6513 MiB | 99% / 100% |
| Dual run 3 | 39.665 | 6993 MiB | 6513 MiB | 99% / 100% |
| Dual run 4 | 40.400 | 6993 MiB | 6513 MiB | 99% / 100% |
| Dual run 5 / warm | 39.790 | 6993 MiB | 6513 MiB | 99% / 100% |

Warm speedup is `1.653x`, or `39.51%` lower time. Every one of the five runs has zero MSE and maximum absolute error, infinite PSNR, and finite values. Rank-1 CPU preload takes 15.898 seconds and overlaps sampling without loading the GPU. End physical-RAM growth from dual run 1 to run 5 is 2506.16 MiB, below the 4096 MiB gate; pagefile use is zero. Every configured gate passes.

## Director integration result

The two-segment one-step R2V regression completes in 209.095 seconds at the harness level and 207.64 seconds at the ComfyUI prompt level. Worker samples take 42.024/18.134 seconds and dual VideoVAE decodes take 9.191/9.210 seconds. Board peaks are 9431/9409 MiB with 100%/100% utilization. The output is a valid 864x480, 24 fps, 146-frame, 6.083-second MP4. Both operations report two ranks; the next diffusion segment reloads after VAE decode; no native fallback, Gloo timeout, actor death, or residual process remains.

The complete automated suite reports `797 passed, 235 warnings, 35 subtests passed`. Every Python file changed since the O15.1 baseline passes Ruff. A repository-wide Ruff scan still exposes 27 historical unused imports/variables in untouched Wan, GGUF, debug, and legacy SeedVR2 files; this feature release does not mechanically alter those unrelated paths.

## Rejected attempts and fixes

- Unloading rank-1 VAE after every decode was correct but produced a roughly 42.97-second second run, missing the absolute gate. The shipped lifecycle reuses it within one Director execution and releases it before the next diffusion sample.
- A standalone harness accidentally retained autograd state. Both benchmark and production decode now run under `torch.inference_mode()`.
- The first full integration exposed a rank-1-only auxiliary VAE load entering the normal two-rank model-budget collective and timing out after 60 seconds. The shipped bypass is scoped only to this auxiliary VAE load; diffusion/FSDP synchronization is unchanged.
- Independent review found native fallback running while the failed hybrid traceback still held temporary CUDA tensors. Fallback now runs after leaving the exception scope. It also required per-run quality gates for all five benchmark outputs and common auxiliary cleanup of the CPU preload cache.
- Modifying ComfyUI/Director or implementing general VAE FSDP was rejected as a larger, riskier compatibility surface.

## Boundary

This is temporal parallelism for standard Director VideoVAE `VAEDecode`, not VAE FSDP. Standard-node primary and Refine/FaceRefine decodes can use it. AudioVAE, direct internal SelfLift `vae.decode()`, non-Director, single-GPU, and unknown-VAE paths remain native. Failure logs a warning and invokes the original decoder.

Reproduce with `scripts/minimax-h3/benchmark_dual_vae_decode.py`. Long telemetry and generated media remain local artifacts rather than repository content.

