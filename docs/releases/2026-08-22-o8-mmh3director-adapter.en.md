# 1.10.0: MiniMaxH3Director MODEL adapter

[English](2026-08-22-o8-mmh3director-adapter.en.md) | [Simplified Chinese](2026-08-22-o8-mmh3director-adapter.zh-CN.md)

## What ships

1.10.0 adds the explicit opt-in `Ray MODEL Adapter (MMH3Director)` node. It sends Director-prepared conditioning, latent, noise, sigmas, and sampling arguments to the existing dual-V100 FSDP2 workers and returns latent through the CUDA P2P/NVLink data plane. Director remains the only panel for seed, steps, sampler, scheduler, shifts, geometry, FPS, duration, story segments, and output settings.

The bridge intercepts only Raylight's exact proxy type. Ordinary `MODEL` objects, legacy `XFuserSamplerCustom`/`Advanced`, old JSON, and public node schemas remain unchanged. An unsupported proxy request fails explicitly: it never silently falls back to local single-GPU execution and never samples once in Director and again in Raylight. MiniMaxH3Director source is unmodified.

## Validated capability

- t2v, i2v, fl2v, r2v, v2v, and rv2v.
- Same-checkpoint Refine, multi-segment R2V, and warm worker/FSDP-shard reuse.
- Paired worker reclamation and recreation on FL2VA/REF2VA checkpoint changes.
- Final `RayKill` reclamation of workers, Ray control processes, CUDA contexts, VRAM, and most process commit.
- Identical video/audio bitstream and final decoded-frame MD5 across same-seed cold and warm runs.

The 1280x704, 124-frame, 24 FPS, 5.167-second T2V runs at 595.10 seconds and 42.2 s/it cold, then 470.82 seconds and 42.0 s/it warm. The upstream single-V100 cold reference is 41:21 and 271.81 s/it. This is about 4.17x/6.44x faster cold end-to-end/per-step and about 5.27x/6.47x on the interactive warm path.

One-cold/three-warm regression covers MiniMax I2V Turbo8, REF2VA Turbo4, LTX 2.3, O6 safe-FP16, and O6 FP32. No unexplained warm sampling median regression exceeds 5%. LTX stage two is about +4.82%, close to but inside the gate.

See the [O8 English validation summary](../testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.md) for the complete method and evidence.

## Key fixes

- Release main-process preprocessing models through standard ComfyUI APIs after Director text encoding and before remote sampling, preventing CLIP/VAE overlap with rank 0 on GPU 0.
- Add opt-in phase timing for preprocessing unload, Python GC, and CUDA cache cleanup. This identified REF2VA E2E outliers as main-process host offload rather than rank skew or P2P degradation.
- Strictly validate operation ID, rank, shape, dtype, finite values, and rank agreement. Each multi-segment pass has one remote operation.

## Boundaries

- The Diffusion Model is FSDP-sharded. CLIP/Text Encoder, Video VAE, Audio VAE, preprocessing, mux, and save remain in the main process.
- Multi-segment generation reuses workers, while the Director/ComfyUI main process can retain reloadable CLIP/VAE weights. Longer jobs must be sized against host RAM and commit/pagefile as well as VRAM.
- Same-model Refine is validated. Custom refine paths that require another checkpoint, different weights, or cannot be represented by standard `comfy.sample.sample()` are outside this guarantee.
- Preview consumes the returned latent in Director/ComfyUI; remote workers do not expose a separate per-frame preview stream.
- Compatibility is pinned to MiniMaxH3Director 1.0.0 at commit `d655912`; an incompatible seam fails explicitly.

## Compatibility and rollback

Old workflows need no migration. Use tag `v1.9.0-pre-mmh3director` for a full pre-adapter rollback. The repository example is `example_workflows/MMH3Director_Windows_V100_FSDP.json`; a local user copy normally lives at `ComfyUI/user/default/workflows/MMH3Director.json`.
