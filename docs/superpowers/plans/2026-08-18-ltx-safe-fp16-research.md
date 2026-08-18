# O7 LTX/LTXAV Model-Specific Safe FP16 Research Plan

**Status:** preliminary feasibility plan; implementation is not scheduled until O6 closes and this plan is recalibrated from O6 evidence.

## Goal

Determine whether the MiniMax H3 strategy of FP32 numerical islands around FP16 matrix-multiplication branches can be adapted safely to LTX/LTXAV on native Windows dual V100 FSDP/P2P. This is not permission to enable global FP16 inference for LTX.

## Evidence already available

- Global `weight_dtype=fp16` was tested on LTXAV and rejected: video was black and the audio path produced NaN/Inf.
- Adding FP16 to the global ComfyUI inference-dtype allowlist did not make the result valid and was reverted.
- The accepted LTX path uses BF16/FP32 inference semantics plus V100-specific FP8/BF16 fallbacks.
- LTX and LTXAV use `comfy.ldm.lightricks` blocks, not MiniMax H3 `DiTBlock`/`MLP`, so the MiniMax patch cannot be copied structurally.
- LTXAV contains separate video and audio residual streams, self/cross attention, AdaLN/RMSNorm modulation, cross-modal gates and separate output heads. A valid video-only tensor cannot establish audio correctness.
- Raylight replaces the LTX model forward and attention path for context parallelism, while FSDP wraps the Diffusion Model in each actor. Any precision policy must therefore be installed in both actors before wrapping and remain compatible with the Raylight forward replacements.

## Working hypothesis

LTX may benefit from keeping residual accumulation, normalization/reduction, timestep/AdaLN modulation, positional calculations, cross-modal gates and final projections in FP32 while casting only selected attention and feed-forward matrix multiplications to FP16. The exact boundaries must be measured; MiniMax H3 constants and patched methods are not assumed to apply.

## Proposed stages

### O7.1: dtype and finite-value trace

- Add opt-in diagnostics around every LTX/LTXAV transformer block without changing compute dtype.
- Record dtype, shape, maximum absolute value and finite status for video/audio residuals, normalized inputs, attention/MLP branches, gates and outputs on both ranks.
- Locate the first divergence or non-finite value when a reduced experimental FP16 path is enabled.

### O7.2: explicit precision policy

- Define named FP32 islands and FP16-eligible branches per LTX model family.
- Keep the default BF16/FP32 workflow byte-for-byte unchanged.
- Expose an LTX-specific experimental loader option; do not edit the global `supported_inference_dtypes` allowlist.
- Reject unsupported LTX revisions and non-V100 combinations with an actionable error.

### O7.3: Raylight FSDP/P2P integration

- Install the policy in both Ray actors before model construction/FSDP wrapping.
- Prove FP8 storage, logical dtype metadata, FSDP wrapper count and P2P all-gather remain unchanged.
- Prove both ranks return element-consistent finite tensors and no collective falls back to host-staged tensor transport.

### O7.4: separate video and audio acceptance

- Validate video latent, Video VAE output, audio latent, Audio VAE/vocoder output and final mux separately.
- Reject black/frozen video, NaN/Inf, silence, clipping, rank mismatch or dtype leakage into an FP32 island.
- Test LTX video-only before LTXAV, but never infer LTXAV acceptance from video-only success.

### O7.5: performance and release decision

- After O6, create a fresh matched local LTX baseline and record model load, preprocessing, every sampler's s/it, VAE/audio decode, mux/write, GPU/host memory and P2P data.
- Set numeric O7 performance gates only from that local baseline and the measured O6 overhead model.
- Keep the feature experimental unless correctness, memory, both-rank and performance gates all pass.

## Quality gates

- No change to accepted default LTX/LTXAV workflows or their dtype policy.
- No global FP16 allowlist change.
- FP32-island and FP16-branch behavior proven by automated dtype assertions on both ranks.
- Finite rank-matched video and audio latent outputs.
- Valid, temporally varying, non-black video and finite non-silent audio.
- FSDP sharding and CUDA P2P/NVLink data path remain active.
- No performance claim until a matched local baseline and full stage timings exist.

## Recalibration gate

After O6 completes, revise this plan using the proven worker patch mechanism, dtype tracing overhead, LoRA behavior, FSDP/FP8 interaction and measured stage bottlenecks. O7 implementation does not begin from this preliminary document alone.
