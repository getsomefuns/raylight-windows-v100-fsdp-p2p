# O11: Efficient attention for MiniMax H3 dual-V100 Text Encoder

> Status: the architecture direction is user-approved; this written specification awaits user review; implementation has not started.
>
> Date: 2026-09-01
>
> Canonical worktree: `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`

## 1. Goal

O11 fixes the O10 dual-GPU Qwen3-VL Text Encoder path silently falling from PyTorch SDPA to math attention, materializing quadratic attention tensors, and running out of VRAM before diffusion sampling on long multimodal inputs.

The final capacity gate is a real Text Encoder workload, not a synthetic attention probe. It must contain:

- one reference video whose measured duration is 15.0 seconds within one source-frame tolerance;
- seven distinct reference images;
- one meaningful prompt containing exactly 7,000 Unicode characters after normalization;
- the MiniMax H3 Qwen3-VL-32B NVFP4/AWQ Text Encoder;
- native Windows, two Tesla V100-SXM2-16GB GPUs in TCC mode, PyTorch 2.7.0+cu126, and xFormers 0.0.30.

Completion means that a real ComfyUI workflow returns finite and structurally correct conditioning without a Text Encoder OOM, NaN/Inf, math-attention fallback, or residual TP2 language weights after cleanup. Passing only a 6,992- or 18,000-token microbenchmark is not completion.

## 2. Non-goals

- O11 does not guarantee that the maximum conditioning workload can subsequently complete diffusion sampling; sampler capacity is a separate boundary.
- The vision tower remains GPU0-only.
- ComfyUI core, MiniMaxH3Director, the NVIDIA driver, Python, PyTorch, CUDA, xFormers, and model formats are not upgraded or forked.
- Training, backward propagation, more than two GPUs, cross-host Text Encoder execution, and NCCL are out of scope.

## 3. Confirmed root cause and baseline

O10 commit `eb7e04b` introduced the TP2 Qwen3-VL language block and directly called `F.scaled_dot_product_attention(..., enable_gqa=True)`.

Each rank has 32 query heads but only four key/value heads. The installed PyTorch Efficient kernel requires matching Q/K/V head counts. Flash is unavailable on this V100 build and cuDNN Attention is disabled, so the dispatcher selects math attention.

For the observed 6,992-token request, one FP32 attention tensor per rank is:

```text
32 × 6992 × 6992 × 4 bytes ≈ 5.83 GiB
```

This exactly matches the failed allocation. The 221-token O10 acceptance workload hid the defect because the corresponding tensor was only about 6 MiB.

TP2 also materializes a `(sequence, sequence)` FP32 causal mask. It is about 1.21 GiB at 18,000 tokens and 2.92 GiB at 28,000 tokens and must be removed.

Read-only local probes already measured:

| Sequence | Backend | Attention peak | One-call time |
|---:|---|---:|---:|
| 6,992 | PyTorch Efficient with expanded K/V | 468 MiB | 0.056 s |
| 6,992 | native xFormers GQA | 360 MiB | 0.047 s |
| 18,000 | PyTorch Efficient with expanded K/V | 1,200 MiB | 0.264 s |
| 18,000 | native xFormers GQA | 917 MiB | 0.256 s |

The xFormers GQA small-case comparison against math SDPA produced `max_abs=1.67e-6`, `mean_abs=6.44e-8`, and passed at `atol=rtol=1e-4`.

## 4. Preserved two-stage lifetime

Vision and language execution remain strictly sequential:

```text
GPU0: token embedding and Qwen3-VL vision tower
GPU1: idle
        ↓
move visual embeddings, DeepStack features, and token tags to CPU
        ↓
destroy vision and token-embedding weights and clear GPU0
        ↓
GPU0 + GPU1: load half of the language weights each and execute 50 TP2 layers
        ↓
move conditioning to CPU, destroy language shards, return worker GPU leases
```

Efficient attention changes only the language backbone. The GPU0 vision peak and duration remain separate acceptance measurements because a 15-second video and seven images may expose a new preprocessing bottleneck.

## 5. Backend architecture

Add a Text Encoder-specific backend enum:

```text
AUTO
XFORMERS_GQA
TORCH_EFFICIENT
MATH_DEBUG
```

One backend is selected once per encode and used for all 50 layers. Per-layer silent switching is forbidden.

### 5.1 XFORMERS_GQA

This is the preferred backend on the target dual-V100 machine.

- Q uses `[B, S, 4, 8, 128]` five-dimensional GQA layout.
- K/V use `[B, S, 4, 1, 128]` and view/expand broadcasting instead of materializing 32 heads.
- Call `xformers.ops.memory_efficient_attention`.
- Use `xformers.ops.LowerTriangularMask()` for causality.
- Forward inference only; no GQA backward dependency.

### 5.2 TORCH_EFFICIENT

This is the supported fallback when native xFormers GQA is unavailable.

- Materialize the local four K/V heads as 32 heads.
- Force `SDPBackend.EFFICIENT_ATTENTION` with `torch.nn.attention.sdpa_kernel`.
- Use `is_causal=True` and no dense square mask.
- Fail explicitly if Efficient is unavailable; never continue to Math.

### 5.3 MATH_DEBUG

Math is limited to numerical diagnostics with a fixed 2,048-token ceiling. Longer inputs fail with an actionable message and cannot use Math as a workflow fallback.

### 5.4 AUTO

AUTO is the recommended default:

1. Run and cache a small device-capability probe before loading 50 language layers.
2. Select XFORMERS_GQA when GQA plus causal bias succeeds.
3. Otherwise probe TORCH_EFFICIENT with expanded K/V.
4. If both fail, report both rejection reasons.
5. Never select Math automatically.

AUTO must select XFORMERS_GQA on the target machine.

## 6. Node API and compatibility

Only `Load MiniMax H3 Text Encoder (Dual V100 TP2)` changes:

- add optional `text_encoder_attention`, defaulting to `AUTO`;
- O10 workflows without the field load as AUTO;
- preserve the existing `execution_mode`, `clip_name`, and `ray_actors_init` semantics;
- do not change the RayInitializer schema.

`RayInitializer.XFuser_attention=TORCH_EFFICIENT` continues to control diffusion/xFuser sampling only. Tooltips and documentation must make clear that it does not select Text Encoder attention.

## 7. Causality and valid tokens

Remove the current dense FP32 `_causal_mask(sequence)`.

- xFormers uses `LowerTriangularMask`.
- Torch uses `is_causal=True`.
- Existing single-batch MiniMax conditioning semantics must remain unchanged.
- Padding, if present, must be trimmed or represented by a structured non-materialized bias. Reintroducing a float `[S,S]` mask is forbidden.
- Returned sequence positions and `minimax_token_tags` must remain tokenizer-compatible.

## 8. Observability

Each normal encode emits one aggregate record containing:

- requested and selected backend;
- total, text, image, and video token counts;
- source/reference frame counts, 2-FPS Qwen frames, and video block count;
- per-image visual token count;
- preprocessing, vision, backbone load, language, and cleanup durations;
- GPU0 vision peak and both language-stage GPU peaks;
- whether Math fallback was detected;
- post-cleanup GPU usage.

Per-layer diagnostics remain off by default. Full prompts and large tensors are never logged.

## 9. Error handling and cleanup

- Backend capability failures occur before loading language weights.
- OOM, non-finite values, rank divergence, and shape mismatches preserve their original cause.
- Failure paths release the backbone, Q/K/V temporaries, visual GPU copies, and external worker leases.
- A failed run may be retried with another explicit backend without restarting ComfyUI.
- xFormers failures never trigger a hidden Math retry on an already-full GPU.

## 10. Frozen acceptance workload

Before the first formal workflow run, create a local manifest containing input-relative paths, SHA-256 hashes, media metadata, and workflow hashes. Do not store absolute user paths.

### 10.1 Video

- exactly one real, decodable reference video;
- measured duration within one source frame of 15.0 seconds;
- at least 854×480;
- not a black, static, or meaningless repeated-frame capacity dummy;
- record source FPS/frames, 24-FPS frames, aligned consumed frames, and final video visual tokens.

MiniMax may align the source to the closest `17k+5` frame boundary. Reports must distinguish the 15-second input from the actually consumed aligned duration.

### 10.2 Images

- exactly seven real images with distinct content and SHA-256 values;
- each short edge at least 480 pixels;
- `ref_image_size=match`;
- record every preprocessed size and visual-token count.

### 10.3 Prompt

- normalize line endings to LF, then require `len(prompt) == 7000`;
- contain meaningful subject, action, scene, style, audio, shot, or continuity instructions;
- repeated single characters, whitespace, random text, and meaningless duplicated paragraphs are forbidden;
- record character count, tokenizer text-token count, and SHA-256;
- keep the prompt body out of public summaries.

### 10.4 Workflow

- execute through ComfyUI, not a standalone Python probe;
- use `MiniMax H3 Reference to Video` and the O11 dual-GPU CLIP Loader;
- use 1056×608, `length=362`, and 24 FPS;
- connect one reference video and seven images;
- terminate at conditioning and do not connect a diffusion sampler;
- save a reproducible workflow under `example_workflows/` and deploy a copy to the user workflows directory;
- hash both UI workflow and API prompt JSON.

## 11. Acceptance criteria

### Gate A: backend correctness

- Test XFORMERS_GQA, TORCH_EFFICIENT, AUTO, and MATH_DEBUG routing.
- At 64, 221, 256, and 2,048 tokens, both efficient backends match Math within `max_abs <= 1.25e-4` for FP16 (covering the measured `1.220703125e-4` quantization step), `max_abs <= 1e-4` for FP32, and `mean_abs <= 1e-5` for both.
- All outputs are finite.
- MATH_DEBUG rejects 2,049 tokens.
- AUTO selects Torch when xFormers is simulated unavailable and fails explicitly when neither backend works.

### Gate B: V100 attention capacity

Run 6,992-, 18,000-, and 28,000-token probes on a real V100:

- both efficient backends complete without a dense `[S,S]` mask or Math call;
- at 18,000 tokens, xFormers peak is at most 1.10 GiB and Torch at most 1.40 GiB;
- at 28,000 tokens, xFormers peak is at most 1.65 GiB and Torch at most 2.10 GiB;
- xFormers must use less peak memory than Torch at equal sequence length.

These are kernel gates, not substitutes for the full 50-layer gate.

### Gate C: GPU0 vision stage

Using the frozen 15-second video, seven images, and 7,000-character prompt:

- GPU0 completes token embedding, vision, and DeepStack without OOM or non-finite values;
- GPU1 does not incorrectly load a full language model during vision;
- all prepared tensors move to CPU before language load;
- vision and token-embedding GPU references are released before language weights load;
- token breakdown, duration, and GPU0 peak are recorded.

### Gate D: real maximum Text Encoder hard gate

Use the exact frozen ComfyUI workflow and hashed inputs. Run AUTO/XFORMERS_GQA once cold and once with warm file cache, and force TORCH_EFFICIENT at least once.

- The input is verifiably one 15-second video, seven distinct images, and a meaningful 7,000-character prompt.
- `encode_from_tokens_scheduled` returns conditioning without entering sampling or OOM.
- Conditioning is `[1, S, 5120]`, with `S` equal to the logged expanded sequence.
- Conditioning and token tags are finite and length-consistent.
- Both GPUs participate in all 50 layers and post-collective rank outputs agree.
- Per-GPU Text Encoder peak is no more than 15.0 GiB.
- Logs prove no Math fallback and no dense square mask.
- Language weights and temporaries are released after encoding.
- Two consecutive executions do not increase final process commit by more than 512 MiB.

Any failed item prevents O11 from claiming support for the maximum workload.

### Gate E: integration and regression

- The original 11.33-second, 854×480, 6,992-token OOM case passes.
- The O10 221-token TP2 smoke still passes.
- After maximum conditioning succeeds, complete one reduced 3-second/480p/four-step sampling smoke to validate the conditioning contract. It does not replace Gate D.
- Existing CLIPLoader, O6, O8, O9, LTX, MiniMax I2V/REF2VA, and MMH3Director regressions pass.
- O10 workflows without the new field load as AUTO.

### Gate F: release and records

- AUTO is default and selects xFormers GQA on the target machine.
- English and Chinese README, user guide, release notes, and testing summaries match code and measured behavior.
- Testing records stage timings, per-GPU peaks, RAM/commit/pagefile peaks, and token breakdowns, including failed experiments.
- Code changes occur only in the canonical worktree and deploy only after review and tests.
- Deployed tracked-file hashes match the canonical worktree before any GitHub push.

## 12. Stop conditions

- If attention passes but Gate D OOMs in the MLP, preserve the workload and begin an MLP-lifetime optimization stage; do not lower the 15-second/seven-image/7,000-character requirement.
- If vision OOMs first, optimize streaming/lifetime of visual and DeepStack outputs; do not reduce references or truncate text.
- If maximum conditioning passes but sampling OOMs, O11 may close only as “Text Encoder capacity passed; sampler boundary not passed.”
- Increasing pagefile, moving production CUDA attention to CPU, disabling references, or truncating tokens cannot satisfy acceptance.

## 13. Next step after design review

After the user reviews this design, write a separate executable O11 implementation plan. Development begins with a failing regression test, then implements backend abstraction and removal of the dense mask before progressing through Gates A–F. This document does not authorize code changes before review.
