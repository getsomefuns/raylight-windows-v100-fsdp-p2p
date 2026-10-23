# Raylight Windows Dual-V100 FSDP + CUDA P2P

[简体中文](README.md) | [English](README_EN.md)

> **Experimental Preview**
> Native-Windows Raylight branch for two Tesla V100-SXM2-16GB GPUs in TCC mode, with FSDP2 weight sharding and CUDA IPC/P2P communication over NVLink.

This branch extends the previous Windows P2P/Ulysses work with the CUDA `all_gather_into_tensor` data path required by FSDP2. The validated LTX 2.3 and MiniMax H3 Diffusion Models are genuinely sharded across two V100 GPUs. Temporary weight gathering and Ulysses tensor exchange use CUDA IPC, cross-process CUDA events, and GPU P2P/NVLink. Gloo/TCPStore remains the rendezvous and control plane.

Version 1.18.3 fixes a long-reference SelfLift OOM caused by shared QKV backing storage overlapping a second full-size Ulysses `all_to_all` output. MiniMax H3 now uses bounded independent Q/K/V projection, an in-place two-rank Windows CUDA P2P exchange, and consumed-Query output reuse for dense xFormers Attention. The 362-frame 768×1344 hardware gate completes at roughly 233,224 high-resolution positions with observed board peaks of 16,083/16,213 MiB. A full-lifecycle recorder now writes Windows commit, process-private, PyTorch allocator, and whole-device VRAM summaries to JSONL. This is a capacity-stability release, not a speed optimization. See the [1.18.3 English release note](docs/releases/2026-10-23-long-reference-oom-memory-recording.en.md).

Version 1.18.2 fixes a native Windows `0xC0000005` access violation in MiniMax H3 FP8 FSDP CPU offload. The old loader built every quantized local shard on CUDA and immediately copied it back to CPU through `comfy_kitchen QuantizedTensor.cpu()`. CPU-offloaded ranks now build their local payload directly on CPU and convert only the outer DTensor wrapper, eliminating the large CUDA-to-CPU round trip. A real REF2VA FP8 checkpoint plus 0.65 Turbo4 LoRA preflight registers all 684 FSDP wrappers on both V100s at about 60.10 GiB/s P2P bandwidth. The full suite reports `800 passed, 35 subtests passed`. See the [1.18.2 English release note](docs/releases/2026-10-19-windows-fsdp-quant-cpu-offload-fix.en.md).

Version 1.18.1 fixes a native Windows `0xC0000005` access violation that could terminate a long multi-segment Director run after repeatedly loading the 15.7 GB NVFP4 Text Encoder. On Windows, TP2 embedding, vision-tower, and 50-layer language-backbone weights are now read one tensor at a time into owned CPU buffers instead of mapping the whole checkpoint through `safetensors.safe_open()`. Linux retains the native mmap path. Real dual-V100 text-only and REF2VA probes pass at about 7.35 GiB peak per GPU and release to about 17–25 MiB. Model values, TP2 sharding, attention, FSDP, and sampling math are unchanged. See the [1.18.1 English release note](docs/releases/2026-10-19-windows-text-encoder-non-mmap-fix.en.md).

Version 1.18.0 (O15.2) parallelizes standard MiniMax H3 Director `VAEDecode` VideoVAE temporal chunks between the GPU0 main process and the GPU1 Ray worker. The official ComfyUI `vae.decode()` reference for 107 frames at 1920x1088 falls from 65.78 to 39.79 seconds warm (1.653x, 39.51% lower), with all five dual-GPU runs passing per-run quality gates and element-exact output. A two-segment Director hardware regression decoded in 9.191/9.210 seconds and produced 146 frames at 864x480 and 24 fps. The bridge is automatic only inside a Ray MODEL proxy Director execution and falls back to native decode for unsupported or failed paths. AudioVAE and direct internal SelfLift decode remain native. The full suite reports `797 passed, 35 subtests passed`. See the [1.18.0 English release note](docs/releases/2026-10-18-o15.2-dual-v100-mmh3-vae-decode.en.md) and [O15.2 test record](docs/testing/minimax-h3/O15_2_DUAL_V100_VAE_DECODE_2026-10.en.md).

Version 1.17.1 (O15.1) moves the validated MiniMaxH3Director coordinate to upstream `4c6245e`. The one commit since `8151b29` is frontend-only: Director-pack import/export now reports upload, extract, pack, download, save, and timeline-apply progress. Sampling, Refine, SelfLift, FaceRefine, VAE, and cache contracts are unchanged, so Raylight adds no new sampling branch. The local two-segment hardware regression produced 146 frames at 864x480 and 24 fps with 99%/97% peak GPU utilization; the full suite reports `769 passed, 35 subtests passed`. This release does not speed up neural VideoVAE decode: PR #182 optimizes only post-decode continuity opening-grade matching. See the [1.17.1 English release note](docs/releases/2026-10-18-o15.1-mmh3director-4c6245e.en.md) and [O15.1 test record](docs/testing/minimax-h3/O15_1_MMH3DIRECTOR_4C6245E_2026-10.en.md).

Version 1.17.0 (O15) moves the validated MiniMaxH3Director coordinate from upstream `576424f` to `8151b29`. Director's Refine realignment, FL2V selected-run/continuity fixes, lossless FFV1 cache, custom cache namespaces, decoded-audio sample-rate preservation, segment cleanup, and UI changes coexist with the existing dual-GPU proxy. Raylight additionally bridges per-segment LoRA descriptors to both FSDP workers: the main process loads no weights, and each segment's stack is applied only to a temporary worker model clone, never the next segment or base model. The two-segment hardware acceptance produced 146 unique frames at 864x480 and 24 fps. See the [1.17.0 English release note](docs/releases/2026-10-12-o15-mmh3director-8151b29.en.md) and [O15 test record](docs/testing/minimax-h3/O15_MMH3DIRECTOR_8151B29_2026-10.en.md).

Version 1.16.1 fixes a lifecycle race in multi-segment MiniMaxH3Director runs. Cleanup or finalization of an old FSDP `ModelPatcher` clone could clear the newer segment's active `current_patcher`, typically failing in the SelfLift high-resolution stage before `prepare_state`. Cleanup now releases that reference only when the clone is still its owner. Model precision, FSDP/P2P transport, and sampling math are unchanged. Director's `confirm_first_pass=true` still intentionally defers Refine/FaceRefine until the confirmed follow-up execution. See the [1.16.1 English release note](docs/releases/2026-10-12-multisegment-patcher-lifecycle-fix.en.md).

Version 1.16.0 (O14) moves MiniMaxH3Director compatibility to upstream `576424f`, adds SelfLift final-state, zero-noise, and fixed-noise forwarding, and fixes valid video with broadband-noise audio. The Ray MODEL facade now mirrors MiniMaxH3's AV audio-flow scale (`12/3=4` by default). The full dual-V100 SelfLift + SOL + Refine run produces 864x480, 124 frames at 24 fps in 288.55 seconds, with zero audio NaN/Inf and 124/124 unique frames. See the [1.16.0 English release note](docs/releases/2026-09-21-o14-mmh3director-576424f-selflift.en.md) and [O14 acceptance record](docs/testing/minimax-h3/O14_MMH3DIRECTOR_576424F_SELFLIFT_2026-09.en.md).

Version 1.15.0 (O13) adapts MiniMaxH3Director `19b6a73`. Director-native Refine, continuity remasking, Refine spatial tiling, and step-wise live previews now execute through the dual-V100 workers instead of being silently dropped. The new `shift_cache` argument is accepted explicitly, but the main-process proxy cache is not serialized; Raylight retains lifecycle ownership of the real remote weights. Refine tiling is a VRAM feature and the untiled path is normally faster when memory permits. Preview events come only from rank 0 through a size-one non-blocking queue, and disabling preview creates no transport. See the [1.15.0 English release note](docs/releases/2026-09-12-o13-mmh3director-19b6a73-full-compatibility.en.md) and [O13 test record](docs/testing/minimax-h3/O13_MMH3DIRECTOR_19B6A73_2026-09.zh-CN.md).

Version 1.14.0 (O12) moves the preferred runtime to Python 3.12.10 and PyTorch 2.12.0+cu126, and adds optional worker-side replay bridges for TE-Speed 3.3 and Sol-Attn V100. Existing workflows enable neither accelerator by default. In the matched 5.17-second 864x480 REF2VA matrix, Sol reduces 4/8-step warm worker sampling from 85.23/170.90 seconds to 69.28/138.06 seconds, about 19%. TE does not show a stable standalone benefit on these short trajectories. The Python312 full suite reports `725 passed` and `35 subtests passed`. See the [1.14.0 English release note](docs/releases/2026-09-10-o12-python312-torch212-h3-acceleration.en.md) and [Chinese O12 test report](docs/testing/minimax-h3/O12_PYTHON312_H3_ACCELERATION_2026-09.zh-CN.md).

Version 1.13.1 is an O11 compatibility patch for MiniMaxH3Director. It retains the legacy `comfy.sample.sample()` bridge used by commit `d655912` while adapting the `sample_single_stage → BasicGuider/CFGGuider → SamplerCustomAdvanced` path in `b8f721c`. Official MiniMax AV `NestedTensor` inputs, external SIGMAS, `apply_shift`, and Director sampling controls are forwarded correctly to the dual-V100 workers, and remote failures never fall back silently to GPU0. The full automated suite reports `701 passed`; a current-Director hardware smoke test produced 56 frames at 864×480 and 24 fps with both GPUs reaching 100% utilization. Director, ComfyUI core, model precision, FSDP, and P2P data paths are unchanged. See the [1.13.1 English release note](docs/releases/2026-09-03-mmh3director-b8f721c-compatibility.en.md).

Version 1.13.0 (O11) fixes long multimodal Text Encoder OOMs caused by SDPA Math fallback and a dense square causal mask. The dedicated dual-GPU Loader now exposes independent `AUTO`, `XFORMERS_GQA`, `TORCH_EFFICIENT`, and `MATH_DEBUG` routing; AUTO selects xFormers GQA on the target V100 system. One 15-second video, seven distinct images, and a meaningful 7000-character prompt expand to 24,236 positions and pass in 112.94/112.55 seconds cold/warm. The follow-up Diffusion-lifetime work also completes the real 15-second, 365-frame, 0.4 MP, seven-image + video + audio REF2VA Turbo4 case: sampling falls from 1999.41 to 1456.12 seconds and sampling-window board peaks fall from 16215/16230 to 13005/12873 MiB. The 365-frame output decodes successfully. The 0.6 MP boundary remains unaccepted. See the [O11 English release note](docs/releases/2026-09-o11-text-encoder-efficient-attention.en.md) and [Chinese test record](docs/testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md).

Version 1.12.0 (O10) adds an explicit MiniMax H3 Qwen3-VL-32B NVFP4 Text Encoder TP2 capacity path. GPU 0 performs vision preprocessing; the seven Linear operations in each of the 50 language layers are tensor-sharded and execute concurrently on both V100s, with CUDA P2P/NVLink SUM for partial results. Formal peaks are 9.25/9.23 GiB and GPU 0 falls 31.9% from the single-GPU peak. I2V, REF2VA, Director three/five-second, and five-segment outputs pass numerical and media quality gates. Text Encoder cold/warm medians are nevertheless 27.17/26.91 seconds versus 20.76/10.37 seconds for one V100, so this remains an explicit, default-off experimental capacity mode rather than a recommended performance mode. See the [O10 English release note](docs/releases/2026-08-29-dual-v100-text-encoder-tp2.en.md) and [test record](docs/testing/minimax-h3/DUAL_GPU_TEXT_ENCODER_TP2_2026-08.zh-CN.md).

Version 1.11.0 (O9) turns the Windows CUDA P2P buffer from a logical-payload limit into a fixed VRAM budget. The default is now 128 MiB per rank; larger Ulysses/FSDP collectives are automatically split into CUDA P2P chunks without CPU/Gloo payload fallback. The historical 239,826,944-byte remote payload, MiniMax H3 I2V/REF2VA, five-segment Director + Refine, and LTX five-second cold/warm runs all pass. See the [O9 English release note](docs/releases/2026-08-o9-adaptive-p2p-chunking.en.md) and [acceptance record](docs/testing/windows-p2p/ADAPTIVE_CHUNKING_2026-08.zh-CN.md).

Version 1.10.2 hardens the single-source layout. The only editable Git worktree is now strictly separated from the plain ComfyUI deployment copy, which records a verifiable source commit. Tests, FSDP probes, and benchmark identity checks support the canonical worktree, a plain `custom_nodes` deployment, and a clean Git clone. Ray's local working directory no longer lives inside the node source tree. See the [1.10.2 English release note](docs/releases/2026-08-28-single-source-hardening.en.md).

Version 1.10.1 fixes a Director multi-segment continuity failure where worker-side reference images replaced motion-context keyframe conditioning. It also restores the intended timeline positions for continuity keyframes and audio. A two-segment main-sample plus Refine run passed on the dual-V100 machine. See the [1.10.1 English release note](docs/releases/2026-08-28-mmh3director-continuity-fix.en.md).

Version 1.10.0 adds the explicit opt-in `Ray MODEL Adapter (MMH3Director)`. Director remains the compact generation-control panel while its diffusion sample executes on the dual-V100 FSDP/P2P workers. Real-machine acceptance covers t2v, i2v, fl2v, r2v, v2v, rv2v, same-model Refine, and multi-segment R2V. See the [O8 English release note](docs/releases/2026-08-22-o8-mmh3director-adapter.en.md) and [validation summary](docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.md).

MiniMax H3 O6 now includes the explicit `fp16_h3_safe` mode: FP32 numerical islands protect residuals and sensitive outputs while dominant attention/MLP branches use FP16. Full I2V/REF2VA sampling reaches 3.299x/3.416x, and the best current REF2VA experimental configuration reaches 3.714x. Quality gates pass, but the initial 4x gate does not. See the complete [English upgrade record](docs/releases/2026-08-18-o6-minimax-h3-safe-fp16.en.md) or [Chinese upgrade record](docs/releases/2026-08-18-o6-minimax-h3-safe-fp16.zh-CN.md) for shipped changes, resource data, and rejected experiments.

The 2026-08-19 manual capacity choices remain historical documentation. Version 1.11.0 supersedes that default with 128 MiB auto-chunking; 256/512 MiB are advanced performance-comparison choices, not requirements for larger workflows. Rank/P2P diagnostics remain off by default.

This is not Windows NCCL and not a general PyTorch `ProcessGroup`. It is a targeted compatibility layer for a single machine, exactly two ranks, Windows V100 inference.

## Upstream and authorship

This repository is derived from [Raylight](https://github.com/komikndr/raylight), created and maintained upstream by **Komikndr / Micko Lesmana**. Raylight uses Ray to manage ComfyUI GPU workers and integrates xDiT/xFuser, yunchang, and FSDP parallelism.

- Upstream repository: <https://github.com/komikndr/raylight>
- Upstream baseline: Raylight 1.9.0, commit `9a7c33d52b3d35e29f75ecff3c227de987f0d4cf`
- Current branch version: 1.18.3; pre-adapter rollback tag: `v1.9.0-pre-mmh3director`
- License: Apache License 2.0
- Current FSDP changes: [WINDOWS_FSDP_CHANGES.md](WINDOWS_FSDP_CHANGES.md)
- Previous P2P/Ulysses changes: [WINDOWS_P2P_CHANGES.md](WINDOWS_P2P_CHANGES.md)
- FSDP test and acceptance record: [docs/WINDOWS_V100_FSDP_TESTING.md](docs/WINDOWS_V100_FSDP_TESTING.md)
- Historical P2P/Ulysses record: [docs/TESTING.md](docs/TESTING.md)
- MiniMax H3 O6 safe-FP16 upgrade record: [English](docs/releases/2026-08-18-o6-minimax-h3-safe-fp16.en.md) / [中文](docs/releases/2026-08-18-o6-minimax-h3-safe-fp16.zh-CN.md)
- Launcher capacity and diagnostic controls: [English](docs/releases/2026-08-19-launcher-controls.en.md) / [中文](docs/releases/2026-08-19-launcher-controls.zh-CN.md)
- MiniMaxH3Director MODEL adapter 1.10.0: [English](docs/releases/2026-08-22-o8-mmh3director-adapter.en.md) / [中文](docs/releases/2026-08-22-o8-mmh3director-adapter.zh-CN.md)
- MiniMaxH3Director continuity fix 1.10.1: [English](docs/releases/2026-08-28-mmh3director-continuity-fix.en.md) / [中文](docs/releases/2026-08-28-mmh3director-continuity-fix.zh-CN.md)
- Single-source deployment and test-layout hardening 1.10.2: [English](docs/releases/2026-08-28-single-source-hardening.en.md) / [中文](docs/releases/2026-08-28-single-source-hardening.zh-CN.md)
- O9 adaptive CUDA P2P chunking 1.11.0: [English](docs/releases/2026-08-o9-adaptive-p2p-chunking.en.md) / [中文](docs/releases/2026-08-o9-adaptive-p2p-chunking.zh-CN.md)
- O10 dual-V100 Text Encoder TP2 1.12.0: [English](docs/releases/2026-08-29-dual-v100-text-encoder-tp2.en.md) / [中文](docs/releases/2026-08-29-dual-v100-text-encoder-tp2.zh-CN.md)
- O11 efficient Text Encoder attention 1.13.0: [English](docs/releases/2026-09-o11-text-encoder-efficient-attention.en.md) / [中文](docs/releases/2026-09-o11-text-encoder-efficient-attention.zh-CN.md)
- MiniMaxH3Director `b8f721c` compatibility patch 1.13.1: [English](docs/releases/2026-09-03-mmh3director-b8f721c-compatibility.en.md) / [中文](docs/releases/2026-09-03-mmh3director-b8f721c-compatibility.zh-CN.md)
- Python 3.12/Torch 2.12 and H3 external accelerator bridge 1.14.0: [English](docs/releases/2026-09-10-o12-python312-torch212-h3-acceleration.en.md) / [中文](docs/releases/2026-09-10-o12-python312-torch212-h3-acceleration.zh-CN.md)
- MiniMaxH3Director `19b6a73` full compatibility 1.15.0: [English](docs/releases/2026-09-12-o13-mmh3director-19b6a73-full-compatibility.en.md) / [中文](docs/releases/2026-09-12-o13-mmh3director-19b6a73-full-compatibility.zh-CN.md)
- Multi-segment FSDP patcher lifecycle fix 1.16.1: [English](docs/releases/2026-10-12-multisegment-patcher-lifecycle-fix.en.md) / [中文](docs/releases/2026-10-12-multisegment-patcher-lifecycle-fix.zh-CN.md)
- MiniMaxH3Director `576424f` / SelfLift 1.16.0: [English](docs/releases/2026-09-21-o14-mmh3director-576424f-selflift.en.md) / [中文](docs/releases/2026-09-21-o14-mmh3director-576424f-selflift.zh-CN.md)
- MiniMaxH3Director `8151b29` / per-segment LoRA 1.17.0: [English](docs/releases/2026-10-12-o15-mmh3director-8151b29.en.md) / [中文](docs/releases/2026-10-12-o15-mmh3director-8151b29.zh-CN.md)
- MiniMaxH3Director `4c6245e` compatibility certification 1.17.1: [English](docs/releases/2026-10-18-o15.1-mmh3director-4c6245e.en.md) / [中文](docs/releases/2026-10-18-o15.1-mmh3director-4c6245e.zh-CN.md)
- Dual-V100 MiniMax H3 VideoVAE decode 1.18.0: [English](docs/releases/2026-10-18-o15.2-dual-v100-mmh3-vae-decode.en.md) / [中文](docs/releases/2026-10-18-o15.2-dual-v100-mmh3-vae-decode.zh-CN.md)
- Windows multi-segment Text Encoder non-mmap stability fix 1.18.1: [English](docs/releases/2026-10-19-windows-text-encoder-non-mmap-fix.en.md) / [中文](docs/releases/2026-10-19-windows-text-encoder-non-mmap-fix.zh-CN.md)
- Windows FSDP quantized CPU-offload native-crash fix 1.18.2: [English](docs/releases/2026-10-19-windows-fsdp-quant-cpu-offload-fix.en.md) / [中文](docs/releases/2026-10-19-windows-fsdp-quant-cpu-offload-fix.zh-CN.md)
- Long-reference Attention OOM and lifecycle memory records 1.18.3: [English](docs/releases/2026-10-23-long-reference-oom-memory-recording.en.md) / [中文](docs/releases/2026-10-23-long-reference-oom-memory-recording.zh-CN.md)

The upstream license, copyright, and attribution are retained. The Windows P2P/FSDP layer, scripts, tests, and documentation are experimental branch work and do not imply upstream support for this Windows configuration.

## What this branch does

It is intended for users who:

- must run native Windows rather than WSL or Linux;
- have two Tesla V100-SXM2-16GB GPUs in TCC mode with working NVLink/CUDA P2P;
- need validated LTX 2.3 or MiniMax H3 Diffusion Model weights split across both 16 GB GPUs;
- want matched CUDA collectives to travel directly between GPUs instead of through host RAM;
- accept the narrower scope and higher process overhead of Ray on Windows.

It provides:

- persistent FSDP2 weight sharding for a validated LTX 2.3 or MiniMax H3 Diffusion Model loaded by `RayUNETLoader`;
- MiniMax H3 FP8-scaled checkpoints, CPU offload, official Turbo LoRAs, same-checkpoint reuse, and worker recycling when checkpoints change;
- a CUDA P2P `all_gather_into_tensor` path for FSDP2;
- the previous CUDA P2P `all_to_all_single` path for Ulysses;
- startup correctness and bandwidth gates;
- bounded V100 compatibility paths for FP8/BF16 model execution;
- worker-session reuse when topology and configuration are unchanged;
- checked-in workflow, API payloads, probes, and test records.

It does not:

- replace or fork yunchang;
- implement the complete NCCL API;
- turn two 16 GB GPUs into a transparent 32 GB device;
- shard every model or every ComfyUI node;
- support GGUF with FSDP, training/backward collectives, multi-node execution, or arbitrary GPU counts.

## O12 optional MiniMax H3 accelerators

O12 does not copy third-party algorithms into Raylight. External nodes patch a MODEL through their normal ComfyUI interface; Raylight serializes only their configuration and calls the installed node implementation inside both real workers.

```text
RayUNETLoader -> Ray MODEL Adapter (MMH3Director)
              -> [TE-Speed, optional]
              -> [Sol-Attn (V100), optional]
              -> MiniMaxH3Director
```

- Omitting both nodes is the unchanged Base path.
- Sol-only is the most repeatable sampling-speed option in the current 4/8-step matrix, about 19% faster than Base.
- TE-only has no stable benefit in this short-trajectory acceptance. TE+Sol reproduces the combined topology but is not guaranteed to beat Sol-only.
- Sol is approximate sparse attention and is not pixel-identical to Base. Validate subjective quality with production inputs.

Third-party source and binaries are not bundled. Install compatible versions of [TE-Speed-MiniMaxH3-OSS](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS) and [ComfyUI-MiniMaxH3-SolAttn-V100](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100); Sol also requires a kernel compatible with Windows, Python 3.12, and V100.

Eight matched Turbo4/Turbo8 `Base`, `TE`, `Sol`, and `TE_Sol` JSON files live in `example_workflows/o12`. On the project development machine they are deployed to `ComfyUI/user/default/workflows/O12`, and all outputs use `ComfyUI/output/video/raylight_o12`.

## MiniMax H3 dual-V100 Text Encoder TP2 (experimental)

### Node wiring and modes

This path supports only `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` and requires the explicit `Load MiniMax H3 Text Encoder (Dual V100 TP2)` node. Connect `RayInitializer.ray_actors_init` to it, route its `clip` output into the existing MiniMax H3 conditioning/Director path, and continue its second `ray_actors_init` output into the Ray Diffusion chain. Existing `CLIPLoader` nodes, workflow JSON, and ordinary ComfyUI CLIP execution remain unchanged.

| `execution_mode` | Behavior | Status |
|---|---|---|
| `tp2` | vision tower on GPU 0; every Linear in the 50-layer language backbone is split across both V100s and merged with CUDA P2P SUM | explicit, default-off experimental capacity mode |
| `single_gpu_fallback` | invokes ComfyUI's native MiniMax CLIP loader | recommended compatibility path; no silent fallback |

`text_encoder_attention` controls only Text Encoder. `RayInitializer.XFuser_attention` continues to control only Diffusion sampling; they are independent settings.

| Text Encoder Attention | Behavior |
|---|---|
| `AUTO` | recommended default; resolves to `XFORMERS_GQA` on the tested V100 and never selects Math |
| `XFORMERS_GQA` | five-dimensional GQA with structured causal bias; bounds long-sequence KV-group workspace and reuses dead Q storage |
| `TORCH_EFFICIENT` | explicitly expands K/V and forces PyTorch Efficient; fails explicitly when unavailable |
| `MATH_DEBUG` | numerical reference only up to 2048 tokens; rejected for real long contexts |

Before TP2 encoding, the main process unloads current ComfyUI GPU models and acquires a synchronized Text Encoder GPU lease from both Ray workers. The workers release resident Diffusion weights while retaining rebuildable checkpoint/mmap state, and reject sampling-model load during the lease. Success, exceptions, and interruption all return both leases in `finally`. This prevents vision/language and Diffusion weights from competing at the same instant; it does not create a transparent 32 GB pool.

### Capability, measurements, and limits

- The vision tower runs on GPU 0. Both GPUs overlap work in all 50 language layers and all seven Linear operations per layer.
- This is not Text Encoder FSDP or contiguous-layer pipeline parallelism. Each Linear is row/column tensor-sharded and both GPUs compute the same layer.
- A warm encode performs 100 P2P SUM operations with 4,453.12 MiB aggregate bidirectional payload in 70.70 ms, about 1.96% of language compute.
- Formal three-run medians: TP2 cold/warm 27.17/26.91 seconds; single V100 20.76/10.37 seconds. The speed gate fails.
- TP2 peak VRAM is 9.25/9.23 GiB, below 14 GiB per card; GPU 0 is 31.9% below the 13.58 GiB single-card peak.
- Ordinary I2V/REF2VA, Director three/five-second, and 3-second x5-segment runs finish with decodable unique frames, no black interval, no NaN/Inf, and both GPUs returning to about 10 MiB after worker exit.
- Other Qwen3-VL checkpoints, Text Encoder LoRA, batch>1, more GPUs, multi-node, training, and silent single-GPU fallback are unsupported.
- The current bottleneck is about 16.4 seconds of NVFP4 shard loading per encode, not NVLink/P2P. Use ordinary `CLIPLoader` or `single_gpu_fallback` when speed is the priority.
- O11's maximum Text Encoder acceptance is one 15-second video, seven images, 7000 characters, and 24,236 expanded positions. Cold/warm times are 112.94/112.55 seconds, internal dual-GPU peaks are about 10.70/11.06 GiB, and language weights are released afterward.
- The user's seven-image, one-video, one-audio, 5802-character case expands to 12,302 positions and completes in 62.88 seconds at 9244/9451 MiB allocated peaks, leaving verified headroom.
- This maximum guarantee ends at Text Encoder. Fifteen-second REF2VA Diffusion OOMs in the first step at both 1056×608 and 864×480; the completed integration smoke is three seconds, 864×480, Turbo4.

## MiniMaxH3Director adapter

### Two workflow paths

| Path | Sampling control surface | Best for | Compatibility |
|---|---|---|---|
| Explicit Ray sampler | `RayUNETLoader` plus XFuser sampler/guider/scheduler nodes | LTX and existing MiniMax H3 I2V/REF2VA graphs with decomposed controls | Fully retained; old JSON needs no migration |
| Director MODEL proxy | `RayUNETLoader` → `Ray MODEL Adapter (MMH3Director)` → `MiniMaxH3Director` | One Director node for seed, steps, sampler, scheduler, shifts, duration, FPS, geometry, segments, and output | New in 1.10.0; enabled only when explicitly connected |

The proxy carries no weights. It only marks which Ray workers must execute this MODEL's sampling request. Legacy Director is intercepted at `comfy.sample.sample()`; current Director is intercepted at `sample_single_stage()` before `BasicGuider → SamplerCustomAdvanced`. Conditioning, latent, fixed seed, sampling controls, and optional SIGMAS are executed by both workers and returned as latent so Director can continue VAE, audio, mux, and save. Director therefore does not sample locally and then sample again in Raylight.

Ordinary ComfyUI `MODEL` objects pass through unchanged. A proxy failure is explicit and never silently falls back to GPU 0. MiniMaxH3Director source is unmodified; validated coordinates include legacy `d655912`, `b8f721c`, `19b6a73`, `576424f`, `8151b29`, and current `4c6245e`. SelfLift returns final step state across its low/high transition, while the proxy preserves MiniMaxH3 AV audio-flow scaling.

### Boundaries

- Validated: t2v, i2v, fl2v, r2v, v2v, rv2v, same-checkpoint Refine, and multi-segment R2V.
- One Director pass maps to one remote operation. Segments invoke separate passes but reuse the same workers, Gloo group, and FSDP shards.
- Director `8151b29` per-segment LoRAs are supported. The global `RayLoraLoader` remains the base stack; up to eight active non-zero LoRAs per segment are temporarily stacked in both workers. An empty or different next-segment stack does not inherit previous patches.
- Custom cache namespaces, raw/FFV1 cache, selected-run/segment export, decoded-audio sample rate, and segment prefetch cleanup remain Director main-process features. Raylight does not accelerate those stages, but their interfaces are validated to coexist with remote sampling.
- Director owns all generation arguments. Raylight owns GPU count, parallel degrees, FSDP, CPU offload, attention, and worker lifecycle.
- Only the Diffusion Model is FSDP-sharded. The explicit O10 node can use separate TP2 for the supported Text Encoder language backbone, while its vision tower stays on GPU0. Version 1.18.0 Director standard VideoVAE decode is temporal parallelism between main-process GPU0 and rank1 GPU1, not FSDP. AudioVAE, mux, and save remain in the ComfyUI main process.
- Main-process preprocessing models are released from GPU before remote sampling to avoid rank-0 overlap. Reloadable weights can remain in RAM/commit, so multi-segment and long jobs must be sized against host memory and pagefile too.
- Director-native same-model Refine stays on both workers, including spatial tiling on each real worker sampler. Tiling saves transient VRAM but adds kernel launches, so leave it off when memory permits.
- Continuity remasking is reconstructed on both real worker models. Unknown callbacks fail explicitly instead of silently degrading.
- With Director live preview enabled, rank 0 sends at most 16 video-latent frames through a non-blocking single-slot queue to Director's original callback. Disabled preview creates no queue or preview traffic.
- Version 1.18.0 automatically parallelizes standard `VAEDecode` only inside a Ray MODEL proxy Director `execute` scope. Rank 1 preloads the VAE on CPU while sampling, then splits temporal chunks with GPU0 and merges them with the official five-frame overlap semantics. Repeated standard decodes reuse the loaded VAE; GPU1 VAE memory is released before the next diffusion sample. Failure, unknown VAE, single-GPU, and non-proxy calls use native decode.
- The bridge does not intercept AudioVAE or SelfLift paths that call `vae.decode()` directly instead of using `VAEDecode`; those remain native. Standard-node primary, Refine, and FaceRefine decodes can use the bridge.
- Custom Refine requiring another checkpoint, different weights, or bypassing `sample_single_stage()`/legacy `comfy.sample.sample()` is outside the guarantee.

## Current FSDP capability

### Do both GPUs compute?

Yes, during the FSDP diffusion samplers. In the final visually accepted run with the original BF16 distilled LoRA:

- GPU0 peaked at 16,224 MiB, 100% utilization, and about 354.5 W;
- GPU1 peaked at 16,156 MiB, 100% utilization, and about 348.4 W;
- both ranks returned element-identical video and audio latents in both sampling stages.

The whole workflow is not dual-GPU at every moment:

| Workflow stage | GPU0 | GPU1 | Reason |
|---|---:|---:|---|
| Text encoding and image preprocessing | ordinary path primary; TP2 vision + language shard | TP2 language backbone participates | separate experimental TP2, not FSDP |
| Diffusion sampler 1 | high load | high load | two-rank FSDP sampling |
| Latent upscale | primary | mostly idle | ordinary upscaler node |
| Diffusion sampler 2 | high load | high load | two-rank FSDP sampling |
| Director standard VideoVAE decode | temporal chunks | temporal chunks | automatic in 1.18.0 proxy scope; excludes AudioVAE/direct SelfLift decode |
| Audio VAE decode | primary | mostly idle | native Director/ComfyUI path |

The precise claim is: **both GPUs compute during the FSDP diffusion stages; the entire workflow is not continuously dual-GPU.**

### How weights are sharded

This is not pipeline parallelism with half the layers on each GPU. For each FSDP weight tensor:

1. each rank persistently owns half of the tensor;
2. before the layer executes, both halves are temporarily gathered over CUDA P2P/NVLink;
3. both GPUs execute the layer forward with the temporary full weight;
4. the full weight is released and each rank returns to its persistent shard;
5. the process repeats for the next layer.

The validated LTX model registered 2,999 FSDP wrappers and 2,620 DTensors. Each rank retained about 11,203 MiB of Diffusion Model payload. The validated MiniMax H3 path registered 684 FSDP wrappers per rank. Each GPU still needs temporary room for the active full layer, activations, LoRA, the fixed 128 MiB auto-chunking P2P staging buffer, and CUDA workspaces, so peak VRAM can still approach 16 GB without permanently sizing the buffer to one large payload.

### What is and is not sharded

| Component | FSDP-sharded now? | Notes |
|---|---|---|
| LTX 2.3 Diffusion Model | Yes | loaded through `RayUNETLoader` |
| MiniMax H3 Diffusion Model | Yes | validated FP8-scaled I2V/REF2VA checkpoints |
| Video/audio transformers inside LTXAV | Yes | part of the Diffusion Model |
| Text Encoder | No FSDP; experimental TP2 | ordinary CLIP remains unsharded; O10 supports only MiniMax H3 Qwen3-VL-32B NVFP4 and keeps vision on GPU 0 |
| Video VAE | No FSDP; temporal parallelism for standard Director decode | 1.18.0 splits chunks across GPU0/GPU1; unknown/failing paths fall back natively |
| Audio VAE | No | ordinary audio VAE decode path |
| Spatial latent upscaler | No | ordinary loader/node |
| Temporal upscaler | No by default | would require separate integration and validation |
| Distilled LoRA | Supported, but not base-weight sharding | lazy CPU sidecar applied by both workers |
| GGUF Diffusion Model | No | explicitly rejected with current FSDP |

The original LTX BF16 distilled LoRA passed a 121-frame visual, audio, and rank-consistency acceptance run. The official MiniMax H3 FL2V Turbo8 and REF2V Turbo4 LoRAs also pass FSDP loading, dual-rank sampling, and media checks.

### FSDP versus NCCL

FSDP and NCCL serve different roles:

- **FSDP** decides how parameters are sharded, gathered, and resharded.
- **NCCL** is the common Linux/NVIDIA transport for GPU collectives.

Typical upstream path:

```text
PyTorch FSDP2
  -> ProcessGroupNCCL
  -> NCCL collectives
  -> NVLink / PCIe / network
```

This Windows branch:

```text
PyTorch FSDP2
  -> torch.distributed.all_gather_into_tensor
  -> targeted Windows collective router
  -> CUDA IPC / P2P / NVLink
```

Gloo/TCPStore still handles rendezvous, group creation, barriers, and unmatched control operations. A matched FSDP CUDA weight payload is not silently downgraded to host-memory Gloo transport. The branch only implements the inference collectives required by the validated two-rank topology; it does not provide NCCL training, multi-node, or general topology capabilities.

### Comparison

| Capability | Upstream Raylight | Previous Windows P2P branch | This FSDP branch |
|---|---|---|---|
| Primary platform | Linux + NCCL | Windows dual V100 | Windows dual V100 |
| Main parallel mode | USP/FSDP/CFG/DP | Ulysses USP | FSDP2 |
| Persistent model weights | sharded with FSDP | full copy per GPU | Diffusion Model genuinely sharded |
| CUDA data path | NCCL | P2P all-to-all | P2P all-gather plus all-to-all |
| Scale | NCCL multi-GPU/multi-node | exactly two ranks | exactly two ranks |
| Main value | standard general implementation | faster sampling for models that already fit | run a model that does not fit one 16 GB GPU |
| Current performance status | model/hardware dependent | 10.28% fair warm-to-warm gain | LTX correctness accepted; MiniMax H3 safe-FP16 functionality/quality pass at 3.299x-3.714x sampling, below the 4x gate |

## Host RAM and page file

The final FSDP + LoRA run recorded:

| Metric | Peak/result |
|---|---:|
| Physical RAM | about 62.6 GiB |
| Windows committed memory | about 110 GiB |
| Actual page-file use | about 476 MiB down to 466 MiB |
| FSDP CPU offload | false |
| safetensors mmap | true |

The successful run did not move FSDP weight collectives through the page file. High host usage includes the ComfyUI process, two Ray workers, model structure and metadata, ordinary Text Encoder/VAE/Upscaler loading, the LoRA CPU sidecar, mmap mappings, and filesystem cache.

A page file can delay a host OOM, but paging can stall one rank, slow sampling severely, and eventually trigger P2P coordination timeouts. It is not a performance expansion mechanism.

## Validated platform

### Hard requirements for the supported fast path

| Item | Requirement |
|---|---|
| OS | native 64-bit Windows, one machine |
| GPUs | exactly two visible CUDA GPUs |
| Validated model | 2× Tesla V100-SXM2-16GB |
| Driver model | both GPUs in TCC; WDDM is not a release configuration |
| GPU transport | CUDA peer access, CUDA IPC, and cross-process CUDA events |
| Ray topology | two workers / two ranks |
| FSDP topology | LTX uses CPU Offload=false and Ulysses/Ring/CFG=0/0/0; full MiniMax H3 uses CPU Offload=true and 2/1/1; both use FSDP=true and DP=1 |
| Supported collectives | two-rank CUDA `all_gather_into_tensor` and `all_to_all_single` |
| Sharding scope | validated LTX 2.3 and MiniMax H3 Diffusion Models |

The physical carrier-board or PCIe slot arrangement is not a software condition. The real gate is whether CUDA P2P works and the complete correctness/bandwidth probe passes.

### Validated versions

| Component | Version |
|---|---|
| Windows | NT build 22631.6199 / 23H2 |
| NVIDIA driver | 577.00 |
| Python | 3.12.10 x64 preferred; 3.10.11 rollback profile |
| PyTorch | 2.12.0+cu126 preferred; 2.7.0+cu126 rollback profile |
| torchvision | 0.27.0+cu126 |
| torchaudio | 2.11.0+cu126 |
| xformers | 0.0.35 |
| Ray | 2.57.0 |
| xFuser | 0.4.5 |
| yunchang | 0.6.4 |
| ComfyUI | 0.31.0, commit `62b3c94bd45154f6486c7abf1b9efcacee96ea69` |
| Diffusion attention | Base uses `XFORMERS_MEMORY_EFFICIENT`; O12 optionally uses Sol-Attn V100; no FlashAttention |

See [environment-windows-v100.json](environment-windows-v100.json) and [requirements-windows-v100-torch212.txt](requirements-windows-v100-torch212.txt) for current machine-readable pins. [requirements-windows-v100.txt](requirements-windows-v100.txt) remains the Python310/Torch2.7 rollback profile.

## Installation and use

### Single source and deployment rule on the project development machine

The project development machine has one editable, committable, and pushable Git repository:

```text
E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p
```

`E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight` is a generated deployment copy loaded by ComfyUI. It is not a Git worktree and must not be edited directly. After changing the canonical repository, stop ComfyUI and its Ray workers, then run:

```powershell
cd E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p
.\scripts\deploy-to-comfyui.ps1 -ComfyUIRoot E:\ComfyUI-py310\ComfyUI
```

The tool materializes the immutable Git `HEAD` archive (never uncommitted worktree content), records the source commit in `.raylight-deployed-commit`, verifies hashes and source identity in a staging directory, and then replaces the live tree. A failed first deployment removes the invalid target; a failed replacement restores the prior deployment. Check for drift without changing the deployment:

```powershell
.\scripts\deploy-to-comfyui.ps1 `
  -ComfyUIRoot E:\ComfyUI-py310\ComfyUI `
  -VerifyOnly
```

`DEPLOYMENT_VERIFIED` means every managed file and the recorded source commit match. No unmanaged files exist apart from explicit runtime caches such as `__pycache__`, pytest/Ruff state, and deployment identity metadata. Ray's working directory lives under the system temporary directory rather than the node source tree. Regular users may still clone directly into `custom_nodes\raylight` as described below; the single-source layout is the development and local deployment policy for this project machine.

Raw local logs, telemetry, contact sheets, and other evidence excluded from Git are consolidated under `.local/artifacts` in the canonical worktree; `.local` is Git-ignored. Maintainable conclusions, plans, and compact test summaries remain under `docs`.

### 1. Prepare the GPUs

Confirm that both V100 GPUs use TCC, active NVLink links are visible, NVIDIA's `p2pBandwidthLatencyTest` passes peer access and correctness, and no stale ComfyUI/Ray process owns the GPUs.

### 2. Install the pinned environment

Use an isolated Python 3.12.10 environment. Do not modify unrelated ComfyUI installations. O12 keeps the old Python310 environment intact for an explicit rollback.

```powershell
$PY = "<Python 3.12.10 path>\python.exe"
cd <ComfyUI>\custom_nodes
git clone `
  https://github.com/getsomefuns/raylight-windows-v100-fsdp-p2p.git raylight

& $PY -m pip install -r .\raylight\requirements-windows-v100-torch212.txt
& $PY -m pip install -e .\raylight
```

The O12 V100 environment is `torch==2.12.0+cu126`, which retains `sm_70`; `2.7.0+cu126` is the rollback profile. The originally proposed Torch 2.8 Windows wheel fails the minimum Gloo transport probe and cannot provide this branch's control plane. Do not install FlashAttention or replace cu126 with cu128/cu130 builds intended for newer architectures.

### 3. Run preflight

```powershell
cd <ComfyUI>\custom_nodes\raylight
.\scripts\verify-windows-v100.ps1 -PythonPath $PY
.\scripts\verify-windows-v100.ps1 -PythonPath $PY -RunP2PProbe
```

The full probe verifies versions, both TCC GPUs, NVLink visibility, two-Ray-actor CUDA IPC/P2P correctness, and at least 50 GiB/s on the validated 115,343,360-byte payload. Do not proceed to a large workflow if a gate fails.

### 4. Start ComfyUI

Validate paths without starting:

```powershell
.\scripts\start-comfyui-windows-p2p.ps1 `
  -PythonPath $PY `
  -ValidateOnly
```

Start normally:

```powershell
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY
```

The launcher prints the 128 MiB staging budget, auto-chunking state, and advanced `128/256/512 MiB` overrides. The recommended default is 128 MiB and does not need to be sized from resolution, frame count, or model:

```powershell
# Default: larger collectives are auto-chunked
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 128

# Advanced performance comparison; adds 128 MiB persistent VRAM per rank
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 256

# Advanced performance comparison; adds 384 MiB per rank over the default
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 512
```

Detailed Rank/P2P diagnostics are off for normal use. Enable them only while investigating timing or synchronization:

```powershell
.\scripts\start-comfyui-windows-p2p.ps1 `
  -PythonPath $PY `
  -EnableDiagnostics
```

Capacity and diagnostic settings are read when Ray workers start. Stop ComfyUI and stale Ray workers before changing them. `-ValidateOnly` prints the resolved settings without launching ComfyUI. The legacy `-P2PCapacityBytes` parameter remains accepted for existing automation; new commands should use MiB.

Open <http://127.0.0.1:8188>. The essential ComfyUI command is:

```text
main.py --listen 127.0.0.1 --port 8188 --disable-cuda-malloc --disable-dynamic-vram --reserve-vram 3
```

`--disable-cuda-malloc` is required by the validated V100 VAE path to avoid `cudaErrorNotSupported / operation not supported`.
The launcher now defaults to `-ReserveVramGiB 3`. This leaves enough main-process workspace for the
validated 0.9 MP two-reference MMH3Director Qwen3-VL/NVFP4 encode. Use `2` only for historical O6
reproduction or lighter single-reference jobs; `4` offers more headroom for larger reference sets at
the cost of additional text-encoder CPU offload and potentially slower preprocessing.

If Gloo selects a VPN, Hyper-V, WSL, or TUN adapter, specify the physical adapter IPv4:

```powershell
.\scripts\start-comfyui-windows-p2p.ps1 `
  -PythonPath $PY `
  -GlooHost <physical-adapter-ip>
```

### 5. Load the example workflow

- ComfyUI workflow: [example_workflows/LTX2_3_i2v_Raylight_Windows_FSDP_5s.json](example_workflows/LTX2_3_i2v_Raylight_Windows_FSDP_5s.json)
- Upstream example input image: [example_workflows/LTX2_3_i2v_Raylight.jpg](example_workflows/LTX2_3_i2v_Raylight.jpg)
- Model and custom-node manifest: [docs/ltx23-model-manifest.md](docs/ltx23-model-manifest.md)

Copy the example image into ComfyUI's `input` directory or reselect it in the Load Image node. The workflow contains the test prompt. Model weights are not included.

Validated `RayInitializer` settings:

| Setting | Value |
|---|---:|
| GPU | 2 |
| ulysses/ring/cfg degree | 0 / 0 / 0 |
| dp_degree | 1 |
| FSDP / FSDP_CPU_OFFLOAD | true / false |
| XFuser_attention | TORCH_EFFICIENT |
| clear_vram_after_sampling | true |
| skip_comm_test | true |
| use_mmap | true |

`skip_comm_test=true` skips Raylight's original generic communication tester; it does not skip this branch's CUDA P2P correctness and bandwidth gate.

### 6. Confirm the fast path

Initial startup should include messages similar to:

```text
[Raylight] Windows Gloo init OK ...
[Raylight] Windows CUDA P2P enabled: ...
```

An unchanged healthy configuration may later report:

```text
[Raylight] Reusing 2 live Ray workers for unchanged configuration
```

A matched supported CUDA P2P operation fails closed if initialization, correctness, bandwidth, operation IDs, or peer state are invalid; it does not silently continue through host memory while claiming NVLink use.

### 7. Reproduce the MiniMax H3 O6 result on the real machine

Use REF2VA Turbo4 for O6 acceptance. It has half as many sampling steps as I2V Turbo8 and is the
workflow used for the best O6 result. “Fastest” has two meanings:

| Test | Workflow | 5 GiB host registration | Historical sampling | Historical cold E2E |
|---|---|---|---:|---:|
| Main safe-FP16 result | REF2VA Turbo4 safe FP16 | off | 54.2174 s/it | **394.110 s** |
| Fastest sampling | REF2VA Turbo4 safe FP16 | on | **49.8680 s/it** | 424.241 s |
| FP32 control | REF2VA Turbo4 default | off | 185.2034 s/it | 932.031 s |

The 5 GiB option reduces per-step submission waiting but adds registration and initialization work.
It is the fastest sampler, not the fastest whole prompt. Run these as three independent cold starts.

Define reusable paths and stop the previous ComfyUI/Ray instance first:

```powershell
$EnvRoot = "<standalone ComfyUI environment root>"
$ComfyRoot = Join-Path $EnvRoot "ComfyUI"
$RepoRoot = Join-Path $ComfyRoot "custom_nodes\raylight"
$PY = Join-Path $EnvRoot "Python310\python.exe"
$Ray = Join-Path $EnvRoot "Python310\Scripts\ray.exe"

& $Ray stop --force
nvidia-smi --query-gpu=index,name,driver_model.current,memory.used,utilization.gpu --format=csv,noheader
```

Wait until both V100s are in TCC, idle, and released from the previous run. Version 1.11.0 completes the measured 239,826,944-byte remote Ulysses payload with the default 128 MiB staging budget, so the safe-FP16 workflow no longer needs a larger manual setting:

```powershell
Remove-Item Env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER -ErrorAction SilentlyContinue
Remove-Item Env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER_MIB -ErrorAction SilentlyContinue

Set-Location $RepoRoot
.\scripts\start-comfyui-windows-p2p.ps1 `
  -PythonPath $PY `
  -ComfyRoot $ComfyRoot `
  -ReserveVramGiB 2
```

Open <http://127.0.0.1:8188> and load:

[example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json](example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4_FP16_Experimental.json)

Both Load Image nodes use `minimax_h3_ref2va_green_robots.jpg`; the repository source image is
[example_workflows/Minimax_H3_REF2VA_Raylight.jpg](example_workflows/Minimax_H3_REF2VA_Raylight.jpg).
Keep the accepted prompt unchanged. Although its prose says “10 seconds,” the measured node geometry
is 124 frames at 24 FPS, approximately 5.167 seconds.

The GUI workflow still contains a Resolution Selector. The O6 benchmark overrode width and height at
the API layer. For an exact manual run, disconnect the selector's width/height links to
`MiniMaxH3ReferenceToVideo`, then enter 1120 and 768 directly. Verify:

| Node/setting | Fixed value |
|---|---|
| geometry | 1120x768, length=124, duration=5, FPS=24 |
| RayUNETLoader | REF2VA FP8-scaled checkpoint; `weight_dtype=fp16_h3_safe` |
| RayLoraLoader | REF2V Turbo4 LoRA, strength=1.0 |
| scheduler / sampler | simple, 4 steps, denoise=1; `res_multistep` |
| seed | `547879687678090`, control set to fixed |
| GPU / Ulysses / Ring / CFG / DP | 2 / 2 / 1 / 1 / 1 |
| sync Ulysses | true |
| FSDP / FSDP CPU offload | true / true |
| clear VRAM after sampling | false |
| attention / skip comm / mmap | `TORCH_EFFICIENT` / true / true |
| SaveVideo prefix | `video/raylight_o6/manual_ref2va_o6_safe_fp16_nohost` |

Queue exactly once. The first run should log safe FP16 on both ranks, FSDP registration, and Windows
CUDA P2P. Historical main-run stages were Ray 37.614 s, model load 19.400 s, preprocessing 36.134 s,
Sampler 246.844 s, VAE 42.984 s, save 8.965 s, and 394.110 s total.

`fp16_h3_safe` workers now enable the accepted block-boundary CUDA cache trim automatically; the launcher does not need another environment setting. Set `RAYLIGHT_H3_BLOCK_CACHE_TRIM=0` only for an explicit diagnostic control. Disabling it materially raises board memory on the 15-second maximum input.

For the 5 GiB fastest-sampling run, stop ComfyUI, run `& $Ray stop --force`, wait for GPU release, and
set these variables before starting new workers:

```powershell
$env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER = "1"
$env:RAYLIGHT_FSDP_CPU_OFFLOAD_HOST_REGISTER_MIB = "5120"
```

Start with the same default launcher, then use the identical workflow, geometry, input, prompt, four steps, and
fixed seed. Use SaveVideo prefix `video/raylight_o6/manual_ref2va_o6_safe_fp16_hostreg5g`. The log
must show `FSDP_HOST_REGISTER` and approximately 5120 MiB. Historical results were 49.8680 s/it,
249.977 s for the Sampler node, and 424.241 s E2E. Instant tqdm values and total Sampler-node time are
not the formal s/it definition; the formal value is the slowest rank's complete sampling interval / 4.

For the FP32 control, restart again with both host-registration variables removed, keep the default 128 MiB auto-chunking budget, and load:

[example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json](example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP_Turbo4.json)

Keep all inputs and parameters identical. The only core precision difference is
`RayUNETLoader weight_dtype=default`. Use prefix `video/raylight_o6/manual_ref2va_o6_baseline_fp32`.
The historical control was 185.2034 s/it, 774.778 s Sampler, and 932.031 s E2E.

Historical local media names are under `<ComfyUI>\output\video\raylight_o6`:

```text
minimax_h3_ref2va_o6-baseline-fp32-p2p256_run0_00001_.mp4
minimax_h3_ref2va_o6-safe-fp16-full-reviewed_run0_00001_.mp4
minimax_h3_ref2va_o6-safe-fp16-hostreg5g-scoped-full_run0_00001_.mp4
```

Check 1120x768 H.264, 124 unique frames, continuous robot/cloth motion, no black/noise/frozen section,
and no audio NaN/Inf symptom. Container hashes need not match. Record prompt and Sampler times, log
s/it, both-GPU VRAM/utilization, physical/committed/pagefile peaks, output path, and media result.
ComfyUI Manager commonly writes `<ComfyUI>\user\comfyui_8188.log`.

Do not enable the sampling profiler, pin-memory experiment, 6 GiB registration, global FP16 allowlist,
FlashAttention, `--highvram`, or `--disable-smart-memory` when reproducing the accepted O6 result.

### 8. Run the MiniMaxH3Director dual-GPU workflow

1. Start ComfyUI with `start_raylight_p2p.ps1` from the environment root. The accepted defaults are a 128 MiB auto-chunking staging budget, 3 GiB main-process VRAM reserve, and detailed diagnostics off.
2. Load `example_workflows/MMH3Director_Windows_V100_FSDP.json`. A local deployed copy normally lives at `<ComfyUI>\user\default\workflows\MMH3Director.json`.
3. Keep `GPU=2`, `ulysses_degree=2`, `ring_degree=1`, `FSDP=true`, and `FSDP_CPU_OFFLOAD=true` in Ray initialization. Use `fp16_h3_safe` on V100; match the checkpoint and official Turbo LoRA to FL2VA or REF2VA mode.
4. Set task type, seed, steps, sampler, scheduler, video/audio shifts, megapixels, aspect ratio, FPS, duration, prompt groups, segments, and output directly on Director. Do not add another XFuser sampler/guider/scheduler stack.
5. Verify the single model path: `RayInitializer → RayLoraLoader → RayUNETLoader → Ray MODEL Adapter (MMH3Director) → MiniMaxH3Director`. An ordinary MODEL connection does not activate the dual-GPU proxy.
6. The first run is cold. An unchanged checkpoint reuses both workers and FSDP shards; changing seed or Director controls does not require a ComfyUI restart. Switching FL2VA/REF2VA checkpoints recycles both workers as a pair.
7. When the session is finished, use `RayKill` connected to the same `RAY_ACTORS` chain for final reclamation. A generic ComfyUI VRAM-cleanup node cannot guarantee termination of Ray workers, raylet, and GCS.
8. During sampling, both V100s should show high utilization and each Director pass should have one operation with no second local sampler. The complete procedure is in the [O8 validation summary](docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.md).

Same-checkpoint repeated generation, same-model Refine, and multi-segment generation pass. Segments reuse workers, but main-process CLIP/VAE weights and intermediate results can still increase physical RAM and commit. For strict isolation between two large workflows, run `RayKill`, wait for workers/GPU contexts to disappear, then start the next workflow.

#### Reproduce the O12 accelerator workflows

Use explicit release-matrix arguments on the project machine:

```powershell
cd E:\ComfyUI-py310
.\start_raylight_p2p.ps1 `
  -PythonPath E:\ComfyUI-py310\Python312\python.exe `
  -P2PCapacityMiB 128 `
  -ReserveVramGiB 4
```

Under the ComfyUI `O12` workflow directory, select `MMH3Director_O12_REF2VA_Turbo4_Sol.json` or `...Turbo8_Sol.json` for the recommended repeatable sampling gain. Use `...TE_Sol.json` to reproduce the chained topology, `...Base.json` for the exact Base control, and `...TE.json` only as an experimental TE-only control.

The first execution includes model IO and Ray startup. Keep ComfyUI open and the checkpoint unchanged, then change the seed and run again for the warm measurement. The O12 JSON files already fix the MODEL chain, `fp16_h3_safe`, two-rank FSDP, and adaptive 128 MiB P2P settings; do not add another sampler stack. Keep diagnostics off for ordinary use. Outputs are written to `E:\ComfyUI-py310\ComfyUI\output\video\raylight_o12`.

## Reproducible benchmark inputs

The repository includes 5-second API prompt payloads under `benchmark_payloads/` for:

- ordinary single-GPU ComfyUI;
- Ray single-GPU control;
- dual-GPU Ulysses;
- dual-GPU FSDP using the visually accepted configuration.

With the validated folder layout, run for example:

```powershell
& $PY .\tests\windows_ltx_mode_benchmark.py --mode fsdp --runs 2 --port 8188
```

Alternative layouts can set `RAYLIGHT_COMFY_ROOT`, `RAYLIGHT_PYTHON`, `RAYLIGHT_BENCHMARK_RESULT_ROOT`, or `RAYLIGHT_BENCHMARK_PAYLOAD_ROOT`. The benchmark starts its own ComfyUI instance, so the selected port must be free.

## Data path and launch settings

```text
ComfyUI -> Raylight workers -> FSDP all-gather / Ulysses all-to-all
                              -> CUDA IPC/P2P/NVLink data plane
TCPStore + Gloo -------------> rendezvous, groups, control, unmatched operations
```

The launcher sets UTF-8 logging, `USE_LIBUV=0`, localhost rendezvous, Ray memory-monitor overrides, two visible GPUs, a 128 MiB per-rank P2P staging budget by default, a 3 GiB main-process VRAM reserve, a 50 GiB/s minimum P2P gate, and diagnostics off. It exposes 128/256/512 MiB advanced P2P choices, `-ReserveVramGiB`, and `-EnableDiagnostics`. See the Chinese README for the complete environment-variable table and memory trade-offs.

P2P capacity is a persistent CUDA staging-VRAM budget per GPU, not a direct conversion from video duration or resolution and no longer a logical payload ceiling. Larger valid collectives use multiple CUDA P2P/NVLink chunks inside one logical call. Increasing the budget does not reserve host RAM/pagefile, but it consumes the selected difference in persistent VRAM. The O9 median retained 98.91%/98.50% of matched-large-buffer bandwidth for 256/512 MiB payloads while using the 128 MiB default.

The memory-monitor override prevents Ray from terminating a large workflow early. It can allow Windows to enter paging under pressure; monitor physical and committed memory rather than treating it as a free optimization.

## Test results

### Communication and FSDP gates

- Existing all-to-all: about 59.3 GiB/s per rank.
- FSDP all-gather: about 60.0-61.6 GiB/s per rank for 64-384 MiB synthetic shards.
- A 256 MiB shard over the 128 MiB staging buffer passed correct two-chunk transfer.
- O9 all-to-all completed a 239,826,944-byte remote payload in two chunks with a fixed 128 MiB endpoint.
- Five interleaved sessions retained 98.91%/98.50% of matched-large-buffer median bandwidth for 256/512 MiB payloads.
- FP32, FP16, BF16-as-bytes, and uint8 transport passed, including tail chunks.
- A 512 MiB synthetic FSDP2 parameter retained 256 MiB per rank and resharded after forward.
- LTX registered 2,999 FSDP wrappers and retained about 11,203 MiB model payload per rank.

BF16 in the transport test means byte-preserving communication only; it is not a claim that V100 supports native BF16 compute.

### End-to-end LTX 2.3 results

Previous P2P/Ulysses baseline:

| Scenario | End-to-end |
|---|---:|
| Single V100 cold | 519.94 s |
| Single V100 warm | 316.60 s |
| Dual-V100 P2P reused-session median | 284.06 s |

The fair P2P/Ulysses warm-to-warm gain was 10.28%, but that mode keeps a complete model copy on each GPU.

Current visually accepted FSDP baseline:

| Scenario | Cold end-to-end | Sampler 1 | Sampler 2 | GPU0/1 peak VRAM | Visual result |
|---|---:|---:|---:|---:|---|
| FSDP without LoRA | 479.83 s | about 11.0 s/it | about 41.7 s/it | 16,218/16,208 MiB | coherent, PASS |
| FSDP + original BF16 LoRA | 551.82 s | about 15.0 s/it | about 45.6 s/it | 16,224/16,156 MiB | coherent, PASS |
| O9 default 128 MiB, cold | 517.13 s | 14.9 s/it | 45.9 s/it | 16,224/16,152 MiB | 121/121 unique frames, PASS |
| O9 default 128 MiB, same-process warm | 386.36 s | 14.9 s/it | 47.8 s/it | 16,234/16,156 MiB | 121/121 unique frames, PASS |

These results prove sharding, dual-GPU computation, and correct output. LTX performance optimization remains a separate task.

### MiniMax H3 accepted state

| Capability | Current result |
|---|---|
| I2V / REF2VA FSDP | dual rank, CUDA P2P, and FP8-scaled checkpoints pass |
| Worker lifecycle | same checkpoint reuses actors; changed checkpoints recycle actors and reclaim committed memory |
| Turbo workflows | loadable Turbo8 I2V and Turbo4 REF2VA workflows pass node/API validation |
| Accepted full profiles | I2V 640x640/56 frames; REF2VA 864x480/124 frames |
| Current compute policy | Default uses FP8 storage with FP32 compute; `fp16_h3_safe` uses FP32 numerical islands with FP16 attention/MLP |
| O6 | Function and quality pass; I2V/REF2VA reach 3.299x/3.416x and optional REF2VA reaches 3.714x; the 4x gate remains open |
| O7 | preliminary model-specific safe-FP16 research for LTX/LTXAV after O6 performance closure |
| O8 Director MODEL proxy | t2v/i2v/fl2v/r2v/v2v/rv2v, Refine, multi-segment, media, reuse, reclamation, and legacy performance gates pass |
| O9 adaptive P2P chunking | 128 MiB default; historical large payload, I2V, REF2VA, Director multi-segment/Refine, and LTX cold/warm pass |
| O10 Text Encoder TP2 | shipped in 1.12.0 after Gates A-F: capacity, numerical, video, and dual-compute gates pass; performance fails, so it remains default-off experimental capacity mode |
| O11 Text Encoder and Diffusion lifetime work | shipped in 1.13.0 after Gates A-F; the Text Encoder maximum and the 15-second/0.4 MP/seven-image+video+audio Turbo4 end-to-end case pass; 0.6 MP remains unaccepted |

The matched local FP32 baselines are locked. I2V Turbo8 takes 1463.67 s end to end and 160.72 s/it; REF2VA Turbo4 takes 932.03 s and 185.20 s/it. The initial 4x limits are 40.1799 and 46.3008 s/it, and current results remain above them. The 11x value is a later research target, not a current capability. Older O5 geometry is not an O6 denominator.

- [MiniMax validation summary](docs/testing/minimax-h3/README.md)
- O6 matched FP32/safe-FP16 report: [English](docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md) / [中文](docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.zh-CN.md)
- [Turbo workflow usage](docs/testing/minimax-h3/TURBO_WORKFLOW_USAGE.md)
- O6 implementation and baseline plan: [English](docs/superpowers/plans/2026-08-18-minimax-h3-safe-fp16-fsdp.md) / [中文](docs/superpowers/plans/2026-08-18-minimax-h3-safe-fp16-fsdp.zh-CN.md)
- [O7 preliminary LTX research plan](docs/superpowers/plans/2026-08-18-ltx-safe-fp16-research.md)
- O8 Director adapter validation: [English](docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.md) / [中文](docs/testing/minimax-h3/MMH3DIRECTOR_ADAPTER_2026-08.zh-CN.md)
- O8 implementation plan: [English](docs/superpowers/plans/2026-08-22-mmh3director-model-adapter.md) / [中文](docs/superpowers/plans/2026-08-22-mmh3director-model-adapter.zh-CN.md)
- [O9 adaptive P2P acceptance](docs/testing/windows-p2p/ADAPTIVE_CHUNKING_2026-08.zh-CN.md)
- [O10 Text Encoder TP2 plan](docs/superpowers/plans/2026-08-29-o10-minimax-h3-dual-v100-text-encoder-tp2.zh-CN.md)
- [O11 efficient-attention plan](docs/superpowers/plans/2026-09-01-o11-minimax-h3-text-encoder-efficient-attention.zh-CN.md)
- [O11 test record](docs/testing/minimax-h3/TEXT_ENCODER_EFFICIENT_ATTENTION_O11_2026-09.zh-CN.md)

## Important fixes retained in this branch

The early FSDP output could be encoded and remain finite while every frame after the first was colored noise. The final root cause was an in-place chunked RMSNorm path overwriting a tensor still needed as an LTX residual. The corrected path writes to a separate bounded output tensor.

Two additional correctness fixes preserve FP8 quantization scale during Ray mmap loading and restore the full gathered logical shape after FP8 FSDP all-gather. The activation path also distinguishes shape-preserving functions from shape-changing functions such as SwiGLU.

A playable file, non-black pixels, and absence of NaN/Inf are only basic gates. FSDP output acceptance also requires multi-frame visual inspection and rank-consistency checks.

## Known limitations

- One native-Windows machine, exactly two ranks, inference only.
- TCC is the release mode; WDDM is unsupported for the CUDA IPC/P2P fast path.
- Only the Diffusion Model loaded through `RayUNETLoader` is FSDP-sharded.
- Text Encoder, Video/Audio VAE, and upscalers are not FSDP-sharded; only the supported MiniMax H3 Text Encoder can explicitly select separate TP2.
- GGUF does not use this FSDP implementation.
- Do not force the LTX 2.3 model globally to FP16: the validated experiment produced black video and audio NaN/Inf. Keep the default BF16/FP32 inference policy and the branch's V100 fallbacks.
- Ray on Windows is Beta and has more startup/host-memory overhead than Linux + NCCL.
- The Director adapter retains the legacy MiniMaxH3Director `d655912` sample bridge and validates the `b8f721c`, `19b6a73`, and `576424f` stage bridge. Other interface coordinates require new acceptance and incompatible versions fail explicitly.
- Multi-segment reuse reduces model reloads but cannot eliminate main-process CLIP/VAE, intermediate-latent, RAM, or commit/pagefile pressure.
- Remote step previews carry only sparse rank-0 CPU latents; Director/ComfyUI still owns TAE/WebP rendering and final decode.
- Models, GPU families, drivers, quantization layouts, and longer workflows outside the validation matrix need independent correctness, memory, and visual acceptance.

## Troubleshooting

- Web UI unavailable: confirm the ComfyUI process is still alive and test `Invoke-WebRequest http://127.0.0.1:8188/ -UseBasicParsing`.
- `use_libuv was requested`: use the repository launcher; it sets `USE_LIBUV=0` and an explicit non-libuv TCPStore.
- Gloo shows a host name or wrong adapter: pass `-GlooHost <physical-adapter-ip>`; do not hard-code a development-machine address in shared scripts.
- P2P below 50 GiB/s: check TCC, active NVLink links, peer access, topology, and competing GPU processes.
- VAE `operation not supported`: retain `--disable-cuda-malloc`.
- FSDP error: use the matching repository workflow with `GPU=2`, `FSDP=true`, and `DP=1`; accepted LTX uses CPU Offload=false with Ulysses/Ring/CFG=0/0/0, while full MiniMax H3 uses CPU Offload=true with 2/1/1.

## Repository layout

```text
raylight/
├─ src/raylight/                         Raylight plus Windows P2P/FSDP layer
│  └─ integrations/mmh3director/         Director MODEL proxy, bridge, and contracts
├─ scripts/                              launcher and hardware preflight
├─ tests/                                unit tests, hardware probes, benchmark tool
├─ example_workflows/                    LTX, MiniMax H3, and Director workflows and upstream inputs
├─ benchmark_payloads/                   checked-in 5-second API prompts
├─ docs/                                 technical plans, diagnostics, and results
│  ├─ testing/minimax-h3/                MiniMax staged validation and usage
│  ├─ releases/                          bilingual release and upgrade notes
│  └─ superpowers/plans/                 O1-O8 plans and acceptance gates
├─ environment-windows-v100.json         machine-readable validation matrix
├─ requirements-windows-v100-torch212.txt O12 preferred Windows V100 dependencies
├─ requirements-windows-v100.txt         Python310/Torch2.7 rollback dependencies
├─ WINDOWS_P2P_CHANGES.md                previous-stage changes
└─ WINDOWS_FSDP_CHANGES.md               current FSDP changes
```

## Redistributable content

The repository contains source code, scripts, tests, configuration, workflow JSON, benchmark prompt payloads, and the upstream example workflow images/prompts. It does not contain model weights, LoRA files, VAEs, text encoders, generated videos, or a complete Python/ComfyUI environment snapshot.

## License and acknowledgements

- Upstream Raylight: [komikndr/raylight](https://github.com/komikndr/raylight)
- xDiT / xFuser: [xdit-project/xDiT](https://github.com/xdit-project/xDiT)
- yunchang: [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention)
- Ray: [ray-project/ray](https://github.com/ray-project/ray)
- PyTorch: [pytorch/pytorch](https://github.com/pytorch/pytorch)

Distributed under the upstream Apache License 2.0 terms. See [LICENSE](LICENSE).
