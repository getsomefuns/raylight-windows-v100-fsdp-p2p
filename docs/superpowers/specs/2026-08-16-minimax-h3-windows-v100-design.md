# MiniMax H3 on Windows Dual V100 — Design Specification

Date: 2026-08-16

## 1. Objective

Run the upstream Raylight MiniMax H3 I2V and REF2VA workflows on native Windows with two Tesla V100-SXM2-16GB GPUs in TCC mode, using the existing Raylight Windows CUDA P2P/FSDP branch.

The result must prioritize, in this order:

1. numerically correct video and audio output;
2. stable execution within two 16GB VRAM devices and a 64GB host;
3. verified participation of both GPUs through CUDA P2P/NVLink;
4. best warm-run performance that does not materially reduce output quality;
5. reproducibility inside `E:\ComfyUI-py310` without changing the NVIDIA driver or other Python environments.

## 2. In-scope workflows and input

- `example_workflows/Minimax_H3_I2V_Raylight.json`
- `example_workflows/Minimax_H3_REF2VA_Raylight.json`
- User-provided green-robot image as the I2V input.
- The same image will initially be supplied to both REF2VA image-reference inputs because the two original pasted source images are not present in the repository. This is a pipeline-validation input, not the final creative reference set.

The original workflow files remain unchanged. Adapted and benchmark workflows will be stored separately with explicit Windows V100 names.

## 3. Fixed environment boundary

The work is restricted to:

- ComfyUI and Python: `E:\ComfyUI-py310`
- Shared model root: `E:\ComfyUI-aki-v3\ComfyUI\models`
- Development worktree: `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`

The NVIDIA driver remains 577.00. No changes are made to environments outside `E:\ComfyUI-py310`.

Validated baseline:

- Windows 23H2
- 2× Tesla V100-SXM2-16GB, TCC, NVLink/P2P
- Python 3.10.11
- PyTorch 2.7.0+cu126 with sm_70
- xformers 0.0.30
- Ray 2.57.0
- xFuser 0.4.5-compatible environment
- yunchang 0.6.4
- comfy-kitchen 0.2.30
- ComfyUI launch includes `--disable-cuda-malloc`

PyTorch cu128/cu130 must not replace the cu126 build unless a separately proven sm_70-compatible wheel exists. The current driver is not a reason to change the PyTorch runtime.

## 4. Resource assessment

At design time:

- E: free space: 194.39 GiB
- physical RAM: 63.91 GiB
- current commit limit: 111.91 GiB
- pagefile: 48–96 GiB on E:

The exact five workflow-requested files occupy about 59.1 GiB:

- two pruned INT8 ConvRot diffusion models: about 39.06 GiB total;
- Qwen3-VL-32B NVFP4 AWQ text encoder: about 14.61 GiB;
- video VAE: about 4.85 GiB;
- audio VAE: about 0.56 GiB.

Model loading must use mmap. Downloads must be resumable and validated before execution. Only one diffusion precision family is downloaded initially to preserve disk and pagefile headroom.

## 5. Precision and kernel strategy

### 5.1 Text encoder

Use `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` as supplied by Comfy-Org. Its NVFP4 format supports a non-Blackwell fallback, but on V100 the text encoder is expected to dequantize through a compatibility path.

The text encoder is not sharded by Raylight FSDP. It remains a main-process ComfyUI stage and is a separate RAM/VRAM risk from the diffusion model.

### 5.2 Diffusion model baseline

Use the pruned FP8-scaled FL2VA and REF2VA diffusion models first:

- `minimax_h3_fl2va_pruned_fp8_scaled.safetensors`
- `minimax_h3_ref2va_pruned_fp8_scaled.safetensors`

Reasons:

- the current FSDP branch has already passed end-to-end scaled-FP8 validation on LTX;
- V100 has no native FP8 or BF16 compute, so numerically safe FP32 fallback is expected;
- FP8-scaled dequantization is a lower-risk first target than ConvRot on sm_70;
- MiniMax H3 officially supports BF16/FP32 inference, not FP16.

No change will add FP16 to MiniMax H3 `supported_inference_dtypes`. Previous LTX testing established that forcing FP16 for BF16/FP32-trained models can produce invalid latent/audio values and black output.

### 5.3 INT8 ConvRot comparison

After FP8 end-to-end correctness, test the exact workflow INT8 ConvRot models as a controlled comparison.

The current comfy-kitchen fast INT8 path requires sm_75, while V100 is sm_70. Therefore the INT8 checkpoint may run through layer-wise dequantization rather than fast INT8 Tensor Core math. Raylight FSDP can shard and communicate INT8 payloads, but it cannot create missing V100 INT8 kernels.

INT8 becomes the default only if it passes correctness and is measurably better in memory or warm-run performance.

### 5.4 Rejected baseline

Pruned BF16 models are not the initial route. V100 cannot execute BF16 natively, and their larger checkpoint/resident footprint would fall back toward FP32, increasing both memory and transfer cost.

## 6. Distributed topology

Primary topology:

- workers / GPUs: 2
- FSDP: enabled
- FSDP CPU offload: disabled initially
- Ulysses degree: 2
- Ring degree: 1
- CFG degree: 1
- DP degree: 1
- synchronized Ulysses: enabled
- attention backend: `TORCH_EFFICIENT`
- mmap: enabled

This is the branch's supported Windows hybrid topology. FSDP shards diffusion weights across both GPUs, while Ulysses splits the packed video/audio/text sequence during attention.

The custom Windows CUDA ProcessGroup must carry matched CUDA collectives through IPC/P2P. Gloo is retained only for control-plane operations or explicitly unmatched CPU collectives; a silent CUDA payload fallback to host memory is a failure.

## 7. Execution stages

### Stage M0 — Reproducibility and model manifest

- capture package, ComfyUI, Raylight and hardware versions;
- verify TCC and CUDA peer access;
- validate required node registrations;
- create a model manifest with official URLs, expected sizes and local destinations;
- install only missing dependencies inside the dedicated Python 3.10 environment.

### Stage M1 — FP8 model-load preflight

- download the NVFP4 text encoder, both VAEs and FL2VA FP8 diffusion model;
- run safetensors header/model-detection checks;
- validate mmap behavior and peak host commit during text encoding/model initialization;
- run Raylight FSDP preflight without a full video.

### Stage M2 — I2V smoke and correctness

- create a reduced-resolution, reduced-frame workflow with the supplied image;
- verify both ranks initialize identical model metadata and produce finite outputs;
- inspect per-rank FSDP diagnostics, P2P transport counters and GPU telemetry;
- scale to the upstream 0.4 MP, 2-second, 20-step workflow after smoke success.

### Stage M3 — REF2VA correctness

- initially duplicate the supplied image into both reference-image inputs;
- begin with a shorter/lower-resolution workload;
- validate video and audio output separately;
- scale toward the upstream 0.4 MP, 5-second workflow.

### Stage M4 — INT8 comparison

- download one INT8 ConvRot diffusion model at a time;
- compare against FP8 at identical seed, prompt, input, resolution, frames and steps;
- record whether the execution uses a fast kernel or dequantized fallback;
- retain the better stable variant and remove no downloaded model without explicit user approval.

### Stage M5 — Performance/effect optimization

- measure cold and warm runs separately;
- evaluate base 20-step sampling and official Turbo LoRA variants where applicable;
- tune frame count, resolution, attention synchronization and load/unload timing;
- keep CPU offload disabled unless required for correctness;
- select the final configuration by the speed/quality Pareto result, not speed alone.

## 8. Observability

Each substantial run records:

- workflow and model variant;
- seed, resolution, frame count, duration and steps;
- model load, text encode, sampling, video VAE, audio VAE and total time;
- sampler seconds/iteration;
- GPU0/GPU1 utilization, power and peak VRAM;
- physical RAM, commit usage and pagefile growth;
- FSDP wrapper/shard diagnostics;
- P2P collective type, byte count, bandwidth and fallback count;
- output filename and compact visual/audio validation result.

Long raw logs and telemetry series are stored as artifacts. The maintained Markdown report contains summarized results and links to artifacts.

## 9. Acceptance criteria

### Functional

- both adapted workflows load without missing nodes or files;
- I2V and REF2VA both finish at an agreed validation setting;
- video contains coherent multi-frame motion rather than black frames, first-frame-only output or noise;
- audio is finite and decodes into a valid stream;
- no NaN or infinity is present in sampled video/audio latent tensors.

### Distributed

- both ranks initialize and complete every sampling step;
- the diffusion model is genuinely sharded, not fully replicated on both GPUs;
- both GPUs show sustained compute participation during sampling;
- CUDA collectives use the custom P2P/NVLink path with zero silent host fallback;
- rank fingerprints/checkpoints agree where equality is required.

### Resource safety

- neither GPU exceeds usable VRAM or triggers OOM;
- host commit remains below the active limit with a safety margin;
- pagefile behavior is recorded and does not cause rank skew/deadlock;
- worker and model cleanup return GPU/host memory close to the measured idle baseline.

### Performance

- at least one cold run and two warm runs are recorded for the selected I2V and REF2VA settings;
- FP8 and INT8 comparisons use identical workload variables;
- the final default is the fastest configuration that passes all functional and output-quality checks;
- no performance claim is published without matching telemetry and output evidence.

## 10. Deliverables

- Windows V100 MiniMax H3 model/download manifest;
- adapted I2V and REF2VA workflows;
- any required Raylight FSDP/quantization fixes with tests;
- reproducible launch/preflight commands;
- summarized MiniMax H3 validation and benchmark document;
- retained output artifacts for correctness review;
- README updates describing support scope and limitations after validation.

## 11. Non-goals

- changing the NVIDIA driver;
- modifying Python or ComfyUI installations outside `E:\ComfyUI-py310`;
- claiming native V100 FP8, BF16, NVFP4 or fast ConvRot execution;
- sharding the Qwen text encoder or VAEs unless later evidence shows that this is required and a separate design is approved;
- treating a longer timeout or larger pagefile as a substitute for finding rank skew or memory faults.
