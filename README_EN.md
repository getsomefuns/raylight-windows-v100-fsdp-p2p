# Raylight Windows Dual-V100 FSDP + CUDA P2P

[简体中文](README.md) | [English](README_EN.md)

> **Experimental Preview**
> Native-Windows Raylight branch for two Tesla V100-SXM2-16GB GPUs in TCC mode, with FSDP2 weight sharding and CUDA IPC/P2P communication over NVLink.

This branch extends the previous Windows P2P/Ulysses work with the CUDA `all_gather_into_tensor` data path required by FSDP2. The validated LTX 2.3 and MiniMax H3 Diffusion Models are genuinely sharded across two V100 GPUs. Temporary weight gathering and Ulysses tensor exchange use CUDA IPC, cross-process CUDA events, and GPU P2P/NVLink. Gloo/TCPStore remains the rendezvous and control plane.

This is not Windows NCCL and not a general PyTorch `ProcessGroup`. It is a targeted compatibility layer for a single machine, exactly two ranks, Windows V100 inference.

## Upstream and authorship

This repository is derived from [Raylight](https://github.com/komikndr/raylight), created and maintained upstream by **Komikndr / Micko Lesmana**. Raylight uses Ray to manage ComfyUI GPU workers and integrates xDiT/xFuser, yunchang, and FSDP parallelism.

- Upstream repository: <https://github.com/komikndr/raylight>
- Upstream baseline: Raylight 1.9.0, commit `9a7c33d52b3d35e29f75ecff3c227de987f0d4cf`
- License: Apache License 2.0
- Current FSDP changes: [WINDOWS_FSDP_CHANGES.md](WINDOWS_FSDP_CHANGES.md)
- Previous P2P/Ulysses changes: [WINDOWS_P2P_CHANGES.md](WINDOWS_P2P_CHANGES.md)
- FSDP test and acceptance record: [docs/WINDOWS_V100_FSDP_TESTING.md](docs/WINDOWS_V100_FSDP_TESTING.md)
- Historical P2P/Ulysses record: [docs/TESTING.md](docs/TESTING.md)

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

## Current FSDP capability

### Do both GPUs compute?

Yes, during the FSDP diffusion samplers. In the final visually accepted run with the original BF16 distilled LoRA:

- GPU0 peaked at 16,224 MiB, 100% utilization, and about 354.5 W;
- GPU1 peaked at 16,156 MiB, 100% utilization, and about 348.4 W;
- both ranks returned element-identical video and audio latents in both sampling stages.

The whole workflow is not dual-GPU at every moment:

| Workflow stage | GPU0 | GPU1 | Reason |
|---|---:|---:|---|
| Text encoding and image preprocessing | primary | mostly idle | ordinary ComfyUI nodes |
| Diffusion sampler 1 | high load | high load | two-rank FSDP sampling |
| Latent upscale | primary | mostly idle | ordinary upscaler node |
| Diffusion sampler 2 | high load | high load | two-rank FSDP sampling |
| Video/audio VAE decode | primary | mostly idle | ordinary VAE nodes |

The precise claim is: **both GPUs compute during the FSDP diffusion stages; the entire workflow is not continuously dual-GPU.**

### How weights are sharded

This is not pipeline parallelism with half the layers on each GPU. For each FSDP weight tensor:

1. each rank persistently owns half of the tensor;
2. before the layer executes, both halves are temporarily gathered over CUDA P2P/NVLink;
3. both GPUs execute the layer forward with the temporary full weight;
4. the full weight is released and each rank returns to its persistent shard;
5. the process repeats for the next layer.

The validated LTX model registered 2,999 FSDP wrappers and 2,620 DTensors. Each rank retained about 11,203 MiB of Diffusion Model payload. The validated MiniMax H3 path registered 684 FSDP wrappers per rank. Each GPU still needs temporary room for the active full layer, activations, LoRA, a 128 MiB P2P buffer, and CUDA workspaces, so peak VRAM can still approach 16 GB.

### What is and is not sharded

| Component | FSDP-sharded now? | Notes |
|---|---|---|
| LTX 2.3 Diffusion Model | Yes | loaded through `RayUNETLoader` |
| MiniMax H3 Diffusion Model | Yes | validated FP8-scaled I2V/REF2VA checkpoints |
| Video/audio transformers inside LTXAV | Yes | part of the Diffusion Model |
| Text Encoder | No | ordinary ComfyUI path |
| Video VAE | No | ordinary `VAELoader` / tiled decode path |
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
| Current performance status | model/hardware dependent | 10.28% fair warm-to-warm gain | LTX correctness accepted; MiniMax H3 O1-O5/Turbo accepted; new O6 baseline pending |

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
| Python | 3.10.11 x64 |
| PyTorch | 2.7.0+cu126 |
| torchvision | 0.22.0+cu126 |
| torchaudio | 2.7.0+cu126 |
| xformers | 0.0.30 |
| Ray | 2.57.0 |
| xFuser | 0.4.5 |
| yunchang | 0.6.4 |
| ComfyUI | 0.31.0, commit `62b3c94bd45154f6486c7abf1b9efcacee96ea69` |
| Attention | `TORCH_EFFICIENT`; no FlashAttention |

See [environment-windows-v100.json](environment-windows-v100.json) and [requirements-windows-v100.txt](requirements-windows-v100.txt) for machine-readable pins.

## Installation and use

### 1. Prepare the GPUs

Confirm that both V100 GPUs use TCC, active NVLink links are visible, NVIDIA's `p2pBandwidthLatencyTest` passes peer access and correctness, and no stale ComfyUI/Ray process owns the GPUs.

### 2. Install the pinned environment

Use an isolated Python 3.10.11 environment. Do not modify unrelated ComfyUI installations.

```powershell
$PY = "<Python 3.10.11 path>\python.exe"
cd <ComfyUI>\custom_nodes
git clone `
  https://github.com/getsomefuns/raylight-windows-v100-fsdp-p2p.git raylight

& $PY -m pip install -r .\raylight\requirements-windows-v100.txt
& $PY -m pip install -e .\raylight
```

The validated V100 environment is `torch==2.7.0+cu126`, which retains `sm_70`. Do not install FlashAttention for this configuration, and do not replace the wheel with the cu128/cu130 builds used by newer architectures.

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

Open <http://127.0.0.1:8188>. The essential ComfyUI command is:

```text
main.py --listen 127.0.0.1 --port 8188 --disable-cuda-malloc
```

`--disable-cuda-malloc` is required by the validated V100 VAE path to avoid `cudaErrorNotSupported / operation not supported`.

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

The launcher sets UTF-8 logging, `USE_LIBUV=0`, localhost rendezvous, Ray memory-monitor overrides, two visible GPUs, a 128 MiB per-rank P2P buffer, and a 50 GiB/s minimum P2P gate. See the Chinese README for the complete environment-variable table.

The memory-monitor override prevents Ray from terminating a large workflow early. It can allow Windows to enter paging under pressure; monitor physical and committed memory rather than treating it as a free optimization.

## Test results

### Communication and FSDP gates

- Existing all-to-all: about 59.3 GiB/s per rank.
- FSDP all-gather: about 60.0-61.6 GiB/s per rank for 64-384 MiB synthetic shards.
- A 256 MiB shard over the 128 MiB staging buffer passed correct two-chunk transfer.
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

These results prove sharding, dual-GPU computation, and correct output. LTX performance optimization remains a separate task.

### MiniMax H3 accepted state

| Capability | Current result |
|---|---|
| I2V / REF2VA FSDP | dual rank, CUDA P2P, and FP8-scaled checkpoints pass |
| Worker lifecycle | same checkpoint reuses actors; changed checkpoints recycle actors and reclaim committed memory |
| Turbo workflows | loadable Turbo8 I2V and Turbo4 REF2VA workflows pass node/API validation |
| Accepted full profiles | I2V 640x640/56 frames; REF2VA 864x480/124 frames |
| Current compute policy | FP8 storage with FP32 V100 diffusion compute |
| O6 | Task 0 matched local baselines complete; safe FP16 is not implemented |
| O7 | preliminary model-specific safe-FP16 research for LTX/LTXAV after O6 |

The matched local FP32 baselines are now locked. I2V Turbo8 takes 1463.67 s end to end and 160.72 s/it; REF2VA Turbo4 takes 932.03 s and 185.20 s/it. Safe FP16 must be below 14.6109 and 16.8367 s/it respectively to be strictly more than 11x faster. Model load, preprocessing, VAE decode, and media write must not regress. Older O5 geometry is not an O6 denominator.

- [MiniMax validation summary](docs/testing/minimax-h3/README.md)
- [O6 matched local FP32 baseline](docs/testing/minimax-h3/SAFE_FP16_FSDP_2026-08.md)
- [Turbo workflow usage](docs/testing/minimax-h3/TURBO_WORKFLOW_USAGE.md)
- [O6 implementation and baseline plan](docs/superpowers/plans/2026-08-18-minimax-h3-safe-fp16-fsdp.md)
- [O7 preliminary LTX research plan](docs/superpowers/plans/2026-08-18-ltx-safe-fp16-research.md)

## Important fixes retained in this branch

The early FSDP output could be encoded and remain finite while every frame after the first was colored noise. The final root cause was an in-place chunked RMSNorm path overwriting a tensor still needed as an LTX residual. The corrected path writes to a separate bounded output tensor.

Two additional correctness fixes preserve FP8 quantization scale during Ray mmap loading and restore the full gathered logical shape after FP8 FSDP all-gather. The activation path also distinguishes shape-preserving functions from shape-changing functions such as SwiGLU.

A playable file, non-black pixels, and absence of NaN/Inf are only basic gates. FSDP output acceptance also requires multi-frame visual inspection and rank-consistency checks.

## Known limitations

- One native-Windows machine, exactly two ranks, inference only.
- TCC is the release mode; WDDM is unsupported for the CUDA IPC/P2P fast path.
- Only the Diffusion Model loaded through `RayUNETLoader` is FSDP-sharded.
- Text Encoder, Video/Audio VAE, and upscalers are not automatically sharded or dual-GPU.
- GGUF does not use this FSDP implementation.
- Do not force the LTX 2.3 model globally to FP16: the validated experiment produced black video and audio NaN/Inf. Keep the default BF16/FP32 inference policy and the branch's V100 fallbacks.
- Ray on Windows is Beta and has more startup/host-memory overhead than Linux + NCCL.
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
├─ scripts/                              launcher and hardware preflight
├─ tests/                                unit tests, hardware probes, benchmark tool
├─ example_workflows/                    LTX and MiniMax H3 workflows and upstream inputs
├─ benchmark_payloads/                   checked-in 5-second API prompts
├─ docs/                                 technical plans, diagnostics, and results
│  ├─ testing/minimax-h3/                MiniMax staged validation and usage
│  └─ superpowers/plans/                 O1-O7 plans and acceptance gates
├─ environment-windows-v100.json         machine-readable validation matrix
├─ requirements-windows-v100.txt         pinned Windows V100 dependencies
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
