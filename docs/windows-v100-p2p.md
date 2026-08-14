# Native Windows dual-V100 CUDA P2P transport

This fork provides an experimental CUDA IPC/P2P data path for Raylight on a
single native-Windows machine with two Tesla V100 GPUs in TCC mode. The tested
LTX 2.3 workflow uses Ulysses sequence parallelism while Gloo remains the
process-group, rendezvous, control, barrier, and unsupported-collective backend.

The implementation is intentionally strict. A supported synchronous CUDA
`all_to_all_single` call uses the P2P endpoint. A P2P failure poisons the
endpoint instead of silently retrying the same operation through host memory.

## What this is and is not

Supported fast path:

- Native 64-bit Windows, one machine.
- Exactly two Ray workers and two CUDA devices.
- TCC mode with CUDA peer access and CUDA IPC.
- Equal-split, synchronous CUDA `all_to_all_single`.
- Ulysses degree 2, ring/CFG/DP degree 1, FSDP disabled.

Not implemented:

- NCCL or a general PyTorch ProcessGroup.
- FSDP, weight sharding, multi-node, or more than two ranks.
- Asynchronous collectives or explicit unequal split sizes.
- Multi-GPU VAE/text encoding/decoding.
- A performance guarantee for WDDM or non-V100 hardware.

## Validated configuration

| Component | Validated value |
|---|---|
| GPU | 2x Tesla V100-SXM2-16GB, six active NVLink links per GPU |
| Driver mode | TCC |
| NVIDIA driver | 577.00 |
| Windows | NT build 22631.6199 / 23H2 |
| Python | 3.10.11 x64 |
| ComfyUI | 0.31.0, commit `62b3c94bd45154f6486c7abf1b9efcacee96ea69` |
| Raylight base | 1.9.0, commit `9a7c33d52b3d35e29f75ecff3c227de987f0d4cf` |
| PyTorch | 2.7.0+cu126 |
| xformers | 0.0.30 |
| Ray | 2.57.0 |
| xFuser / yunchang | 0.4.5 / 0.6.4 |

`nvidia-smi` showing CUDA 12.9, an installed CUDA 12.9 Toolkit, and PyTorch's
CUDA 12.6 runtime are different facts. Runtime use of this Python backend does
not require a locally installed 12.9 Toolkit. The Toolkit is needed only to
compile CUDA samples or other native extensions.

## Installation

Clone this fork under `ComfyUI/custom_nodes` and select the P2P branch:

```powershell
cd <ComfyUI>\custom_nodes
git clone https://github.com/getsomefuns/raylight-windows-v100-p2p.git raylight
cd raylight
git switch windows-v100-p2p
```

Use Python 3.10.11 and install the validated package set:

```powershell
<python.exe> -m pip install -r requirements-windows-v100.txt
<python.exe> -m pip install -e .
```

Do not install FlashAttention for the validated V100 configuration. Do not
upgrade this environment to a CUDA 12.8/13.0 PyTorch wheel: the tested wheel is
`torch==2.7.0+cu126`, including `sm_70` support.

## Verify before starting ComfyUI

Run the quick identity and version check:

```powershell
.\scripts\verify-windows-v100.ps1 -PythonPath <python.exe>
```

Run the real two-Ray-actor correctness and P2P probe as the release gate:

```powershell
.\scripts\verify-windows-v100.ps1 -PythonPath <python.exe> -RunP2PProbe
```

Also validate the machine with NVIDIA's `p2pBandwidthLatencyTest`. Physical
PCIe carrier layout does not need to match the development machine, but CUDA
peer access and real bandwidth must pass.

## Start ComfyUI

From the Raylight repository:

```powershell
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath <python.exe>
```

To validate path discovery and environment setup without opening ComfyUI:

```powershell
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath <python.exe> -ValidateOnly
```

The script automatically derives the ComfyUI root from
`ComfyUI/custom_nodes/raylight`. It enables:

- `USE_LIBUV=0` and a local TCPStore rendezvous.
- Ray memory-monitor overrides used by the validated large workflow.
- Windows CUDA P2P mode with one persistent 64 MiB send buffer per rank.
- A strict 50 GiB/s startup health threshold.
- `CUDA_VISIBLE_DEVICES=0,1`.
- ComfyUI `--disable-cuda-malloc` for the V100 VAE path.

`RAYLIGHT_GLOO_HOST` is normally auto-selected. If Windows chooses the wrong
Hyper-V, VPN, WSL, or tunnel adapter, supply the physical adapter explicitly:

```powershell
.\scripts\start-comfyui-windows-p2p.ps1 -GlooHost 192.0.2.10
```

Replace the documentation address with the machine's real IPv4 address.

## RayInitializer settings

Use these settings for the validated workflow:

| Setting | Value |
|---|---:|
| GPU | 2 |
| ulysses_degree | 2 |
| ring_degree / cfg_degree / dp_degree | 1 / 1 / 1 |
| sync_ulysses | true |
| FSDP / FSDP_CPU_OFFLOAD | false / false |
| XFuser_attention | TORCH_EFFICIENT |
| skip_comm_test | true |

`skip_comm_test=true` skips Raylight's older general communication tester. It
does not skip this fork's strict CUDA P2P health check.

Expected startup messages include:

```text
[Raylight] Windows Gloo init OK ...
[Raylight] Windows CUDA P2P enabled: ...
```

An unchanged later prompt should also report worker reuse:

```text
[Raylight] Reusing 2 live Ray workers for unchanged configuration
```

The validated workflow is included at
`example_workflows/LTX2_3_i2v_Raylight_Windows_P2P.json`. Replace its image
input with your own file and install the models listed in
[`ltx23-model-manifest.md`](ltx23-model-manifest.md).

## Measured results on the development machine

- CUDA P2P actor health check for a real 110 MiB-sized collective: about
  59.3 GiB/s remote payload per direction.
- Native single-V100 LTX 2.3 cold run: 519.94 s.
- Native single-V100 LTX 2.3 hot run: 316.60 s.
- Dual-V100 P2P reused-session median: 284.06 s.
- Fair hot-to-hot end-to-end improvement: 10.28%.

These results prove correctness and a real NVLink/P2P data path, but the current
release has not reached the project's future 20% end-to-end performance target.
Small-collective synchronization and per-call Python/control overhead remain
important optimization areas.

## Failure interpretation

- `use_libuv was requested`: start with `USE_LIBUV=0`; the supplied script does.
- P2P bandwidth below threshold: verify TCC, NVLink state, peer access, and GPU
  topology. Lowering the threshold changes the guardrail, not the hardware.
- Wrong Gloo adapter or timeout: pass the physical IPv4 with `-GlooHost`.
- FSDP requested: unsupported on this Windows P2P path; disable it.
- Payload exceeds capacity: increase `-P2PCapacityBytes` only after checking
  available VRAM. The validated maximum input was 115,343,360 bytes.

Ray on Windows remains a beta platform. Treat this branch as an experimental,
single-machine compatibility implementation rather than a drop-in Linux/NCCL
replacement.
