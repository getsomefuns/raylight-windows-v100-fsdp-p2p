# comfy-aimdo Windows TCC compatibility patch

This directory records the minimal source patch used by the Raylight Windows
dual-V100 runtime. It is not a fork or a vendored binary.

- Upstream: https://github.com/Comfy-Org/comfy-aimdo
- Upstream tag: `v0.4.13`
- Upstream commit: `3f4259a502e7df7508dca7c30bd07eb936edf5a6`
- Local version: `0.4.13+tcc1`
- Detours tag: `v4.0.1`
- Detours commit: `e4bfd6b03e50de46b47abfbd1e46b384f0c5f833`

The patch keeps WDDM as the preferred Windows backend. A CUDA TCC device is
accepted only when AIMDO has successfully initialized its NVML pressure
monitor. Without WDDM and without NVML the native library fails closed.

The patch also backports the upstream 0.4.15 NVML-specific 512 MiB pressure
headroom. The 0.4.13 WDDM 512 MiB headroom is unchanged.

No wheel or compiled DLL is committed here. Use the runtime build script and
retain the generated build manifest with artifact hashes.
