# MiniMax H3 Windows Dual V100 Implementation Plan

> Execute this plan inside the dedicated worktree. Keep the installed ComfyUI node checkout as a deployment target, not as the source of truth.

**Goal:** Run the upstream MiniMax H3 I2V and REF2VA Raylight workflows correctly and efficiently on native Windows with two V100-SXM2-16GB GPUs, using the branch's CUDA P2P FSDP+Ulysses topology.

**Architecture:** Use the existing Raylight Ray workers and xFuser MiniMax sequence-parallel injection. Shard only the MiniMax diffusion model with FSDP2 and carry CUDA collectives through the custom Windows CUDA IPC/P2P ProcessGroup. Keep Qwen3-VL and the VAEs in normal ComfyUI stages. Start with pruned FP8-scaled diffusion weights and FP32 compute fallback, then compare INT8 ConvRot only after correctness.

**Fixed stack:** Windows 23H2, driver 577.00, 2× V100 TCC, Python 3.10.11, torch 2.7.0+cu126, xformers 0.0.30, Ray 2.57.0, yunchang 0.6.4, comfy-kitchen 0.2.30.

---

## Task 1: Capture a reproducible MiniMax baseline

**Files:**

- Create: `scripts/minimax-h3/inspect-environment.ps1`
- Create: `tests/test_minimax_h3_assets.py`
- Create: `docs/testing/minimax-h3/README.md`

**Step 1: Write the failing asset-manifest test**

The test must define the required FP8-baseline filenames, official repository paths, approximate byte sizes and target ComfyUI model directories. It must initially fail because no manifest module exists.

Run:

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_minimax_h3_assets.py -q
```

Expected: FAIL with missing MiniMax asset manifest.

**Step 2: Implement environment and model inventory**

The PowerShell script records, without secrets or machine identifiers:

- GPU model, TCC/WDDM mode, VRAM and driver;
- Python, torch/CUDA, Ray, xformers, yunchang and comfy-kitchen versions;
- physical RAM, commit limit and E: free space;
- required model presence and local file sizes;
- the five required workflow node classes.

The test owns the canonical manifest constants so later workflow and downloader tests share one source of truth.

**Step 3: Run tests and inventory**

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_minimax_h3_assets.py -q
powershell -ExecutionPolicy Bypass -File scripts/minimax-h3/inspect-environment.ps1
```

Expected: tests PASS; inventory reports models missing but environment compatible.

**Step 4: Commit**

```powershell
git add scripts/minimax-h3/inspect-environment.ps1 tests/test_minimax_h3_assets.py docs/testing/minimax-h3/README.md
git commit -m "test: add MiniMax H3 environment and asset inventory"
```

## Task 2: Build deterministic Windows V100 workflows

**Files:**

- Create: `scripts/minimax-h3/build_workflows.py`
- Create: `tests/test_minimax_h3_workflows.py`
- Create: `example_workflows/Minimax_H3_I2V_Windows_V100_FSDP.json`
- Create: `example_workflows/Minimax_H3_REF2VA_Windows_V100_FSDP.json`

**Step 1: Write failing workflow-transform tests**

Tests load the two upstream workflows and require the generated variants to have:

- two Ray workers;
- Ulysses 2, Ring 1, CFG 1, DP 1;
- synchronized Ulysses enabled;
- FSDP enabled and CPU offload disabled;
- `TORCH_EFFICIENT` attention;
- mmap enabled;
- pruned FP8-scaled diffusion filenames;
- user input filename for I2V;
- the same initial validation image connected to both REF2VA image inputs;
- upstream prompts, sampler, seed and scheduler preserved;
- unique output prefixes identifying FP8/FSDP validation.

Run:

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_minimax_h3_workflows.py -q
```

Expected: FAIL because the builder/output files do not exist.

**Step 2: Implement the workflow builder**

Transform by node type and input name, never by fragile visual coordinates. Reject unexpected upstream schemas with a clear message.

Generate both full validation workflows and accept command-line overrides for smoke resolution, duration, frame count and steps without modifying upstream files.

**Step 3: Generate and verify**

```powershell
E:\ComfyUI-py310\Python310\python.exe scripts/minimax-h3/build_workflows.py
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_minimax_h3_workflows.py -q
```

Expected: PASS and two deterministic JSON files.

**Step 4: Commit**

```powershell
git add scripts/minimax-h3/build_workflows.py tests/test_minimax_h3_workflows.py example_workflows/Minimax_H3_*_Windows_V100_FSDP.json
git commit -m "feat: add Windows V100 MiniMax H3 FSDP workflows"
```

## Task 3: Add a resumable, validated model acquisition script

**Files:**

- Create: `scripts/minimax-h3/models.json`
- Create: `scripts/minimax-h3/download-models.ps1`
- Extend: `tests/test_minimax_h3_assets.py`

**Step 1: Add failing download-manifest tests**

Require official Comfy-Org URLs, target subdirectories, remote byte sizes, `.part` staging, HTTP resume and a no-overwrite rule for complete files.

**Step 2: Implement the downloader**

The script accepts named groups:

- `i2v-fp8`: text encoder, video/audio VAEs, FL2VA FP8 model;
- `ref2va-fp8`: REF2VA FP8 model only;
- `i2v-int8` and `ref2va-int8`: later comparison models;
- `turbo`: optional official LoRAs.

It downloads into `E:\ComfyUI-aki-v3\ComfyUI\models`, uses resumable `.part` files, verifies final byte size and checks that the safetensors header is readable before renaming.

**Step 3: Test without downloading**

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_minimax_h3_assets.py -q
powershell -ExecutionPolicy Bypass -File scripts/minimax-h3/download-models.ps1 -Group i2v-fp8 -WhatIf
```

Expected: PASS and a four-file I2V plan totaling about 39.5 GiB.

**Step 4: Commit**

```powershell
git add scripts/minimax-h3/models.json scripts/minimax-h3/download-models.ps1 tests/test_minimax_h3_assets.py
git commit -m "feat: add resumable MiniMax H3 model acquisition"
```

## Task 4: Download and validate the I2V baseline assets

**Files:**

- Runtime models under `E:\ComfyUI-aki-v3\ComfyUI\models`
- Runtime input under `E:\ComfyUI-py310\ComfyUI\input`
- Create artifact: `docs/testing/minimax-h3/artifacts/model-inventory-i2v-fp8.txt`

**Step 1: Copy the supplied image into ComfyUI input**

Use a stable filename such as `minimax_h3_green_robots.jpg`. Preserve the source image.

**Step 2: Start resumable I2V downloads**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/minimax-h3/download-models.ps1 -Group i2v-fp8
```

Do not start ComfyUI sampling until all four final files pass size/header validation.

**Step 3: Re-run inventory**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/minimax-h3/inspect-environment.ps1
```

Expected: I2V FP8 group complete; REF2VA diffusion still intentionally missing.

## Task 5: Add MiniMax FSDP quantized-load preflight coverage

**Files:**

- Create: `tests/test_minimax_h3_fsdp_quant.py`
- Modify only if tests prove necessary:
  - `src/raylight/comfy_dist/fsdp_utils.py`
  - `src/raylight/comfy_dist/model_patcher.py`
  - `src/raylight/comfy_dist/kitchen_patches/fp8.py`
  - `src/raylight/distributed_worker/ray_worker.py`

**Step 1: Write focused failing tests**

Cover MiniMax-relevant shapes and metadata:

- FP8 QuantizedTensor row sharding and reconstruction;
- odd/scalar FP32 islands replicated rather than incorrectly split;
- 5376-wide hidden layers and 14336-wide FFN tensors;
- FP32 compute policy on sm_70;
- hybrid FSDP+Ulysses topology validation;
- custom P2P process group selected for CUDA all-gather/all-to-all.

**Step 2: Run the focused suite**

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_minimax_h3_fsdp_quant.py -q
```

If it passes without source changes, retain the test as evidence. If it fails, implement the smallest source correction and rerun.

**Step 3: Run regression tests**

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests -q
```

Expected: all repository tests pass; standalone-path assumptions must be fixed in tests rather than bypassed.

**Step 4: Commit**

```powershell
git add tests/test_minimax_h3_fsdp_quant.py src/raylight
git commit -m "test: validate MiniMax H3 quantized FSDP topology"
```

## Task 6: Deploy the tested worktree revision to ComfyUI

**Files:**

- Create: `scripts/minimax-h3/deploy-to-comfyui.ps1`
- Extend: `tests/test_minimax_h3_assets.py`

**Step 1: Implement safe deployment checks**

The script must:

- require the source worktree and destination custom node to be Raylight repositories;
- refuse deployment if the destination has tracked modifications;
- synchronize only tracked project files required at runtime;
- never copy `.git`, tests, outputs, models or local logs;
- record the deployed source commit in a local runtime marker ignored by Git.

**Step 2: Test dry-run and deploy**

```powershell
powershell -ExecutionPolicy Bypass -File scripts/minimax-h3/deploy-to-comfyui.ps1 -WhatIf
powershell -ExecutionPolicy Bypass -File scripts/minimax-h3/deploy-to-comfyui.ps1
```

Expected: installed `custom_nodes/raylight` matches the tested source without altering its Git metadata.

## Task 7: Run I2V model-load and distributed preflight

**Files:**

- Runtime log: `E:\ComfyUI-py310\ComfyUI\user\comfyui_8188.log`
- Create artifacts under: `docs/testing/minimax-h3/artifacts/i2v-preflight/`

**Step 1: Establish idle baseline**

Stop stale ComfyUI/Ray workers, verify both GPUs near idle, and record RAM/commit/VRAM.

**Step 2: Launch with the validated Windows P2P script**

Use `--disable-cuda-malloc`. Do not add `--highvram` initially because the unsharded 32B text encoder must be unloaded before diffusion sampling.

**Step 3: Execute a load/preflight workflow**

Require:

- both actors initialized;
- mmap enabled;
- FSDP wrapper/shard diagnostics present;
- P2P health probe passes at the configured floor;
- no CUDA tensor collective silently uses Gloo;
- no OOM or commit-limit exhaustion.

If initialization fails, capture the smallest reproducer before source changes.

## Task 8: Run and validate I2V smoke output

**Files:**

- Generated smoke workflow under `E:\ComfyUI-py310\ComfyUI\user\default`
- Artifacts under `docs/testing/minimax-h3/artifacts/i2v-smoke/`

**Step 1: Generate a reduced workload**

Use the same image, prompt, sampler and seed, but reduce resolution and frames. Keep enough sampling steps to distinguish a correct video from numerical noise.

**Step 2: Monitor the full run**

Record stage timings, both GPU traces, RAM/commit/pagefile and rank progress.

**Step 3: Validate tensors and media**

Require finite video/audio latents, valid video/audio containers, temporal variation and coherent frames at the beginning, middle and end.

**Step 4: Scale to upstream I2V settings**

Run 0.4 MP, 2 seconds and 20 steps only after smoke acceptance.

For any failure, add a focused regression test before changing Raylight source.

## Task 9: Add and validate REF2VA

**Files:**

- Download REF2VA FP8 model to shared model root
- Artifacts under `docs/testing/minimax-h3/artifacts/ref2va/`

**Step 1: Download and verify `ref2va-fp8`**

**Step 2: Run reduced REF2VA with duplicated green-robot references**

Validate text/image conditioning, video latent, audio latent, distributed progress and media output independently.

**Step 3: Scale toward 0.4 MP and 5 seconds**

Do not solve rank skew by only increasing timeout. Correlate any skew with GPU, RAM, commit, page faults and collective traces and correct its cause.

## Task 10: Compare INT8 and performance variants

**Files:**

- Artifacts under `docs/testing/minimax-h3/artifacts/benchmarks/`
- Extend: `docs/testing/minimax-h3/README.md`

**Step 1: Download one INT8 model and run an identical workload**

Confirm whether sm_70 uses eager/dequantized fallback. Compare peak VRAM, host commit, model-load time, sampler iteration time and output correctness.

**Step 2: Select precision default**

Keep FP8 unless INT8 is stable and materially improves a measured resource or speed metric.

**Step 3: Evaluate official Turbo LoRA variants**

Compare 20-step base quality against applicable 8-step/4-step variants. Use matched input and seed where the workflow permits.

**Step 4: Record cold and warm results**

At least one cold run and two warm runs for the selected I2V and REF2VA settings.

## Task 11: Final regression, review and documentation

**Files:**

- Modify: `README.md`
- Modify: `README_CN.md` if present
- Update: `docs/testing/minimax-h3/README.md`
- Create: `docs/testing/minimax-h3/ACCEPTANCE.md`

**Step 1: Run the complete automated suite**

```powershell
E:\ComfyUI-py310\Python310\python.exe -m pytest tests -q
```

**Step 2: Run release smoke checks**

Re-run the selected I2V and REF2VA validation workflows from an idle state and verify cleanup.

**Step 3: Review code and security/reproducibility boundaries**

Ensure no model, input, output, absolute local runtime path, token or machine-specific log is staged for Git.

**Step 4: Commit documentation and acceptance evidence**

```powershell
git add README.md README_CN.md docs/testing/minimax-h3
git commit -m "docs: publish MiniMax H3 Windows V100 validation"
```

Do not push or alter the public repository until the user has reviewed the local result.
