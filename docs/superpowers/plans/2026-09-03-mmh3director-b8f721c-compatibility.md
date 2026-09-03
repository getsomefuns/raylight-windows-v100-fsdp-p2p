# MiniMaxH3Director b8f721c Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This execution is explicitly single-agent; do not dispatch subagents.

**Goal:** Restore Raylight dual-V100 remote sampling for the installed MiniMaxH3Director commit `b8f721c` without modifying Director or regressing the previously supported `d655912` path.

**Architecture:** Keep the existing legacy `comfy.sample.sample` bridge for old Director releases. Add a capability-detected stage bridge at Director's `sample_single_stage` boundary for the new `BasicGuider -> SamplerCustomAdvanced` pipeline; only `RayMMH3DirectorModelProxy` is intercepted, while ordinary ComfyUI models continue through the untouched Director implementation. The stage bridge prepares deterministic noise and latent metadata in the main process, then sends the same sampling controls and optional external SIGMAS to the existing Ray workers.

**Tech Stack:** Python 3.10.11, PyTorch 2.7.0+cu126, ComfyUI 0.31.0, Ray, pytest, two Tesla V100-SXM2 GPUs in TCC mode.

**Spec:** Approved bounded design in the 2026-09-03 project conversation; this document records the complete implementation and acceptance contract.

## Global Constraints

- Modify only `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`; deploy only after automated tests pass.
- Do not modify `ComfyUI_MiniMaxH3_Director` or ComfyUI core.
- Keep the formal runtime at Python 3.10.11, PyTorch 2.7.0+cu126, xFormers 0.0.30, and comfy-aimdo 0.4.13.
- Preserve all non-Director Raylight workflows and the legacy Director `d655912` bridge.
- Never silently fall back to local GPU0 sampling when the model is a Ray proxy.
- Execute and review in one agent only.

---

### Task 1: Reproduce the new Director routing break

**Files:**
- Modify: `tests/test_mmh3director_bridge.py`
- Modify: `tests/test_mmh3director_nodes.py`

**Interfaces:**
- Consumes: installed Director `sample_single_stage(*, model, positive, negative, latent, seed, cfg, steps, sampler_name, scheduler, shift_video, shift_audio, on_phase, on_step_preview, preview_every, denoise, phase_name, sigmas, apply_shift)`.
- Produces: failing tests proving a Ray proxy is routed before `BasicGuider` can treat it as a full `ModelPatcher`.

- [ ] **Step 1: Add a new-contract regression test**

  Build a real proxy fixture and call the stage wrapper with a latent dictionary, explicit SIGMAS, fixed seed, and phase callback. Assert that the remote dispatcher receives the exact controls and that the returned latent preserves unrelated metadata.

- [ ] **Step 2: Add pass-through and installation tests**

  Assert that a normal non-Ray model calls the original Director function unchanged, and that installation replaces aliases already imported into both `executor_core` and `refine_sampling` without modifying unrelated modules.

- [ ] **Step 3: Add adapter-time installation test**

  Assert that `RayMMH3DirectorModelAdapter.adapt_model()` refreshes the stage bridge after all custom nodes have loaded, so import order cannot disable compatibility.

- [ ] **Step 4: Run the focused tests and verify RED**

  Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_mmh3director_bridge.py tests/test_mmh3director_nodes.py -q`

  Expected: the new stage-routing tests fail because the installed bridge only wraps `comfy.sample.sample`.

### Task 2: Implement the capability-detected stage bridge

**Files:**
- Modify: `src/raylight/integrations/mmh3director/bridge.py`
- Modify: `src/raylight/integrations/mmh3director/nodes.py`

**Interfaces:**
- Produces: `dispatch_mmh3director_stage(...) -> dict`, an idempotent stage-bridge installer, and a strict new Director contract validator.
- Preserves: `dispatch_mmh3director_sample(...) -> Tensor` and the legacy global bridge.

- [ ] **Step 1: Implement the explicit stage wrapper**

  For a Ray proxy, normalize the latent, prepare deterministic noise from the Director seed, preserve the optional noise mask and external SIGMAS, apply the requested MiniMax shift to proxy metadata, dispatch to all workers, and rebuild the Director latent dictionary. Call Director phase callbacks at start and successful completion.

- [ ] **Step 2: Implement safe pass-through**

  For every object other than `RayMMH3DirectorModelProxy`, call the exact original function with all positional and keyword values unchanged.

- [ ] **Step 3: Install against imported aliases**

  Patch the canonical `core_sampling.sample_single_stage` and only aliases that still reference that exact original function. Mark wrappers, retain originals for tests, and make repeated installation idempotent.

- [ ] **Step 4: Refresh at adapter execution**

  Call the bridge installer from `adapt_model()` after worker validation. Director remains an optional dependency during Raylight import, but must be present when this adapter node executes.

- [ ] **Step 5: Run focused tests and verify GREEN**

  Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_mmh3director_bridge.py tests/test_mmh3director_nodes.py tests/test_mmh3director_optional_dependency.py -q`

  Expected: all pass; legacy and new paths coexist.

### Task 3: Preserve new sampling controls inside workers

**Files:**
- Modify: `src/raylight/distributed_worker/ray_worker.py`
- Modify: `tests/test_mmh3director_worker.py`

**Interfaces:**
- Consumes: request fields `sigmas` and `apply_shift` from the main-process stage bridge.
- Produces: worker sampling that honors external SIGMAS and does not apply a shift when Director explicitly disables it.

- [ ] **Step 1: Write worker RED tests**

  Add one test with literal external SIGMAS and one with `apply_shift=False`. Assert the exact values reaching `comfy.sample.sample`; the false branch must not invoke `MiniMaxH3SigmaShift`.

- [ ] **Step 2: Run worker tests and verify RED**

  Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_mmh3director_worker.py -q`

  Expected: the `apply_shift=False` test fails because the current worker always applies the shift.

- [ ] **Step 3: Implement minimal request handling**

  Default `apply_shift` to true for backward compatibility. When false, clone the worker model patcher without applying SigmaShift; merge model options exactly as in the shifted path. Continue passing external SIGMAS directly to ComfyUI sampling.

- [ ] **Step 4: Run worker and lifecycle tests**

  Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_mmh3director_worker.py tests/test_mmh3director_lifecycle.py -q`

  Expected: all pass, including cleanup after success and failure.

### Task 4: Update compatibility documentation and regression scope

**Files:**
- Modify: `README.md`
- Modify: `README_EN.md`
- Create: `docs/releases/2026-09-03-mmh3director-b8f721c-compatibility.zh-CN.md`
- Create: `docs/releases/2026-09-03-mmh3director-b8f721c-compatibility.en.md`
- Modify: `tests/test_mmh3director_bridge.py`

**Interfaces:**
- Produces: user-facing compatibility coordinates and an accurate description of dual legacy/new sampling seams.

- [ ] **Step 1: Update current README claims**

  State that `d655912` uses the legacy `comfy.sample.sample` bridge and `b8f721c` uses the stage bridge. Preserve the statement that Director itself is not forked or modified.

- [ ] **Step 2: Write bilingual release notes**

  Record the root cause, implemented routing, supported controls, unchanged runtime, tests, deployment status, and known limits. Do not rewrite historical O8 release records.

- [ ] **Step 3: Run documentation-adjacent contract tests**

  Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest tests/test_mmh3director_bridge.py tests/test_mmh3director_optional_dependency.py -q`

  Expected: all compatibility messages name the current validated coordinate.

### Task 5: Full verification, deployment, and real workflow smoke test

**Files:**
- Source: `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`
- Deploy to: `E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight`
- Test workflow: `E:\ComfyUI-py310\ComfyUI\user\default\workflows\MMH3Director.json`

**Interfaces:**
- Produces: a clean deployed node package that routes the installed Director `b8f721c` to both Ray workers.

- [ ] **Step 1: Run the complete automated suite**

  Run: `E:\ComfyUI-py310\Python310\python.exe -m pytest -q`

  Expected: zero failures.

- [ ] **Step 2: Review the complete diff**

  Confirm no Director, ComfyUI core, formal Python environment, model, input, output, or unrelated workflow file changed.

- [ ] **Step 3: Deploy from the unique worktree**

  Use the repository deployment script to replace only `ComfyUI\custom_nodes\raylight`, then verify deployed hashes match the worktree.

- [ ] **Step 4: Run import/startup verification**

  Start with the formal launcher and verify Raylight imports, the installed Director imports, and no DynamicVRAM candidate path appears in the command line or module paths.

- [ ] **Step 5: Run a minimal real Director generation**

  Use the existing `MMH3Director.json` configuration at a short duration and low resolution. Require both ranks to enter sampling, matching finite outputs, successful VAE decode/save, and no local-GPU fallback.

- [ ] **Step 6: Verify cleanup and record evidence**

  Stop ComfyUI after the smoke test. Confirm no Ray worker remains and both GPUs return to idle context usage. Record timings and outcome in the bilingual release notes.

- [ ] **Step 7: Commit locally**

  Commit the tested compatibility implementation and documentation. Do not push to GitHub unless the user explicitly requests it.

## Self-review

- Coverage: environment rollback, old/new Director routing, external SIGMAS, shift control, optional dependency, lifecycle cleanup, documentation, deployment, and hardware smoke testing each have an explicit task.
- Scope: only Raylight worktree files are modified; Director and ComfyUI core remain untouched.
- Failure safety: Ray proxies fail explicitly; normal models always pass through; no fallback path samples silently on GPU0.
- Compatibility: legacy `d655912` remains supported while installed `b8f721c` gets a dedicated stage seam.
- Test quality: every production behavior begins with a failing regression test; real Director/ComfyUI functions are used where practical and Ray/GPU boundaries alone are substituted in unit tests.
- Placeholders: none. All paths, commands, interfaces, and acceptance conditions are specified.

