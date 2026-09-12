# O13 MiniMaxH3 Director 19b6a73 Full Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Director to `19b6a73` and support its continuity, native Refine, spatial tiling, step previews, and new `shift_cache` call contract through Raylight.

**Architecture:** Extend the bridge-to-worker protocol, reconstruct recognized continuity hooks on real worker models, install spatial tiling on worker samplers, and carry rank-0 preview events through a bounded non-blocking Ray queue.

**Tech Stack:** Python 3.12, PyTorch 2.12+cu126, Ray 2.57, ComfyUI, pytest, PowerShell, dual Tesla V100 TCC.

**Spec:** `docs/superpowers/specs/2026-09-12-o13-mmh3director-19b6a73-full-compatibility-design.md`

## Global Constraints

- The only source worktree is `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`.
- Deploy only through `scripts/deploy-to-comfyui.ps1`.
- Do not patch ComfyUI, Director, SOL Attention, TE-Speed, or other nodes.
- Follow red-green TDD for production changes.
- Run one four-step GPU acceptance workflow unless a failure requires diagnosis.

---

### Task 1: Director contract and continuity protocol

**Files:** `tests/test_mmh3director_bridge.py`, `src/raylight/integrations/mmh3director/bridge.py`

- [x] Add failing tests for the current signature, remask capability, unknown-callback rejection, and worker request fields.
- [x] Run bridge tests and observe the intended failures.
- [x] Extend the contract and request protocol without serializing closures.
- [x] Rerun bridge tests to green.

### Task 2: Worker remask and spatial tiling

**Files:** `tests/test_mmh3director_worker.py`, `src/raylight/distributed_worker/ray_worker.py`

- [x] Add failing tests for post-shift remask, tile arguments, guaranteed restore, and unchanged untiled routing.
- [x] Run worker tests and observe the intended failures.
- [x] Reconstruct the hook and use `sample_custom` only for tiled execution.
- [x] Rerun worker tests to green.

### Task 3: Non-blocking step preview transport

**Files:** new `preview_transport.py` and its test; modify bridge and worker.

- [x] Add failing tests for video-only extraction, 16-frame cap, CPU transfer, non-blocking full-queue behavior, and disabled zero overhead.
- [x] Run affected tests and observe the intended failures.
- [x] Implement the bounded queue, rank-0 callback, driver drain loop, and guaranteed final callback.
- [x] Rerun affected tests to green.

### Task 4: Upstream update and real-contract regression

- [x] Stop ComfyUI/Ray and verify workers exited.
- [x] Fast-forward Director and verify clean `19b6a73` HEAD.
- [x] Add the installed-signature regression and verify the real `shift_cache` parameter.
- [x] Update the compatibility constant and `shift_cache` boundary, then run the Director test group.

### Task 5: Version, bilingual docs, commit, and deployment

- [x] Publish Raylight `1.15.0` docs covering tiling, preview overhead, `shift_cache`, and fallback behavior.
- [x] Run Director/O11/O12/deployment/contract tests and Ruff (`741 passed + 35 subtests`; Ruff passes).
- [x] Review `git diff --check`, paths, secrets, and unrelated files; commit canonical source.
- [x] Deploy with the canonical script and verify the marker matches the source commit.

### Task 6: Dual-V100 four-step acceptance

- [x] Start ComfyUI with the production launcher and verify clean imports.
- [x] Run one four-step Director workflow with live preview and two-tile Refine.
- [x] Verify both GPUs, finite media, animated previews, tile quality, and one worker pair.
- [x] Record evidence and stop the test process; the first failure returned to TDD and passed an identical rerun.
