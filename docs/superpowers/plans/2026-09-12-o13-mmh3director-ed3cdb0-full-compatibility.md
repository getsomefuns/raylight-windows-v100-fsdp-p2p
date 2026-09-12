# O13 MiniMaxH3 Director ed3cdb0 Full Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Director to `ed3cdb0` and support its continuity, native Refine, spatial tiling, and step preview features through Raylight.

**Architecture:** Extend the bridge-to-worker protocol, reconstruct recognized continuity hooks on real worker models, install spatial tiling on worker samplers, and carry rank-0 preview events through a bounded non-blocking Ray queue.

**Tech Stack:** Python 3.12, PyTorch 2.12+cu126, Ray 2.57, ComfyUI, pytest, PowerShell, dual Tesla V100 TCC.

**Spec:** `docs/superpowers/specs/2026-09-12-o13-mmh3director-ed3cdb0-full-compatibility-design.md`

## Global Constraints

- The only source worktree is `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`.
- Deploy only through `scripts/deploy-to-comfyui.ps1`.
- Do not patch ComfyUI, Director, SOL Attention, TE-Speed, or other nodes.
- Follow red-green TDD for production changes.
- Run one four-step GPU acceptance workflow unless a failure requires diagnosis.

---

### Task 1: Director contract and continuity protocol

**Files:** `tests/test_mmh3director_bridge.py`, `src/raylight/integrations/mmh3director/bridge.py`

- [ ] Add failing tests for the `ed3cdb0` signature, remask capability, unknown-callback rejection, and worker request fields.
- [ ] Run bridge tests and observe the intended failures.
- [ ] Extend the contract and request protocol without serializing closures.
- [ ] Rerun bridge tests to green.

### Task 2: Worker remask and spatial tiling

**Files:** `tests/test_mmh3director_worker.py`, `src/raylight/distributed_worker/ray_worker.py`

- [ ] Add failing tests for post-shift remask, tile arguments, guaranteed restore, and unchanged untiled routing.
- [ ] Run worker tests and observe the intended failures.
- [ ] Reconstruct the hook and use `sample_custom` only for tiled execution.
- [ ] Rerun worker tests to green.

### Task 3: Non-blocking step preview transport

**Files:** new `preview_transport.py` and its test; modify bridge and worker.

- [ ] Add failing tests for video-only extraction, 16-frame cap, CPU transfer, non-blocking full-queue behavior, and disabled zero overhead.
- [ ] Run affected tests and observe the intended failures.
- [ ] Implement the bounded queue, rank-0 callback, driver drain loop, and guaranteed final callback.
- [ ] Rerun affected tests to green.

### Task 4: Upstream update and real-contract regression

- [ ] Stop ComfyUI/Ray and verify workers exited.
- [ ] Fast-forward Director and verify clean `ed3cdb0` HEAD.
- [ ] Add the installed-signature regression and observe failure with the old compatibility constant.
- [ ] Update the constant and run the Director test group.

### Task 5: Version, bilingual docs, commit, and deployment

- [ ] Publish Raylight `1.15.0` docs covering tiling, preview overhead, and fallback behavior.
- [ ] Run Director/O11/O12/deployment/contract tests and Ruff.
- [ ] Review `git diff --check`, paths, secrets, and unrelated files; commit canonical source.
- [ ] Deploy with the canonical script and verify the marker matches the source commit.

### Task 6: Dual-V100 four-step acceptance

- [ ] Start ComfyUI with the production launcher and verify clean imports.
- [ ] Run one four-step Director workflow with live preview and two-tile Refine.
- [ ] Verify both GPUs, finite media, animated previews, tile quality, and one worker pair.
- [ ] Record evidence and stop the test process; return to the failing TDD task if needed.

