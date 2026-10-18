# Raylight 1.17.1 / O15.1 MiniMaxH3Director 4c6245e Compatibility Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fast-forward the clean upstream MiniMaxH3Director checkout from `8151b29` to `4c6245e` and certify that Raylight dual-V100 FSDP/P2P sampling, Refine, SelfLift, FaceRefine, per-segment LoRA, live preview, and cache behavior do not regress.

**Architecture:** The single upstream commit changes only Director-pack import/export progress UI. The Python sampling contract is unchanged. Raylight therefore does not copy frontend code or invent a new worker protocol; it updates the validated compatibility coordinate and proves it with upstream signature checks, node startup, automated regression, and one real dual-GPU workflow.

**Tech Stack:** Windows, Python 3.12.10, PyTorch 2.12.0+cu126, ComfyUI, Ray 2.57.0, CUDA P2P, FSDP2, pytest, MiniMaxH3Director `4c6245e`.

## Scope and conclusion

- PR #182 batches the post-decode continuity opening-grade match on GPU. It does not execute neural VAE decode and cannot remove the measured roughly 66-second VideoVAE decode for 107 frames at 1920x1088.
- `8151b29..4c6245e` is one commit touching two frontend JavaScript files. It adds pack upload, processing, packing, download, save, and timeline-apply progress; no Python, node schema, sampling, VAE, Refine, or cache-format contract changes.
- Do not modify ComfyUI or Director source. Keep the canonical Raylight worktree as the only editable source and deploy from a committed revision.

## Tasks

1. Change the contract test to expect `4c6245e518dd8a494f3afc4f3cccd53f22d82d10`, prove RED, then minimally update `SUPPORTED_DIRECTOR_COMMIT` and prove GREEN.
2. Fast-forward the clean official Director checkout, verify the exact two-file diff, run upstream tests, compile checks, live signature checks, and ComfyUI node startup.
3. Run Raylight Director bridge/worker/LoRA/lifecycle/acceleration/deployment tests plus the full suite, deploy from source, and execute one short real dual-V100 Director workflow.
4. Release as `1.17.1`, update bilingual README/release/test records, review the diff, commit, and push `fsdp/main`.

## Acceptance gates

1. Director HEAD and `origin/main` are clean at `4c6245e`.
2. The real upstream sampling signature and all legacy Raylight paths pass.
3. ComfyUI loads the latest Director and Raylight nodes without import errors; the new Director-pack progress assets are present.
4. One real workflow completes through exactly one remote operation with both ranks, decodable frames, valid audio, and no NaN/Inf.
5. Code, version, bilingual documentation, test records, deployment, and GitHub all describe the same state.

