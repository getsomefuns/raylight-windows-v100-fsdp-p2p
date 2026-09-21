# O14 MiniMaxH3 Director 576424f Compatibility Implementation Plan

**Goal:** Adapt upstream Director `576424f` through the smallest Raylight bridge extension, without modifying Director or ComfyUI, and preserve reproducible pre/post hardware evidence.

**Stack:** Windows, Python 3.12, PyTorch 2.12+cu126, Ray 2.57, 2× Tesla V100 TCC, pytest, PowerShell.

**Design:** `docs/superpowers/specs/2026-09-21-o14-mmh3director-576424f-compatibility-design.md`

## Task 1: Audit, rollback snapshot, and pre-upgrade baseline

- [x] Fetch upstream and audit the 13 commits from local `52f8fb7` to `576424f`.
- [x] Archive Director source and identity metadata plus the current Raylight deployment marker.
- [x] Run one cold fixed O13 workflow baseline and retain JSON, CSV, logs, and media metadata.

## Task 2: Add failing contract tests first

- [x] Test `on_step_state`, `zero_noise`, `noise_override`, final-state return, and rank validation.
- [x] Cover SelfLift stages, Semantic Bridge conditioning, FaceRefine proxy sampling, and legacy contracts.
- [x] Confirm failures represent missing production behavior.

## Task 3: Upgrade Director and implement the minimal bridge

- [x] Fast-forward Director to `576424f` and install required dependencies without editing upstream source.
- [x] Extend the bridge request and worker callback to return only SelfLift's required final full state.
- [x] Preserve previews, continuity remask, tiling, SOL/TE options, and worker lifecycle.
- [x] Run focused tests and fix real API differences.

## Task 4: Dual-V100 workflow acceptance

- [x] Re-run the identical O13 workflow for strict pre/post comparison.
- [x] Run one minimal SelfLift workflow and prove both stages use the same two workers.
- [x] Hardware-test Semantic Bridge/FaceRefine when local optional assets permit; otherwise record contract-only coverage.
- [x] Validate media metadata, finite output, representative frames, process cleanup, and VRAM release.

## Task 5: Release, deploy, and synchronize

- [x] Update version, bilingual READMEs, changelog/release notes, and O14 test report.
- [x] Run focused Director tests, O11/O12/O13 regressions, full pytest, Ruff, and `git diff --check`.
- [ ] Commit the unique source worktree, deploy through the script, and verify the deployed marker.
- [ ] Push GitHub and verify local/remote equality while Director remains a clean upstream checkout.
