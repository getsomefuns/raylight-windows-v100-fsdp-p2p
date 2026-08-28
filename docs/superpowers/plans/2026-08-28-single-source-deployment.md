# Raylight Single-Source Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one independent Raylight Git repository and a reproducible, hash-verified ComfyUI deployment copy.

**Architecture:** The canonical repository is converted from a linked worktree to a standalone clone. A PowerShell deployment tool materializes the immutable Git `HEAD` archive through a verified staging directory and swaps the live node directory with first-install cleanup and replacement rollback support.

**Tech Stack:** Git, PowerShell 7/Windows PowerShell, Python 3.10, pytest, ComfyUI custom-node loader.

**Spec:** `docs/superpowers/specs/2026-08-28-single-source-deployment-design.md`

## Global Constraints

- The only Git repository and editable source is `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`.
- The live node path is `E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight` and contains no Git metadata.
- Deployment copies only Git-tracked files and validates SHA-256 equality.
- Existing workflows, models, inputs, Python installation, and root launcher remain untouched.

---

### Task 1: Preserve repository history in one standalone checkout

**Files:**
- Replace Git layout at: `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p\.git`

**Interfaces:**
- Consumes: current linked worktree and configured remotes.
- Produces: standalone repository on `windows-v100-fsdp-p2p` at the current canonical commit.

- [x] Record commit, branch, remotes, status, Git directory, and common directory.
- [x] Clone the canonical branch locally into a temporary sibling path without local hardlinks.
- [x] Restore the original remote names and URLs in the temporary clone.
- [x] Verify commit identity and clean status.
- [x] Remove the linked worktree registration and move the independent clone into the canonical path.
- [x] Verify `.git` is a directory and `git worktree list` contains only the canonical path.

### Task 2: Implement reproducible deployment

**Files:**
- Create: `scripts/deploy-to-comfyui.ps1`
- Create: `DEPLOYMENT_NOTICE.md`
- Create: `tests/test_windows_deployment.py`
- Modify: `README.md`
- Modify: `README_EN.md`

**Interfaces:**
- Consumes: `-SourceRoot`, `-ComfyUIRoot`, optional `-VerifyOnly`.
- Produces: live node copy containing exactly the source repository's Git-tracked files.

- [x] Write functional tests for clean deployment, stale-file removal, `.git` exclusion, drift detection, and source/target path guards.
- [x] Run the tests and confirm they fail because the script does not exist.
- [x] Implement manifest collection, staged copy, SHA-256 validation, atomic swap, rollback, and verify-only mode.
- [x] Add the bilingual operating rule and deployment commands to the READMEs.
- [x] Run the deployment tests and confirm they pass.

### Task 3: Migrate and deploy the live node

**Files:**
- Replace contents: `E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight`

**Interfaces:**
- Consumes: the canonical tracked-file manifest.
- Produces: a non-Git runtime copy with matching hashes.

- [x] Identify and stop only ComfyUI and Ray worker Python processes using this environment.
- [x] Run the deployment script from the canonical repository.
- [x] Run `-VerifyOnly` and require zero missing, changed, or extra files.
- [x] Confirm the live directory has no `.git` entry.

### Task 4: Validate and commit

**Files:**
- Commit all new deployment governance files in the canonical repository.

**Interfaces:**
- Consumes: migrated repository and deployed live node.
- Produces: verified local commit and documented operating procedure.

- [x] Import the deployed custom node with the project Python and ComfyUI root.
- [x] Verify version, node count, and MMH3Director adapter registration.
- [x] Run deployment tests and MMH3Director regression tests.
- [x] Check clean deployment verification and repository status.
- [x] Commit the deployment tooling and documentation locally.

### Task 5: Harden portable validation and consolidate local evidence

- [x] Add layout-aware ComfyUI and Python discovery for canonical and deployed trees.
- [x] Remove ambiguous absolute imports of the `tests` package.
- [x] Record and verify source identity in plain deployments.
- [x] Accept clean Git deployments in benchmark identity checks.
- [x] Move Ray runtime working files out of the custom-node source tree.
- [x] Move maintained historical plans under `docs` and raw local evidence under ignored `.local/artifacts`.
- [ ] Complete final full-suite, deployed-copy, GPU probe, node-import, and reduced-workflow release verification for 1.10.2.
