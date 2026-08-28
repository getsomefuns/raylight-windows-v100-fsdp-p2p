# Raylight Single-Source Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one independent Raylight Git repository and a reproducible, hash-verified ComfyUI deployment copy.

**Architecture:** The canonical repository is converted from a linked worktree to a standalone clone. A PowerShell deployment tool copies only Git-tracked files through a verified staging directory and swaps the live node directory with rollback support.

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

- [ ] Record commit, branch, remotes, status, Git directory, and common directory.
- [ ] Clone the canonical branch locally into a temporary sibling path without local hardlinks.
- [ ] Restore the original remote names and URLs in the temporary clone.
- [ ] Verify commit identity and clean status.
- [ ] Remove the linked worktree registration and move the independent clone into the canonical path.
- [ ] Verify `.git` is a directory and `git worktree list` contains only the canonical path.

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

- [ ] Write functional tests for clean deployment, stale-file removal, `.git` exclusion, drift detection, and source/target path guards.
- [ ] Run the tests and confirm they fail because the script does not exist.
- [ ] Implement manifest collection, staged copy, SHA-256 validation, atomic swap, rollback, and verify-only mode.
- [ ] Add the bilingual operating rule and deployment commands to the READMEs.
- [ ] Run the deployment tests and confirm they pass.

### Task 3: Migrate and deploy the live node

**Files:**
- Replace contents: `E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight`

**Interfaces:**
- Consumes: the canonical tracked-file manifest.
- Produces: a non-Git runtime copy with matching hashes.

- [ ] Identify and stop only ComfyUI and Ray worker Python processes using this environment.
- [ ] Run the deployment script from the canonical repository.
- [ ] Run `-VerifyOnly` and require zero missing, changed, or extra files.
- [ ] Confirm the live directory has no `.git` entry.

### Task 4: Validate and commit

**Files:**
- Commit all new deployment governance files in the canonical repository.

**Interfaces:**
- Consumes: migrated repository and deployed live node.
- Produces: verified local commit and documented operating procedure.

- [ ] Import the deployed custom node with the project Python and ComfyUI root.
- [ ] Verify version, node count, and MMH3Director adapter registration.
- [ ] Run deployment tests and MMH3Director regression tests.
- [ ] Check clean deployment verification and repository status.
- [ ] Commit the deployment tooling and documentation locally.
