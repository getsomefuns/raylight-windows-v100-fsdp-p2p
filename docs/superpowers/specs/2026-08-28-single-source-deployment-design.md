# Raylight Single-Source Deployment Design

## Goal

Make `E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p` the only Git repository and the only directory where Raylight is edited. Keep `E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight` as a generated deployment copy whose Git-tracked content is identical to the source repository.

## Current problem

The canonical development directory is a linked Git worktree whose Git metadata is stored under the live ComfyUI node directory. The two worktrees are on different commits, and selected runtime files have been copied manually. This permits version, documentation, test, launcher, and source drift.

## Architecture

1. Convert the canonical development path into a standalone clone on branch `windows-v100-fsdp-p2p` at commit `60cd2efe1bc2bff9fdad6529f3737e458e7635ed`, retaining the existing GitHub remotes.
2. Remove Git metadata from the live ComfyUI node path. It is no longer a repository or worktree.
3. Add `scripts/deploy-to-comfyui.ps1` to the canonical repository. It obtains the deployment manifest from `git ls-files`, builds a staging copy, verifies SHA-256 hashes, atomically swaps the live directory, and supports a read-only verification mode.
4. Add `DEPLOYMENT_NOTICE.md` to make the single-source rule visible in both source and deployed copies.
5. Deploy all tracked files. Do not deploy `.git`, ignored caches, bytecode, pytest state, Ruff state, or other local artifacts.

## Safety and recovery

- ComfyUI and Ray workers must be stopped before deployment.
- The deployment script stages and verifies the complete copy before replacing the live directory.
- The old live directory is renamed to a backup during the swap and restored automatically if the swap fails.
- Migration first creates and validates an independent clone before the old Git metadata is removed.
- Destructive operations are limited to the exact live Raylight directory and validated absolute paths.

## Acceptance criteria

- The canonical directory has a normal `.git` directory, not a linked-worktree pointer.
- `git worktree list` reports only the canonical directory.
- The live node directory contains no `.git` entry.
- Every `git ls-files` entry in the canonical repository exists in the live directory with the same SHA-256 hash.
- The live directory has no unmanaged files immediately after deployment.
- Raylight imports as version `1.10.1`, registers its node mappings, and exposes `Ray MODEL Adapter (MMH3Director)`.
- Deployment script functional tests and relevant MMH3Director regression tests pass.

## Operating rule

All future changes are made, tested, committed, and pushed from the canonical repository. Before ComfyUI testing, run the deployment script. Direct edits under `ComfyUI\custom_nodes\raylight` are unsupported and must be detected as deployment drift.
