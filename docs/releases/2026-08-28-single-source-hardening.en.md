# 1.10.2: Single-source deployment and test-layout hardening

## Goal

This patch separates the editable repository, ComfyUI runtime copy, local test evidence, and maintained project documentation. It prevents another split-source state and removes assumptions that tests only run from one directory layout.

## Shipped changes

- `raylight-windows-v100-fsdp-p2p` is the only editable Git worktree. `custom_nodes/raylight` is a generated deployment without `.git`.
- Deployment materializes the immutable `HEAD` object tree through `git archive`, eliminating worktree changes between validation and copy. It writes `.raylight-deployed-commit` and validates tracked-file hashes and source identity during deployment and `-VerifyOnly`.
- Replacement tracks old-tree backup and new-target installation independently. A failed first deployment removes the invalid target, while a failed replacement restores the prior deployment. Final verification is the commit point: a Windows file lock that prevents old-backup cleanup leaves the verified new deployment active and emits a warning instead of restoring a partly deleted backup.
- Test discovery supports a canonical worktree beside ComfyUI and a deployment below `ComfyUI/custom_nodes/raylight`.
- Test modules use package-relative imports, avoiding collision with ComfyUI's own `tests` package.
- The FSDP probe no longer hard-codes `REPO_ROOT.parent/ComfyUI`.
- The MiniMax H3 benchmark accepts a marked plain deployment or a clean Git clone. A plain-copy marker no longer implies clean files: deployed runtime content is checked independently. A Git checkout always reports its real `HEAD` and ignores stale private deployment markers.
- Plain-copy file identity now compares Git blobs rather than raw deployment bytes against the filtered Windows worktree. Real changes still fail, while Git line-ending conversion or mixed CRLF/LF no longer produces false deployment drift.
- The legacy `scripts/minimax-h3/deploy-to-comfyui.ps1` remains as a compatibility entry point but delegates to the canonical deployer and never requires or mutates a Git repository at the runtime path.
- Ray's local runtime working directory moved to the system temporary tree and no longer creates `_ray_runtime_env` inside the node source.
- Historical plans moved to `docs/superpowers/plans/legacy`; raw local evidence moved to Git-ignored `.local/artifacts`.

## Validation contract

Regression coverage now includes canonical and deployed layouts, plain-copy identity, clean Git checkout identity, launcher resolution, and the external Ray working directory. Release verification also requires the complete test suite, direct pytest from the deployment, P2P/FSDP probes, node import, and a real reduced workflow.
