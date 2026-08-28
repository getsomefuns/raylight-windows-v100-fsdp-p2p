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
- Plain-copy file identity now compares Git blobs with each repository-relative path, applying `.gitattributes` and Git clean filters before comparison. Real changes still fail, while CRLF/LF conversion no longer produces false deployment drift.
- The legacy `scripts/minimax-h3/deploy-to-comfyui.ps1` remains as a compatibility entry point but delegates to the canonical deployer and never requires or mutates a Git repository at the runtime path.
- Ray's local runtime working directory moved to the system temporary tree and no longer creates `_ray_runtime_env` inside the node source.
- Historical plans moved to `docs/superpowers/plans/legacy`; raw local evidence moved to Git-ignored `.local/artifacts`.

## Validation contract

Regression coverage now includes canonical and deployed layouts, plain-copy identity, clean Git checkout identity, launcher resolution, and the external Ray working directory. Release verification also requires the complete test suite, direct pytest from the deployment, P2P/FSDP probes, node import, and a real reduced workflow.

## Final acceptance results

- Canonical worktree: `444 passed, 1 warning, 30 subtests passed`; deployed copy: `444 passed, 3 warnings, 30 subtests passed`.
- All 394 deployed files matched commit `c0b3f1b51942690638a62c1627761ae260561028`. The runtime copy contained no `.git`, and no second Raylight backup remained under `custom_nodes`.
- Both Tesla V100-SXM2-16GB GPUs were in TCC mode. CUDA P2P all-gather passed at 64, 128, 256, and 384 MiB with approximately 59.3-61.6 GiB/s per direction.
- The real MiniMax H3 REF2VA acceptance run used 576x352, 24 fps, a 2-second request, Turbo 4-step, FSDP, and CPU offload. Both worker ranks participated. Sampling took 39.38 seconds and the complete prompt took 200.25 seconds. Recorded phases were 120.19 seconds before the sampler, 0.93 seconds for model-to-GPU, 71.21 seconds in the sampler node, 6.47 seconds for VAE decode, and 2.38 seconds for video save.
- GPU0/GPU1 peak VRAM was 15949/4603 MiB and peak utilization was 100%/96%. Peak physical memory was 95847 MiB, peak committed memory was 134431 MiB, and pagefile usage remained zero.
- The H.264/AAC output was 576x352 at 24 fps, 56 frames, and 2.33 seconds. Full video and audio decode passed; all 56 frame hashes were unique, and the first/middle/last frames contained valid non-black, non-noise imagery.
- The benchmark stopped ComfyUI and both Ray workers. No Python compute process or GPU compute allocation remained afterward.

PyTorch 2.7's legacy Windows TCPStore still emits one `WorkStudio:29500` IPv6-candidate connection warning. Existing isolation tests show that it continues through IPv4 loopback and completes group creation; this two-rank workflow and valid output again confirm that the warning is neither a P2P data-path fallback nor an execution failure.
