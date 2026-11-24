# RelayVSR production migration / 正式环境迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax. User approved route A and continuous execution on 2026-11-24; do not request approval again between tasks.

**Goal:** Preserve existing Director workflows while adding native RelayVSR with exact dual-V100 Flash decoding to production.
**Architecture:** Pin official core `2fcb6ce`, add old/new H3 compatibility and scoped VSR GPU ownership in Raylight, validate in a separate runtime, then deploy the accepted versions with rollback backups.
**Tech Stack:** Windows TCC, 2×V100 16GB, Python 3.12.10, Torch 2.12.0+cu126, Ray, ComfyUI, Director aff2579 + preserved patches.
**Spec:** `docs/superpowers/specs/2026-11-24-relayvsr-production-migration.zh-CN.md` (English companion beside it).

## Global Constraints

- Unique source checkout remains this repository; existing branch windows-v100-fsdp-p2p. Isolated runtimes are deployments, not source worktrees.
- No manual ComfyUI core edits, no Director fork, driver/Torch changes, user workflow/cache overwrites or external bundle changes.
- Production changes and GitHub push are authorized only after acceptance. Keep DynamicVRAM disabled.
- Never hide the 14 known failures with skips, remove masks or destroy Ray sessions to admit VSR.
- Full resource/timing and raw-frame correctness evidence, honest unfinished cases, bilingual documentation and recoverable rollback are mandatory.

## Review Focus

1. Partial audio/video masks across rank boundaries must retain per-row timestep and output semantics (Task 1).
2. Foreign sessions, failed cleanup and physically remapped GPUs must not enter local decoding (Task 2).
3. Director without VSR and unrelated native VAE calls must remain untouched (Task 3).
4. VSR parameters invalidate final caches without decoding FFV1 during inspection; first-pass reuse and subsequent segments remain valid (Tasks 3–4).
5. Failed migration, interruption or partial dependency upgrade must have a tested recovery path (Task 5).

### Task 1: New-core H3 compatibility and isolated candidate

**Files:** `src/raylight/diffusion_models/minimax/core_compat.py` (new focused helpers), `xdit_context_parallel.py`, `tests/test_minimax_core_compat.py`, existing AdaLN/parallel tests.
**Interfaces:** layout builder accepts payload including optional frame_count; modulation builder returns timestep values, packed modulation segments and final video/audio row indices. Rows can be integers or per-token LongTensors; split/projection consumers must support both.

- [ ] Reproduce 14 failures on existing isolated new core; preserve log and full traceback.
- [ ] Add real numerical/mask, rank-split and per-row projection regressions; observe RED.
- [ ] Implement signature-aware layout construction and full native mask/modulation semantics in our code while retaining FP32 curve interpolation and existing memory-saving split/projection.
- [ ] Run old and new full suites, not just focused files. Expected: no unexplained failures; legacy-native feature absences explicitly reported.
- [ ] Create `.local/environments/production-relayvsr-candidate` from pinned official core and overlay dependencies; copy installed custom nodes excluding user data and deploy only committed Raylight. Preserve Director overlays. Record exact manifests and all import failures.
- [ ] Commit compatibility with its tests, record task completion only after verification.

### Task 2: Scoped GPU ownership for local VSR

**Files:** `src/raylight/relayvsr/local.py`, `admission.py` (new), worker lease boundaries as required, `tests/test_relayvsr_admission.py`.
**Interfaces:** a context-managed authorized lease applies only to current driver/session/generation and both validated physical GPUs; standalone no-Ray admission remains supported. Unknown sessions remain refused. Cleanup occurs before release.

- [ ] RED tests for active foreign Ray, partial acquisition, wrong device identity, cleanup failure, cancellation and next-operation reuse.
- [ ] Reuse worker lease coordination with actual post-cleanup verification, not a boolean bypass of is_initialized. Keep model/cache objects and actor lifetime.
- [ ] GREEN focused tests and both full suites; commit and ledger.

### Task 3: Director VSR bridge and cache safety

**Files:** `src/raylight/integrations/mmh3director/relay_vsr.py` (new), integration install/context boundaries, tests `test_mmh3director_relay_vsr.py` and cache characterization fixtures.
**Interfaces:** wrap only `apply_segment_relay_vsr(frames, pack)` inside an active Ray MODEL Director context; forward all pack parameters; use native conditioning/sampling and local dual Flash; restore GPU ownership in finally.

- [ ] RED tests for no context/unconnected passthrough, exact pack forwarding, x2/x4 mismatch, frame count, audio passthrough, cancellation and cleanup order.
- [ ] Add scoped hook including loaded aliases; unknown contracts fail explicitly. Do not globally replace all VAE decodes or discard Ray/Director caches.
- [ ] Characterize first/final cache fingerprints and metadata inspection against current Director. Fix relevant compatibility in our integration only; document inability rather than claiming a false cache match.
- [ ] GREEN focused + complete old/new suites; commit and ledger.

### Task 4: Workflow/resource acceptance

**Files:** maintained workflow fixtures under `example_workflows`, campaign driver under `scripts/minimax-h3/`, public compact data/report under `docs/reports`; private inputs/logs/videos under `.local/artifacts/relayvsr-production-migration-20261124`.
**Interfaces:** actual ComfyUI workflow API graphs; 500ms whole-GPU and host resource traces; complete frame/audio validation and stage timings.

- [ ] Test campaign validation RED→GREEN before running graphs; refuse altered seeds/cached sampler outputs being reported as warm GPU work.
- [ ] Baseline production-version old graphs in isolation; candidate old H3 non-Director and FL2VA/Ref2VA 4/8-step, TP2, LoRA/segment LoRA, SelfLift, Refine, FaceRefine, dual H3 VAE, silent/normal output and preview.
- [ ] At least three Director segments with VSR and selective cache rerun; verify H3→VSR→H3, cache correctness and idle cleanup.
- [ ] Standalone 480p/10s/x2 three matched runs, 1MP/5s/x2/x4; all-frame exact Flash reference, media integrity and resource records. Existing reference Flash/total=71.363/110.428s; median regression ≤10%. Investigate old graph regressions >5%.
- [ ] Full suite and independent whole-change Python/code review; fix important findings RED→GREEN. No production migration if any required gate remains unmet.
- [ ] Commit acceptance fixtures/reports, not media, credentials, absolute machine paths or full logs.

Task4 acceptance addendum: fresh31-frame1MP/x4 A/B/A controls identified the launcher's `max_split_size_mb:128` allocator policy, not the independent128MiB P2P staging capacity. Allocated Flash peak stays about6.08GB, while reserved memory grows from7.67GB to16.60GB and windows slow. Add a Raylight-only, reentrant VSR allocator scope: read live native settings, temporarily remove only max_split_size_mb during native VSR sampling/Flash, restore and verify the original settings on success/failure before H3 reuse; quarantine on unprovable restoration. Preserve all other allocator options, H3 worker configuration, precision and P2P capacity. RED→GREEN unit/routing tests, native API restoration probe, full suites, actual integrated/cache regressions and full raw/performance verification remain required. This is a root-cause fix to Task4, not an unconditional global startup change.

补充验收：仅在VSR执行期间临时解除PyTorch拆分限制，正常或异常结束均恢复原配置；恢复不可证实则保持worker租约、不允许等待任务继续。Director覆盖VSR采样和Flash；独立解码节点仅覆盖Flash，不宣称已加速此前运行的原生KSampler。与P2P的128MiB通信预算无关。

### Task 5: Backup, production deployment and release

**Files:** `scripts/` migration/rollback helpers with tests if newly needed, README.md/README.zh-CN.md as present, CHANGELOG and bilingual release notes; private verified backups below `.local/backups/production-relayvsr-20261124`.
**Interfaces:** accepted source commit/core/dependency manifest; immutable backup manifest and rehearsed rollback commands.

- [ ] Test backup/restore logic with temporary fixtures; cover partial failure without touching user caches/media.
- [ ] Capture recoverable Python/packages/core/Director/Raylight/startup state and restore into a disposable isolation; verify hashes and original workflow.
- [ ] Check production queue idle, stop only scoped server processes, update official core and exact dependencies, replay/preserve Director overlays and deploy accepted Raylight commit.
- [ ] Smoke old Director and VSR in production, verify deployment hashes; on failure restore prior runtime immediately and report the failed gate.
- [ ] Update bilingual installation/usage/version/test records; full regression status and safety review, then non-force GitHub push/version record as authorized. Keep recovery backups and original isolated VSR environment.

## 执行记录 / Execution record

Native inline execution; no extra source worktree. Scratch ledger and task briefs belong to this plan only. Archive evidence before cleaning scratch. The approved spec remains the authority; record deviations with reasons and risk. 本次用户已明确授权方案 A 和测试通过后的迁移/推送，不再逐任务请求确认。
