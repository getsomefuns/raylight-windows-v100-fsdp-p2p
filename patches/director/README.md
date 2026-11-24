# Preserved local Director overlays / Director 本地补丁

Official base / 官方基线：`aff2579` (RelayVSR).

These are recovery snapshots, not complete Pika/VSR acceptance. Known follow-ups:
the cache-status UI does not yet reconstruct FaceRefine/VSR fingerprints, and
metadata reference paths require input-directory containment hardening. Review
the dated compatibility report before applying; do not treat passing legacy
tests as verification of these new cases.

Apply the numbered patches in order to a **clean checkout of that exact base**:

```powershell
git status --short
git am --3way <path-to-0001.patch> <path-to-0002.patch>
```

Do not apply twice to the already-patched local install. If a future update conflicts,
stop and use `git am --abort` or review the merge; never choose upstream wholesale.

1. PR231 metadata-only cache inspection, Refine cancellation/deduplication and the
   FL2V fingerprint correction. Original source: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/231
2. Existing local export-tail continuity and face tracking extensions, preserved
   from the five modified files found before this update. They are not official
   upstream features. Left/right face choice is initial position selection, not
   identity recognition.

Future updates: finish the queue, close ComfyUI, back up with `git bundle create`,
fetch upstream, then `git rebase origin/main` on the local patched branch. Run
regressions before restarting. Manager Update All/reinstall may switch branches
or discard the overlay: do not use it for this node. There is no automatic
guarantee against every upstream semantic change; the backup, identifiable
commits, patches and tests provide recovery and a reviewed upgrade path.

中文：当前安装已带补丁，**不要重复应用**。以本地补丁分支持续 rebase 官方更新，
发生冲突必须审查，不能直接覆盖。完整备份留在唯一 Raylight 工作树的忽略目录；
模型、视频、日志与 Git bundle 不提交仓库。补丁保留 Apache-2.0 上游归属，
随 Raylight 仓库的许可证及原 Director 许可证使用。
