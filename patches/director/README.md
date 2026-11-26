# Preserved local Director overlays / Director 本地补丁

Official bases / 官方基线：`3cbc048` (2026-11-25) and historical `aff2579`.

The patch series preserves local fixes and adds two narrow compatibility fixes.
It is not a substitute for running the relevant workflows. The cache-status UI
does not reconstruct FaceRefine/VSR fingerprints; see the release notes for the
workflows actually verified.

Apply 0001 through 0004 in order to a **clean checkout of `3cbc048`**.
For the historical `aff2579` base use **0001 and 0002 only**. Patch 0003
requires the new Director toggle and preserves explicitly saved booleans.
The preserved maintenance patches contain both LF and CRLF hunks; keep the
checkout and mail patch line endings instead of normalizing them.

```powershell
# For a new checkout, set this before checkout; do not change an existing
# modified working tree to this setting and then try to apply patches.
$PatchDir = 'C:\path\to\raylight-windows-v100-fsdp-p2p\patches\director'
$PatchFiles = @(Get-ChildItem -LiteralPath $PatchDir -Filter '000*.patch' |
    Sort-Object Name | ForEach-Object FullName)
if ($PatchFiles.Count -ne 4) { throw 'Expected four Director patches' }
git clone -c core.autocrlf=false --no-checkout https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git
cd ComfyUI_MiniMaxH3_Director
git switch --detach 3cbc048b3ed5fbf993f4d7819557782f4a5ebb0c
git switch -c local/director-3cbc048
git status --short
git am --keep-cr --3way @PatchFiles
```

Do not apply twice to the already-patched local install. If a future update conflicts,
stop and use `git am --abort` or review the merge; never choose upstream wholesale.

1. PR231 metadata-only cache inspection, Refine cancellation/deduplication and the
   FL2V fingerprint correction. Original source: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/231
2. Existing local export-tail continuity and face tracking extensions, preserved
   from the five modified files found before this update. They are not official
   upstream features. Left/right face choice is initial position selection, not
   identity recognition.
3. Legacy workflow restore for the new pre-decode DiT cleanup toggle: positional
   placeholders are not booleans; restore invalid values to the default off.
4. SelfLift full-frame fallback restores the original conditions in reverse
   order, even if tile layout validation fails after changing the mask.

中文：新版固定 `3cbc048`，依次应用 0001 至 0004；历史 `aff2579`
只适用前两份。新检出应从一开始关闭自动换行转换，重放时用
`git am --keep-cr --3way`。现有已打补丁的安装不要重复应用。

Future updates: finish the queue, close ComfyUI, back up with `git bundle create`,
fetch and pin the new upstream commit, then use `git rebase --onto <new-upstream> <previous-upstream>`
on an isolated copy of the local patched branch. Inspect `git range-diff` and run
regressions before restarting. Manager Update All/reinstall may switch branches
or discard the overlay: do not use it for this node. There is no automatic
guarantee against every upstream semantic change; the backup, identifiable
commits, patches and tests provide recovery and a reviewed upgrade path.

中文：当前安装已带补丁，**不要重复应用**。以本地补丁分支持续 rebase 官方更新，
发生冲突必须审查，不能直接覆盖。完整备份留在唯一 Raylight 工作树的忽略目录；
模型、视频、日志与 Git bundle 不提交仓库。补丁保留 Apache-2.0 上游归属，
随 Raylight 仓库的许可证及原 Director 许可证使用。
