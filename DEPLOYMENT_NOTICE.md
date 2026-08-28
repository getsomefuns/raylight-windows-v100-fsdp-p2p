# Deployment source of truth / 部署唯一来源

This repository is the editable source of truth. A ComfyUI installation may contain an immutable-commit copy generated at `custom_nodes/raylight`, but that deployed copy is not a Git worktree and must not be edited directly.

本仓库是唯一允许修改、提交和推送的源码目录。ComfyUI 的 `custom_nodes/raylight` 可以保存由本仓库不可变提交生成的部署副本，但部署副本不是 Git 工作树，不得直接修改。

The generated copy contains `.raylight-deployed-commit`, which records the exact source commit and is checked together with tracked-file hashes. Ray runtime working files live under the system temporary directory. Local raw evidence belongs under the Git-ignored `.local/artifacts` tree in this repository.

生成副本中的 `.raylight-deployed-commit` 记录准确来源提交，并与跟踪文件哈希一同校验。Ray 运行工作文件位于系统临时目录。本机原始测试证据统一放在本仓库已被 Git 忽略的 `.local/artifacts` 下。

Deploy from the repository root:

```powershell
.\scripts\deploy-to-comfyui.ps1 -ComfyUIRoot <ComfyUI目录>
```

Check for deployment drift without changing files:

```powershell
.\scripts\deploy-to-comfyui.ps1 -ComfyUIRoot <ComfyUI目录> -VerifyOnly
```
