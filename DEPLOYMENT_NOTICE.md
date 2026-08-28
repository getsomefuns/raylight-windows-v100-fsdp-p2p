# Deployment source of truth / 部署唯一来源

This repository is the editable source of truth. A ComfyUI installation may contain a generated copy at `custom_nodes/raylight`, but that deployed copy is not a Git worktree and must not be edited directly.

本仓库是唯一允许修改、提交和推送的源码目录。ComfyUI 的 `custom_nodes/raylight` 可以保存由本仓库生成的部署副本，但部署副本不是 Git 工作树，不得直接修改。

Deploy from the repository root:

```powershell
.\scripts\deploy-to-comfyui.ps1 -ComfyUIRoot <ComfyUI目录>
```

Check for deployment drift without changing files:

```powershell
.\scripts\deploy-to-comfyui.ps1 -ComfyUIRoot <ComfyUI目录> -VerifyOnly
```
