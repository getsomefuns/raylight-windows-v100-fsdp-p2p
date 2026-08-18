# 2026-08-19 启动脚本容量与日志开关更新

## 更新目标

让正常使用和性能测试采用同一套明确、可复现的启动配置，同时避免大量
`[RAYLIGHT_RANK_DIAG]` / `[RAYLIGHT_P2P_DIAG]` 信息干扰日常终端输出。

## 用户可见变化

- `scripts/start-comfyui-windows-p2p.ps1` 默认选择每张 GPU `256 MiB` P2P staging buffer。
- `-P2PCapacityMiB` 只接受 `128`、`256`、`512` 三档，启动时会显示全部可选值和当前值。
- Rank/P2P 详细诊断默认关闭；使用 `-EnableDiagnostics` 才会开启。
- `-ValidateOnly` 会显示最终容量和诊断状态，但不会启动 ComfyUI。
- 旧的 `-P2PCapacityBytes` 参数继续保留，避免已有自动化失效；新命令优先使用 MiB 参数。

## 使用示例

```powershell
# 默认：256 MiB，诊断关闭
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY

# 显式选择容量
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 128
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 256
.\scripts\start-comfyui-windows-p2p.ps1 -PythonPath $PY -P2PCapacityMiB 512

# 性能或同步诊断
.\scripts\start-comfyui-windows-p2p.ps1 `
  -PythonPath $PY `
  -P2PCapacityMiB 256 `
  -EnableDiagnostics
```

切换容量或日志状态前必须停止当前 ComfyUI 和旧 Ray worker，再重新启动。

## 容量含义

容量是每张 GPU 的持久 CUDA staging buffer 上限，不等同于视频时长或分辨率。
更大档位不会自动加速，也不会固定占用更多系统内存或分页文件，但会增加每张 GPU
的常驻显存：256 相比 128 MiB 多约 128 MiB/GPU，512 相比 256 MiB 多约
256 MiB/GPU。已验收的 1120×768 MiniMax H3 O6 工作流包含 239,826,944 字节
Ulysses 远端 payload，因此需要 256 MiB 或更大档位。

底层 worker 在没有启动脚本配置时仍回退到 128 MiB；这保留库级兼容性，但不是当前
仓库启动脚本的默认值。容量不足会明确失败，不会自动通过 CPU 主存绕过 NVLink。

## 验证范围

发布配置测试覆盖默认值、三档合法选择、非法 `384 MiB` 拒绝、诊断显式开启、诊断
默认关闭以及旧字节参数兼容。此更新不修改 CUDA P2P/FSDP 数据路径、模型精度或工作流。
