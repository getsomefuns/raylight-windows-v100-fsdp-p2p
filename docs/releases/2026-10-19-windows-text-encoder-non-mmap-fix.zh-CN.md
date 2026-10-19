# Raylight 1.18.1：Windows 多段 Text Encoder 非 mmap 稳定性修复

## 故障

73 段 MiniMax H3 Director 任务在第 42 段完成采样、双卡 VideoVAE 解码和 AudioVAE 后，于第 43 段重新加载 Qwen3-VL-32B NVFP4 Text Encoder 时使整个 Python 进程以 Windows `0xC0000005 access violation` 退出。堆栈顶位于 `torch.storage.__getitem__` 与 `safetensors.safe_open().get_tensor()`；这不是 VideoVAE、FSDP worker 或 CUDA OOM。

## 根因与修复

原 TP2 Text Encoder 每段会为同一个 15.7GB checkpoint 多次创建 `safetensors.safe_open()` 映射。Windows 会把大文件 copy-on-write mmap 计入系统提交额度；长时间反复装卸后，Rust/PyTorch storage 交接可能直接访问冲突，而不会先产生可捕获的 Python 内存异常。

1.18.1 在 Raylight 内新增有界的 safetensors reader。Windows 按 tensor 的 header offset 使用 `readinto` 读取到自有 `bytearray`，随后沿用原有 GPU 分片和复制流程；任一时刻只有当前 tensor 的 CPU buffer 在途。embedding、视觉塔和 50 层语言主干全部走该路径。Linux 继续使用官方 mmap reader。ComfyUI、MiniMaxH3Director、checkpoint 和工作流均未修改。

## 验证

- 新回归测试先验证旧代码不存在非 mmap reader，再验证关闭 reader 后，已返回 tensor 仍有效且 Windows 可立即重命名 checkpoint。
- 自动化：非 mmap reader 与 TP2 CLIP proxy 共 17 项通过；全仓为 `799 passed, 240 warnings, 35 subtests passed`。警告来自既有的 PyTorch 弃用提示和 Windows pytest 临时目录清理，不含测试失败。
- 已部署版本对实际 checkpoint 的 20 MiB `q_proj` NVFP4 tensor 连续执行 64 次打开、读取、关闭；进程 RSS 只增加 3.58 MiB、VMS 只增加 2.66 MiB，文件句柄未按轮次累积。
- 实际 15.7GB checkpoint、双 V100 纯文本探针：29 tokens，19.61 秒，结果有限值，峰值 7347.60/7347.60 MiB，关闭后 16.25/16.25 MiB。
- 实际 REF2VA 视觉探针：77 tokens，20.89 秒，结果有限值，峰值 7370.09/7361.80 MiB，关闭后 25.05/16.92 MiB。

该修复改变的是 Windows 权重读取和内存提交行为，不改变模型数值、TP2 切分、注意力、扩散采样或 VideoVAE 输出。完整 73 段任务耗时很长，因此本次自动验收使用映射生命周期测试与两条真实模型路径；后续生产续跑仍用于验证超长时稳定性。
