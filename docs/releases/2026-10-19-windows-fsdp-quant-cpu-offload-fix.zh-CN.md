# Raylight 1.18.2：Windows FSDP 量化 CPU-offload 原生崩溃修复

## 故障

MiniMax H3 Director SelfLift 在进入第一次双卡采样前，两名 Ray worker 都于 FSDP 权重注册阶段以 Windows `0xC0000005 access violation` 退出。原生堆栈位于 `comfy_kitchen.tensor.base._handle_to()`，由 `load_from_full_model_state_dict()` 对量化分片执行 `.cpu()` 触发。worker 退出信息又包含 Windows C10d 的非 UTF-8 本地化文本，触发 Ray raylet 的 UTF-8 序列化缺陷，最终前端只看到 `ActorUnavailableError: Connection reset`。

这与 1.18.1 修复的 Text Encoder safetensors mmap 生命周期不是同一个故障，也不是 CUDA OOM。

## 根因与修复

启用 FSDP CPU offload 时，旧实现仍先在 CUDA 上构造每个量化本地分片，再立即通过 `QuantizedTensor.cpu()` 搬回 CPU。MiniMax H3 共注册 684 个 FSDP wrapper；真实 FP8 checkpoint 的大量往返复制会进入 comfy-kitchen 的 Tensor subclass `to_copy` 路径，并在本机触发原生访问冲突。

1.18.2 改为：

1. CPU-offload 模式直接在 CPU 上切出并构造本 rank 的量化 payload；
2. 仍对外层 DTensor 执行 `.cpu()`，满足 PyTorch FSDP2 对参数公开设备的校验；
3. 此时本地 qdata 已在 CPU，外层转换只产生安全的 CPU→CPU 操作，不再搬运大型 CUDA qdata；
4. 非 CPU-offload 路径保持原有 CUDA 构造方式，模型数值、分片边界、LoRA、采样和 P2P collective 均未改变。

## 验证

- 新测试先在旧代码上因缺少 CPU 直接构造路由而失败，修复后通过。
- 定向测试：`7 passed, 2 subtests passed`。
- 全仓自动化：`800 passed, 35 subtests passed`。
- 双 V100 量化 FSDP CPU-offload 前向探针：两 rank 输出与参考值逐元素一致，最大误差 `0.0`，本地 qdata 均驻留 CPU。
- 真实模型预检：`minimax_h3_ref2va_pruned_fp8_scaled.safetensors`、`fp16_h3_safe`、0.65 Turbo4 LoRA、FSDP CPU offload、Ulysses 2；两 rank 均完成 684 个 wrapper、526 个 DTensor 和 200 个 FP8 layout 注册。本地 payload 为 10,477,671,768 bytes/rank，P2P 中位数约 60.10 GiB/s，预检正常退出。

完整 73 段生产任务没有作为自动验收重跑；它会在后续正常续跑中继续验证长时间稳定性。本次真实预检已经完整覆盖原崩溃所在的模型装载与 FSDP 注册阶段。
