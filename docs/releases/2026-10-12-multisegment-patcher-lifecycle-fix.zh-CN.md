# Raylight 1.16.1：多段 FSDP patcher 生命周期修复

## 故障现象

MiniMaxH3Director 连续执行多段视频时，前几段可以正常生成和保存，后续片段可能在 SelfLift 高清阶段报错：

```text
AttributeError: 'NoneType' object has no attribute 'prepare_state'
```

本次真实故障发生在连续选择的第 6 个片段。低清采样已完成，错误出现在高清阶段进入 ComfyUI sampler 前；它不是显存不足，也发生在 FaceRefine 执行之前。

## 根因

FSDP `ModelPatcher` 的多个克隆共享同一个底层模型。新片段会把自己的 patcher 注册到 `model.current_patcher`，但旧克隆稍后执行 `cleanup()` 或被 Python 回收时，旧代码会无条件把这个共享引用清空。

因此，实际仍然有效的新 patcher 被一个过期克隆误删。下一次 sampler 读取 `current_patcher.prepare_state` 时只会得到 `None`。

## 修复

显式清理和 finalizer 现在都先核对所有权：只有 `model.current_patcher is self` 时，当前克隆才可以解除这个引用。旧克隆不能再破坏新片段的活动状态。

该改动：

- 不修改 ComfyUI 或 MiniMaxH3Director；
- 不修改模型权重、dtype、Safe-FP16 或量化；
- 不修改 FSDP 分片、CUDA P2P/NVLink 通信或 staging 容量；
- 不修改采样公式、噪声、seed、SOL、TE-Speed、SelfLift 或 Refine 参数。

## 验证

- 新增四个定向回归测试：两个覆盖旧克隆的显式 `cleanup()` 和 `__del__()`，两个保证真正的当前所有者仍可释放自身引用。
- RED 阶段：修复前两项均稳定复现活动引用被清空。
- GREEN 阶段：修复后两项通过。
- `tests/test_fsdp_mixed_precision.py`：`22 passed`，另有 `2 subtests passed`。
- Director worker/lifecycle 定向回归：`44 passed`，另有 `2 subtests passed`。
- 全项目自动化套件：`757 passed`，另有 `35 subtests passed`。
- 修改文件 Ruff 无缓存检查：通过。

双 V100 部署验证结果以本次提交和部署来源标记为准。

## FaceRefine 说明

这次崩溃不是 FaceRefine 造成的。Director 的 `confirm_first_pass=true` 表示先完成并缓存首采，等待用户确认后再次执行才进入 Refine/FaceRefine。若希望一次执行直接进入 Refine 和 FaceRefine，应关闭该开关；若保留该开关，则按 Director 设计确认后重跑。

## 版本与兼容性

- Raylight 版本：`1.16.1`
- 基于已发布的 `1.16.0` O14 SelfLift 适配继续修补；未回退 O14 能力。
- 旧工作流不需要改线，修复在 FSDP patcher 生命周期内部生效。
