# Windows 双 V100 DynamicVRAM 运行时升级设计

## 状态

- 日期：2026-09-03
- 目标环境：`E:\ComfyUI-py310`
- 唯一源码工作树：`E:\ComfyUI-py310\raylight-windows-v100-fsdp-p2p`
- 部署副本：`E:\ComfyUI-py310\ComfyUI\custom_nodes\raylight`
- 当前基线：Python 3.10.11、PyTorch 2.7.0+cu126、torchvision 0.22.0+cu126、torchaudio 2.7.0+cu126、xFormers 0.0.30、comfy-aimdo 0.4.13、ComfyUI `62b3c94b`

## 问题与根因

ComfyUI 启动时打印：

```text
Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later.
Falling back to legacy ModelPatcher.
```

这不是日志误报。ComfyUI 在自动启用 DynamicVRAM 时明确拒绝 PyTorch 2.8 以下版本；Raylight worker 中也有相同门禁。当前主进程和 worker 因而都使用 legacy `ModelPatcher`。

`comfy-aimdo` 官方同时声明需要 PyTorch 2.8+、CUDA 12.8+。因此只把 2.7.0+cu126 改成 2.8.0+cu126 不属于本项目可发布的完整支持组合；也不允许用 `--enable-dynamic-vram` 强制绕过版本门禁来冒充验收通过。

## 目标

在不修改 NVIDIA 577.00 驱动、不影响 `E:\ComfyUI-py310` 以外环境的前提下：

1. 让 ComfyUI 主进程真正启用 `ModelPatcherDynamic`；
2. 让两个 Ray worker 真正启用 DynamicVRAM，而不是只消除启动警告；
3. 保持 Tesla V100 `sm_70`、TCC、CUDA P2P/NVLink、Windows Gloo 控制面、FSDP2、O6 safe-FP16、O9 动态 P2P 分块和 O11 高效注意力可用；
4. 保持 MiniMax H3 输出有限、可解码且无黑屏、NaN/Inf；
5. 保留完整、快速、可验证的 PyTorch 2.7 回退路径。

## 非目标

- 本阶段不适配 2026-09 更新后的 MiniMaxH3Director 新采样入口；该问题已经定位，但在 DynamicVRAM 验收后单独修复。
- 不升级 NVIDIA 驱动。
- 不修改 `E:\ComfyUI-py310` 外的 Python、CUDA 或 ComfyUI 环境。
- 不把多卡显存描述为透明的 32 GiB 统一显存池。
- 不启用 V100 不支持的 FlashAttention 2/3 路径。

## 方案选择

### 采用：规范路径候选副本 + 整体回退

先停止 ComfyUI/Ray，完整保留当前 `Python310` 目录。候选环境仍运行在规范路径 `E:\ComfyUI-py310\Python310`，避免 Windows console launcher、Ray 子进程和嵌入式 Python 中潜在的绝对路径问题。只有候选通过全部门禁后，才删除旧环境备份。

候选版本：

| 组件 | 候选版本 | 原因 |
|---|---:|---|
| Python | 3.10.11 | 保持不变，避免引入第二个迁移维度 |
| torch | 2.8.0+cu128 | 同时满足 ComfyUI 的 2.8 门槛和 AIMDO 的 CUDA 12.8 门槛 |
| torchvision | 0.23.0+cu128 | 与 torch 2.8.0 配套 |
| torchaudio | 2.8.0+cu128 | 与 torch 2.8.0 配套 |
| xformers | 0.0.32.post2 | 官方为 torch 2.8.0 提供的对应系列 |
| comfy-aimdo | 0.4.13 | 当前已安装版本，先不增加无关升级变量 |
| comfy-kitchen | 0.2.30 | 保持版本，重新验证其 CUDA 二进制与 torch 2.8 的运行兼容性 |
| Ray | 2.57.0 | 保持版本，避免同时改变调度层 |

`flash-attn 2.8.3` 不属于 V100 发布路径。若其二进制与 torch 2.8 冲突，候选环境中移除；Raylight 必须继续通过 xFormers CUTLASS/GQA 或 Torch Efficient 路由运行。

### 不采用：原地无备份升级

下载和切换更快，但 Torch、xFormers、torchvision、torchaudio 以及 CUDA 二进制一旦出现 ABI 冲突，恢复容易遗漏残留文件，不满足本机长时间任务所需的回退可靠性。

### 不采用：PyTorch 2.7 强制打开 DynamicVRAM

这只绕过 ComfyUI 的保护分支，不能补充 AIMDO 要求的 PyTorch/CUDA allocator 能力，可能造成崩溃、错误卸载或静默坏图。

## 运行时和代码改动边界

### 环境改动

- 只改 `E:\ComfyUI-py310\Python310` 内依赖；旧目录以带版本和日期的名称保存在 `E:\ComfyUI-py310\runtime-backups`。
- 更新 Torch 四件套和 xFormers，必要时移除不适用的 flash-attn。
- 不依赖系统安装的 CUDA Toolkit；以 PyTorch wheel 自带 CUDA runtime 为准。驱动保持 577.00。

### Raylight 改动

- 更新 `requirements-windows-v100.txt` 中已验收的运行时版本。
- 增加运行时兼容探针：Torch/CUDA 版本、`sm_70`、xFormers V100 forward、AIMDO 初始化、CUDA IPC metadata round-trip。
- 只有实测暴露兼容差异时，才修改 Raylight 的 AIMDO 或 Torch 2.8 适配代码；每项修改必须先有失败测试。
- 保持唯一工作树修改、提交、部署、核验流程；不得直接编辑部署副本。

## 风险与处理

### V100 架构缺失

安装后必须同时满足：`torch.cuda.get_arch_list()` 包含 `sm_70`，并且在 V100 上实际完成 FP16 GEMM、CUDA event 和 IPC 操作。仅看到设备名称不算通过。

### xFormers wheel 不含 V100 kernel

必须在两张 V100 上实际运行项目使用形状的 `memory_efficient_attention`。若官方 0.0.32.post2 wheel 不支持 `sm_70`，先评估针对 `TORCH_CUDA_ARCH_LIST=7.0` 的源码构建；构建成本或稳定性不达标则停止升级并回退，不允许静默改用会导致 OOM 的 SDPA Math。

### Torch 私有 CUDA IPC 接口变化

Windows P2P 依赖 `torch.multiprocessing.reductions.reduce_tensor/rebuild_cuda_tensor` 的 CUDA IPC metadata。必须执行双进程、双卡 IPC、all-to-all 和 all-gather 探针；不能只跑 CUDA P2P samples 工具。

### DynamicVRAM 与 FSDP shard 生命周期冲突

主进程的普通 `ModelPatcherDynamic` 和 worker 内的 `FSDPModelPatcher` 并非同一对象。验收必须证明 worker 的 AIMDO 开启不会破坏 shard 注册、量化权重解量化、LoRA、采样后回收和下一次热运行。

### 性能倒退或错误输出

DynamicVRAM 的目标是降低显存/RAM压力，但按需换入可能增加数据搬运。本阶段不接受通过显著降低采样速度换取“能运行”。若相同工作流采样速度稳定倒退超过 10%，或出现黑屏、噪声、NaN/Inf，即判失败并回退。

## 验收门禁

### Gate A：静态依赖一致性

- `torch==2.8.0+cu128`
- `torchvision==0.23.0+cu128`
- `torchaudio==2.8.0+cu128`
- `xformers==0.0.32.post2`
- `pip check` 不新增与 Torch 栈有关的冲突；现有无关依赖冲突单独记录，不混作本阶段回归。

### Gate B：V100 与核心 CUDA

- 两张卡均为 Tesla V100-SXM2-16GB、TCC、compute capability 7.0；
- `torch.cuda.get_arch_list()` 包含 `sm_70`；
- 每张卡 FP16 GEMM、stream、event 正常；
- 保留 `--disable-cuda-malloc` 后 VAE 不出现 `operation not supported`。

### Gate C：xFormers 与量化路径

- V100 上 xFormers memory-efficient attention 实际 forward 通过且输出有限；
- O11 `AUTO` 仍选择已验收的 xFormers GQA 路径；
- NVFP4/FP8 权重继续按 safe-FP16 路径解量化，不触发 V100 原生 FP8 计算；
- 不安装或调用 FlashAttention 2/3 作为 V100 必需依赖。

### Gate D：AIMDO 真实性

- ComfyUI 主进程打印 `DynamicVRAM support detected and enabled`；
- `comfy.model_patcher.CoreModelPatcher is ModelPatcherDynamic`；
- 两个 Ray worker 都打印 `[Raylight][AIMDO] DynamicVRAM enabled in worker`；
- `host_buffer.lib` 有效，两个设备均成功注册；
- 用真实模型加载制造显存压力，确认权重按需驻留/卸载，而不是只验证启动日志。

### Gate E：P2P/FSDP/Raylight 回归

- Windows CUDA IPC metadata round-trip 通过；
- 双卡 P2P `all_to_all_single` 和 `all_gather_into_tensor` 正确性与有限值通过；
- FSDP 注册、分片加载、Ulysses collective、O9 大 payload 自动分块通过；
- 两个 rank 输出一致，失败时不回退 CPU/Gloo 数据面；
- 自动化测试全量通过。

### Gate F：真实工作流

在新版 Director 适配尚未完成期间，使用不依赖 Director 的 MiniMax H3 Raylight 工作流：

- 至少完成一个短 I2V 和一个 REF2VA；
- 两卡均参与采样，输出视频逐帧可解码；
- 视频无整段黑屏/噪声，latent/audio 无 NaN/Inf；
- 记录模型加载、Text Encoder、采样、VAE/音频解码、保存时间，以及显存、RAM、commit/pagefile 峰值；
- 与 torch 2.7 基线相比，采样速度稳定倒退不得超过 10%；
- 工作流结束及 worker 退出后的显存回收符合既有验收口径。

## 回退规则

任一发布门禁失败即停止推广候选环境：

1. 停止 ComfyUI 和 Ray；
2. 保留失败候选的版本清单和最小诊断记录；
3. 将候选 `Python310` 移出规范路径；
4. 将 PyTorch 2.7+cu126 备份恢复为 `E:\ComfyUI-py310\Python310`；
5. 重新执行 Torch、xFormers、P2P 和启动冒烟测试，证明恢复完成。

在新环境完成 Gate A-F 前，不删除回退副本，不更新 GitHub，不把 README 的正式环境版本改为 2.8。

## 文档与发布

验收完成后同步更新：

- 中英文 README 的环境矩阵、安装命令、启动方式和限制；
- `requirements-windows-v100.txt`；
- 中英文升级说明；
- DynamicVRAM 专项测试汇总；
- 部署来源提交号。

GitHub 推送仍遵循用户明确授权；本阶段本地提交与部署不自动等同于远端发布。

## 后续顺序

1. 完成并验收本设计的 DynamicVRAM 运行时升级；
2. 再适配 MiniMaxH3Director `cd8f56b` 之后改为 `BasicGuider/CFGGuider + SamplerCustomAdvanced` 的新入口；
3. 对更新后的 Director 工作流重新执行双卡、质量、显存与多段回归。
