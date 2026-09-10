# 1.14.0 / O12：Python 3.12、Torch 2.12 与 H3 外部加速桥

1.14.0 将默认候选运行时升级到 Python 3.12.10 + PyTorch 2.12.0+cu126，并在不修改 ComfyUI、MiniMaxH3Director、TE-Speed 或 Sol-Attn 源码的前提下，让 Raylight 的 Director MODEL 代理可选串联 TE-Speed 3.3 和 Sol-Attn V100。

## 主要变化

- 启动和验证脚本优先选择 `Python312`，找不到时仍可回退 `Python310`。
- 新增 `requirements-windows-v100-torch212.txt`；旧 Torch2.7 锁定文件保留用于回滚。
- 外部节点写入主进程 MODEL 的 patch 会被归一化为纯数据配置，再由两个真实 Ray worker 重放；不向 Ray 序列化闭源对象或函数。
- TE-Speed 3.3 的精确模型类型检查通过无权重视图适配 composable FSDP 动态子类，不复制模型权重。
- Sol-Attn 在 Ulysses all-to-all 之后处理每个 rank 的局部序列；Raylight 保留 FP16 数值安全控制，避免重复缩放。
- 修复 Torch2.12 下多分块 CUDA IPC event 复用竞态和 xFuser Windows subgroup Gloo 路由。
- 新增 8 个可直接比较的 4/8 步 Base、TE、Sol、TE+Sol REF2VA 工作流。

## 性能结论

同一 864×480、124 帧、24 fps、约 5.17 秒 REF2VA 输入下，热启动结果如下：

| 步数 | Base | TE | Sol | TE+Sol |
|---:|---:|---:|---:|---:|
| 4 步端到端 | 142.04 s | 149.27 s | 136.49 s | **130.67 s** |
| 4 步采样 | 85.23 s | 88.88 s | **69.28 s** | 73.32 s |
| 8 步端到端 | 228.86 s | 228.49 s | 210.54 s | **200.05 s** |
| 8 步采样 | 170.90 s | 162.52 s | **138.06 s** | 138.13 s |

Sol 的采样收益约为 19%，是当前确定的加速来源。TE 在该短轨迹矩阵中未表现出稳定独立收益；组合组端到端最快，但采样没有快于 Sol。详细资源、冷启动、质量和三连数据见 [O12 中文测试汇总](../testing/minimax-h3/O12_PYTHON312_H3_ACCELERATION_2026-09.zh-CN.md)。

## 兼容与边界

- 新加速默认关闭。未串联外部节点的旧 JSON 和 Base 工作流行为不变，未安装外部节点也不影响旧路径加载。
- 外部节点不是本仓库的一部分，需分别安装 [TE-Speed-MiniMaxH3-OSS](https://github.com/HELPMEEADICE/TE-Speed-MiniMaxH3-OSS) 和 [ComfyUI-MiniMaxH3-SolAttn-V100](https://github.com/aaalll12322/ComfyUI-MiniMaxH3-SolAttn-V100) 的兼容版本及其 V100/Python3.12 kernel。
- Sol 是稀疏近似注意力，输出不与 dense Base 像素等价。TE-only、Sol-only 与组合均必须显式插入节点，不会暗中改变普通工作流。
- PyTorch 2.8/2.9 Windows Gloo 不能满足当前控制面；本阶段最终采用已通过真机探针的 Torch2.12，而不是强行停留在 2.8。
