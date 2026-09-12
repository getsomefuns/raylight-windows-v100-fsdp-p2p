# Raylight 1.15.0：MiniMaxH3Director 19b6a73 完整兼容

## 升级目标

本版把已验证的 MiniMaxH3Director 坐标从 `b8f721c` 更新到 `19b6a73`，并补齐此前 Ray MODEL 代理会遗漏的四类能力：Director 自带 Refine、连续段动态 remask、Refine 空间分块和采样逐步实时预览。旧显式 Ray sampler 工作流和普通 ComfyUI MODEL 路径保持不变。

## 已实装

- 扩展 `sample_single_stage()` 契约，接收 `after_shift`、`enable_tiling`、`tile_count`、`tile_overlap` 和新版 `shift_cache`。
- 只把已知的 `install_continue_prefix_remask` 转换为稳定能力标识；两个 worker 在真实 MODEL 上分别恢复回调。未知回调立即报错，不静默降级。
- Director 自带 Refine 仍由双 worker 计算。开启空间分块时，每个 worker 在实际 sampler 上安装 Director 原生 tile wrapper，并确保异常后恢复。
- 实时预览只由 rank 0 抽取最多 16 个视频 latent 帧，通过容量为 1 的非阻塞 Ray 队列交回 Director 原回调。队列满时丢弃过期中间帧，不阻塞采样；预览关闭时不创建队列。
- 接收 Director `19b6a73` 新增的 `shift_cache` 参数。该缓存只管理主进程 MODEL 克隆，而 Raylight 的真实权重在远端，因此不跨进程序列化，避免两套生命周期互相干扰。
- 增加真实已安装 Director 签名回归测试，兼容错误会显示当前验证提交。
- 修复 Ray worker 无法导入同级 Director 分块模块的问题：通过 ComfyUI 配置发现节点目录，以私有运行时包加载 `director` 子包；远端异常现在会在 Director 警告中保留根因末行。

## Director 上游更新覆盖

从 `b8f721c` 到 `19b6a73` 不只有 Refine 与预览：还包括混合时间轴任务、分段导出内存回收、参考音频槽位修复、自定义 FPS 修复、段间“引导+重绘/保完整”、latent 时间分块、参考图最长边档位以及 SigmaShift/死槽显存泄漏修复。Raylight 不复制或篡改这些功能；Director 继续负责计划、条件、解码、拼接和导出，Raylight 只在它进入扩散采样时接管代理 MODEL。

## 性能与边界

- 空间分块的目标是降低 Refine 瞬时显存，不是加速。显存足够时关闭通常更快。
- 预览会增加少量 rank 0 到 CPU 的 latent 拷贝与主进程 TAE/WebP 开销；关闭后没有预览传输。
- Director 的段间显存清理修复仍作用于主进程官方路径；远端模型、remask hook、worker 复用和回收继续由 Raylight 管理。
- 另一 checkpoint、不同权重，或绕过 `sample_single_stage()`/旧 `comfy.sample.sample()` 的第三方采样路径不在本版保证范围。

## 验证

自动化覆盖 bridge、worker remask、空间分块、异常恢复、预览队列、真实 Director 签名以及旧路由。双 V100 实机结果维护在 [O13 测试记录](../testing/minimax-h3/O13_MMH3DIRECTOR_19B6A73_2026-09.zh-CN.md)。

同参实机工作流（864×480、124 帧/5.17 秒、首采 4 步 + Refine 2 步、2 块）端到端 299.85 秒。首采与 Refine 两次 operation 均由同一对 worker 完成，两卡利用率峰值均 100%，显存峰值 9223/9201 MiB；输出为 124 帧、24 fps 的 H.264 + AAC，抽检无黑屏、随机噪声或明显 tile 接缝，结束后两卡回落至约 10 MiB。
