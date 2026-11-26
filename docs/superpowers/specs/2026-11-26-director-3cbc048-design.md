# Director 11 月 25 日更新：Raylight 1.19.1 兼容设计

日期：2026-11-26。发布结果见[升级说明](../../releases/2026-11-26-director-3cbc048.zh-CN.md)。

目标：升级 AIMixer Director 的官方基线 `aff2579` → `3cbc048b3ed5fbf993f4d7819557782f4a5ebb0c`，保留 PR231 元数据缓存检查、保尾、人脸跟踪 overlay。正式 Raylight 为 1.19.0（`3a6be82`），本地 Director HEAD `13eba63`。ComfyUI 0.46.0（`2fcb6ce`）、Torch 2.12.0+cu126、驱动 577.00、双 V100 TCC、启动参数均不变，不启用 DynamicVRAM。

| 上游变化 | Raylight 影响 |
|---|---|
| [#284](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/284)：SelfLift＋引用上段（引导＋重绘）＋空间分块的模型侧视频去噪遮罩随瓦片裁切及补齐 | 两 rank 的 `_sample_mmh3director_spatial_tiles` 直接调用安装的 `wrap_sampler_spatial_tiles`，预期继承修复。须用真实遮罩和双 rank 瓦片工作流验证；不复制裁切逻辑。 |
| [#287](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/287)：可选解码前只卸 DiT，默认关闭 | 上游只检查 ComfyUI 主进程的模型列表，不会卸载 Ray worker 的 FSDP 分片。双卡 VAE 已有 worker 显存交接，保留原生命周期。纯 Ray 主进程无 DiT 时允许释放 0 GiB，不增加重复 worker 清理，不预报提速。 |
| [#285](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director/pull/285)：分段 MP4 从 1 编号 | 只改新导出文件名，不改缓存索引或历史产物。选段仍用时间轴素材组编号；检查所有后缀、本地保尾补丁和导出调用方。 |

**推荐路线：** 固定官方 SHA，隔离重放本地补丁，先验证原 Raylight 路径。只回移某项修复会额外维护一个行为分支，留作回滚思路；一并升级核心或重写双卡分块/卸载没有必要。若实测没有适配缺口，1.19.1 只更新可复现补丁、版本元数据、兼容提示（如适用）和双语说明，不人为加运行逻辑。需修时用可复现的失败限定范围，旧版工作流仍可运行。

**隔离验收后的落地设计：** Raylight 运行逻辑不改。Director 补丁 0003 在工作流恢复时只修正新版卸载开关的非布尔值；0004 将瓦片安装撤销列表放到调用者，异常或正常退出都逆序恢复，防止同一条件在多个分支被引用时错误回滚。两项都通过原入口复现旧行为；按四份补丁从 `3cbc048` 重放。补丁本身禁止 Git 文本换行转换。

正式切换前备份并核验回滚；真实工作流和媒体通过后才发布。新测试脚本、夹具、工作流、测试视频、素材、原始日志、缓存及备份只留本地忽略目录，GitHub 只发必要实现、可重放的维护补丁和不含私有内容的文档。
