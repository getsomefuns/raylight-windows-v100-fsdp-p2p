# Raylight 1.18.4：Director 静音视频保存兼容修复

## 问题与原因

Director 关闭音频时返回 `waveform.shape=[1,1,0]` 的 AUDIO 字典。官方 `CreateVideo → SaveVideo` 根据字典是否存在来创建音轨，随后将零采样点音频送入 PyAV 18.1.0 重采样器，触发 `av.error.MemoryError: [Errno 12] Cannot allocate memory`。该错误能够在不加载模型、LoRA 或 CUDA 的最小编码测试中复现，不能凭异常名称判断系统内存耗尽。

用户案例中，第二段的独立 LoRA、采样和双卡 VAE 解码均已完成；Director 内部两段 MP4 与外接保存节点写出的最后一段视频都可完整解码。外接节点在视频编码完成后处理空音频时失败。

## 修复行为

- 只在 Ray MODEL 代理的 Director 执行返回处处理零长度 AUDIO，将其补为与对应片段画面等长的 CPU 静音波形，保持原采样率、声道和 AUDIO 字典结构。
- 普通音频、非空静音音频均直接透传，不复制或修改波形，也不重采样。
- 分段列表逐项对应；“先确认一采”时使用 `images_pre_refine` 确定静音长度。不同画面输出共用音频时取最长可用片段长度，由官方保存节点按所选画面裁剪。
- 双卡 VAE 成功、预加载失败回退及不可并行的代理路径均覆盖。非 Ray Director 执行保持原行为。
- 不修改 ComfyUI 和 Director 源文件，不增加节点、开关或新依赖。仍可保持 `Director AUDIO → CreateVideo audio` 连线。

静音波形很小：15 秒、32 kHz、单声道 FP32 约 1.83 MiB；不会分配额外显存。

## 验证

- 修复前新增测试为 `4 failed, 2 passed`，其中官方 `VideoFromComponents.save_to()` 准确复现用户相同堆栈的 PyAV MemoryError。
- 修复后 9 项定向测试通过，包含静音、正常双声道音频、混合分段、正常双卡 VAE、预加载回退、首采确认、非 Ray 执行与真实 MP4 编码/解码。
- 编码测试验证视频 6/6 帧可解码、静音解码振幅为 0、正常音频解码保留非零信号。正常音频输入对象及波形保持原样。
- 全量回归：`817 passed, 35 subtests passed`。另有 19 条上游弃用及 Windows 临时测试目录清理警告，没有测试失败。

本次验证针对发生错误的音频输出和保存阶段，不重新执行耗时的模型采样。分段导出目录和只向外接保存节点返回最后一个保留片段的上游行为保持不变。
