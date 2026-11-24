# 未发布：RelayVSR 双 V100 Flash 隔离预览

只在隔离环境验证；正式核心、Python、Director、部署节点均未更改，未推送GitHub，不改变正式版本号。

- 新增 `RelayVSR Flash Decode (Ray)`：保留官方单卡关键帧采样，独立Flash窗口在两张GPU计算。
- FP32和原版窗口history/重叠端点不变，合并按帧编号写入连续CPU输出。
- 默认同进程执行，不创建Ray worker或集群；默认结束释放自有decoder模型。
- 可显式保留decoder权重，空闲计算缓存始终归还；失败/取消回收，不返回半成品。
- 准入锁、共享Ray会话拒绝、确定性patcher清理、future寿命、失败后重建均有回归。
- 按实际VAE执行输出后处理，不假设所有版本永远为identity。
- RelayVSR误用普通Wan USP时清晰报错；不改变普通Wan/H3/LTX注册与原节点ID。
- 提供无素材/权重的静音基准API工作流：`example_workflows/RelayVSR_DualV100_Raylight.json`。

Ray跨进程原型没有达到冷启动门槛，且共享集群所有权需要额外工程，故不开放它或借用worker接口。
相关失败、修复、完整时间与资源对比保留在[实测报告](../reports/2026-11-24-relayvsr-dual-flash.zh-CN.md)。

原正式ComfyUI没有原生RelayVSR；仅更新Raylight不能增加核心算子。隔离新版核心原有14项
H3相关基线失败仍单独列明，不把本次RelayVSR验收当成正式环境整体升级验收。
