# 2026-10-23 长参考视频 OOM 与内存记录修复计划

## 问题边界

- 目标故障是 MiniMax H3 Director 15 秒 REF2VA 在 SelfLift 高分辨率采样阶段，由 Ulysses `all_to_all` 额外申请完整输出张量而触发的 CUDA OOM。
- 当前输入会把 Attention 序列推到约 152,976；Q/K/V 是同一 QKV 投影的视图，源存储无法随单个分量完成通信而释放。
- 主内存需要区分输入工作集、Ray 请求副本、worker 私有内存和系统 commit，不能再依据任务管理器瞬时值判断版本回归。

## 实施方案

1. MiniMax H3 QKV 按 4,096 token 分块投影到独立 Q/K/V 存储；FSDP 权重每次 attention 只 unshard/reshard 一次。
2. Windows 双 rank P2P 的 4D Ulysses 交换复用通信布局张量作为收发输出，依赖既有 128 MiB staging 的“先保存发送块、再覆盖目标块”顺序，不再创建第二个完整 `empty_like`。
3. 非 Windows P2P 路径继续使用 yunchang 原实现。
4. xFormers 密集注意力把结果写回已消费的 Query 存储，减少一个完整 context 输出。
5. 每次 Director 采样在主进程和两个 worker 启动 1 秒周期的本地记录器；无论成功还是异常，都向 `ComfyUI/user/raylight_memory/<日期>/` 写入 JSONL 汇总。

## 验收标准

- 单元测试证明 QKV 分块有界、三个输出不共享底层存储、Windows P2P all-to-all 输入输出复用同一 data pointer。
- 内存记录在采样成功和异常退出时均能写入，并包含系统可用内存最低值、Windows commit 峰值、进程 RSS/private 峰值和 CUDA allocated/reserved 峰值。
- 原 15 秒、双图、354 帧抠像参考视频工作流必须越过原 OOM 点；最终验收需要完成全部采样并确认输出有限值和画面质量无退化。
- 不修改 ComfyUI 与 MiniMax H3 Director 源码。
