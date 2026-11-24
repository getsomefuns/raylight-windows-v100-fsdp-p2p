# RelayVSR 双 V100 Flash 加速设计

日期：2026-11-24。用户已要求先落盘计划，再在隔离环境开发和测试。
实现选择由维护者作出；没有把未经实测的加速比写成已经实现的收益。

## 目标和约束
- 唯一开发树：本 Raylight 仓库。正式 ComfyUI、Director、Python、驱动不变。
- 开发后只部署到 .local/environments/relayvsr-v100/ComfyUI/custom_nodes/raylight。
- 不修改 ComfyUI/core、comfy-kitchen、上游 Raylight 或其他节点源码。
- 双 V100-SXM2-16GB/TCC；Python3.12、Torch2.12/cu126；保持原有H3/LTX功能。
- 本轮主要交付：显式可选择的双GPU Flash解码节点，原版单卡关键帧采样不变。
- 不强迫全部步骤双卡，不把权重分片或GPU占用当成加速。不承诺理论2倍。
- 暂不实现RelayVSR专用Ulysses/窗口注意力kernel，不改全局注意力/精度。
- 默认FP32 Flash；FP16仅列后续实验，不能混入FP32对照而隐藏变量。
- 本轮不自动接管Director VSR tail，不修改其工作流或缓存。先交付独立RelayVSR工作流；
  新节点可借用明确传入的现有Ray worker，但绝不抢占/杀死无关worker。

## 已有实证
官方单卡480×864/240帧/24fps/x2：
工作流183.482s，采样14.314s，Flash133.272s，保存27.177s；
整卡峰值11249MiB。输入图像、seed42、1步CFG1/euler/simple、detail1、CRF16固定。
纯FSDP输出240帧像素一致，但采样23.220s、总229.672s、两卡各16135MiB。
FSDP+Ulysses错误来自WAN21父类误匹配；不是硬件/NVLink故障。
这些结果在.local/artifacts/relayvsr-single-v100-20261124与relayvsr-dual-v100-20261124。

## 方案比较
1. 继续通用Wan USP：必须重写Relay专用lq投影、因果窗口、INT8 KV缓存；
   且只能覆盖约8%的采样时间，优先级最低。
2. 两进程各解码完整视频：无工作量分摊，不接受。
3. 选择：按官方独立Flash窗口分配到两GPU。官方每个窗口重置history，
   仅共享端点条件；因此可保持窗口内原版计算，合并时去重。
   窗口间无需GPU collective：采用本机Ray共享内存传CPU输入/输出，
   GPU计算均在对应V100；不是Gloo经CPU替代GPU计算。NVLink采样能力不被移除。

## 数学和边界
官方keyframes=range(0,F-1,15)+[F-1]；F>=1。
每个窗口接收左右端点latent、闭区间原图与detail，调用原版vae.decode(vae_options含lq_images)。
窗口0输出全部帧；其余窗口仍计算首帧建立history，但合并时丢弃首帧。
240帧分16个窗口：0..15、15..30、...、225..239，最后输出226..239。
F=1只用一个latent，F=16一个窗口，F=17尾窗口15..16，F=31两个完整窗口。
不按任意秒切片，不改变RoPE、缓存量化、重叠权重或插值方式。
只支持单视频batch=1的B,C,T,H,W latent和F,H,W,3图像；
显式验证latent数、空间尺寸、模型x2/x4与参考图一致，不静默裁剪或降采样。
不支持的原模型继续原路径；新节点无导入期副作用，可被旧ComfyUI加载但执行时清晰报缺能力。

## 模块/API
- relayvsr/planning.py：不可变RelayWindow及plan_relay_windows(frame_count, interval=15)；
  merge_window_result(output,written,job,frames)校验并写入指定CPU片段。
- relayvsr/worker.py：RelayFlashDecoder(vae_path,dtype='fp32')缓存官方小VAE；
  decode_window(latents,images,detail)返回CPU F,H,W,3 tensor。
- relayvsr/runtime.py：run_parallel_flash(vae_path,samples,images,detail,worker_count=2,
  keep_workers=False,borrowed_workers=(),interrupt_check=None)；
  最多一个在途任务/worker，完成即复制并释放引用，不累积全部窗口结果。
- relayvsr/nodes.py：RayRelayVSRDecode，显示RelayVSR Flash Decode (Ray)；
  inputs samples/images/vae_name/detail/worker_count(1或2)/keep_workers，
  optional ray_actors；return IMAGE。替换VSRDecode与VAELoader，不再加多层代理。
- distributed_worker/ray_worker.py：可选借用worker的方法，独立VSR缓存，不覆盖现有H3 VAE。
- distributed_modules/usp.py：RelayVSR在通用Wan规则前清晰拒绝不兼容USP，并指导使用新节点。
- 根__init__.py：只增加新节点映射；旧节点ID与行为不变。

## 资源与生命周期
没有Ray集群时，专用轻量actor每卡1GPU，num_cpus=2，object store按任务估算：
round_up_256MiB(max(1GiB,2*(最大窗口输入字节+输出字节)+256MiB))。
FP32输出字节=窗口帧数*输入H*输入W*倍率平方*3*4；
480p/x4单窗口已经约1.19GiB，不能硬塞1GiB的对象存储。
自建存储上限min(8GiB,可用系统RAM的25%)，不足则提交前清晰报错；
既有共享集群容量不足也报错，不偷偷溢写磁盘、不销毁别人的集群扩容。
禁用dashboard，不进行无关FSDP/USP/P2P健康初始化。CPU数据通道由Ray本机共享内存承担。
已有集群但GPU被占且无借用payload时立即报资源冲突，不能无限排队或抢占。
传入现有worker时只借用它们，不销毁其扩散模型CPU缓存或整个集群。
默认结束释放自有actor/自建Ray集群；keep_workers显式开启才保留自有池。
异常/取消销毁自有worker；借用worker只释放本模块VAE，取消中保留并等待已发窗口退出，
不得假称立即中断长CUDA kernel，也不杀死外部worker。
所有创建的GPU资源必须有明确owner，不能在清理时全局ray stop。
完成帧缓冲属于ComfyUI正常IMAGE缓存，本节点不能承诺在返回后清空它。
不做OOM后静默改精度/缩小输入/回退单卡；失败必须可观察。

## 验收（不可在实现后悄悄放宽）
- TDD覆盖窗口边界、乱序返回、重复/缺失/错误帧、模型倍率不匹配、取消与失败清理。
- 用真实官方RelayFlash的缩小CPU模型，x2/x4、1/16/17/31帧验证窗口合并与单次decode一致。
- 真实480p/31帧FP32原始像素：max_abs<=1e-5，RMSE<=1e-6；优先逐像素一致。
- 完整480p/10s/x2：240帧/24fps/960×1728完整解码；
  相对单卡成品SSIM>=0.9999、PSNR>=50dB（这是同配置一致性，不是无真值画质分）。
  单独检查15/30/...窗口接缝与尾帧；无新增黑帧、重复、漏帧。
- 双GPU必须分别处理非空且不重叠的输出窗口，记录任务ID、物理GPU与每窗口耗时。
- FP32 Flash节点总耗时（包括自有池冷启动、模型加载与结果合并）<=88.848s；
  工作流总耗时<=146.786s（相对183.482s降低至少20%）。
- 冷/热进程条件分开；至少2次完整2x运行，不能从一次波动声称稳定加速。
  所有计时工作流使用--cache-none，保持同seed/输入并确认每次输入/采样/解码/保存节点均有
  新执行事件、每卡窗口日志；缓存命中不能算热池加速。测默认释放池的冷运行，
  另测keep_workers=true的冷/热池对照，清楚区分。
  重新跑同cache-none条件的单卡对照；门槛取历史与配对对照中更快者，不能靠较慢基线放宽门槛。
- 整卡GPU0/1每0.5s采样，后端树工作集/私有提交、系统RAM/提交/分页同时记录；
  全阶段物理峰值，不只报告Torch active。初始资源预算每卡<=12GiB、系统RAM<=40GiB，
  超预算须解释并优化，不能用“权重分片较小”替代峰值。
- x4短视频/尾窗口回归；若2x门槛通过再验证已测试的1MP/x2和x4输入，不需先重复全部长x4基线。
  小尺寸x4通过不能声称原尺寸x4已通过。对象存储容量必须按真实输出尺寸计算并先验准入。
- 运行裸pytest全套并逐名记录失败；旧ComfyUI缺RelayVSR不能造成旧节点导入失败。
- 无性能达标不得标记加速功能验收通过。只保留隔离部署，不自动部署正式目录、不推GitHub。

## English summary
Use exact official Flash temporal windows on two V100 actors, preserve warm-up overlap frames,
drop only duplicate output endpoints, bound in-flight work and explicitly own lifecycle.
Keep native single-GPU keyframe sampling. No upstream/core edits, no global attention changes.
FP32 remains the baseline. Acceptance requires numerical continuity, complete frames, actual dual-GPU
work distribution, at least 1.5x Flash-node throughput and 20% end-to-end latency reduction.

## 实测后修订：同进程双GPU执行器

首个修正传输后的完整Ray冷运行：Flash108.537s、全流程149.240s，均未达到门槛。
独立窗口计算成立，但Windows创建进程、导入核心和CPU结果序列化不应成为默认必付成本。
增加 `execution_mode=local_threads|ray_actors`，默认local_threads：同一进程两个GPU执行器，
每个执行器独占一张卡和一个官方FP32 Flash模型，仍使用同一个窗口计划和有界合并器。
显式设置VAE.device；加载后只调用其原版RelayFlash.decode，不在并发循环中触发全局模型调度器。
该分支仅在验证RelayFlash类型、process_output为其原生identity语义后启用，绝不用于普通VAE。
不是把解码交给CPU，也不改ComfyUI核心。保留Ray模式用于显式借用或跨进程对照。
local_threads拒绝借用ray_actors，避免静默忽略用户连接。keep_workers在本地模式保留GPU decoder池。
增加真实小型官方CPU Flash数学等价测试与真实双V100 raw像素/性能测试；验收门槛不变。
所有失败Ray结果保留，若本地执行器也未达标，仍按实验记录，不能标记加速验收通过。

## 审查后交付收敛（优先于前文的Ray借用/双路由接口）

同进程三轮完整测试已成功，Flash约71秒；Ray冷108秒未达到同一门槛。
独立审查发现共享集群的driver/session/locality/容量所有权边界不适合作为本轮附带功能交付。
因此本轮公开节点只交付local_threads，不开放Ray actor后端或借用既有worker。
跨进程原型、原测试和提交保留在历史/本地产物中，不把未完成的安全边界留给用户。
新解码器不创建、不销毁任何Ray集群，检测到本进程已连接Ray时直接报错；包括暖池复用时重新检查。
用户需先用现有Ray生命周期节点结束原Ray会话，再运行本轮独立RelayVSR工作流。
采用一个操作锁覆盖准入、Comfy模型卸载、池变更、调度和回收；拒绝并发时不得改全局状态。
显式detach/cleanup自有VAE patcher，解除模型循环引用后释放显存；不能依赖稍后的GC。
消费后立即解除完成future和结果引用；暖池trim失败必须失效并回收该池，诊断记失败。
直接Flash解码仍执行该VAE的process_output，避免假定未来核心永远identity。
这不是新增普适性研究；是缩小交付面以满足双V100的安全、性能、可维护性目标。

