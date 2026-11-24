# RelayVSR 双 V100 Flash 加速实现计划

最终状态（2026-11-24）：窗口规划、显式节点、同进程双GPU解码、隔离验收已完成。
以下Task2/Task3的Ray actor与借用接口是保留的初稿实施历史，已按末尾审查修订撤回，不属于交付功能。
最终范围以spec末节与[验收报告](../../reports/2026-11-24-relayvsr-dual-flash.zh-CN.md)为准。
正式部署和远端推送未执行；新版核心既有14项H3失败不能视为整个正式环境已通过升级验收。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 只修改本Raylight，让RelayVSR Flash解码实际分工到双V100并获得可验证的端到端加速。
**Architecture:** 保留官方单卡关键帧采样，按官方Flash窗口调度两个GPU；
新增显式解码节点与轻量池，必要时借用明确传入的Ray worker，不改ComfyUI源码。
**Tech Stack:** Python3.12 / Torch2.12 cu126 / Ray / Windows TCC / 原生ComfyUI RelayVSR。
**Spec:** ../specs/2026-11-24-relayvsr-dual-flash-design.md

## Global Constraints
- 唯一工作树E:/ComfyUI-py310/raylight-windows-v100-fsdp-p2p；不创建另一开发树。
- 只部署.local/environments/relayvsr-v100/ComfyUI/custom_nodes/raylight，不动正式环境。
- 保持FP32 Flash、原输入/seed/步数/倍率/帧数；任何实验精度变化单独记录。
- 计划和测试结果留在本树；不上传模型、素材、视频、凭据，不推送GitHub。
- 原Raylight节点、H3/Director/LTX路径保持原功能；新节点默认显式使用、非全局拦截。

## Review Focus
1. 共享端点被过早删除导致下一窗口history错误：真实Flash模型1/16/17/31帧对照。
2. 异步结果到达乱序或尾帧缺失：CPU合并器重复/缺失/越界结果测试。
3. 原模型x2/x4同形权重配错：metadata/latent/image形状验证测试，失败不得自动改输入。
4. 外部Ray集群占用GPU、取消或worker异常：所有权/失败路径测试，不能抢占或保留孤儿。
5. 旧ComfyUI没有RelayVSR类：节点导入安全且普通Wan/H3注册不变。

## Task 1: 窗口计划与数值契约
**Files:** Create src/raylight/relayvsr/{__init__,planning}.py；
tests/test_relayvsr_windows.py。
**Interfaces:** plan_relay_windows(frame_count:int,interval:int=15)->tuple[RelayWindow,...]；
merge_window_result(output,written,job,frames)->None。
RelayWindow fields index/pixel_start/pixel_stop/latent_start/latent_stop/discard_prefix；
输出范围为pixel_start+discard_prefix到pixel_stop。
- [ ] 先写字面量边界预期：1/16/17/31/240；240共16窗口，输出0..239恰好一次。
- [ ] 运行新测试，确认缺少实现而失败。
- [ ] 最小实现planner与CPU合并器；拒绝F<=0、重复写、形状/范围不符。
- [ ] 真实缩小RelayFlash CPU模型测试x2/x4、1/16/17/31帧：
  原decode与分窗decode+合并逐像素一致，测试不得使用被测planner推导预期帧号。
- [ ] 运行任务测试与全套pytest；保存结果，提交本地实现。

## Task 2: GPU窗口decoder与显式worker所有权
**Files:** Create src/raylight/relayvsr/{worker,runtime}.py；
Modify src/raylight/distributed_worker/ray_worker.py（仅增加可选方法）；
tests/test_relayvsr_runtime.py。
**Interfaces:** RelayFlashDecoder(vae_path,dtype='fp32').decode_window(latents,images,detail)；
run_parallel_flash(vae_path,samples,images,detail,worker_count=2,keep_workers=False,
borrowed_workers=(),interrupt_check=None)->torch.Tensor。
- [ ] 先测试错误倍率/latent帧数、乱序合并、任务失败不返回部分结果、可取消、释放所属资源；
  480p/x4的16帧输出>1GiB时必须扩大自有store或预先报错，不可提交后无限等待。
- [ ] 运行RED；再实现单视频/batch1验证、原版VAE调用和CPU返回。
- [ ] 实现专用Ray actor，每卡1GPU，最多一项在途任务/worker；只传当前窗口。
- [ ] 自建池num_cpus=2；store按2个窗口输入/输出+256MiB估算、256MiB进位、最小1GiB，
  上限min(8GiB,可用RAM的25%)；共享集群容量不足先报错。独立临时目录，隔离runtime_env。
- [ ] 外部集群GPU冲突立即报错；借用worker只使用独立VSR缓存、不销毁其他模型/集群。
- [ ] 默认释放自有actor/集群；keep_workers明确才保留；失败/取消路径有测试。
- [ ] 跑GREEN、完整pytest，本地提交；不得改ComfyUI或comfy-kitchen。

## Task 3: 工作流节点与兼容性隔离
**Files:** Create src/raylight/relayvsr/nodes.py；
Modify 根__init__.py、src/raylight/distributed_modules/usp.py；
tests/test_relayvsr_nodes.py。
**Interfaces:** RayRelayVSRDecode输入samples/images/vae_name/detail/worker_count/keep_workers，
optional ray_actors；return IMAGE。旧node ID不变。
- [ ] RED测试节点参数、旧ComfyUI缺Relay能力可导入、只拒绝RelayVSR的通用USP误匹配。
- [ ] 新节点替换VAELoader+VSRDecode，不修改官方Sampler；保留单worker对照模式。
- [ ] 给RelayVSR误用USP增加清晰错误，不重写或伪装其forward适配完成。
- [ ] 验证普通Wan/H3/LTX无新增拦截；跑完整pytest并本地提交。

## Task 4: 隔离实机验收与文档
**Files:** .local/artifacts/relayvsr-flash-dev-20261124/（全部原始数据/视频）；
example_workflows/RelayVSR_DualV100_Raylight.json；
docs/reports/2026-11-24-relayvsr-dual-flash.{zh-CN,en}.md；
README.md/README.zh-CN.md的实验功能说明与CHANGELOG。
- [ ] 先提交已测试代码，再用deploy-to-comfyui.ps1部署到隔离ComfyUI；保留旧隔离包备份。
- [ ] 原480p/31帧真实模型比较raw像素，检查1e-5 max_abs/1e-6 RMSE。
- [ ] 用ComfyUI工作流（非绕过节点脚本）测试480p/240帧/x2；
  --cache-none+固定输入/seed，确认每次采样/解码/保存均重新执行；
  测默认释放池冷运行、keep_workers冷/热池，至少2次完整双卡运行。
- [ ] 重跑同cache-none的单卡对照，阈值使用历史与配对对照较快者，防止缓存或基线选择虚报收益。
- [ ] Flash节点包括冷启动<=88.848s，工作流<=146.786s；未通过则只记实验，不降标准。
- [ ] 每0.5秒记录两卡物理显存/负载/功耗及进程树和系统内存；任务日志证明两卡分工。
- [ ] 输出240帧/24fps/960×1728全解码，单卡成品SSIM>=0.9999、PSNR>=50dB及接缝检查。
- [ ] x4短输入和不完整尾窗口回归；记录未运行的长x4场景，不伪称全矩阵通过。
- [ ] 裸pytest全套、代码审查、旧节点导入回归；所有失败逐项记录。
- [ ] 更新双语报告/README/变更记录，提交本地；正式部署和GitHub同步留待用户后续指令。

## 执行说明

Task4实测修订（先于开发）：Ray冷运行Flash108.537s/全流程149.240s未达标。
补充同进程双GPU decoder池作为默认，Ray actor作为显式选项；窗口、FP32、原模型数学不变。
先写真实小型Flash对照和节点路由RED测试；实现独占device、不调用并发全局调度的本地执行器；
再重复31帧raw和完整240帧冷/热测试。不能降低验收门槛或覆盖Ray失败数据。
详见spec的“实测后修订”。这是Task4的证据驱动迭代，不是另开阶段或修改正式环境。

独立审查后收敛：只交付已达性能门槛的local_threads；Ray原型归档，不开放共享集群或借用。
按spec末节修复操作准入锁、自有模型循环引用、完成future寿命、暖池失败失效和process_output语义。
所有Ray实验数据/历史保留；最终回归和实机数据以收敛后的提交为准。

用户已指定计划后直接在隔离环境开发，采用主代理逐任务执行，不另派实现代理。
设计选择、测试失败、偏离与验收数据写入本计划专属ledger；
不将基线采样数据当作成功的优化数据，不删除此前实验附件。

