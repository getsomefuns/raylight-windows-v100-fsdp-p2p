# 1.19.0 — 官方核心兼容与 Director 双 V100 RelayVSR

状态：隔离验收与官方核心正式迁移通过；两条正式真实工作流、完整帧／音轨媒体检查通过，旧环境可恢复副本保留。

## 本次实现

- 固定官方ComfyUI `2fcb6ce`（0.46.0），前端1.59.11、kitchen0.2.42、aimdo0.6.0、模板0.11.118、文档0.5.17。保留Python3.12.10、Torch2.12.0+cu126、驱动577.00及现有Director `aff2579`+本地补丁，不手改ComfyUI或Director源码。
- 兼容新旧H3 PackedLayout、音视频去噪mask、逐token时间调制和patcher的`fast_disk`克隆参数；保留安全FP16与既有H3 FSDP/P2P能力。
- Ray MODEL支持的Director连接原生RelayVSR参数包后，关键帧采样仍走原生单卡，Flash独立窗口以原版FP32在双V100并行。不是给VSR模型做FSDP，也不是32GB显存池或Windows NCCL。
- 向已有H3 workers取得两张真实物理GPU的独占租约，严格卸载非活跃显存；Flash清理、分配器恢复验证后才归还。不销毁H3 CPU patcher／actor或Director缓存，默认释放Flash GPU模型池。
- 仅在VSR作用域临时移除原生`max_split_size_mb`，结束后恢复完整实时配置。Director覆盖VSR采样和Flash，独立节点仅覆盖Flash。不改H3策略、精度、环境变量和P2P128MiB预算。清理／恢复不可信时隔离进程，等待线程或换actor也不能继续，必须重启。
- TE3.3改为在保留mask的USP循环中调用其原生残差控制器，双rank一致同意后复用，不安装不兼容的旧整模型forward。无关原生TE及未连接TE的工作流不变。
- VSR最终缓存身份包含模型／VAE来源戳和可识别patch设置；选择运行时拒绝不兼容旧成片，不删除可用的一采AV缓存。作者检查接口不含完整VSR参数，因此界面明确显示最终匹配未验证，不误报命中。
- 加入取消／所有权边界、严格资源监测验收和可校验的新目录恢复工具。不创建第二源码工作树或长期后端服务。

## 实测与限制

完整案例、失败实验、各阶段时间与资源口径见[中文报告](../reports/2026-11-24-relayvsr-production-migration.zh-CN.md)／[English](../reports/2026-11-24-relayvsr-production-migration.en.md)。

- 现有Director Ref2VA／FL2VA4／8步、原非Director XFuser、TP2／原生CLIP、分段LoRA、SelfLift、缓存Refine、FaceRefine56／56帧、双卡H3 VAE、静音／正常音轨和真实实时预览均有隔离工作流记录。
- 同图同种子864×480、124帧、24fps、5.1667秒：旧／新冷启动319.887／310.911秒，热启动241.586／234.263秒。仅一组对照，不宣称普遍稳定提速。新旧H3像素不相同（SSIM0.784984），不承诺广义审美一致或无损。新核心冷启动主存较高，热态接近。
- 修复后480×864、240帧、24fps、10秒、2×三轮纯流程112.913／112.084／111.362秒，日志Flash71.371／71.357／71.278秒。相对已有双卡优化基线，总耗时中位数+1.50%，Flash基本不变，仍在原定10%门槛内。
- 审查后真实三段VSR／段间引导、变参采样前拒绝错误旧成片、同参只重跑第二段均通过。原生最短帧数／保留尾部最终输出141帧、24fps、5.875秒。冷态功能轮与备份复制重叠，不当成匹配性能比较。
- 完整480p／2×的240帧、1MP／2×和4×的120帧原始输出均为有限FP32，与独立官方同latent Flash逐像素一致。修复后1MP／4× Flash538.205秒、历史534.571秒，完整媒体检查通过。最终串行新／旧核心965／951通过、旧核心14项原生接口缺失，两侧35子测试；前端5项通过。包含新增的节点／包版本一致性测试：先复现1.18.5与1.19.0冲突，再统一1.19.0通过；没有改动采样／解码算法。
- 可恢复运行快照72,115文件，另有297文件用户状态备份。真正恢复出的旧Python／核心／全部节点及其真实Ray workers完成124帧正常视频，不是只有依赖清单或未验证复制。

原生参考诊断总耗时与内存还包括额外CPU／参考图像，不是纯生产吞吐。整卡NVML峰值与Torch allocated／reserved分开；系统commit不等于物理RAM或实际分页流量。

## 使用与迁移

仍用环境根目录`start_raylight_p2p.ps1`，导入[Director＋VSR界面样例](../../example_workflows/Minimax_H3_Director_DualV100_RelayVSR.json)。本次候选测试使用4GiB预留，启动器默认3GiB未改。x2／x4 UNet与FlashVAE必须配对。示例固定42、H3 Turbo4／四步、864×480／56帧24fps，VSR一步／euler／simple／CFG1／denoise1／detail1。模型放diffusion_models／vae。完整连线及独立超分用法见中英文README。

正式迁移只切换固定官方核心与十个精确离线配套wheel，未替换Torch、驱动、其他ComfyUI环境、Director补丁、用户工作流／媒体或分段缓存；仅向共享模型目录新增四份VSR权重，不覆盖已有文件。34项官方依赖及不可变622文件候选部署均已核验。前端升级后重启并强制刷新浏览器。

原启动器正式烟测：原Director864×480／124帧／24fps总353.221秒；公开56帧／24fps Director＋VSR图208.896秒，输出1728×960。两者均真实双rank采样、完整视频及对齐AAC；VSR图实际运行双卡Flash，未连接的旧图没有进入该路线。这两个不同长度的功能烟测不是速度对照或加速承诺。随后已关闭服务及workers；关闭前整卡空闲1138／1094MiB，并非保留Flash权重。详细阶段及资源见报告／附件。

保留`.local/backups/production-relayvsr-20261124`备份及`.local/environments/production-relayvsr-recovery-rehearsal`可启动旧环境。回滚先停止本次服务／workers，保留失败状态，再由副本恢复记录中的旧核心62b3c94、Python／依赖和部署节点c403c91。不要整体覆盖用户工作流、媒体或缓存；仅在必要时恢复变化的数据库／配置。原始独立VSR隔离环境也保留作手测参考。

## 明确不宣称的范围

只验收Windows、双16GB V100、TCC、NVLink。DynamicVRAM、PDD／多头Flash、任意第三方CUDA并发、多GPU数量、完整480p／10秒／4×及任意长片／高分辨率都不在本轮验收内。Pika仍未实机验收。LTX相关兼容测试保留，但本次没有重跑全套LTX GPU生成。未来核心／Director／TE版本仍需重新核验。
