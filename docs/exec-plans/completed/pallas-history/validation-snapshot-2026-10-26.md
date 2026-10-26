# Pallas 后端观察与阶段验收索引

> 状态：2026-10-26 归并时核对的历史证据，未重跑这些批次。后端限制按锁定版本记录；当前准入和证明义务见[验证规则](../../../design-docs/current/pallas/validation.md)。

## 已确认的源后端限制

锁定环境为 Python 3.12、JAX/jaxlib 0.11.2；近期 AOT 使用 libtpu 0.0.48.1、离线 `v5:2x2x1` megacore。
没有 TPU 实机精度/性能结论；用户确认图与编译工作无需实机。

- TPU lowering 与 TPU 专用 CPU interpreter 仍拒绝 `load_p(mask=...)`；通用 CPU Pallas 可做相应数值对照。
  不改写原源码绕过后，再声称该原源码编译成功。
- TPU CPU swap race checker 使用整个 view 的写足迹，未按 mask 裁剪；阶段 26 一个不相交 store/DMA
  样例仅关闭这项检测做数值对照，该次运行不能称为解释器无竞争证明。
- 交错串并轴的 CPU interpreter 有 prefix 限制；阶段 27 同源码 native AOT 成功，两个限制回归不计数值成功。
- 默认/IMMEDIATE 非连续重访、双缓冲 ANY、部分 Ref transpose 等源限制与本项目拒绝分别记录。
- 有界 cast 的 4 份 float16 输出样例通过 IR 校验，但当前 TPU 后端拒绝 `tpu.pack_subelements`；
  同批 float32/bfloat16 与闭合 cast 共 17 份编译成功，详见[验收](../pallas-bounded-integer-casts.md)。

## 最近验证与详细日志入口

| 阶段 | 准确摘要 | 打开原文的理由 |
| --- | --- | --- |
| [有界动态页 DMA](../pallas-bounded-page-dma.md) | 最终六批去重 721 项通过；真实动态页 attention 等价、倍 V、错 K 页数学结论均在默认预算内完成 | 逐笔在途 DMA、三输出 oracle、地址/反例安全检查与缓存边界 |
| [顺序输出映射与纯死计算](../pallas-output-map-and-pure-lowering.md) | 最终两片 438 项通过；独立多维/尾块/无普通输入全坐标对照通过 | 输出窗口绑定、覆盖与重访，活跃依赖保留，反序坐标修复及测试命令 |
| [设备局部等价收口](../pallas-local-equivalence-closure.md) | 冻结源码后四片去重 1,288 项全部通过；真实数学 6 例在默认预算/60 秒内完成；Ruff/格式/类型/tracing 通过 | 局部公开接口、完整 paged attention、求和域、条件初始化、历史失败与源码哈希 |
| [27：远端多维 grid](stage-27.md) | 分批去重 1,183 项；6 份 AOT；CPU 20 项中 18 数值/race + 2 源限制 | 两份协议、单 core 反例、绑定篡改、原始日志 |
| [26：尾块/mask](stage-26.md) | 分批去重 2,167 项；最终受影响 423 项 + 共享 core 368 项 | 初次 2 失败及修复、四缓冲预算、mask oracle、原生拒绝 |
| [25：并行 ANY 尾块](stage-25.md) | 1,975 不同 pytest ID；9 份 AOT | 72 次完整参考、有限域 13,966 点、owner 证明与篡改 |
| [24：ANY/远端原存储](stage-24.md) | 原 Ref、多维足迹与恢复的独立对照 | padding 生命周期、线性协议地址与真实 shape |

这些是当时的验证事实，不是 2026-10-26 文档整理重跑的测试；更早记录按[历史索引](index.md)定位。
原始工件多位于当时 `/tmp` 目录，路径保留供本机复查，但长期可复现范围以仓库源码/固定测试与记录命令为准。
