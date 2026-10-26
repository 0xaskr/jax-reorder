# Pallas 近期阶段证据摘要

> 状态：各阶段历史证据导航，不能相加为当前源码全量验收。当前设计见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。

## 最近证据

- 2026-10-22 的[设备局部等价收口](../pallas-local-equivalence-closure.md)、
  [顺序输出映射](../pallas-output-map-and-pure-lowering.md)及
  [有界动态页 DMA](../pallas-bounded-page-dma.md)已完成；准确批次与限制见[验证专题](../../../design-docs/current/pallas/validation.md)。
- [远端多维 grid](stage-27.md)：分批去重 1,183 项通过；6 份离线 TPU AOT。
  CPU worker 的 20 项中 18 项为数值/race 检查，2 项只确认解释器限制。
- [普通尾块与 mask](stage-26.md)：分批去重 2,167 项通过；最终受影响 423 项及共享核心 368 项复测。
  锁定 TPU 后端拒绝 masked load 的源码单列，未计为编译成功。
- [并行 ANY 尾块](stage-25.md)：独立有限 oracle、完整值参照、源证据篡改反例及 9 份 AOT。

这些是各阶段原始结果，不是压缩文档时重跑全量测试，也不能跨阶段相加成为新的覆盖总数。
