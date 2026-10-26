# 历史：9.4 M1：分片与执行参数扩展

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.4 M1：分片与执行参数扩展

已增加类型化设备排列、mesh 轴、抽象分片和 `shard_map` 契约；原始/局部形状与输入输出 spec 的
对应在适配器内按锁定 JAX 规则校验。恢复手动轴环境后再适配子区域，保留 `check_vma` 与设备顺序。
`jit(shard_map(...))` 和 `shard_map(jit(...))` 均保留调用边界。不同 mesh 排列、不同分片绑定不能
只因局部输出公式相同而获得 normalized 结论。

Pallas 在 `check_vma=True` 时清除 kernel Ref 的 manual-axis variation；现在分别保留 BlockMapping
中的 Ref 类型、kernel 的形参类型及该绑定政策，不能要求两者原样相同，也不能把变化性整体丢弃。
对各 shard 独立的 kernel，根据纯值 DAG 的实际输入依赖证明输出在未装配轴上不变；关闭 `check_vma`
也不豁免此检查。分片输入产生不同输出却声明 replicated 的近似反例明确拒绝。

TPU `CompilerParams` 的全部 17 个字段纳入 schema 和视图。默认 grid 顺序按锁定 Mosaic lowering 的
`arbitrary` 语义记录。只有完成无跨 program 状态、完整覆盖及唯一写入证明后，才允许默认/显式
`arbitrary`/`parallel` 调度之间的跨分块规范化；其它数值、barrier、side-effect、资源策略保持契约，
非默认策略尚不应用跨分块规则。未知 flags 逐键审查前拒绝，core/subcore 调度仍需硬件绑定证明。
非空 input/output alias 已完整编码并可结构自比较，进入存储规范化时仍明确拒绝。

验证见 [test_pallas_sharding.py](../../../../tests/test_pallas_sharding.py)：两 CPU 设备的符号比较与独立 interpret
样本、设备换序/分片变化/错误复制声明反例、编译参数策略及证据篡改。当前与既有 slice、分层合计
**83 passed**；不作为 TPU 执行证据。
