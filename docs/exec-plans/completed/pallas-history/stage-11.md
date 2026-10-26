# 历史：9.11 通信计数与必要依赖固定点（2026-10-19）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.11 通信计数与必要依赖固定点（2026-10-19）

执行计划：[通信协议扩展](../communication-protocols.md)。本节扩展 9.10 的单代限制，
目标涵盖普通信号量、跨代复用、部分等待及混合 DMA 四项；沿用有限静态单轴/每设备单程序域。

上游 [TPU DMA semaphore 文档](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#dma-semaphores)
明确它记录异步传输进度，因此一笔 DMA 在完整完成之前就可能贡献部分计数。
锁定 JAX 提交 `32544801e26115ac1794926d027148abf3baf009` 的
[TPU lowering](https://github.com/jax-ml/jax/blob/32544801e26115ac1794926d027148abf3baf009/jax/_src/pallas/mosaic/lowering.py)
丢弃 `is_wait_send` 后降低为计数等待，不能把 wait 的源/目标 descriptor 当作 producer 身份。
普通 signal 只建立自己的程序事件依赖，不默认完成此前仅发起的 DMA。

每个 semaphore 实例的 wait 在所属设备的单一顺序流中执行。设此前已消费量为 C，当前等待 n，
其启用条件是累计产量达到 C+n；成功后累计消费变为 C+n。静态非负数量使有限事件的启用条件单调，
饱和可达固定点配合有效传输/信号最终推进的假设证明进展。不是选择一条安全调度来替代所有调度。

依赖固定点从程序顺序和发起/完整完成边开始，重复以下充分规则：

1. 只有已证明 wait 先于 producer 的 **发起**，才能排除该 producer 对本次 wait 的贡献。
   wait 先于完整 completion 不能排除 DMA 的部分进度；整个发送完成先于整个接收完成，
   也不意味着接收计数只能在整个发送完成后增加。
2. 若等待累计阈值恰好耗尽所有仍可能贡献的 producer 总量，则这些 producer 必须完整完成。
   部分 DMA wait 本身不识别某一 producer；必须保留未证明完成的源/目标保护。
3. 普通 signal 若缺少该项贡献就达不到阈值，可推出其发起先于 wait；这不表示此前 DMA 完成。
4. 新的必要边又能排除后续代次的 producer。带确认信号的复用因此可闭合；没有确认的未来代次
   可能提前贡献计数，不能按源代码相邻位置强行分代。

存储冲突仍要求每对交叠访问有确定的结束→开始关系。合法调度见证只用于版本求值；
信号量读取必须证明每次观察值确定，不能采用见证中碰巧读到的值。
数量/类型、单消费者、初始零与退出零、计数范围及源/目标绑定均纳入可重放证据。
`decrement=False` 被锁定 TPU lowering/interpreter 明确拒绝，不能把仅 Python API 可追踪当作后端支持。

必要反例：累计数相同但候选 producer 不同；部分等待后读某个目标；未发送完成便覆盖源；
只 signal 而未 DMA wait 就宣称目标就绪；未来代次冒充旧 credit；两条等待流竞争同一计数；
计数不足/残留/越界；依赖环；不确定 semaphore_read；混合计数类型；篡改阈值、边或观察值证据。

接入发现并修复的源顺序问题：锁定 JAX 的 `semaphore_read` abstract_eval 声明空 effects，
但它观察可变资源。原公共结构规范化可能把 wait 之后的 read 移到 wait 之前，导致实际
`3 + 0` 的两次观察在适配图里变成 `3 + 3`。源适配器现在把该 primitive 明确加入资源
顺序链；原图对应校验和代数前置规范化共同复用此事实。回归同时检查 canonicalize 保持
read/signal/wait 的原顺序，以及提前/延后观察不能被结构快路径判为相同。

实现与证据：`ProtocolEvidence.producers` 保存实例、发起和完整完成事件及贡献量；每次 wait
保存累计阈值、此前消费、被排除的未来贡献和必需依赖；`order` 是实际满足计数的合法执行，
`ancestors` 只包含必要因果，二者不互换。`observations` 保存已证明的 counter 观察值。
普通 counter 可保留确定的非零值；DMA counter 的硬件单位不公开，只接受稳定零观察。
存活计数上界按相邻消费区间检查，允许多代累计总量超过单次 counter 上限。

`remote-memory-to-global-values:v2` 支持有限单轴通信，`communication-memory-to-values:v1`
处理无 mesh 的单参与者普通信号量。混合 DMA 在相同 dtype、128 元素对齐、已审查 Ref
路径内合并计数，本地 DMA 只产生一次目标计数；远端 send/receive 标签不作为 wait 身份。
本地 DMA、远端 DMA 与远端 signal 的进展假设按实际操作分别附在公共结果上。

独立验证分为三层：小域机器逐单位增加 DMA/send/receive/signal 计数，检查所有可达交错
下的必要边、读数与进展；源测试核对完整投影、公共 CAS 及篡改证据的拒绝；官方 TPU
interpreter 用 `eager` / `on_wait` 核对数组输出及 race 报告。后者模拟整段进度，并在
[shared_memory.py](https://github.com/jax-ml/jax/blob/32544801e26115ac1794926d027148abf3baf009/jax/_src/pallas/mosaic/interpret/shared_memory.py)
注明超额信号的 happens-before 精度 TODO，因此不能替代独立的分块进度反例。

调研进一步参考 [IREE scheduling vs. causality](https://iree.dev/developers/design-docs/async-scheduling/#scheduling-vs-causality)：
已发生的调度不能自动升级为 wait 必需的因果历史。但 IREE 是单调 timeline，Pallas 是
消耗型 counter，不能照搬其 producer 身份。
[锁定 semaphore 源码](https://github.com/jax-ml/jax/blob/32544801e26115ac1794926d027148abf3baf009/jax/_src/pallas/primitives.py)
及 TPU lowering 是类型、target、decrement 与 read schema 的适配依据。

验证边界：本机没有 TPU。锁定版本 `pltpu.InterpretParams` 的 CPU 路径对 `semaphore_read`
缺少 MLIR lowering；本地/远端读取两类都固定为明确的上游拒绝测试，不改写源码或静默跳过。
signal/wait、复用、部分等待和混合 DMA 的官方对照正常执行；读取值由独立计数机器和
源规范化回归验证，未宣称其 TPU 或 CPU interpreter 执行已通过。
多轴/每设备多程序、动态控制流、由 read 决定路由、数组/变换 semaphore、自定义 barrier、
外部 alias 与一般多维布局仍不在本规则内；无法证明唯一计数观察或充分完成顺序时明确拒绝。
