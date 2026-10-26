# 历史：9.5 M2 本地 DMA：有限直线片段

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.5 M2 本地 DMA：有限直线片段

已实现无远端设备、无 alias 的整 Ref 直线式本地 DMA，显式建模 start 与 wait，适配阶段不配对事件。
当前分析限单个 program：scratch DMA semaphore 在调用入口的零初态及退出时归零作为可核对规则，
避免未经证明地把共享 scratch 当成每个 grid program 私有。本节记录首次整块实现；累计等待与本地
拆分合并已在 §9.6 扩展，多 program 与远端协议仍待实现，不能因本地片段通过宣称 M2 完成。

存储分析跟踪发起时的源版本、目的缓冲区和信号量；源被提前修改、目的被提前读写、等待缺少已发起传输、
退出仍有未完成 DMA 都应失败。完成等待后才能发布目标新版本，并记录传输量与信号量消费。对无依赖的
算术允许 start 提前/延后，终端纯值比较只在内存安全与有限进展前提完成之后进行。反例与官方 CPU
interpreter 对照分别报告；不能用一次模拟成功替代静态证明。

[communication.py](../../../../src/jax_reorder/core/communication.py) 跟踪每个信号量的一次未完成传输、
源版本、目标 Ref、传输范围、start/wait 来源和复用代次。等待消耗完整传输后，信号量可以再次使用。
不同信号量可同时传输到互不重叠的目的 Ref；共享只读源允许，等待完成顺序可以不同。
`logical_bytes` 仅记录逻辑数据量，不解释为特定硬件代际的信号量计数编码。
normalized 结果明确携带“运行时最终完成有效本地 DMA”的平台进展假设；缺少 start/wait、未初始化源、
提前读写与未消费计数是必须检查的程序条件，不能放入假设来绕过。
计数语义参见 [JAX TPU 分布式文档](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#dma-semaphores)；
锁定版本的 `mosaic/interpret/shared_memory.py::Semaphore` 使用零初态，wait 完整消费完成量。

同一规则已扩展到 call 生命周期的 VMEM 数组 scratch：入口未初始化，完整块写入后产生新版本，
允许串行覆写和多次复用；每次 DMA 的源版本须保持到完成等待。仍只允许单个 program，以免把调用级
scratch 隐含改成每个 grid program 私有；输出继续要求一次完整写入。验证覆盖直接复制与经 scratch
中转的相同值、等待后的合法复用，以及未等待就覆盖源或读取目标的反例。

VMEM 的 `MemoryRef` 构造器使用空的默认抽象 mesh，即使处于 shard_map 内也是如此；这不代表
多个设备共享同一临时缓冲区。视图同时保留原始 scratch 类型和绑定到当前内核局部 mesh 后的数组
类型。只允许默认空描述符或已经匹配的局部 mesh，显式指向其它 mesh 的 scratch 拒绝，不能直接
丢弃分片字段。该路径分别验证 `check_vma=True/False`，输出仍需通过实际输入依赖的复制性检查。

验证入口为 [test_pallas_dma.py](../../../../tests/test_pallas_dma.py)。官方 CPU interpreter 在 eager/on_wait
模式下对串行复制、独立计算重排与 scratch 复用做数值对照；适配器仍要求原生 `interpret=False`
源图，模拟模式不作为待比较的执行契约。符号测试另检查多个在途读取共享源时，一个 wait 不能解除
其它传输的保护；100 次 scratch 版本更新在预算 512 内比较，不进行调度搜索。

当前 Pallas/分片/DMA/包分层与既有结构语义组合回归 **454 passed**；Ruff、格式、类型检查、
adapter/lax inventory、运行示例和小域 oracle 通过。全量回归仍有 SymPy worker 10 秒超时，
不能报告全量通过；具体失败与命令见 [active 计划](../../active/pallas-semantic-layer.md)。
