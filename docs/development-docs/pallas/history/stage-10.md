# 历史：9.10 远端协议到全局值：实现规则（2026-10-19）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.10 远端协议到全局值：实现规则（2026-10-19）

本节推进 9.9 的远端缺口，执行记录见[远端 DMA 计划](../../../exec-plans/completed/remote-dma-normalization.md)。
实现限定一个 concrete mesh 轴、每设备一个完整向量块、静态设备分支及路由、单代全量 DMA 等待。
旧协议证据只证明进展与冲突次序；新的规范化证据必须额外保存设备存储版本、源快照及全局分片映射。

1. 从源程序顺序、发送/接收完成、信号量计数生成依赖。对每个交叠且至少一方写的区间，
   必须证明前一访问结束先于后一访问开始。无序则拒绝，不能选择某个“恰好安全”的拓扑顺序或补边。
2. 按已验证的事件拓扑解释值。start 读取已初始化源区间，保存逻辑值和版本；源在 send_complete 前
   受上述冲突证明保护。send_complete 核对版本；receive_complete 才写入目标新版本。
   中继必须先接收再发起下一次传输。wait 仍消费数量，不猜测一次 wait 对应哪一次 start。
3. 每个存储由设备、根 Ref 身份及区间定位，区域实参别名解析回根 Ref。输入只读；scratch 生命周期
   限于调用、初值未定义，允许已证明有序的复用。每个输出区间必须完整初始化且只有一个写入者。
4. 输入分片映射到全局输入槽位和坐标；输出按真实 shard_map 端口映射拼接。声明复制的输出必须逐设备
   证明值相同，不能依赖 check_vma=False。元素不枚举，工作量随设备、操作和区间边界增长。
5. 终端值图和完整证据可从源视图重放。公共 CAS 只接收完成组装的图；远端进展平台假设随结果传递。

反例包括：在途改写源（包括写回旧值）、发送完成后提前读目标、无序重叠目标写、未接收就中继、
计数相同但路由/切片不同、缺少输出分片、未证明的复制输出，以及生命周期或证据被篡改。
测试分开记录规范化比较、官方 CPU interpreter 的 eager/on_wait 对照、规模预算与未验证的 TPU 硬件条件。

上游调研依据（不是本库正确性证明）：[IREE Stream 资源 SSA/timepoint](https://iree.dev/reference/mlir-dialects/Stream/)
提供资源版本和可见性建模参考；[AIR 依赖分析](https://github.com/Xilinx/mlir-air/blob/main/mlir/lib/Transform/AIRDependency.cpp)
处理 RAW/WAR/WAW，但它从顺序程序生成依赖的编译行为不能用于修复待验证异步程序；
[TVM 固定提交的流水线实现](https://github.com/apache/tvm/blob/bdcfa01eae3ffe8c6d39aa26d0d1e5b311d47efb/src/tir/transforms/inject_software_pipeline.cc)
区分物理槽位与逻辑版本，commit/wait 数量不表示单次拷贝配对；
[MLIR Async 20.1.8](https://github.com/llvm/llvm-project/blob/llvmorg-20.1.8/mlir/include/mlir/Dialect/Async/IR/AsyncOps.td)
区分异步 token 和结果就绪。IREE/AIR 引用为本次核查时的 main，未作为固定依赖引入。

实现位于 [remote_memory.py](../../../../src/jax_reorder/core/remote_memory.py)，规则名
`remote-memory-to-global-values:v1`。每个设备的值解释器在访问/通信命令之前暂停，
仅由协议事件推进；因此纯运算可以提前解释，Ref 读取不会跨过尚未完成的 wait。
`RewriteEvidence.remote` 保存协议、区间版本、传输快照、端口组装及已履行的协议外义务；
重放逐字段核对，同时用共享 DAG 遍历比较值，避免把共享表达式展开成指数树。

已验证等待换序、请求—回复、三设备直接/中继、同信号量多个不交叠 producer 的全量等待、
发送完成后的 scratch 源复用、普通连续切片读写、分区/复制端口、多输入/多输出和条件区域绑定。
128 与 `2**24` 元素块使用相同分析预算，生产路径不逐元素分配。
公共数学入口可比较远端交换、请求—回复、切片交换与复用算术的完整 JAX 参考；
普通 JAX 必须在相同 mesh 下追踪，使输入输出抽象契约一致，不会删去分片属性来强行比较。

仍有明确边界：每设备一个程序、一个 concrete mesh 轴、默认执行策略、静态路由/设备分支、
DMA 信号量单代一次全量消费。普通信号量、多代/部分远端等待、混合本地/远端协议、
外部 alias、多维/动态控制流须新增规则。浮点 iota 没有投影规则；整数 iota 后转浮点的
分段坐标例可以完成 core 投影及 CPU 对照，但现有 CAS 无法建立原始转换条件，仍报
`condition_undecided`，未弱化条件。本轮没有真实 TPU、多主机执行证据。
