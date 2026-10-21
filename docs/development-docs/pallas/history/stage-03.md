# 历史：9.3 内部垂直 slice 实施（2026-10-18）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.3 内部垂直 slice 实施（2026-10-18）

用户已要求开始实现。本节是当前代码状态，前述实验/讨论记录保留其历史时点含义；
slice 通过不代表 M1、通信或整个目标完成。

| 实现 | 当前行为 |
| --- | --- |
| [core/model.py](../../../../src/jax_reorder/core/model.py) | 不可变 Program/Region/Operation/Value/Ref/effect、来源和独立 SourceHandle；视图内没有 JAX 或 SymPy 对象。 |
| [core/adapters/jax.py](../../../../src/jax_reorder/core/adapters/jax.py)、[Pallas call](../../../../src/jax_reorder/core/adapters/pallas/call.py) | 锁定版本和 Primitive 身份，校验 SSA、参数 schema、抽象值、上下文、Ref effects 与区域绑定；完整适配后才交给消费者。 |
| [core/index_domain.py](../../../../src/jax_reorder/core/index_domain.py) | 精确整数表达式、逐操作溢出检查、blocked 正向/反向双射及逆映射证明；生产路径不枚举程序或元素。 |
| [core/memory.py](../../../../src/jax_reorder/core/memory.py) | 输入只读、输出未初始化、swap 旧值、当前存储版本；尾块读写 guard 和有效域完整覆盖；共享纯值 DAG 不含 CAS 对象。 |
| [core/view_graph.py](../../../../src/jax_reorder/core/view_graph.py) | 内部结构快路径和原视图复核；目前保留源方程顺序，不替代旧 JAX BFS/歧义搜索。 |
| [core/normalization.py](../../../../src/jax_reorder/core/normalization.py) | `compare_program_views` 先结构匹配，再重分块规则与终端纯值匹配；规则证据可重放。公式不同通常抛数学未决，仅输入无关整数输出的精确坐标反例可判不等价。 |

已验证的正例包括 `x + 1`、组合逐元素非线性计算的 128/256 分块与反向 grid、非整除尾块，
以及写入一个输出后读取该新版本并生成另一个输出。mask 在所有有效输出坐标为真、在越界尾部为假
时才可消除；`other` 保留在源事实中并检查操作数，不能用补零代替输出初始化。
纯值投影只改写地址与 Ref 状态，保持浮点操作和操作数顺序；目前没有浮点交换/结合规则。

初始 slice 接受直接或默认 JIT 包装下的单个 `interpret=False` Pallas 调用、一维静态 blocked grid、同形输入输出、
块内整片或 lane 索引、默认 Ref 内存空间/布局/分片/上下文。不同读取映射即使覆盖同一集合也不会直接
判等。默认 JIT 的全部参数、区域实参/结果绑定和调用边界进入视图及规则证据；非默认 JIT 明确拒绝。
后续扩展已保留 alias/CompilerParams 源事实，并支持有限 shard_map 与单 program 本地 DMA/scratch
（§9.4–9.5）。循环/atomic、一般索引/部分切片、不完整 mask、未知字段/原语/effect 仍明确拒绝；
这些都是后续覆盖缺口，未缩减 §1 的目标。
结构自比较不承诺存储初始化和无竞争，只有规范化路径具有对应的域/存储证明。

源适配和规则实现是共同信任基础。原视图对应复核独立于 matcher；规则重放共享规则实现，不是另一套
形式化证明器。图不匹配不能推出数学不等价。`compare_functions` / `compare_algebraic_jaxprs` 仍按原先
行为拒绝 Pallas，接入留在 M3；不增加公开 Pallas API。

验证入口：

- [test_pallas_semantic_slice.py](../../../../tests/test_pallas_semantic_slice.py)：§8.2 的六个命名验收、近似反例、
  证据篡改、同名假 Primitive、完整图读取和 DAG 规模回归。大形状 `2**24 + 13` 只用 ShapeDtypeStruct；
  比较期间禁止 `eval_jaxpr` 且分析预算上限 256。
- [pallas_domain_oracle.py](../../../../tests/pallas_domain_oracle.py)：独立枚举小域；覆盖/漏写 1,280 个案例；
  pytest 另将生产逆映射证明与 oracle 对照。生产代码不导入 oracle。
- [adapter_inventory.py](../../../../tests/adapter_inventory.py)：Pallas call、grid、block、Ref、indexer 和读写参数
  纳入原有锁定版本快照；未知版本/字段仍须重新审查。
- CPU interpret 对完整块和尾块做独立数值对照；这是有限样本证据，未提供 TPU 编译或执行证据。

初始 slice 验证：Pallas slice 与包分层检查合计 **57 passed**，较早的结构/语义/对应关系组合回归 **382 passed**；
Ruff、格式检查和 basedpyright 通过。
后续扩展和全量回归的命令、失败及结果以 active 计划为准。
