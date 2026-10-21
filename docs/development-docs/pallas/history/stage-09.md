# 历史：9.9 官方构造覆盖与公共代数接入（2026-10-19）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.9 官方构造覆盖与公共代数接入（2026-10-19）

按用户指定顺序，先完成覆盖专项再接入 CAS。本节更新 9.8 中“代数尚未接入”的历史状态。

[覆盖清单](../../../../tests/pallas_inventory.py)逐一分类锁定 JAX 的 119 个公开 callable 导出：
73 个有实际追踪用例，25 个是参数/类型声明，14 个是配置/设备工具，3 个 host helper，
3 个属于当前单 pallas_call 范围外的 MPMD API，1 个被上游前端拒绝。分类不能被解读成
119 个 API 均通过运行验证，更不能代表任意语法组合已覆盖。

[固定构造](../../../../tests/pallas_coverage_cases.py)含 40 类，每类均有实际 sin/cos 换序正例和
将减法改成加法的反例；另有动态 num_programs、4 类 BlockDim、捕获/索引变化与未知
HiPrim 拒绝。48 项原生 Pallas/state 参数 schema 均有实际追踪来源（包括官方 pipeline
lowering 生成的 pipeline_body）。其中 17 类另跑 CPU interpret，原图、正例、反例各执行一次。
Ref transpose 的 CPU discharge 不支持、Indirect 数组索引的前端拒绝分别固定为测试。
MXU ACC/conv 使用官方 abstract TPU mesh 元数据完成源码追踪，没有模拟硬件执行成功。

修复内容：run_state 身份/schema；精确 Einshape 高层原语参数；emit_pipeline/pipeline_body
的父操作数捕获绑定；显式 BufferedRef allocation 的索引函数语义；默认展开 pipeline 的
唯一返回类别排序。输入输出角色、Ref/信号量别名和完整副作用顺序仍保留。

公共 `compare_functions` 的默认 auto 路径和 `compare_algebraic_jaxprs` 现在支持：

- 一维静态 Blocked 的 128/256 分块、正反向网格、带正确 mask 的尾块；大域不逐元素枚举。
- 完整输出的加法、乘法、分配律等改写及非线性公式，并可和普通 JAX 或默认 JIT 比较。
- 多输入/多输出位置契约；局部 scratch/DMA 的完整版本和 wait 证据。
- `program_id` 作为输出数据时保留具体索引含义，不能因分块变化而误判等价。

结构不同仍返回 `different`；只有中间层投影完成并经隔离 SymPy 查询后才给出数学
`equivalent` / `inequivalent`。`AlgebraComparison.normalizations` 保存左右投影证据，
`backend_result.queries` 保存 CAS 查询；DMA 路径附带 progress 假设，所有数学改写保留
理想实数与浮点舍入警告。完整输出坐标上的反例才支持不等价结论。

分段 DMA 的 `select_n` 投影可与 JAX 拼接比较。定义域条件和输出值一样，只在合法
输出坐标内比较：例如长度 256 的拼接产生 `i < 256`，与完整投影的 `True` 在该域
等价。比较不同条件前对两侧同时加入输出坐标边界，不删除 `log` 等输入值条件；
部分函数与无条件输出的反例仍须拒绝，不能借坐标约束扩大输入定义域。

远程 DMA 和普通 semaphore 的缺口归属 **core 语义图及规范化层**。该层负责设备/Ref
绑定、传输区间、start/wait 与信号量依赖、源/目标存储版本和可见性，以及初始化、
生命周期、输出覆盖与跨设备组装。只有这些关系已经确定，才能生成交给代数层的完整
纯值表达式；SymPy 不负责推断通信拓扑、传输完成或读取的是哪个存储版本。

9.9 完成时，[remote.py](../../../../src/jax_reorder/core/remote.py) 与
[protocol.py](../../../../src/jax_reorder/core/protocol.py) 已有受限路由、计数、进展和冲突分析，
尚未与值版本传播接成完整远端输出规范化；后续接通结果见 9.10。此项属于语义层工作，
不能归为等待 CAS 解决的代数问题。原图结构比较
继续保留通信操作及拓扑，不以完成这些规范化证明作为结构准入条件。

多维/一般控制流投影、一般分片组装、远程协议扩展和非默认执行选项仍有未支持范围。
GLM5 absorbed 大图的歧义接口预算和真实 TPU/多主机编译执行仍是后续边界，不把它们
记作本轮已验证。

```bash
uv run python -m tests.pallas_inventory
uv run pytest -q tests/test_pallas_coverage.py tests/test_pallas_algebra.py
```

验证及已知边界见[本轮记录](../../../exec-plans/completed/pallas-coverage-and-algebra.md)。
