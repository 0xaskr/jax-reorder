# 代码导读与公共入口

开发入门、支持状态概览与后续路线图见[贡献者指南](../../../CONTRIBUTING.md)；本页提供代码与专题导读。

本页维护当前实现导读。截至 2026-10-26，源码 `93e93ca` 已接入 JAX 完整图比较、
函数/一阶求导契约、SymPy 代数比较，以及 Pallas 的结构、资源/协议和输出值比较。
公共接口以 [包导出](../../../src/jax_reorder/__init__.py) 为准；[比较契约](../../../ARCHITECTURE.md) 是架构维护入口，
其中尚未同步的旧接口说明记录为 [DOC-002](../../exec-plans/index.md#待处理)。

## 从输入选择入口

| 手里的输入与目标 | 公共入口 | 结论范围 |
| --- | --- | --- |
| Python 函数 | `compare_functions` | 核对 pytree、静态参数和环境；默认 forward、`equivalence="auto"`，结构不同后尝试数学比较 |
| 分别追踪后比较 | `trace_function` → `compare_function_traces` | 保存并核对同一模式的函数契约；JVP/VJP 按请求分别追踪 |
| 带常量绑定的 jaxpr，只查结构 | `compare_jaxprs` | 完整图、绑定与效果顺序；相同结果附原图对应证据 |
| 单图规范化 | `canonicalize` | 返回重排后的 jaxpr、规范结构及原变量/方程来源 |
| 带常量绑定的 jaxpr，查数学关系 | `compare_algebraic_jaxprs` | 先查结构，再对受支持的完整输出及定义条件调用 SymPy |
| 每侧一个 Pallas kernel，查局部结构/语义 | `compare_pallas_kernels` | 默认局部结构；`check_protocol=True` 加资源检查，`equivalence="semantic"` 比较已证明的值投影 |
| Pallas 设备局部数学比较 | `compare_algebraic_jaxprs(..., scope="device_local")` | 先证明资源义务，再查结构、规范化值或数学关系；函数入口的相同 scope 仅接受 forward |

`equivalent` 的依据由结果类型和 `equivalence_kind` 区分：`structural`、`normalized`、`algebraic`。
`different` 只表示结构不同；`inequivalent` 表示声明数学解释下存在完整输出差异。
失败通过 `ComparisonError` 子类及其 `stage/path/failure_kind/diagnostics` 报告，没有 `Status.UNKNOWN`。
代数浮点结论包含理想实数假设与警告，不保证逐位一致。详见[函数契约](function-and-derivative-contract.md)
和 [Pallas 契约](pallas/contracts.md)。

## 代码阅读路线

调用依赖从 `functions` 到 `algebra`、`core`；`core` 不反向导入上层。
`errors.py` 与 `status.py` 提供共享结果词汇；[包结构测试](../../../tests/test_package_layout.py)检查导入环、分层和公开接口。

| 代码位置 | 职责与下一步 |
| --- | --- |
| [functions.py](../../../src/jax_reorder/functions.py) | 追踪、契约检查、模式与 scope 分流、可选编译、结果汇总；从 `compare_functions` 看起 |
| [core/canonical.py](../../../src/jax_reorder/core/canonical.py) | SSA/版本检查、区域接口同步置换、输出 BFS、依赖排序与完整图比较；[unrooted.py](../../../src/jax_reorder/core/unrooted.py)处理无根图细化/搜索 |
| [core/semantics.py](../../../src/jax_reorder/core/semantics.py)、[params.py](../../../src/jax_reorder/core/params.py) | 精确编码常量/抽象值/参数，提供共享参数视图；[derivatives.py](../../../src/jax_reorder/core/derivatives.py)物化注册的一阶规则 |
| [core/correspondence.py](../../../src/jax_reorder/core/correspondence.py) | 构造并独立复核原图双射；复核共享语义编码，不等于独立证明整个适配器正确 |
| [core/adapters/](../../../src/jax_reorder/core/adapters/)、[model.py](../../../src/jax_reorder/core/model.py) | Pallas 结构描述符与语义 `ProgramView` 适配；普通 JAX 结构消费者仍直接操作 jaxpr，尚未全部迁移到统一视图 |
| [core/local.py](../../../src/jax_reorder/core/local.py)、[normalization.py](../../../src/jax_reorder/core/normalization.py) | 分别组织设备局部与完整函数的 Pallas 值投影；grid/存储/协议的细分路线见 [Pallas 导读](pallas-jax-equivalence.md) |
| [algebra/compare.py](../../../src/jax_reorder/algebra/compare.py)、[adapter/](../../../src/jax_reorder/algebra/adapter/) | 结构快路径、共同纯 JIT 身份、Pallas 投影和跨进程冻结；局部数学编排在 [local.py](../../../src/jax_reorder/algebra/local.py) |
| [algebra/symbolic.py](../../../src/jax_reorder/algebra/symbolic.py)、[expression.py](../../../src/jax_reorder/algebra/expression.py) | 构造带 shape/dtype、定义条件与内存有效性的逐输出公式；[trusted.py](../../../src/jax_reorder/algebra/trusted.py)组织完整输出查询和可终止 worker |

结构主线为 `compare_jaxprs → canonicalize → difference → build_correspondence → verify_correspondence`。
普通 JAX 数学主线为 `compare_algebraic_jaxprs → 结构比较 → CommonCalls/freeze → isolated_compare`；
worker 内部由 `SymbolicBuilder` 构造公式，`sympy_ops` 查询恒等关系，`witnesses` 寻找精确负例。
Pallas 在进入同一数学后端前，先由 core 证明地址、初始化、内存版本与通信，并生成 `MemoryProjection`。
完整函数的结构快路径不自动执行协议证明，局部数学路径则始终先检查资源义务。

## 设计与状态入口

设计主题与替代关系统一维护在[设计索引](../index.md)。当前支持、尚未完成的能力及主动排除项见
[贡献者指南](../../../CONTRIBUTING.md)，已复现缺陷见[复现与修复要求](../../exec-plans/active/pallas-semantic-layer.md#2026-10-26-审查确认的正确性缺陷)。
实施、实验与验收过程见[执行记录索引](../../exec-plans/index.md)；历史批次不能相加称为当前源码全量通过。
