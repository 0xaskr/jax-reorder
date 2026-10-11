# BFS 计算图规范化与比较

对应[执行计划](../exec-plans/completed/bfs-reordering.md)。第一轮实现与自审已完成。下文区分目标支持域、逐项验证证据和明确的后续工作。

本文保留第一轮验收时的范围和数字。后续已补齐完整无根图、函数契约、注册一阶求导与深层迭代；当前状态以[系统验证矩阵](verification-matrix.md)和[架构](../../ARCHITECTURE.md)为准，下文的“未实现”不代表这些功能的最新状态。

## 接口与边界

- `canonicalize(jaxpr)` 返回新 jaxpr、规范结构、BFS 与最终方程索引、深度及未完成部分。输入必须是 JAX 0.11.2 的有效 jaxpr，常量绑定随图传入。
- `compare_jaxprs(left, right)` 返回 `equivalent`、`different` 或 `unknown`，并包含差异路径或未完成原因。只比较追踪后的计算图，不判断一般 Python 函数或代数等价。
- 支持目标是 `jax.lax` 及其公开子模块在 CPU 上支持的原语，包括原语的全部合法 dtype、shape 和参数。测试的代表值不作为额外准入限制。
- 函数级 pytree、静态参数和执行环境需由调用方保持一致。jaxpr 已展平输入输出；本接口不能恢复已丢失的 Python 接口信息。

## 表示与算法

输入位置固定。每个结果值由 `(BFS 节点编号, 结果端口)` 引用，字面量保存类型与精确内容，捕获常量按首次引用编号并保存内容及共享关系。所有输出按原位置记录。算子名称不参与排序，也不合并重复节点。

从有序输出开始，FIFO 遍历各节点的有序操作数；节点在入队时编号。依赖深度沿合法方程列表计算，按 `(深度, BFS 编号)` 排列输出可达节点。未使用纯计算按原相对顺序保留在末尾，结果标记为不完整，不用输出子图相同宣称全图等价。

嵌套 jaxpr 在各自边界内规范化，保持分支、循环携带值和调用边界。cond 内部实参/形参另做同步置换，消除自动捕获顺序差异，见 [绑定规范化](cond-bindings.md)；外部函数接口固定。重建必要的参数容器，保留其他语义字段，不内联或展开。token 作为值依赖保留；隐式顺序语义须显式处理，未知 effects 明确失败。

参数与抽象值使用带类型的结构表示。数组比较完整 dtype、shape 和内容字节，区分有符号零与不同 NaN 表示；不比较打印摘要或哈希。JAX 的已知参数描述符保存完整语义字段；未知对象、回调或未审核字段不能凭 `repr` 相同匹配。调试源位置不属于计算语义；执行上下文属于比较内容。

方程链的 BFS、深度和比较表示均为迭代形式，不沿每条依赖递归展开。单图排序为 O(V log V + E)，常量内容与嵌套图另计。

## 验证与 review

实现测试检查合法顺序置换和变量重命名不变性、幂等性、连接与端口保留、常量内容、多输出、共享与重复计算、嵌套图及显式拒绝。预期来自独立的图规格与预先给出的 AI 判断，不能用待测 BFS 生成答案。

模型基准来自计划中固定版本的 SGLang-JAX。正例保留图，仅改变独立语句顺序；负例改变算子、参数或连接，实际计算不等价的例子提供 CPU 反例。生成与判断在开发阶段完成，CI 只重放固定案例。

最终逐原语记录覆盖、CPU 执行和比较结果；未实现语义、未验证原语和 review 缺陷均保留为未完成事项，不能通过缩小原语合法域消除。

## CPU 清单与 JAX 适配

[逐项清单](lax-coverage.md)登记 171 个公开原语、182 个固定原语及参数用例，以及实际追踪到的 11 个辅助原语。169 个公开原语有 CPU 执行证据；`stage` 被 JAX 自身消去，`ragged_all_to_all` 的 CPU 拒绝另有断言。`pbroadcast`、`psend`、`precv` 的 CPU 拒绝同样单独记录，不计为执行通过。

实现按照 Primitive 对象身份发现原语，名称不能冒充身份；dtype、shape 和参数值没有采样白名单。JAX 自身的抽象求值规则负责类型与参数合法性，CPU 编译/执行测试负责后端证据。`canonicalize` 本身不会编译图，不能据此证明某一调用可在 CPU 执行。

适配以下语义信息：

- ShapedArray 的 shape、dtype、weak_type、sharding、manual_axis_type、memory_space、layout；token 抽象值；忽略的 `_stripped_weak_type` 仅为派生缓存。
- 完整 EqnContext 已设置的槽位，包括 mesh、layout mode、计算类型与元数据。源位置、变量名和调试回溯不参与语义比较，但保留在返回图中。
- NumPy/JAX CPU 数组的 dtype、shape、内容字节；字面量按声明 dtype 解释。PRNG key 使用完整 KeyTy/PRNGImpl 身份和 key_data，避免 NumPy 转换丢失 key 语义。
- mesh、设备、分片、PartitionSpec、Layout、枚举、已知 JAX dataclass、FlatTree 和 pytree。pytree 辅助数据逐项编码，未知对象不能凭自定义 `__eq__` 获得匹配。
- `cond`、`scan`、`while`、`custom_linear_solve`、reducer/comparator、composite、jit、shard_map 等参数内的子图分别处理。shard_map 子图校验恢复其 named-axis 与 check_vma 环境。

两个经过源码核对的参数适配例外：

1. jit 的 `name` 是诊断标签，不参与计算；仅从比较键中排除，返回 jaxpr 保留原参数。donation、sharding、layout、compiler options、inline 等其他参数仍比较。
2. JAX **0.11.2** 的 `reduce.computation` 保存 Python callable，同时提供已追踪的 `jaxpr`。该版本的 lowering/JVP/batching 使用后者；比较以完整 reducer 子图表示 callable，返回图仍保留原 callable。该例外按版本限制，其他不透明 callable 明确拒绝。

这里使用了 JAX 私有 IR 和描述符接口，兼容性以锁定版本为准。升级 JAX 时须重跑清单并审查新字段和规则；不能把自动发现的 Primitive 身份视为全部语义已经通过审核。

## 顺序约束与拒绝路径

`rng_uniform` 即使 effects 为空也有隐式顺序要求，`dce_sink` 有 NoDCEEffect。实现将这些节点及包含它们的调用/控制流节点作为顺序根，并保存原相对顺序；保守地不交换这些根。NoDCEEffect 没有额外字段，named-axis effects 保存轴身份，token 作为普通值边。其他 effects 明确拒绝。

不属于 lax 包装范围的 `jax` program_order API 暂不支持，它的子图带额外调度含义，不能用普通子图排序处理。自定义原语、未适配的抽象值或不透明参数报告明确路径；这不改变 CPU lax 合法域的目标。遇到新的 CPU 合法调用而编码失败时应登记和修复，不能把测试集反过来当作准入清单。

未使用纯分支与常量绑定返回部分完成；其唯一标号属于后续任务。一般函数级 pytree 比较、持久化格式、跨 JAX 版本兼容及代数规范化也未实现。深层区域/参数仍受 Python 递归深度限制，超过时明确报错；普通长计算链不递归遍历。

## Review 与验证记录

2026-10-11 完成实现自审，核对了输入/输出、节点及端口身份、常量绑定、字段编码、嵌套边界、顺序依赖、错误路径与测试预期来源。修复了以下实际问题并保存回归检查：

| 审查问题 | 修复与证据 |
| --- | --- |
| 单独检查 shard_map 子图会丢失轴环境 | 在外层语义环境下校验；CPU 双设备集合通信通过 |
| jit 内 RNG 的外层调用可能被当作纯节点交换 | 传播顺序约束；交换两个调用被识别为图差异 |
| dtype.str 会合并不同 float8 dtype；普通标量比较会丢失 NaN/signed-zero 信息 | 保存精确 dtype 与字节；位模式和 dtype 反例通过 |
| reduce 的 Python callable 身份导致同一 reducer 无法比较 | 采用仅限 0.11.2 的子图适配；同义独立 callable 匹配，改动 reducer 不匹配 |
| 参数容器、子图可能形成环；pytree 辅助数据可能是不透明对象 | 在 JAX 递归类型检查前验证；显式返回错误或 unknown |
| 只追踪原语不足以证明 CPU 支持，某些 API 会被消去 | 检查目标 primitive 实际出现，再编译执行；消去与 CPU 拒绝单独报告 |
| 用例 oracle 可能重复被测遍历算法 | AI 先保存标签和数值反例；另用独立合法调度枚举及 CPU 执行复核 |

当前未发现需要在第一轮验收前修复的遗留缺陷。此结论限于源码审查、固定回归集与锁定环境，不等于穷举全部合法参数或组合。

实际检查：`uv sync --locked --offline`、Ruff lint/format、`uv run pytest`（516 通过）、两个可运行示例和原语清单一致性检查通过。离线同步使用了现有缓存；初次在线构建曾因网络 DNS 失败。CPU CI 已配置相同验证，但未推送或触发远程工作流。

[AI 案例与模型来源](bfs-cases.md)保存 18 个图等价/不等价判断、9 个具体数值反例和固定源版本。未使用子图等已声明的后续工作见[技术债](../exec-plans/tech-debt-tracker.md)。
