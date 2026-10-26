# 索引乘加与有限求和的代数比较

状态：当前生效的数学后端设计。代码路径与结果字段见“实现与结果接口”，支持边界见“当前边界”；
两节按 2026-10-26 源码核对。仍待实现的设计决定单独标明，不表示能力已经完成。
旧证明器及 CAS 候选方案已移至[设计归档](../archive/algebraic-equivalence.md)，
规模实验、迁移和历史验收见[执行记录](../../exec-plans/completed/history/algebraic-equivalence-validation.md)。

## 2026-10-19 决定：在代数 lowering 中删减未使用的纯计算

状态：**Pallas 数学投影已实现受限删减，普通 JAX 路径尚未完成**。Pallas 先检查完整源图及
内存/通信义务，再只把活跃输出值依赖交给数学构造器；未使用纯计算可以不进入 SymPy。
普通 JAX 的 `freeze → SymbolicBuilder.graph/finish` 仍转换死计算并收集其定义条件。
以下保留目标比较边界，不能据此把普通 JAX 的目标行为视为已经实现。

结构比较继续保留完整原图，包括未参与输出的纯计算和常量。代数比较只保留影响输出、
副作用及其依赖的计算；能够确认无副作用、且不影响这些根的纯计算可以忽略。
因此，两份图可以在结构阶段返回 `different`，再在数学阶段返回 `equivalent`，
并以 `equivalence_kind="algebraic"` 标明结论来源。

### 所在阶段与保留规则

- 删减放在代数 lowering 中、构造 SymPy 表达式之前；tracing、结构规范化与原图对应证据
  继续处理完整图，原始 jaxpr 不被修改。结构不同的诊断仍保留在函数模式报告中。
- 以有序输出和具有副作用或顺序约束的节点为根，反向保留全部数据依赖、控制依赖及必要绑定。
  其余节点只有在能够确认是无副作用的纯计算时才可删除；仅供这些节点使用的内部常量也可删减，
  外部输入输出契约不变。
- 嵌套调用和控制流中的副作用必须检查并保留，不能仅凭外层结果无人使用就删除整个区域。
  `rng_uniform` 等隐式有序操作及 `dce_sink` 等显式保留标记仍是保留根。
- 未知 effects、不透明语义或无法确认纯性的节点继续明确拒绝。删减不能绕过已有支持域检查，
  也不意味着数学后端新增了副作用支持；保留后仍无法解释的语义继续通过异常报告。
- 被删除的纯计算不参与符号构造，其定义条件也不附加到最终输出。仍参与输出或副作用的计算
  保留原有条件检查；例如返回 `log(x)` 或 `0 * log(x)` 时，不能用本规则消除 `x > 0` 条件。
  后者在源图中仍有输出依赖，属于代数消项问题，与未使用计算的删减分开处理。

### 示例与后续验证

以下两个普通 JAX 函数的原始图不同；按本决定应判为代数等价，但当前实现仍保留左侧定义条件：

```python
def left(x):
    jnp.log(x)  # 结果未使用，且无副作用。
    return jnp.zeros(2, dtype=jnp.float32)


def right(x):
    return jnp.zeros(2, dtype=jnp.float32)
```

目标行为是左侧未使用的 `log(x)` 不再引入 `x > 0` 的比较前提。纯死代码的定义条件被忽略是
本次明确选择的比较范围，不能据此声称两份完整计算图相同；普通 JAX 路径仍待实现和验证。

后续回归须覆盖：上述正例及空输出变体、未使用纯计算的依赖链、仍被输出使用的部分函数、
未使用结果的副作用节点及其生产者、嵌套区域中的副作用，以及未知 effects 的拒绝路径。
每组分别检查结构结论与代数结论，避免把代数删减结果当作原图结构证据。

Pallas 的实现与验收见[顺序输出映射与纯死计算](../../exec-plans/completed/pallas-output-map-and-pure-lowering.md)，
[语义切片回归](../../../tests/test_pallas_semantic_slice.py)分别检查死标量运算可删、活跃输出/地址依赖不可删，
以及结果未使用的未初始化读取仍拒绝。普通 JAX 的
[SymbolicBuilder](../../../src/jax_reorder/algebra/symbolic.py) `graph` / `finish` 后续仍需按本节调整。
下文的历史行为与验证记录按当时实现保留。

## 2026-10-16 三种结论与异常通道

[结果契约](../../references/equivalence-results.md)已落实：完整数学比较正常返回结构等价、数学等价、数学不等价；删除 `Status.UNKNOWN`。
结构入口仍提供匹配/不匹配阶段诊断；未完成、契约冲突和运行故障都抛出分类异常。
`ComparisonError` 是统一基类，保存阶段、路径、失败分类和原始原因；查询/资源诊断放在 `diagnostics`。
`SympyResult` 和正常函数模式报告删除 `failure_kind`，该字段改由异常提供。
worker 用独立失败消息传输异常分类、诊断与远端 traceback；超时仍终止并回收进程。
多模式比较遇到失败立即抛错，批处理捕获后写单独错误记录；不以其他模式的负结论掩盖失败。
原来返回 unknown 的测试改为异常断言；增加 None 查询、worker 故障、异常链及后续请求恢复验证。
迁移时定位到一个被旧 unknown 掩盖的错误：`max(x, 0) / max(x, 0)` 的定义条件可被 SymPy
化成布尔分支的 `Piecewise`，该对象继承 `Expr`，旧查询误将其与 `True` 相减。现在仅对全部
分支均为逻辑值的 Piecewise 使用 SymPy 的 `rewrite(ITE)`，再走逻辑比较，并计入转换预算。
数值型 Piecewise 继续走算术查询；正反操作数、逻辑等价和数值分支分别回归。
详细取舍、迁移范围与验证记录见[执行计划](../../exec-plans/completed/comparison-exceptions.md)。

下文的历史状态描述按其实现时期保留；当前结果接口以本节和结果契约为准。

## 2026-10-16 可信 SymPy 默认数学后端

用户已明确选择完全信任 SymPy；随后暂停生产迁移，要求先按“可以本地修改开源库”的前提
更新[迁移计划](../../exec-plans/completed/trusted-sympy-backend.md)，评估转换器和公式资产。
新的默认数学路径已将完整索引表达式及定义条件交给 SymPy；它的数学判断直接成为结果，
不再要求自写独立证明器批准。用户随后明确放弃旧证明器，旧代数证书及验证 API 已删除。
转换仍保留输入、输出、轴、常量、条件和区域绑定，无法表达的语义明确报告。
用户已要求恢复实施。下文“仅候选、独立复核”的内容保留为旧后端的设计记录；
架构评估与本地 fork 探针见[复用报告](../../exec-plans/completed/research/equivalence-backends/architecture-reuse.md)，迁移进展在计划中记录。

新转换路径已将 jax2sympy 的 SSA 遍历和 36 个逐点映射直接并入 SymbolicBuilder，精确常量转换一并保留。
构造器返回 SymPy 的 Indexed/Sum 元素公式及原始定义条件，shape 和输出坐标独立保存；
轴布局和区域绑定逐项适配。转换缺失抛出 UnsupportedJaxprError，不删除不透明依赖。
原有完整数组映射及逐元素转换入口现位于 [algebra/explicit.py](../../../src/jax_reorder/algebra/explicit.py)，供公式基准与后续扩展使用；
它不跟踪定义条件，不参与生产数学判等。两个承载文件的开头保留来源、固定 revision 和完整 MIT 声明，
不再保留 vendor 包或 manifest。重构记录见[迁移计划](../../exec-plans/completed/inline-symbolic-converter.md)。

查询在可终止的独立进程中运行，覆盖符号构造和化简的墙钟预算；传入的是已经结构校验的图数据，
不执行 JAX 数组。所有输出及条件通过才判等，完整最终输出的不恒等才作负结论。
符号坐标/抽象变量上的 False 不能直接推断源函数不恒等；负结论须落实到合法输出坐标的完整符号公式。
原始条件在约分前保存；无法确认的条件、CAS None、超时与转换错误分别报告。

验收包含独立 Nerva 公式对照、两份手写 JAX 的公共入口比较、错误轴/系数/输出反例、
空域/单位轴/部分函数/区域拒绝，以及阻断 JAX 数值执行的测试。
输入规模递增时单独记录表达式节点和转换开销，不将数值抽样当作等价依据。

特殊值和不透明算子仍由转换层保留：复用现有源图有限性分析，完整记录 primitive 参数、输入 tensor 定义、
输出端口和捕获的循环/索引变量；交给 SymPy 的不透明函数仅以标量坐标和捕获变量为参数，
避免将 Tensor/Lambda 容器错误当成普通实数参数。只有 SymPy 规范化后的完整输入定义相同才共享身份。
对可能非有限的值不直接建立 `y-y=0` 等实数公式。定义条件在特殊值消项、cast 和区域之间继续保留。
有限性分析属于已有转换语义的复用，不是对 SymPy 结论的第二次证明批准。

### 实现与结果接口

| 模块 | 当前职责 |
| --- | --- |
| [algebra/compare.py](../../../src/jax_reorder/algebra/compare.py) | 公共图入口：结构快路径、图抽象契约、完整函数/设备局部分流和结果汇总 |
| [algebra/local.py](../../../src/jax_reorder/algebra/local.py) | 设备局部入口：先完成资源证明，保留输入来源与轴绑定，再尝试结构、规范化值和数学比较 |
| [algebra/explicit.py](../../../src/jax_reorder/algebra/explicit.py) | 完整数组映射与逐元素转换参考接口，保留尚未用于生产的转换代码 |
| [algebra/adapter/calls.py](../../../src/jax_reorder/algebra/adapter/calls.py) | 两侧共有的完整相同纯 JIT 身份与端口对应；默认执行契约由同目录 jit.py 核对 |
| [algebra/adapter/transport.py](../../../src/jax_reorder/algebra/adapter/transport.py) | 校验后的图、常量、参数与区域接口冻结为独立进程输入 |
| [algebra/adapter/pallas.py](../../../src/jax_reorder/algebra/adapter/pallas.py) | 连接 core 的内存/协议证明，将完整或局部值投影冻结为数学图 |
| [algebra/expression.py](../../../src/jax_reorder/algebra/expression.py) | SymbolicTensor、精确常量、坐标实例化与避免捕获的绑定重命名 |
| [algebra/symbolic.py](../../../src/jax_reorder/algebra/symbolic.py) | SSA 遍历、精确常量、直接标量映射、Indexed/Sum、原始条件与词法作用域 |
| [algebra/pallas.py](../../../src/jax_reorder/algebra/pallas.py) | 把 core 已证明合法的值投影转为完整输出公式，分别传递定义条件与内存有效性 |
| [algebra/regions.py](../../../src/jax_reorder/algebra/regions.py) | cond、独立 scan 累加、受限符号展开和 while 绑定 |
| [algebra/sympy_ops.py](../../../src/jax_reorder/algebra/sympy_ops.py) | 预算受控化简、ArrayExpr/逻辑/恒等查询与定义条件处理 |
| [algebra/witnesses.py](../../../src/jax_reorder/algebra/witnesses.py) | 精确输入特化寻找完整输出反例，验证读取范围、所有原始定义条件及转换可表示性；未找到反例不推出等价 |
| [algebra/trusted.py](../../../src/jax_reorder/algebra/trusted.py) | 组织完整输出与定义条件比较、查询审计、worker 消息协议和超时回收 |
| [algebra/result.py](../../../src/jax_reorder/algebra/result.py) | AlgebraComparison、结论来源、执行前提及浮点警告 |

生产调用链按比较范围分开：

```text
普通 JAX：compare_algebraic_jaxprs
  → compare_jaxprs → 结构不同时 _prepare → CommonCalls → freeze
  → isolated_compare → compare_symbolic → SymbolicBuilder → SymPy 查询

完整 Pallas 函数：compare_algebraic_jaxprs
  → 结构快路径或 _prepare → adapter.pallas.project → core 语义投影
  → 冻结输出值图 → 相同的数学 worker

设备局部：compare_algebraic_jaxprs(scope="device_local") → compare_local_values
  → 规范化/适配 → project_local_view（资源证明与实际来源绑定）
  → 结构/规范化值快路径，或 freeze_local_projection → 数学 worker
```

完整函数结构快路径在没有显式执行绑定时直接返回；显式绑定仍须完成对应协议检查。
设备局部路径始终先检查资源义务；父函数输出组装与后处理不属于该局部结论。
普通 JAX 直接冻结已校验的 jaxpr，Pallas 通过 `ProgramView` 和值投影桥接，两者尚未统一迁入同一视图。
`explicit.py` 仅供参考测试和规模实验使用，不参与上述生产链路，也不承担定义条件、有限性或预算检查。

`compare_functions`、`compare_function_traces` 和 `compare_algebraic_jaxprs` 接受
`algebra_timeout`（默认 60 秒，正有限数），原 `algebra_budget` 默认仍为 10,000。
时间限制覆盖 worker 内的转换与查询，不包括 JAX tracing、父进程结构规范化与端口检查。
超时会 kill 并回收 worker，后续请求可继续；预算在大多项式展开前预检。
这是时间/表达式工作量边界，没有新增进程 RSS 硬上限。

`AlgebraComparison` 只记录当前判定路径：`correspondence` 保存完整函数结构快速路径的
`GraphCorrespondence`，可用 `verify_correspondence(left, right, correspondence, include_derivative_rules=False)`
复核；`backend_result` 保存 SymPy 的 backend/version、实际查询方法链、True/False/None、
输出位置、条件、完整不透明定义、工作量和时间。查询记录用于审计，不是独立证书。
数学路径的 `correspondence` 为 `None`；结构路径的 `backend_result` 为 `None`。
局部结构/规范化快路径使用 `local_evidence`；`normalizations` 保存 Pallas 左右值投影与前提，
`core_executions` 保存局部显式 core 绑定的协议证据。`scope` 区分 `function` 与 `device_local`，
`equivalence_kind` 区分 `structural`、`normalized` 与 `algebraic`，不等价时为 `None`。

此次删除 `AlgebraProof`、`PolynomialInequivalenceProof`、`verify_algebraic_proof`、
`verify_inequivalence_proof` 公开接口，以及结果中的 `proof` / `inequivalence_proof` 字段。
旧多项式 IR、规范化/搜索、域见证、标量候选、证据重放与 `tests/legacy_algebra.py` 均已移除，
不保留兼容别名或回退。调用方应读取当前结果字段；公式正反例迁移到公共 SymPy 入口，
旧证书篡改和旧搜索器实现测试删除，结构证据测试继续保留。

共享辅助代码按实际用途拆分：

- [algebra/budget.py](../../../src/jax_reorder/algebra/budget.py)：确定性工作量预算。
- [algebra/adapter/semantics.py](../../../src/jax_reorder/algebra/adapter/semantics.py)：完整元数据编码与默认执行属性检查。
- [algebra/indices.py](../../../src/jax_reorder/algebra/indices.py)：精确整数坐标适配，不含旧 Factor/Term 作用域与证据规则。
- [algebra/finiteness.py](../../../src/jax_reorder/algebra/finiteness.py)：源值有限性与共同调用的非有限常量检查，独立于旧 Lowerer。

重构范围、测试迁移与验收记录见[清理计划](../../exec-plans/completed/remove-legacy-prover.md)。

异常 `failure_kind` 区分 `unsupported_conversion`、`condition_undecided`、`cas_undecided`、
`budget_exhausted`、`timeout`、`internal_error`；不再返回 `algebra_incomplete` 结果。
例如 `x/x` 与 `1` 保留除零条件，抛出条件未判定异常；`sqrt(x*x)` 与 `x` 可由 SymPy 判为不恒等。
不恒等是“并非同一函数”，不是“所有输入的值都不同”。

### 表示边界与绑定验证

主表示是逐输出元素的惰性 Indexed/Sum；适合的收缩问题再转换为 SymPy 原生 ArrayExpr 查询。
没有另造张量代数证明器，也没有在生产依赖中新增 tensorgrad、AutoHOOT 或 sympy2jax。
归约最大值/最小值、布尔归约及小范围条件可能显式展开并扣预算；大 shape 的任意组合并不保证可判定。
一般 scan 最多符号展开 64 步，独立加法累加可保持 Sum；while 只接受可确定的受限终止。
数据依赖的未知终止、effects、复杂 dtype、非默认精度/分片等保留拒绝路径。

ArrayExpr 只接收完整自由轴、符号轴标签、完整范围求和及纯张量收缩。SymPy 1.14 的
`convert_indexed_to_array` 会把 `x[i+1]`、`x[2*i]` 的坐标整体当作轴标签，直接转换会丢失切片映射。
因此偏移、步长、翻转、广播缺轴、部分求和和坐标权重保留 Indexed 表示；合法大矩阵结合仍可走 ArrayExpr。
用法参照 [SymPy array expression 文档](https://docs.sympy.org/latest/modules/tensor/array_expressions.html)，
本地复现与正反例见 [转换边界测试](../../../tests/test_algebra_array_conversion.py)。

不含 Piecewise、Max、Min、Abs 的 Sum/Product 查询先由 SymPy 分别化简两侧（`doit=False`），再比较差值，避免提前展开
把 `exp(Sum(...))` 变成难以合并的大 Product。未解决且超过展开界限的求和不再调用会隐式
展开的 `Expr.equals`，最终无法判定时抛出 `failure_kind="cas_undecided"` 的异常；与求和变量无关的 Sum 按常数乘法计费。
含分段函数及其 Max/Min/Abs 表示的求和先做差值消项或有预算的有限展开，避免两侧化简造成分支膨胀；worker 超时仍是所有查询的最终资源边界。

绑定转换不能直接使用 SymPy 1.14 的嵌套 `Sum.as_dummy()`：其重名会把外层读取捕获到内层。
本地按词法深度规范命名，嵌入/读取时生成新 Dummy；独立双重求和参考验证矩阵结合两种转换，
防止只比较两份同样错误的表达式而漏检。

浮点 narrowing cast 用逐元素不透明函数及其原始定义条件表示，不应用 cast 线性规则。
`SymbolicTensor.finite` 表示在保留的定义条件下可作有限实数解释；源图 `Source.finite` 表示
无需附加条件的有限性事实，后者用于 guarded-softmax，不能把两者混用。
固定宽度整数算术原则上保留不透明语义；未编码 dtype 值域的无限整数查询不能提供源程序的负结论。
当前归约/点积、`iota` 和空整数极值归约仍有已确认实现缺陷，见下方“当前边界”，不能把该原则视为全部分支已正确落实。
不透明函数残差同样只能提供同余正结论，不能把泛化问题的 False 当作完整程序反例。

## 当前边界

本节维护当前生产实现；迁移前的接口和早期验收已分别归入上方链接的设计与执行归档。
2026-10-26 审查确认三项尚未修复的整数边界问题：归约/点积忽略溢出、`iota` 忽略 dtype
值域、空整数极值归约使用浮点单位元。它们会导致错误等价或不等价结论；归约问题也已在
Pallas 完整函数与设备局部数学入口复现。复现及修复状态维护于
[ALG-006 / ALG-007 / ALG-008](../../exec-plans/active/pallas-semantic-layer.md#2026-10-26-审查确认的正确性缺陷)。这与受支持域之外明确拒绝的情况分开记录。

Pallas 合法输入 padding 另用 `SymbolicTensor.memory_validity` 携带逐 lane 初始化义务，
与 log/sqrt 等数学定义域 `defined` 分开。每侧完整输出必须先证明内存有效；
相同未定义域、`x-x` 或 `0*x` 不能替代初始化证明。受限 masked Sum 的局部区间
收缩只在代数 lowering 进行，保留 mean 分母和非零补充值；前缀差分还须证明
选中分支在新增区间全定义。grid、地址、DMA 和信号量时序始终由 core 处理。
支持范围、原生尺寸用例及 CAS 超时记录见
[尾块有效域与 Ref 组合](../../exec-plans/completed/pallas-history/stage-22.md#922-时序重访尾块有效域与-ref-组合2026-10-20完成)。

`-inf` mask 已加入受限支持，设计与验收见[完成计划](../../exec-plans/completed/negative-infinity-masks.md)。
负无穷及含负无穷的捕获数组保留 dtype、shape、完整内容，不进入有理数系数。
当前 `SymbolicTensor.finite` 和 `Source.finite` 按各自定义追踪有限性，数学定义条件保存在 `defined`；
旧后端的 `Tensor.nonfinite` / `total` 不再是当前接口。比较和索引等整数/布尔结果仍须保留对应调用语义。
可能非有限的乘加、消项、幂、归约和点积保持原语，索引视图及有限子表达式仍可规范化。

共同 JIT、循环体的字面量、捕获数组以及 gather `fill_value` 等参数都检查非有限值。
含这些值的循环保守标记浮点 carry 和区域输出，禁止按独立有限累加器消去循环。
不假设 softmax 输出必定有限，也不删除全 mask 行后续的 where 防护。
NaN / 正无穷字面量及捕获常量仍拒绝展开；原有纯调用中的参数保持精确身份并传播标记。
有限输入和有限代数子表达式仍是证明前提，这不是一般 IEEE 浮点等价算法。

[回归](../../../tests/test_algebra_negative_infinity.py)覆盖有限分支经过 mask/softmax/top-k 的改写，
不同谓词、分支、常量、`x-x` / `0*x` 的错误消项，共同 JIT、循环后续迭代、
gather 非有限填充值和全 mask 行防护。当前数学结论来自可信 SymPy 查询；独立原图对应校验属于结构路径，
不能把数学查询记录称为旧后端的独立可重放证书。

- 新增规则用于实数浮点的理想实数解释。整数、复数、特殊精度、布局和分片保留原边界；布尔谓词及实数类型转换接入显式标量函数；结构等价仍可直接通过。
- 代数转换要求静态 shape 和有限独立矩形求和域。reshape 和静态切片已转换为符号索引，动态索引域、对角读取以及其他未经适配的索引变换仍无完整规则。
- 普通 JAX 的 token、有序操作和未适配的嵌套原语不进入代数展开；原结构层能够证明的图仍通过结构证据接受。Pallas 的内存与通信先由 core 证明，只有完成的值投影进入数学层；转换失败通过异常报告。
- 除已审核的默认纯 JIT 组合及条件/scan 适配外，区域边界、调用参数、反馈与返回接口保持固定。未适配的求导区域结构变化通过分类异常报告；前向局部改写不保证任意变换后的区域接口仍相同。动态 while 的 VJP 由锁定 JAX 拒绝，单独记为 `trace_rejected`。
- 并非任意数学恒等式的完备判定。符号转换、有限展开、CAS 查询和深度均可能触发明确停止；未决或超预算异常不能解释为数学不等价。其他模型改写和 GPU/TPU 执行没有整体保证；真实 GLM 的当前范围见[核心层记录](../../exec-plans/completed/validation/glm52-decode-dsa-mla.md)，Pallas 的范围见[验证专题](pallas/validation.md)。

旧实现的索引与求和域设计保存在[旧索引代数引擎](../archive/algebraic-reduction-gaps.md)，该引擎已被替代，不能据其旧接口或“未支持”条目推断当前状态。
