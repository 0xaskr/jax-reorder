# 索引乘加与有限求和的代数比较

## 2026-10-16 三种结论与异常通道

[结果契约](../references/equivalence-results.md)已落实：完整数学比较正常返回结构等价、数学等价、数学不等价；删除 `Status.UNKNOWN`。
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
详细取舍、迁移范围与验证记录见[执行计划](../exec-plans/completed/comparison-exceptions.md)。

下文的历史状态描述按其实现时期保留；当前结果接口以本节和结果契约为准。

## 2026-10-16 可信 SymPy 默认数学后端

用户已明确选择完全信任 SymPy；随后暂停生产迁移，要求先按“可以本地修改开源库”的前提
更新[迁移计划](../exec-plans/completed/trusted-sympy-backend.md)，评估转换器和公式资产。
新的默认数学路径已将完整索引表达式及定义条件交给 SymPy；它的数学判断直接成为结果，
不再要求自写独立证明器批准。用户随后明确放弃旧证明器，旧代数证书及验证 API 已删除。
转换仍保留输入、输出、轴、常量、条件和区域绑定，无法表达的语义明确报告。
用户已要求恢复实施。下文“仅候选、独立复核”的内容保留为旧后端的设计记录；
架构评估与本地 fork 探针见[复用报告](equivalence-backends/architecture-reuse.md)，迁移进展在计划中记录。

新转换路径已将 jax2sympy 的 SSA 遍历和 36 个逐点映射直接并入 SymbolicBuilder，精确常量转换一并保留。
构造器返回 SymPy 的 Indexed/Sum 元素公式及原始定义条件，shape 和输出坐标独立保存；
轴布局和区域绑定逐项适配。转换缺失抛出 UnsupportedJaxprError，不删除不透明依赖。
原有完整数组映射及逐元素转换入口迁至 algebra_explicit.py，供公式基准与后续扩展使用；
它不跟踪定义条件，不参与生产数学判等。两个承载文件的开头保留来源、固定 revision 和完整 MIT 声明，
不再保留 vendor 包或 manifest。重构记录见[迁移计划](../exec-plans/completed/inline-symbolic-converter.md)。

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
| [algebra/compare.py](../../src/jax_reorder/algebra/compare.py) | 结构快路径、函数契约、共享调用的端口对应和新结果汇总 |
| [algebra/explicit.py](../../src/jax_reorder/algebra/explicit.py) | 完整数组映射与逐元素转换参考接口，保留尚未用于生产的转换代码 |
| [algebra/adapter/transport.py](../../src/jax_reorder/algebra/adapter/transport.py) | 校验后的图、常量、参数与区域接口冻结为独立进程输入 |
| [algebra/symbolic.py](../../src/jax_reorder/algebra/symbolic.py) | SSA 遍历、精确常量、直接标量映射、Indexed/Sum、原始条件与词法作用域 |
| [algebra/regions.py](../../src/jax_reorder/algebra/regions.py) | cond、独立 scan 累加、受限符号展开和 while 绑定 |
| [algebra/trusted.py](../../src/jax_reorder/algebra/trusted.py) | SymPy ArrayExpr/逻辑/恒等查询、完整输出汇总与可终止 worker |

`compare_functions`、`compare_function_traces` 和 `compare_algebraic_jaxprs` 接受
`algebra_timeout`（默认 10 秒，正有限数），原 `algebra_budget` 默认仍为 10,000。
时间限制覆盖 worker 内的转换与查询，不包括 JAX tracing、父进程结构规范化与端口检查。
超时会 kill 并回收 worker，后续请求可继续；预算在大多项式展开前预检。
这是时间/表达式工作量边界，没有新增进程 RSS 硬上限。

`AlgebraComparison` 只记录当前判定路径：`correspondence` 保存结构快速路径的
`GraphCorrespondence`，可用 `verify_correspondence(left, right, correspondence, include_derivative_rules=False)`
复核；`backend_result` 保存 SymPy 的 backend/version、实际查询方法链、True/False/None、
输出位置、条件、完整不透明定义、工作量和时间。查询记录用于审计，不是独立证书。
数学路径的 `correspondence` 为 `None`；结构路径的 `backend_result` 为 `None`。

此次删除 `AlgebraProof`、`PolynomialInequivalenceProof`、`verify_algebraic_proof`、
`verify_inequivalence_proof` 公开接口，以及结果中的 `proof` / `inequivalence_proof` 字段。
旧多项式 IR、规范化/搜索、域见证、标量候选、证据重放与 `tests/legacy_algebra.py` 均已移除，
不保留兼容别名或回退。调用方应读取当前结果字段；公式正反例迁移到公共 SymPy 入口，
旧证书篡改和旧搜索器实现测试删除，结构证据测试继续保留。

共享辅助代码按实际用途拆分：

- [algebra/budget.py](../../src/jax_reorder/algebra/budget.py)：确定性工作量预算。
- [algebra/adapter/semantics.py](../../src/jax_reorder/algebra/adapter/semantics.py)：完整元数据编码与默认执行属性检查。
- [algebra/indices.py](../../src/jax_reorder/algebra/indices.py)：精确整数坐标适配，不含旧 Factor/Term 作用域与证据规则。
- [algebra/finiteness.py](../../src/jax_reorder/algebra/finiteness.py)：源值有限性与共同调用的非有限常量检查，独立于旧 Lowerer。

重构范围、测试迁移与验收记录见[清理计划](../exec-plans/completed/remove-legacy-prover.md)。

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
本地复现与正反例见 [转换边界测试](../../tests/test_algebra_array_conversion.py)。

不含 Piecewise、Max、Min、Abs 的 Sum/Product 查询先由 SymPy 分别化简两侧（`doit=False`），再比较差值，避免提前展开
把 `exp(Sum(...))` 变成难以合并的大 Product。未解决且超过展开界限的求和不再调用会隐式
展开的 `Expr.equals`，返回 `cas_undecided`；与求和变量无关的 Sum 按常数乘法计费。
含分段函数及其 Max/Min/Abs 表示的求和先做差值消项或有预算的有限展开，避免两侧化简造成分支膨胀；worker 超时仍是所有查询的最终资源边界。

绑定转换不能直接使用 SymPy 1.14 的嵌套 `Sum.as_dummy()`：其重名会把外层读取捕获到内层。
本地按词法深度规范命名，嵌入/读取时生成新 Dummy；独立双重求和参考验证矩阵结合两种转换，
防止只比较两份同样错误的表达式而漏检。

浮点 narrowing cast 用逐元素不透明函数及其原始定义条件表示，不应用 cast 线性规则。
`SymbolicTensor.finite` 表示在保留的定义条件下可作有限实数解释；源图 `Source.finite` 表示
无需附加条件的有限性事实，后者用于 guarded-softmax，不能把两者混用。
固定宽度整数算术保留不透明语义；未编码 dtype 值域的无限整数查询不能提供源程序的负结论。
不透明函数残差同样只能提供同余正结论，不能把泛化问题的 False 当作完整程序反例。

### 离线复现与规模记录

Nerva 两套固定源码及许可证仅位于测试 fixture；hash 检查后从临时目录导入，不引入训练栈。
12 组源程序→独立公式检查、12 组公共入口正反比较，以及 5 组独立手写公式改写分别测试。
覆盖 softmax/log-softmax、激活、Hadamard/矩阵乘法、平方损失及梯度；比较期间阻断 JAX 后端执行。
输入为有限实数解释的 float32 `(2,3)` 抽象矩阵；softmax 按行归一化，矩阵乘法右输入为 `(3,2)`。
平方损失输出标量，梯度输出 `(2,3)`；其余输出契约由两端公式逐坐标核对，没有添加正值输入假设。
[逐元素转换基线](../../tests/test_symbolic_explicit.py)保留 23 项探针，
[新后端测试](../../tests/test_trusted_sympy.py)覆盖实际查询、资源恢复、域和绑定边界。

运行 [规模脚本](../../tools/experiments/equivalence_backends/trusted_sympy_scaling.py)：
`uv run python tools/experiments/equivalence_backends/trusted_sympy_scaling.py`。
Python 3.12、JAX 0.11.2、SymPy 1.14.0 下，对 `sum(x*x)` 的一次实测如下：

| 输入长度 | 逐元素/惰性唯一节点 | 逐元素/惰性构造毫秒 | 逐元素/惰性 Python 峰值字节 |
| --- | --- | --- | --- |
| 8 | 18 / 12 | 74.716 / 12.842 | 1,786,235 / 266,150 |
| 32 | 66 / 12 | 18.321 / 4.710 | 49,898 / 38,825 |
| 128 | 258 / 12 | 211.659 / 4.592 | 248,904 / 34,051 |
| 512 | 1,026 / 12 | 1,664.135 / 3.696 | 1,723,645 / 32,811 |

逐元素输入符号数随长度增长；惰性表示只有一个 Indexed 读取、12 个唯一节点。
时间包含首次初始化/缓存影响，峰值来自 tracemalloc，不是进程 RSS；只测表达式构造，没有执行 JAX，
也不意味着任意大图的最终 SymPy 查询具有常数成本。

可信后端首次迁移验收为 **2166 passed，1 个已有上游弃用警告，1403.69 秒**；Ruff、格式、tracing/数学示例、
离线 wheel 构建和独立导入均通过。该次验收仍包含旧证据兼容测试。
清理后全量运行 1930 项，其中 1928 项通过、2 项稳定 softmax 查询超时；定位并修复查询顺序后，
两项在原 15 秒限制下通过，受影响的 8 个测试文件 **299 passed**。最后调整后未再重复全量；
最终 Ruff、格式、示例、文档检查、离线打包与独立导入通过，详细阶段记录见清理计划。

## 历史实现记录（以下为迁移前的后端）

状态：首期实现与全量 1600 项测试已通过，[执行计划](../exec-plans/completed/algebraic-equivalence.md)已归档。用户要求不修改受保护文档，这些入口的同步记录为 [DOC-002](../exec-plans/tech-debt-tracker.md)；后续核心重构见[标量索引 IR](algebraic-reduction-gaps.md)，本文的原始验收数字保留为基线。

## 比较与数值契约

函数入口先检查输入输出 pytree、完整抽象值、静态参数和追踪环境，再执行原有结构比较。默认 `equivalence="auto"` 在结构不同后尝试代数证明；`equivalence="structural"` 保留结构结果。底层结构入口不改变。数学结论限定于实数浮点的理想实数解释，必须报告浮点警告，不能承诺实际舍入、溢出、NaN/Inf 或有符号零相同。

未知语义、effects、无法解释的参数、预算不足、证明不成立均返回 `unknown`。完整图检查先于输出表达式提取，未知节点不能通过乘零或删除无用计算逃过检查。整数、复数和布尔运算保留结构调用，不采用实数多项式规则。

结构等价蕴含可接受的等价结论：函数入口和直接代数入口都先复用已复核的完整结构证据，标记 `structural`。这类图无需再次进入实数代数支持域，也不受代数展开预算影响。`verify_algebraic_proof` 同时接受这种结构直通证据和局部代数推导；只有实际依赖代数规则时才发出相应浮点警告。

## 表示与规范化

张量保存完整输出抽象值和有序自由轴；元素表达式包含精确常量、输入读取、加乘、有限求和及有边界的调用。输入位置和输出轴固定；求和索引允许保持范围和引用的双射重命名。乘积中的多个求和分别使用新绑定，防止索引捕获。

`dot_general` 的收缩维变为求和轴，批次维共享自由轴，左右剩余维按输出位置排列。`reduce_sum` 把对应输入维变为求和轴。广播扩展长度为 1 的输入维读下标 0；转置、reshape 和静态切片按符号整数映射处理。矩阵乘法与广播乘积再求和因此获得同一表示，矩阵乘法的顺序仍体现在各个输入读取的有序下标中。

递归将表达式转换为稀疏项表：加法合并项，乘法做有预算限制的项组合，求和给每项附加独立有限域。每项包含有理数系数、求和范围与标量因子多重集。绑定索引按范围与连接关系分组，在需要时进行有预算的精确编号搜索。项键按完整内容比较；哈希仅加速查找。

转换过程沿有限 DAG 递归，不反向因式分解；每步减少尚未转换的子表达式。搜索空间受静态索引数量限制，预算在分配组合之前扣减。规范项的索引编号由候选集最小序列化决定；不以原节点名或遍历顺序破除平局。维度长度只保留为范围，不枚举张量元素。

## 证明与信任边界

证明保存原图到已适配区域图的结构对应，以及表达式各节点的局部推导。规范化器负责搜索规范索引；校验器只验证证据提供的索引双射，并独立检查所有加法项、乘法组合、求和范围和系数，避免漏掉交叉项。校验器不调用代数规范化器或规范编号搜索。

原语到索引表达式的语义适配、精确值编码和 JAX 图类型检查属于共同信任基础。原图的结构对应由已有独立校验器检查；额外的代数推导由新校验器逐项检查。证明必须绑定给定原图的常量、参数和所有有序输出；篡改中间步骤或最终项表都须拒绝。

未适配的 jit、cond、scan、while 区域保留调用边界、参数、输出槽位和反馈关系，在对应区域内部应用代数规则。已审核的条件包装和独立 scan 累加接入新的标量/递推 IR。JVP/VJP 只检查实际追踪出的程序，前向证明不替代求导证明。

[真实 mHC 层验证](deepseek-v41-layer.md)补充严格默认纯 JIT 的实参代入，使不同私有残差接口可在组合后比较；两侧完整结构相同的默认子调用用原始端口对应证据复用，保留调用求值 barrier。非默认执行属性仍保留边界。分块归约的转置仅在证明读取范围安全后接入静态零扩展，前缀指标保持完整支持域。

## 接口与实现入口

`compare_functions` 和 `compare_function_traces` 默认采用 `equivalence="auto"`，可显式指定 `"structural"`；每个模式的 `algebra_budget` 默认为 10,000。汇总和单模式结果用 `equivalence_kind` 区分 `structural`、`algebraic` 和未证明。单模式的 `comparison` 保留原结构报告，`algebra` 保存规则、前提、警告、证明、工作量和失败原因。

底层 `compare_algebraic_jaxprs(left, right)` 只覆盖已经物化的前向图；JVP/VJP 须先由函数入口分别追踪。`verify_algebraic_proof(left, right, proof)` 从原图复核证据，成功返回 `None`，篡改或无效证据抛出 `CorrespondenceError`。证明是进程内对象，未定义跨进程持久化格式。

| 实现 | 职责 |
| --- | --- |
| [algebra/compare.py](../../src/jax_reorder/algebra/compare.py) | 结构证据直通、代数比较和结果分类 |
| [algebra_ir.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_ir.py) | 不可变 DAG、项表、精确系数和预算 |
| [algebra_lower.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_lower.py) | 原语语义、索引映射与固定区域转换 |
| [algebra_normalize.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_normalize.py) | 展开、项收集和绑定索引规范化搜索 |
| [algebra_proof.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_proof.py) | 原图对应和局部推导的独立复核 |

## 当前边界

`-inf` mask 已加入受限支持，设计与验收见[完成计划](../exec-plans/completed/negative-infinity-masks.md)。
负无穷及含负无穷的捕获数组保留 dtype、shape、完整内容，不进入有理数系数。
内部 Tensor 的 `nonfinite` 标记表示可能依赖显式非有限值，与部分函数的 `total` 独立。
标记沿浮点/复数结果传播；比较和索引等整数/布尔结果仍保留完整调用语义。
可能非有限的乘加、消项、幂、归约和点积保持原语，索引视图及有限子表达式仍可规范化。

共同 JIT、循环体的字面量、捕获数组以及 gather `fill_value` 等参数都检查非有限值。
含这些值的循环保守标记浮点 carry 和区域输出，禁止按独立有限累加器消去循环。
不假设 softmax 输出必定有限，也不删除全 mask 行后续的 where 防护。
NaN / 正无穷字面量及捕获常量仍拒绝展开；原有纯调用中的参数保持精确身份并传播标记。
有限输入和有限代数子表达式仍是证明前提，这不是一般 IEEE 浮点等价算法。

[回归](../../tests/test_algebra_negative_infinity.py)覆盖有限分支经过 mask/softmax/top-k 的改写，
不同谓词、分支、常量、`x-x` / `0*x` 的错误消项，共同 JIT、循环后续迭代、
gather 非有限填充值、全 mask 行防护以及证明篡改。独立校验器从原图重建这些边界。

- 新增规则用于实数浮点的理想实数解释。整数、复数、特殊精度、布局和分片保留原边界；布尔谓词及实数类型转换接入显式标量函数；结构等价仍可直接通过。
- 代数转换要求静态 shape 和有限独立矩形求和域。reshape 和静态切片已转换为符号索引，动态索引域、对角读取以及其他未经适配的索引变换仍无完整规则。
- token、有序操作和未适配的嵌套原语不进入代数展开；原结构层能够证明的图仍通过结构证据接受。公开原语清单中的 27 个此类用例有明确拒绝断言，没有因转换失败而跳过检查。
- 除已审核的默认纯 JIT 组合及条件/scan 适配外，区域边界、调用参数、反馈与返回接口保持固定。未适配的求导区域结构变化仍可能返回 `unknown`；前向局部改写不保证任意变换后的区域接口仍相同。动态 while 的 VJP 由锁定 JAX 拒绝，单独记为 `trace_rejected`。
- 并非任意数学恒等式的完备判定。多项式展开、同色索引搜索和深度均可能触发明确停止；`unknown` 不能解释为数学不等价。其他模型改写和 GPU/TPU 执行没有整体保证；真实 GLM 的当前验收范围见下方后端接入及核心层记录。

详见[后续核心设计及当前边界](algebraic-reduction-gaps.md)。

## 验证策略

正例覆盖逐元素展开、矩阵结合与分配、广播乘加互认、多轴归约、单位/空轴、嵌套区域和独立求导模式。反例覆盖输入输出位置、索引连接、捕获、精度、类型与反馈。证明测试篡改常量、局部推导、轴映射和原图来源，并禁止校验器调用规范化器。压力测试验证预算拒绝、大维度保持符号和原图不变。

数值实验单独保存实数等价但浮点不同的见证，不作为证明替代品。[结构蕴含回归](../../tests/test_algebra_structural.py)把原有 182 组公开原语 IR 与其结构规范化结果成对送入代数入口，全部要求等价；其中 155 组另外直接通过转换、展开和独立证明，其余 27 组明确断言适配边界。加上预算直通、函数三模式和篡改证据，共 371 项通过。

初期代数功能阶段的全量 pytest 为 **1600 passed，1 warning，426.75 秒**；Ruff、格式、tracing/BFS/cond 示例、原语及适配清单、双/四 CPU 各 22 组通信验证均通过。唯一 warning 来自保留的上游 Pallas 弃用接口。具体命令记录在执行计划中。[代数示例](../../examples/algebraic_equivalence.py)的分配律、矩阵与乘加归约均通过 forward/JVP/VJP 证明、独立复核及 CPU 编译。

## SymPy 接入与数学不等价证据（2026-10-15）

本轮延续既有 `Lowerer → Expr/Term → PolynomialNormalizer → proof checker`，不另建实验版 jaxpr 前端。
SymPy 只生成受限数学变换候选；接受结果必须附独立可重放的局部证据。禁止以
`equals()`、随机求值、无条件 `simplify()` 或 CAS 搜索失败作为判等/不等价依据。

- 标量改写在每步原始项表产生后、域双射和绑定轴规范化之前执行；原始 Expr、premises、
  原图对应和有序输出仍绑定源图。证据检查器不导入 SymPy 或调用规范化搜索。
- 正式依赖为 SymPy 1.14。首批覆盖有限实数 exp 的加法/乘法恒等式和零点、非零常数除法、
  两个有限 exp 的商，以及有非零证据的同式相除。`exp_defined` 在子证据已消除部分性后
  复核 exp 的数学总性，使 `exp(x/2)` 与 `exp(x*0.5)` 等组合也能完成证明。
  `total` 不是非零证据；`x/x` 不能无条件化为 1。非有限原子不可转换成可任意消去的实符号。
  SymPy 符号映射保留完整 Factor、Index 和 scope 身份，有限和继续由既有索引证明处理。
- 新的数学不等价状态与结构 `different` 分离。负证据首先支持完整真实输出的实数多项式，
  变量只来自可追溯输入元素，系数精确有理。区域 barriers、未解释函数和中间节点不同均不能
  提供完整程序反例；不支持的数学域继续返回定位明确的 unknown。
  `PolynomialInequivalenceProof` 绑定原图、真实输出槽位、合法坐标、输入元素单项式和非零
  差系数；独立验证器按指定单项式计算系数，不复用候选展开。结论是存在输入使两函数不同，
  不是每个输入都不同。当前要求全部输出是有限实数多项式且没有区域 barriers；张量常量、
  未解释函数、整数/复数和带条件定义域仍拒绝。大输出域惰性取坐标，大归约在枚举前计预算。
- guarded-softmax 的有限性规则必须验证同一 mask、有限 logits、匹配的 max/sum 归约域、
  广播和最后的零分支；全 mask 行依靠最终 guard 定义为零。错 guard/错轴不能获得此事实。
  独立校验从源图重建有限性，不能信任外部传入的 nonfinite 标记。
  新的有限性事实不修改部分求值 `Expr.total`。默认 NaN gather fill、窄化转换和循环 carry
  仍保守处理；只有审核过的有限输入/常量/扫描切片才能提供有限性事实。
- 预算覆盖转换、表达式增长和证据复核。测试包括正负例、空域、未满足定义域、索引捕获、
  非有限值、证据篡改，以及禁用 CAS/规范化器后仍可复核的独立性检查。

完整 GLM 负例还需要精确处理 approximate top-k、动态 gather 和 scan。局部 QK 或 PV 的
结论不提升成完整 layer 结论；若本轮不能建立源输入到最终输出的证据，保留具体 unknown。
工作进度见[执行计划](../exec-plans/completed/symbolic-math-integration.md)。

实现与验收：[CAS 候选](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_sympy.py)、
[标量证据复核](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_scalar_proof.py)、
[数学不等价证据](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_inequivalence.py)、
[有限性规则](../../src/jax_reorder/algebra/finiteness.py)。
对应回归为 [CAS](../../tests/test_algebra_sympy.py)、
[负证据](../../tests/test_algebra_inequivalence.py) 与
[guarded softmax](../../tests/test_algebra_finiteness.py)。

本轮冻结后的全量验证为 **2095 passed，1 warning，944.74 秒**；Ruff、169 个 Python 文件
格式、依赖锁一致性、tracing 示例、文档链接和差异检查通过。GLM 三组抽象签名均为
5 个正例已证明、5 个负例 unknown，且禁止编译/eager 执行。

## 实施中发现并修复的前置问题

函数签名转换必须保留 ShapeDtypeStruct 原始的 sharding 描述：JAX 0.11.2 会依据显式 NamedSharding 的 memory_kind 重设 memory_space。把未指定的 sharding 替换为 aval 默认 sharding 会把 Host 变成 Device；实现保留原描述，并精确比较转换前后的全部抽象属性。实际数组则继续采用其抽象 sharding，不能把样本驻留的单个设备额外写成编译时放置限制；双 CPU shard_map 回归覆盖这一差别。

嵌套求导区域可能重复返回同一个 Var。原结构证据构造按 Var 的出现次数推断输出置换，可能与父调用的结果槽位置换不同，导致合法的规范化图不能通过自身对应校验。构造器现以父调用的结果双射约束 jit、cond、scan 和 while body 的返回槽位；独立校验器仍逐项检查值绑定和循环反馈。回归必须包括重复输出的嵌套求导图与错误反馈反例。

求导产生的无用 slice 和 empty2 已按锁定版本核对：固定 slice 的合法性由 JAX 类型检查保证，empty2 的 CPU lowering 返回零常量。它们可以按纯且全定义节点检查；不能把这一结论扩大到未知原语或部分函数。
