# jaxpr 两轮规范化与比较结果

本文定义完整 jaxpr 比较的契约：**成功返回时只有计算图等价、数学等价、数学不等价三种结论；不能完成判定时直接抛出异常。** “无法判定”不再作为正常比较结果。

这是结果接口的设计决定，不表示比较器能够对任意两份 JAX 程序必定作出结论。可信 SymPy 后端及转换实现见[代数开发记录](../development-docs/algebraic-equivalence.md)。代码已按本文迁移：`Status.UNKNOWN` 已删除，比较失败通过 `ComparisonError` 及其子类抛出。

## 比较范围

比较限定于声明的输入签名、静态参数、执行环境和输入输出 pytree。算子、常量、dtype、shape、primitive 参数、输入输出顺序以及受支持的嵌套语义都属于比较依据。

数学等价使用声明的数学域和定义条件；当前数学后端采用理想实数解释，保留特殊值和部分函数的必要条件。它不等同于实际浮点执行的逐位相等，不承诺跨平台结果一致或误差很小。

本规范约束完成两阶段判定的最终结果。结构检查的“匹配／不匹配”属于阶段诊断；`different` 不能作为完整比较的第四种结论，也不能自动解释为数学不等价。

## 三种正常结果

| 结果 | 判定含义 | 沿用现有字段时的表示 |
| --- | --- | --- |
| **计算图等价** | 结构规范化后，算子、参数、常量、抽象值、操作数位置、输入输出和依赖关系一致，受支持的嵌套语义也匹配。 | `status="equivalent"`，`equivalence_kind="structural"` |
| **数学等价，计算图不同** | 第一阶段已确认结构不同，第二阶段确认完整输出在声明的数学域和定义条件下等价。 | `status="equivalent"`，`equivalence_kind="algebraic"` |
| **数学不等价** | 在声明的比较范围内，可信数学判断确认两个程序并非同一个函数。 | `status="inequivalent"`，`equivalence_kind=None` |

这三个分支互斥：结构匹配时优先返回第一类；只有结构不匹配才进入第二阶段。结果字段可以继续使用 `status` 与 `equivalence_kind` 的组合，不要求为了三类结论另造三个状态字符串。

第二类可能涉及运算顺序、分组、算子数量或整个计算图拓扑的变化，因此不再使用“只有运算顺序不等价”的描述。

第三类对应原文的“完全不等价”：它表示两个程序不是同一个函数，不要求它们对每个输入都给出不同结果。例如 `x` 与 `2*x` 在实数域上不等价，但在 `x=0` 时相等。

## 第一阶段：计算图结构规范化

结构规范化保留节点语义、依赖关系、操作数位置和运算分组，仅消除内部变量编号和独立方程排列等差异。当前采用从输出反向 BFS 的策略：

1. 按函数输出顺序确定起点，按每个节点的操作数顺序回溯依赖并统一编号。
2. 按依赖深度和编号排列节点，保留先定义、后使用的约束；BFS 访问顺序本身不是执行顺序。
3. 精确比较完整结构并复核对应关系。匹配时返回计算图等价；不匹配时进入数学比较。

比较必须覆盖原始完整计算图，包括输出回溯未覆盖的方程。未知语义、无法编码的参数或结构证据校验失败必须抛出异常，不能忽略问题后继续声明等价。

例如，交换两个独立的 `sin(x)` 和 `cos(y)` 的计算顺序，最后仍按同样的操作数顺序相加，可以得到相同结构表示。`a+b` 与 `b+a`、`(a+b)+c` 与 `a+(b+c)` 则改变了操作数位置或依赖连接，需要第二阶段判断。

## 第二阶段：可信 SymPy 数学比较

数学转换保留输入身份、常量、轴、输出位置、区域绑定和原始定义条件，以 Indexed/Sum 等表达式表示完整输出，再交给 SymPy 查询。适用时可使用 ArrayExpr；无需先生成自研多项式项表，也不再经过旧独立代数证明器复核。

SymPy 属于可信基础，数学结论按它对忠实转换后问题的判断接受。转换语义和查询问题仍由本项目负责：对抽象坐标或不透明函数得到的判断，必须适用于原程序的输出和定义域，才能成为程序级结论。

- 全部输出及必要定义条件确认等价，返回数学等价。
- 确认完整程序在声明范围内不恒等，返回数学不等价，并记录输出位置及所依据的查询。
- 仍未取得足以支持上述结论的判断，抛出带原因的异常。

结构不同、化简后表达式不同、未化成零、缺少规则，都不是数学不等价的充分依据。例如分配律两侧 `a*(b+c)` 与 `a*b+a*c` 的初始表达式不同，但在约定实数域内等价。

信任 SymPy 包括尊重它未给出结论的情况。[`Expr.equals` 的官方接口](https://docs.sympy.org/latest/modules/core.html#sympy.core.expr.Expr.equals)允许返回 `True`、`False` 或 `None`；`None` 不能当成 `False`。单次查询未决时可以继续尝试适用的方法，但整个比较仍未决时必须抛出异常。

项目不以执行两份 JAX 程序的数值抽样替代数学判断。浮点结果不同也不能单独证明理想实数公式不等价。依赖数学改写的结果继续报告浮点警告。

## 异常通道

“不能完成比较”通过异常传递，不构造 `status="unknown"` 的正常结果，也不返回 `None` 或空结果掩盖失败。只靠从枚举中删除 `UNKNOWN` 不足以落实这个契约：所有入口、异常捕获、worker 通信及汇总路径都必须一致处理。

| 失败类别 | 典型情况 | 要求 |
| --- | --- | --- |
| 输入或比较范围不成立 | 非法 jaxpr、输入契约不兼容、追踪环境不一致 | 抛出输入／契约异常，指出冲突位置。 |
| 语义无法识别或转换 | 不支持的 primitive、effects、参数或必要的区域语义 | 抛出不支持／转换异常，不丢弃语义继续比较。 |
| 数学查询未决 | SymPy 最终未能给出结论、必要定义条件未确认 | 抛出数学判定未完成异常，保留 `cas_undecided` 或 `condition_undecided` 等原因。 |
| 资源限制 | 工作量预算耗尽或 worker 超时 | 抛出资源／超时异常，保留预算、实际用量或时限。 |
| 实现或运行故障 | 转换器 bug、结构证据校验错误、worker 崩溃或通信损坏 | 抛出内部错误，保留原异常及可定位信息。 |

错误信息应至少包含阶段、位置、原因分类和具体原因；可用时附输出槽位、查询记录及原始异常链。不能把所有异常统一改名为“不支持”，也不能把数学未决或超时一律宣称为程序 bug。

“直接报错”指比较调用失败并向调用方抛出异常，不要求强制终止整个进程。批量工具可以捕获异常、记录失败项目并继续其他任务，但必须把错误记录与三类正常结果分开，不能重新包装为第四类语义结论。

当前默认 `algebra_budget=10000`、`algebra_timeout=10.0` 秒。资源上限继续保留，耗尽时走异常通道；不能为满足三类结果而取消资源边界或猜测结论。

## 完整判定流程

1. 检查输入、比较范围及完整图语义。失败则抛出异常。
2. 进行结构规范化和对应关系检查。确认匹配则返回计算图等价。
3. 结构不同则转换到 SymPy，确认声明域中的数学关系。
4. 确认等价则返回数学等价；确认不等价则返回数学不等价。
5. 没有获得结论或执行失败则抛出分类异常，保留诊断信息。

`compare_jaxprs` 以及显式 `equivalence="structural"` 仅提供结构阶段诊断，返回 `equivalent` 或 `different`；失败仍抛异常。完整比较使用 `compare_algebraic_jaxprs` 或默认 `equivalence="auto"` 的函数入口，它们不会把 `different` 作为最终结果。输入输出契约或环境不兼容时，两种模式都抛出契约异常。

## 决策评价与实现

**把正常结果限制为三类、把失败交给异常通道，是合理的严格接口设计。** 调用方获得的每个正常返回值都有确定的比较含义，不会把一次未完成的检查当成结论。

这个决定改变了失败的传递方式，没有增强数学判定能力。只有在“允许失败抛错”的前提下，三类结果契约才成立；不能据此承诺任意两份 JAX 程序都能成功归入其中一类。[SymPy 官方说明](https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html#equals-signs)也明确指出，通用符号恒等判断并非完备。

当前实现：

- `ComparisonError` 是统一可捕获基类，携带 `stage`、`path`、`reason`、`failure_kind` 和 `diagnostics`。正常结果不再保存 `failure_kind`。
- `InvalidJaxprError` / `ComparisonContractError` 分别表示非法图和不兼容的比较范围；`UnsupportedJaxprError` 表示未知语义或转换缺口。
- `MathematicalUndecidedError` 区分 `cas_undecided` 和 `condition_undecided`；`ComparisonBudgetError` / `ComparisonTimeoutError` 分别表示工作量或时间边界。
- `CorrespondenceError` / `ComparisonInternalError` 表示对应证据或实现/worker 故障；`TraceError` / `CompilationError` 表示追踪或可选编译阶段失败。原始原因通过异常链保留。
- worker 的失败消息与 `SympyResult` 分开传输，父进程恢复异常类型、查询诊断和远端 traceback，不返回失败结果。
- 函数的任一请求模式失败立即抛错；只有所有请求模式完成后才汇总。先前模式的不等价结论不会掩盖后续模式的失败。
- 批量示例捕获 `ComparisonError` 时保存独立 `error` 记录，错误记录没有 `status`，并以非零退出码报告未完成项。

`algebra_budget`、`algebra_timeout`、`modes`、`equivalence` 等调用选项的格式校验仍抛出 `ValueError`；这类调用错误同样不会生成比较结果。

调用方需要处理失败时，可以统一捕获基类，也可以按异常子类分别处理：

```python
from jax_reorder import ComparisonError, compare_algebraic_jaxprs

try:
    result = compare_algebraic_jaxprs(left_jaxpr, right_jaxpr)
except ComparisonError as error:
    print(error.stage, error.path, error.failure_kind, error.reason)
    raise
else:
    print(result.status, result.equivalence_kind)
```

迁移与验证见[执行记录](../exec-plans/completed/comparison-exceptions.md)。受保护的架构文档仍保留旧描述，后续需在获得对应文件授权后同步。
