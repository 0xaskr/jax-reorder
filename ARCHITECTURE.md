# 架构与比较契约

`jax-reorder` 通过 jaxpr 结构规范化，检查 AI 生成或改写的 JAX 代码是否与原始代码等价。本文是架构与比较语义的维护入口。

## 当前实现

已实现第一轮计算图规范化与精确比较，基线为 Python 3.12、锁定的 JAX/jaxlib 0.11.2 和 CPU。实现位于 [src/jax_reorder](src/jax_reorder/)，[tracing 与重排示例](examples/inspect_jaxpr.py)展示 BFS 编号和最终排列，[AI 组合示例](examples/bfs_reordering.py)展示经典案例及 SGLang-JAX LlamaMLP 正反例。

当前没有代数模式，也没有一般 Python 函数级比较入口。[两轮结果参考](docs/references/equivalence-results.md)中的四类结果描述后续完整流程；本轮接口只返回计算图等价、计算图不等价或无法判定。

## 流程与职责

调用方在相同的输入签名、静态参数和执行环境下追踪函数，得到包含常量绑定的 jaxpr，再调用 `canonicalize` 或 `compare_jaxprs`。jaxpr 已展平参数；函数级输入输出 pytree、静态参数及追踪外的 Python 行为由调用方核对，本轮接口不能恢复这些信息。

| 模块 | 职责 |
| --- | --- |
| [canonical.py](src/jax_reorder/canonical.py) | SSA 检查、输出 BFS、依赖深度、嵌套重排、完整结构比较及诊断 |
| [semantics.py](src/jax_reorder/semantics.py) | 参数、抽象值与常量的精确编码，已知描述符适配 |
| [support.py](src/jax_reorder/support.py) | 公开 lax 原语及包装函数辅助原语的身份发现 |
| [errors.py](src/jax_reorder/errors.py) | 非法图与不支持语义的显式错误 |

比较使用完整结构，既不依赖变量名、打印文本，也不以摘要代替精确比较。`canonicalize` 返回新 jaxpr、规范结构、BFS/最终方程索引、原方程深度、变量引用映射及部分完成原因。原图不被修改；不含嵌套子图的方程复用原对象。

## 图规范化契约

从有序输出开始，按操作数位置反向 BFS 编号，再按 `(依赖深度, BFS 编号)` 排列。保留算子身份、全部语义参数、外部输入输出位置、结果端口、共享关系、运算分组、常量、dtype、shape、布局与分片等抽象值信息。独立方程可以换序；普通运算的操作数不交换，不做结合、分配、合并、常量折叠、DCE 或内联。

捕获常量按首次引用编号，同时保留值、类型与绑定关系。字面量与捕获常量不同，共享绑定与两份相同常量也不同。数组比较完整内容，区分有符号零、NaN 位模式和不同 float8 格式。

嵌套子图在原区域内分别规范化，保留分支顺序、循环接口及调用边界。token 依赖照常保留；隐式有序操作及包含它们的调用保留相对顺序。未知 effects 或不透明参数显式拒绝，即使它们不影响输出也不忽略。

`cond`（包括 `lax.switch`）按分支中参数首次使用位置，对数据实参与所有分支形参执行同一个置换，再进行父图 BFS。它消除闭包自动捕获位置变化，保留实参与变量的实际绑定；不改变选择值、分支顺序及外部接口。多个在所有分支中均未使用的参数无法由使用位置唯一排列时，保留并返回 `unknown`。规则与反例见 [cond 绑定规范化](docs/development-docs/cond-bindings.md)。

`shard_map` 内显式集合通信沿用相同图规则，保留通信轴、分组、置换、维度、mesh 与分片接口；`pvary` 辅助节点也完整保留。子图在恢复的轴环境内校验。[通信实验](docs/development-docs/collective-jaxpr.md)覆盖单进程多 CPU 设备，不外推为多主机或 GPU/TPU 执行验证。

未被输出或已处理的顺序根覆盖的纯计算保留在末尾，维持原相对顺序；未使用常量绑定也保留。此时返回部分完成，比较结果为 `unknown`，不宣称全图已有唯一形式。

## CPU 原语支持与结果

目标范围以 `jax.lax` 及公开子模块在 CPU 的实际能力为准，不另建 dtype、shape 或参数值白名单。JAX 能处理而本项目不能解释的情况属于实现缺口，必须报告；JAX 自身拒绝的调用单独记录。[原语清单](docs/development-docs/lax-coverage.md)列出逐项证据，有限样本不代表穷举所有合法调用。

| 接口结果 | 含义 |
| --- | --- |
| `equivalent` | 在已完成的图比较中，节点、连接、语义信息及嵌套结构一致 |
| `different` | 存在确定的图差异，并提供差异路径；不推出公式不等价 |
| `unknown` | 图非法、遇到未支持的语义或存在未规范化部分，并提供原因 |

直接调用 `canonicalize` 遇到非法图或不支持的语义会抛出 `InvalidJaxprError` / `UnsupportedJaxprError`；比较接口把它们转成带原因的 `unknown`。结构校验不替代 CPU 编译或数值执行，公开函数的 CPU 可执行性在独立测试中复核。

## 结论范围与证据

结论仅针对给定 jaxpr 及声明的追踪上下文，不宣称一般 Python 等价、跨平台等价或任意输入下逐位浮点相等。图不匹配仍可能数值相同；实际不等价案例另附有效输入及不同输出。将来如增加浮点代数改写，应单独记录规则、条件和警告，本轮不执行这些规则。

算法论证、JAX 适配与 review 见 [开发设计](docs/development-docs/bfs-reordering.md)，AI 标签、数值证据与模型提取来源见 [案例记录](docs/development-docs/bfs-cases.md)。pytest、Ruff、示例与清单一致性检查已纳入 [CPU CI](.github/workflows/ci.yml)；在线 CI 是否通过需由实际工作流结果确认。

[真实业务案例](docs/development-docs/business-layer-cases.md)另保存五个层级内基准与 20 个 AI 手写候选，预判和源码均 hardcode；用独立 NumPy 参照复核基准，并分别验证候选图标签及两侧重排执行。该有限案例集不代表完整上游模型或全部配置的支持证明。
