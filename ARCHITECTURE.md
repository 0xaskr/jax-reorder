# 架构与比较契约

`jax-reorder` 通过 jaxpr 结构规范化，检查 AI 生成或改写的 JAX 代码是否与原始代码等价。本文是架构与比较语义的维护入口。

## 当前实现

已实现完整计算图规范化、精确比较和函数/一阶求导入口，基线为 Python 3.12、锁定的 JAX/jaxlib 0.11.2 和 CPU。实现位于 [src/jax_reorder](src/jax_reorder/)，[tracing 与重排示例](tests/inspect_jaxpr.py)展示 BFS 编号和最终排列，[AI 组合示例](tests/bfs_reordering.py)展示经典案例及 SGLang-JAX LlamaMLP 正反例。

当前没有代数模式。新增的 `compare_functions` 核对给定签名的函数契约，并分别追踪前向/JVP/VJP；它不证明一般 Python 等价。[两轮结果参考](docs/references/equivalence-results.md)中的四类结果描述后续完整流程；图接口仍返回计算图等价、计算图不等价或无法判定。

## 流程与职责

可使用 `compare_functions` 一并记录输入输出 pytree、静态参数、可微叶子和追踪环境，或在自行核对这些条件后向 `canonicalize` / `compare_jaxprs` 提供包含常量绑定的 jaxpr。底层 jaxpr 已展平接口，不能自行恢复函数契约。函数入口要求调用方提供纯函数；追踪外的 Python 行为仍不在结论范围内。

| 路径 | 职责 |
| --- | --- |
| [core/canonical.py](src/jax_reorder/core/canonical.py) | SSA 检查、输出 BFS、依赖深度、嵌套重排、完整结构比较及诊断 |
| [core/semantics.py](src/jax_reorder/core/semantics.py) | 参数、抽象值与常量的精确编码，已知描述符适配 |
| [core/support.py](src/jax_reorder/core/support.py) | 公开 lax 原语及包装函数辅助原语的身份发现 |
| [core/unrooted.py](src/jax_reorder/core/unrooted.py) | 完整图细化、歧义搜索及明确资源预算 |
| [core/correspondence.py](src/jax_reorder/core/correspondence.py) | 构造原图双射并独立复核全部节点与绑定 |
| [core/derivatives.py](src/jax_reorder/core/derivatives.py) | 自定义一阶规则物化、残差及 symbolic zeros 模式 |
| [core/params.py](src/jax_reorder/core/params.py) | 比较与代数适配共享的参数视图及规则物化入口 |
| [algebra/compare.py](src/jax_reorder/algebra/compare.py) | 结构快路径、代数比较编排与结果汇总 |
| [algebra/adapter/](src/jax_reorder/algebra/adapter/) | 图到数学层的桥接：元数据、默认 JIT、共同调用、进程冻结 |
| [algebra/symbolic.py](src/jax_reorder/algebra/symbolic.py) | 已校验图的 SymPy 表达式构造 |
| [algebra/expression.py](src/jax_reorder/algebra/expression.py) | 符号值模型：坐标、常量、绑定重命名 |
| [algebra/sympy_ops.py](src/jax_reorder/algebra/sympy_ops.py) | 预算受控的化简、定义条件与恒等查询 |
| [algebra/trusted.py](src/jax_reorder/algebra/trusted.py) | 可信 SymPy 结果协议与可终止 worker |
| [errors.py](src/jax_reorder/errors.py) | 非法图与不支持语义的显式错误 |
| [functions.py](src/jax_reorder/functions.py) | 函数契约、追踪上下文、分别报告前向/JVP/VJP |

实现按 `errors → core → algebra → functions` 单向分层：`core` 不导入 `algebra` 或 `functions`，`algebra` 不导入 `functions`；`algebra.expression` 与 `algebra.sympy_ops` 是两个符号共享底层，`algebra.adapter` 只负责桥接。子模块路径属于实现细节，不构成稳定公共 API；对外接口以 [src/jax_reorder/__init__.py](src/jax_reorder/__init__.py) 的 `__all__` 为准。

比较使用完整结构，既不依赖变量名、打印文本，也不以摘要代替精确比较。`canonicalize` 返回新 jaxpr、规范结构、BFS/最终方程索引、原方程深度、变量引用映射及部分完成原因。原图不被修改；不含嵌套子图的方程复用原对象。

## 图规范化契约

从有序输出开始，按操作数位置反向 BFS 编号，再按 `(依赖深度, BFS 编号)` 排列。保留算子身份、全部语义参数、外部输入输出位置、结果端口、共享关系、运算分组、常量、dtype、shape、布局与分片等抽象值信息。独立方程可以换序；普通运算的操作数不交换，不做结合、分配、合并、常量折叠、DCE 或内联。

捕获常量按首次引用编号，同时保留值、类型与绑定关系。字面量与捕获常量不同，共享绑定与两份相同常量也不同。数组比较完整内容，区分有符号零、NaN 位模式和不同 float8 格式。

嵌套子图在原区域内分别规范化，保留分支顺序、循环角色及调用边界。私有接口允许实参/形参和结果槽位同步置换；位置元数据、反馈边和外部绑定必须一起保持。token 依赖照常保留；隐式有序操作及包含它们的调用保留相对顺序。未知 effects 或不透明参数显式拒绝，即使它们不影响输出也不忽略。

`cond`（包括 `lax.switch`）对数据实参与全部分支形参执行同一个置换，再进行父图 BFS。它消除闭包自动捕获位置变化，保留实际绑定；不改变选择值、分支顺序及外部接口。jit、cond、shard_map 的私有结果按父图使用路径排列；scan/while 的捕获与 carry 分角色同步处理。规则与反例见 [区域绑定](docs/design-docs/current/region-bindings.md)。

`shard_map` 内显式集合通信沿用相同图规则，保留通信轴、分组、置换、维度、mesh 与分片接口；`pvary` 辅助节点也完整保留。子图在恢复的轴环境内校验。[通信实验](docs/exec-plans/completed/validation/collective-jaxpr.md)覆盖单进程多 CPU 设备，不外推为多主机或 GPU/TPU 执行验证。

未被输出或顺序根覆盖的纯计算和常量也完整比较，通过全图标签与连接细化、必要时有界精确搜索确定编号。多个没有使用锚点的内部槽位同样有界搜索；默认预算 10,000，超限返回 `unknown`。不删除未使用计算，见[完整图与深度设计](docs/design-docs/current/complete-graph.md)。

每个 `equivalent` 都携带原图对应证据，并经独立校验器复核；规范结构相同而原始绑定校验失败时返回 `unknown`。[对应关系论证](docs/design-docs/current/original-correspondence.md)明确了独立校验与共同语义编码的边界。

自定义 JVP/VJP 注册规则按锁定版本显式物化，残差生产与消费连接起来核对。底层默认检查注册的一阶规则；函数入口按实际请求的前向/JVP/VJP 程序给结论，不外推高阶求导。前向图一致也可能因 JAX 梯度累加顺序变化而得到不同的 VJP 图，见[求导契约](docs/design-docs/current/function-and-derivative-contract.md)。

## CPU 原语支持与结果

目标范围以 `jax.lax` 及公开子模块在 CPU 的实际能力为准，不另建 dtype、shape 或参数值白名单。JAX 能处理而本项目不能解释的情况属于实现缺口，必须报告；JAX 自身拒绝的调用单独记录。[原语清单](docs/references/lax-coverage.md)列出逐项证据，有限样本不代表穷举所有合法调用。

| 接口结果 | 含义 |
| --- | --- |
| `equivalent` | 在已完成的图比较中，节点、连接、语义信息及嵌套结构一致 |
| `different` | 存在确定的图差异，并提供差异路径；不推出公式不等价 |
| `unknown` | 图非法、遇到未支持的语义或存在未规范化部分，并提供原因 |

直接调用 `canonicalize` 遇到非法图或不支持的语义会抛出 `InvalidJaxprError` / `UnsupportedJaxprError`；比较接口把它们转成带原因的 `unknown`。结构校验不替代 CPU 编译或数值执行，函数入口可通过 `check_cpu=True` 另行编译检查。未审查的 JAX/jaxlib 版本和描述符子类明确拒绝；版本升级须更新适配和验证证据。

## 结论范围与证据

结论仅针对给定 jaxpr 及声明的追踪上下文，不宣称一般 Python 等价、跨平台等价或任意输入下逐位浮点相等。图不匹配仍可能数值相同；实际不等价案例另附有效输入及不同输出。将来如增加浮点代数改写，应单独记录规则、条件和警告，本轮不执行这些规则。

算法论证、JAX 适配与 review 见 [开发设计](docs/design-docs/archive/bfs-reordering.md)，AI 标签、数值证据与模型提取来源见 [案例记录](docs/exec-plans/completed/validation/bfs-cases.md)。pytest、Ruff、示例与清单一致性检查已纳入 [CPU CI](.github/workflows/ci.yml)；在线 CI 是否通过需由实际工作流结果确认。

[真实业务案例](docs/exec-plans/completed/validation/business-layer-cases.md)另保存五个层级内基准与 20 个 AI 手写候选，预判和源码均 hardcode；用独立 NumPy 参照复核基准，并分别验证候选图标签及两侧重排执行。该有限案例集不代表完整上游模型或全部配置的支持证明。

[完整 Attention/block](docs/exec-plans/completed/validation/full-attention.md)覆盖原 GLM5 MHA、Llama GQA、CLIP encoder、Flux joint attention、未使用索引器和分模式求导，另记录真实 Pallas/上游拒绝路径。测试矩阵、CPU CI、适配快照和候选升级流程见[系统验证](docs/design-docs/current/verification-matrix.md)。
