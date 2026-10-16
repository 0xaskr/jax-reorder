# 带索引的标量语义 IR 与代数规范化

本页记录已删除的旧索引代数引擎，源码链接固定到清理前提交；当前实现见[可信 SymPy 数学路径](algebraic-equivalence.md)。

状态：实现与最终全量 1775 项测试已通过。[执行计划](../exec-plans/completed/algebraic-reduction-gaps.md)记录验收。沿用[索引代数契约](algebraic-equivalence.md)，结构比较不变。

## 标量索引 IR 核心

本轮按用户确认升级为核心重构：不再采用张量视图来源追溯。每个 Tensor 表示有序输出坐标下的标量 Expr；输入读取带符号整数 Index（加法、整数倍、整除、取余），普通 reshape、transpose、broadcast、slice 仅复合索引映射。标量 call 的参数共享当前索引环境；保留语义的未知 atom/区域边界具有独立参数环境。二者不得混淆。

sum、matmul 和 scan 累加都产生有限求和。规范化对项的求和域执行有证据的双射：合并连续混合进制轴、把一个轴拆成商和余数、平移连续区间。平移后的区间采用前缀差分，使相邻分块通过系数收集合并。上述规则只检查 IR 的索引与域，不识别原始算子组合。

求和域重写记录具体轴、除数和位移见证；独立校验器检查双射条件和所有前缀项，再复核既有绑定置换和多项式推导。指数项数在分配前计入预算。索引整数和范围保持符号，百万长度不产生百万节点。

标量调用内的局部求和与外层绑定通过作用域深度区分。索引替换、绑定重命名及使用分析递归进入标量参数，但不进入未适配张量边界。非零区间的前缀扩展只允许全定义表达式，且每次读取的被平移坐标必须一致，避免扩展到负索引。一般递推、动态索引和未审核原语仍保留边界；无证明返回 unknown。

## scan 的加法递推

检查实际 body 图、输入输出树、carry 槽位、长度、方向、展开参数和抽象属性。每个被转换的递推须为自身 carry 加上不依赖变化 carry 的增量，按序列首轴构造符号有限求和，并保留初值。不得把 `c-x`、`2*c+x` 或错误输入绑定当作同一归约。

所有 ys 和其他 carry 都必须逐项处理；依赖变化 carry 的历史输出不能丢弃。JVP/VJP 生成的多 carry、恒等 carry 及广播输出按实际图分别检查，不从前向结果外推。语义适配属于共同信任基础，证明校验从原图重新核对这些前置条件；多项式与求和推导继续独立复核。

## abs 与 where

在实数解释下，严格核对零比较、比较对象和两条分支引用，把合法的正负选择与 abs 统一到同一个数学表示。仅消除已审核的默认纯 select/broadcast 包装，保留不透明参数、上下文和其他调用边界。

`x > 0` 与 `x >= 0` 在零点选择不同分支；前向的 `x` 和 `-x` 在实数零点相同，不代表它们的 JAX JVP/VJP 约定相同。梯度测试必须包含零点，并拒绝将不同导数强行判为等价。NaN/Inf 和有符号零继续受原有浮点警告约束。

## 验证与边界

用原图追踪、规则正反例、独立证明篡改和资源预算共同验收，抽样执行仅作补充。大维度保持符号，不随元素数量展开。当前设计仅覆盖上述可审查规则，不新增任意索引或一般循环的等价判定。

## 实现分层

| 层 | 表示与职责 | 实现 |
| --- | --- | --- |
| 语义 IR | `Tensor` 的有序自由坐标、`Expr` 标量 DAG、`Recurrence` 的状态/初值/更新/序列输出 | [IR](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_ir.py)、[转换器](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_lower.py) |
| 整数索引 | 常量、轴、加法、整数倍、正整数整除和取余；标量调用参数内的绑定有明确作用域深度 | [索引运算](../../src/jax_reorder/algebra_indices.py) |
| 求和域 | 零基矩形和 `FiniteRange(start, stop)`；混合进制合并/拆分、区间平移、带符号前缀 | [域检查内核](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_domains.py) |
| 原语接入 | 已审核的逐点函数/转换、零比较/选择、有限递推与独立累加求解 | [标量函数](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_scalar.py)、[条件](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_piecewise.py)、[递推](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_scan.py) |
| 证明 | 多项式展开、域见证规划、绑定编号搜索；原图重建后独立复核每个见证与最终项表 | [规范化](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_normalize.py)、[复核](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_proof.py) |

例如 `x.shape == (3, 4)` 时，`reshape(-1)` 的第 `t` 个元素为 `x[t // 4, t % 4]`。完整归约的 `0 <= t < 12` 可由见证拆为 `t = 4*i+j, 0 <= i < 3, 0 <= j < 4`。这条规则同样适用于被 `sin` 或条件函数包住的读取。

切片 `[a:b:s]` 的第 `i` 个元素为 `x[a+s*i]`，部分轴归约无需保留 slice 边界。连续域的相邻前缀抵消保留覆盖重数；缺口、重复、重叠不会被合并为完整求和。步长会留在索引中，本轮不提供任意步长子域的覆盖求解。

证明格式升级为进程内 `version=2`，每个原始展开项记录带分支位置的域指令；校验器拒绝漏项、错轴、非法除数、不安全位移及错误前缀分支。语义转换、整数化简和小型域检查内核是显式共同信任基础；校验器不调用规范化器的域规划或编号搜索。独立小域解释器按整数和有限求和定义检查这些公共内核，不将执行抽样当作函数等价证明。

## 本轮保留的边界

- 已接入的普通实数形状原语统一为索引映射；动态索引、gather/scatter、反转、任意稀疏求和域和未审核原语尚无完整数学适配。
- 类型转换作为显式标量函数保留；同 dtype 的 weak_type 转换不改变元素值，最终输入输出契约仍比较完整抽象属性。低精度完整 reshape 归约已覆盖，但不同位置的舍入转换不自动消去。
- 非默认精度、布局、分片、effects 继续保留或拒绝。选择包装只在严格校验其内容和元数据后透明；任意 jit/cond/while 跨边界内联仍不在本轮范围。后续[真实 mHC 层验证](deepseek-v41-layer.md)接入了严格默认纯 JIT 的有序实参代入、共同调用复用及安全静态零扩展，具体条件见该记录。
- scan 先构建有限递推，再证明独立累加和不变状态。交叉反馈、非线性变化状态、依赖变化状态的历史输出保持原区域语义，不展开长度。
- 当前规则并不完备；复杂索引表达式或域规划预算耗尽仍返回 `unknown`。百万元素测试要求同构程序的工作量不随元素数量增长。

验收用例见[组合与证据测试](../../tests/test_algebra_scalar_ir.py)、[独立索引解释器](../../tests/test_algebra_index_rules.py)、[scan](../../tests/test_algebra_scan_sum.py)和[abs/where](../../tests/test_algebra_abs_where.py)。

最终验收：1775 passed，1 个上游弃用 warning；Ruff、格式、示例、两项清单、双/四 CPU 各 22 组通信验证和文档检查通过。完整命令与保留边界见上方归档计划。
