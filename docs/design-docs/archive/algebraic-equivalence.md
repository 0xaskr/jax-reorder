# 旧代数证明器与 CAS 候选设计

> 状态：已被替代：独立代数证明器及其旧接口已移除；当前生产路径见 [SymPy 设计](../current/algebraic-equivalence.md)。原方案保留供追溯。

## 历史实现记录（以下为迁移前的后端）

状态：首期实现与全量 1600 项测试已通过，[执行计划](../../exec-plans/completed/algebraic-equivalence.md)已归档。用户要求不修改受保护文档，这些入口的同步记录为 [DOC-002](../../exec-plans/index.md#待处理)；后续核心重构见[标量索引 IR](algebraic-reduction-gaps.md)，本文的原始验收数字保留为基线。

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

[真实 mHC 层验证](../../exec-plans/completed/validation/deepseek-v41-layer.md)补充严格默认纯 JIT 的实参代入，使不同私有残差接口可在组合后比较；两侧完整结构相同的默认子调用用原始端口对应证据复用，保留调用求值 barrier。非默认执行属性仍保留边界。分块归约的转置仅在证明读取范围安全后接入静态零扩展，前缀指标保持完整支持域。

## 接口与实现入口

`compare_functions` 和 `compare_function_traces` 默认采用 `equivalence="auto"`，可显式指定 `"structural"`；每个模式的 `algebra_budget` 默认为 10,000。汇总和单模式结果用 `equivalence_kind` 区分 `structural`、`algebraic` 和未证明。单模式的 `comparison` 保留原结构报告，`algebra` 保存规则、前提、警告、证明、工作量和失败原因。

底层 `compare_algebraic_jaxprs(left, right)` 只覆盖已经物化的前向图；JVP/VJP 须先由函数入口分别追踪。`verify_algebraic_proof(left, right, proof)` 从原图复核证据，成功返回 `None`，篡改或无效证据抛出 `CorrespondenceError`。证明是进程内对象，未定义跨进程持久化格式。

| 实现 | 职责 |
| --- | --- |
| [algebra/compare.py](../../../src/jax_reorder/algebra/compare.py) | 结构证据直通、代数比较和结果分类 |
| [algebra_ir.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_ir.py) | 不可变 DAG、项表、精确系数和预算 |
| [algebra_lower.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_lower.py) | 原语语义、索引映射与固定区域转换 |
| [algebra_normalize.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_normalize.py) | 展开、项收集和绑定索引规范化搜索 |
| [algebra_proof.py](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_proof.py) | 原图对应和局部推导的独立复核 |

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
工作进度见[执行计划](../../exec-plans/completed/symbolic-math-integration.md)。

实现与验收：[CAS 候选](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_sympy.py)、
[标量证据复核](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_scalar_proof.py)、
[数学不等价证据](https://github.com/0xaskr/jax-reorder/blob/05abeafa1e85193ea037d9fc182f12f67fa10f9c/src/jax_reorder/algebra_inequivalence.py)、
[有限性规则](../../../src/jax_reorder/algebra/finiteness.py)。
对应回归为 [CAS](../../../tests/test_algebra_sympy.py)、
[负证据](../../../tests/test_algebra_inequivalence.py) 与
[guarded softmax](../../../tests/test_algebra_finiteness.py)。
