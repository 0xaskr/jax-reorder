# Oryx、MLIR 与 StableHLO 参考路线

本项目优先评估 Oryx 的 jaxpr 表达式转换和重写设计，以 MLIR 的结构比较及 StableHLO 的算子语义作为参考。比较对象仍是 jaxpr，判断依据仍是本项目的[比较契约](../../ARCHITECTURE.md)。TVM 不纳入当前路线。

## Oryx：优先验证 jaxpr 表达式转换

- [仓库](https://github.com/jax-ml/oryx)
- [实验性 jax_rewrite 模块文档](https://www.tensorflow.org/probability/oryx/api_docs/python/oryx/experimental/matching/jax_rewrite)
- [PyPI 发行包](https://pypi.org/project/oryx/)

公开文档中的 `jaxpr_to_expressions` 将 jaxpr 转成输出表达式；`Primitive` 保存算子、操作数和参数，`Part` 表达多输出索引，`BoundExpression` 保存闭包常量。模式匹配与重写机制可用于研究如何忽略独立方程的排列差异。

先核对发行包中的实际模块与 JAX 兼容性，再判断是否直接依赖。表达式转换和规则应用本身不保证等价；需要另行检查共享节点、未被输出引用的方程、effects、嵌套计算与函数 pytree 是否满足本项目契约。不能直接把表达式的 `==` 当作函数级比较结果。

早期 Oryx 评估原文当前缺失，不能作为可重放证据；依赖兼容性缺口记录在[待核验记录 IR-001](../exec-plans/index.md#待核验旧记录)。当前实现使用本项目自己的完整图规范化与对应关系校验。

## MLIR：结构比较与变量映射

- [仓库](https://github.com/llvm/llvm-project/tree/main/mlir)
- [OperationEquivalence](https://mlir.llvm.org/doxygen/structmlir_1_1OperationEquivalence.html)
- [Operation Canonicalization](https://mlir.llvm.org/docs/Canonicalization/)

参考 `OperationEquivalence` 如何比较操作及嵌套 region、维护 SSA 值之间的对应关系，并显式配置比较策略。本项目的输入位置和操作数顺序必须保持，不能照搬忽略操作数关系或属性的选项。

MLIR 的 canonicalizer 是尽力而为的重写流程，官方没有定义一个覆盖所有程序的唯一规范形式。它的结构比较机制和规范化 pass 是不同层次；不能从两次规范化结果不同推出程序不等价。

## StableHLO：算子与执行语义参考

- [仓库](https://github.com/openxla/stablehlo)
- [规范](https://openxla.org/stablehlo/spec)
- [参考解释器](https://openxla.org/stablehlo/reference)

StableHLO 定义机器学习程序的操作、类型、参数和执行语义。它适合辅助审核 shape、dtype、精度属性、控制流与有副作用操作的处理，并为后续 lowered IR 实验提供参照。

StableHLO 不提供本项目所需的完整 JAX 函数等价判定。若后续比较 lowering 结果，仍需保存追踪上下文、闭包常量和函数输入输出 pytree，明确 JAX 到 StableHLO 的映射与版本；lowering 或解释器运行成功不能替代等价检查。

## 采用顺序

1. 实测 Oryx 模块与版本，完成最小转换实验并记录反例和支持缺口。
2. 参考 MLIR 设计结构比较及变量映射，保留依赖图的共享关系与所有语义字段。
3. 用 StableHLO 规范辅助审核算子语义；是否增加独立的 StableHLO 比较入口另行设计。

结构模式先于代数模式。浮点交换律或结合律仍须显式启用、记录规则并发出警告，不保证误差很小或逐位相等。
