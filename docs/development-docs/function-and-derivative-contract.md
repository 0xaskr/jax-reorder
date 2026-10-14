# 函数契约与自定义一阶求导

`compare_functions` 捕获展平前的输入输出 pytree、静态参数、可微叶子、抽象值和追踪环境，分别返回请求的前向、JVP、VJP 结果。输入可以是实际 CPU 数组或 `ShapeDtypeStruct`；实际数值仍需另行检查。接口要求调用方提供纯函数，不分析追踪之外的任意 Python 行为。

```python
from jax_reorder import compare_functions

report = compare_functions(
    original,
    candidate,
    x,
    y,
    modes=("forward", "jvp", "vjp"),
    transforms=("jit",),
    check_cpu=True,
)
for mode in report.modes:
    print(mode.mode, mode.status, mode.equivalence_kind, mode.failure_kind)
```

`static_argnums`/`static_argnames` 指定静态接口，`argnums` 选择原函数的可微位置参数；默认对动态浮点、复数叶子求导。整数等叶子仍是普通输入。`transforms` 中的 `jit`/`vmap` 顺序有意义，它们先作用于原函数，再应用所请求的一阶求导；其他变换顺序应显式包装后单独验证。

默认 `equivalence="auto"` 在结构不同后尝试理想实数代数证明；`equivalence="structural"` 保留原完整图结果。`ModeComparison.comparison` 始终保存原结构报告，`algebra` 单独保存代数规则、假设、浮点警告、证明和停止原因；`equivalence_kind` 区分 `structural`、`algebraic` 与未证明。`algebra_budget=10000` 按模式分别计数。输入签名在追踪前保留完整抽象属性，包括 memory_space 与 manual_axis_type；不能忠实还原时报告 `unknown`。

JVP 的全部切向量和 VJP 的每个输出余切向量作为独立动态输入。VJP 在同一个程序里连接前向和 pullback，不固定原始输入，不将余切向量固定为全 1。只有全部请求模式完成且等价才报告整体等价；未请求模式列入 `unverified_modes`。

## 自定义调用节点

锁定 JAX 0.11.2 的 `custom_jvp_call` 和 `custom_vjp_call` 有专用适配：

- 普通 `call_jaxpr` 始终保留调用边界与绑定。
- JVP 延迟规则物化为包含原值及非零切向量接口的图，并记录输入激活模式和输出零标记。
- VJP 延迟前向及反向分别追踪成调用区域，并在同一规则图中连上残差。`out_trees()` 的输入透传索引用实际边表达，不能只比较打印出的 fwd/bwd 名称。
- symbolic zeros 的合法非空激活模式逐项展开。普通非 symbolic 模式使用完整动态种子；整数叶子及反向返回的显式零按 JAX 语义处理。最多物化 64 个输入/余切模式组合，超出预算明确返回 `unknown`。
- 保留原 callable 供返回的 jaxpr 执行；比较键使用规则内容。没有忽略未知 callable 的通用回退。

底层 `compare_jaxprs` 默认同时比较注册的一阶规则。`derivative_scope="primal"` 明确限定为普通前向调用；函数入口使用此范围比较已经通过公开 JAX 变换展开的具体模式，避免从一阶自动递归声称高阶规则相同。规则内部调用自身的普通前向不会无限展开其高阶规则。

规则捕获的动态 tracer 或静态数组若已作为原调用实参，适配器把它还原成实际输入边，不当作固定常量。只在求导规则中捕获、没有出现在普通调用实参中的动态值，底层目前返回 `unknown`；显式函数 JVP/VJP 入口可由 JAX 展开后按实际图比较。

JAX 的延迟 VJP thunk 缓存图而不恢复 `out_trees` 的辅助状态。不同 symbolic 模式可以改变残差树，适配器核对锁定版本的 wrapper 后重新物化原 thunk，避免把上一模式的透传表误用于当前图；没有修改 JAX 缓存。测试覆盖跨模式重复比较。

延迟规则新增的 `rng_uniform` / `dce_sink` 顺序约束不会出现在普通调用的父图中，当前底层明确拒绝此类规则；显式 JVP/VJP 入口追踪实际执行图后单独检查。反向日志、不透明捕获和其他未解释的 effects 也不能按名称或对象地址放行。

`custom_vjp` 的普通 JVP，以及动态 while 的反向求导，是 JAX 本身不适用的组合，报告 `trace_rejected`。规则无法物化、未知结构或预算不足报告 `unsupported`；CPU 编译拒绝单列 `cpu_compile_rejected`。图差异与数值反例是两个不同结论。

## 整数输出的 VJP 物化修复

2026-10-14 审查发现两个误报等价的适配缺口，修复及验证见[执行计划](../exec-plans/completed/review-correctness-fixes.md)。混合浮点/整数输出在 `symbolic_zeros=False` 时，整数输出的余切必须是实际 `float0` 零数组；只有 symbolic 模式才向用户反向规则传入 `SymbolicZero`。类型会影响用户规则的 Python 分支，因此不能用同一个对象表示两种模式。回归已将物化规则与公开 `jax.vjp` 的反向执行交叉核对，并拒绝实际梯度不同的规则。

当全部输出都是整数或布尔时，自定义前向规则仍会在对浮点输入做 VJP 时执行，向 pullback 显式传入 `float0` 还会调用反向规则。必须保留完整前向、残差及反向图，不能返回空规则列表。symbolic 模式下整数余切也可能是显式 `float0` 或 `SymbolicZero`，输出模式枚举因此覆盖全部叶子的显式/符号组合；至少一个输出为显式余切时物化反向规则，并计入原有 64 个组合预算。普通模式将整数余切直接实例化为零数组。

空输出 pytree 的 VJP 仍执行前向规则，但 pullback 不调用反向规则；单独保存其前向图和残差计算即可。回归包括不同前向值、不同反向值、相同规则、空输出内的有序操作，以及普通/symbolic 两种配置；未适配的 effect 仍显式拒绝。没有可微输入时，公开 JAX VJP 直接使用普通前向，不扩展其输入激活模式。

## 梯度累加顺序的边界

不能从前向图匹配推断每种求导后图也匹配。[反馈回归](../../tests/test_feedback_bindings.py)有一个实例：同步交换 scan 的返回槽位，前向及 JVP 保持同图，但 JAX 转置按原槽位顺序累加余切向量，VJP 的 `add_any` 连接发生变化。结构模式分别返回 `equivalent`、`equivalent`、`different`；默认自动模式可以通过独立代数证明接受这次实数加法换序，并保留结构差异与浮点警告。数值抽样接近仍不能代替任何一层证明。

这一边界也出现在[四类完整业务层](full-attention.md)的独立投影换序中。该组历史证据显式使用结构模式，继续保留原验证范围：只对 hidden states 求导，权重仍为动态输入，没有宣称权重梯度及高阶模式都已验证。

## 实现与测试

- [函数入口](../../src/jax_reorder/functions.py)、[规则适配](../../src/jax_reorder/derivatives.py)。
- [AI 固定规则](../../examples/custom_derivative_cases.py)、[求导测试](../../tests/test_custom_derivatives.py)。
- [函数契约测试](../../tests/test_functions.py)：tuple/list、字典键、静态参数、动态余切向量、环境变更、追踪与编译失败。
- [原图对应校验](original-correspondence.md)：重新物化规则并复核原始绑定，语义适配本身仍属于共同信任基础。

验证为给定 CPU 配置和一阶模式的有限证据，不代表已覆盖全部高阶求导或任意 JAX 变换组合。
