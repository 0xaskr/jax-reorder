# 编译器与图转换路线评估

> 状态：历史研究与选型证据，候选版本、探针和建议按原时点保留，不作为当前已接入能力。生产选择见[当前代数设计](../../../../design-docs/current/algebraic-equivalence.md)。

日期：2026-10-15。目标为两份 JAX 图的结构/数学语义比较，不以运行模型、性能或数值误差为验收。本组实际运行 MLIR/StableHLO 与 TVM 符号探针；Alive2、Oryx、Herbie、Daisy 只做官方文档/源码入口核对。

## 实测环境与复现

仓库 Python 3.12.3、JAX/jaxlib 0.11.2；隔离环境安装 `apache-tvm==0.27.0.post1`、`apache-tvm-ffi==0.1.14.post1`。默认 uv 缓存只读，随后使用 `/tmp` 缓存；普通沙箱下载 DNS 失败后，经网络审批完成隔离安装。没有更改项目依赖。

[探针源码](../../../../../tools/experiments/equivalence_backends/compiler_probe.py)：

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline \
  python -m tools.experiments.equivalence_backends.compiler_probe mlir
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv venv --python .venv/bin/python \
  /tmp/jax-reorder-equivalence-eval/compiler/venv
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv pip install \
  --python /tmp/jax-reorder-equivalence-eval/compiler/venv/bin/python \
  apache-tvm==0.27.0.post1 apache-tvm-ffi==0.1.14.post1
/tmp/jax-reorder-equivalence-eval/compiler/venv/bin/python \
  tools/experiments/equivalence_backends/compiler_probe.py tvm
```

## MLIR / StableHLO

官方来源：[OperationEquivalence](https://mlir.llvm.org/doxygen/structmlir_1_1OperationEquivalence.html)、[canonicalization 契约](https://mlir.llvm.org/docs/Canonicalization/)、[StableHLO 语义](https://openxla.org/stablehlo/spec)。

MLIR 提供结构比较和贪心 canonicalization；文档明确 canonicalization 是 best effort，没有全局唯一规范形式。StableHLO 提供带形状和算子语义的交换格式。两者可作为转换与局部整理基础，不能仅凭规范化后不同推出数学不等价。

实测使用 jaxlib 自带绑定：解析 IR，运行 `builtin.module(canonicalize,cse)`，再调用 `is_structurally_equivalent`；仅忽略源码 location，不忽略属性、类型或参数。没有自定义重写规则，没有生成机器码。

| 用例 | 规范化前 | 规范化后 | 能得出的结论 |
| --- | --- | --- | --- |
| i32 `x+0` 与 `x` | 结构不同 | 结构相同 | 上游规则能够证明这一局部恒等式 |
| i32 `(x+y)*z` 与 `x*z+y*z` | 结构不同 | 结构不同 | 本次流程未完成数学证明，unknown |
| StableHLO 2×2 i32 张量的同一分配律 | 结构不同 | 结构不同 | 本次流程未完成数学证明，unknown |

所有 IR 均通过语法/类型 verify；这不是等价证明。初次探针未忽略 location，连消零控制例也返回不同，修正比较选项后控制例通过。因此不能把解析位置差异当数学证据。

适配结论：可复用 IR 和已审核的局部规则，不建议为了当前数学等价瓶颈而替换已经工作的结构比较器。完整 GLM 的 scan、gather、归约和 top-k 还需数学语义适配。

## Apache TVM

官方来源：[安装](https://tvm.apache.org/docs/install/)、[当前 tvm.sym API](https://tvm.apache.org/docs/reference/api/python/sym.html)、[旧 tvm.arith API](https://tvm.apache.org/docs/reference/api/python/arith.html)。

旧资料中的 `tvm.arith` 在本轮 PyPI wheel 中不存在；实际入口为 `tvm.sym.Analyzer` 和 `tvm.tirx`。探针记录真实版本和模块，避免将 API 迁移误报成缺少推理能力。

| 用例 | 实际结果 | 解释 |
| --- | --- | --- |
| E2 int64 分配律 | `can_prove_equal=True`，差为 0 | 原生符号整数规则可用 |
| N1 `2*x` 与 `x`，无约束 | 相等和普遍不等都未证明 | `False` 不是数学不等价的返回值 |
| N1，声明 `x>0` 的可满足子域 | 不等关系可证明 | 有条件的不等价证据 |
| N4，声明 `i=n` | `<n` 与 `<=n` 的选择结果不等 | 索引边界推理可用 |
| float32 分配律 | unknown | 当前 IR 的默认浮点模型不能直接充当理想实数 CAS |
| E5 exp 恒等式 | unknown | 现成 analyzer 未证明 |
| E3 符号长度 n 的缩放归约 | unknown | 对构造的两个 Reduce 表达式未证明相等 |

E3 没有枚举 n 个元素，也没有运行归约。浮点 IR 用例的结果只描述该 API 的默认推理范围，不表示实数恒等式不成立。运行时出现 TE tensor-load 的文本打印警告，不影响 analyzer 返回值，但没有将其隐藏。

适配结论：适合索引、范围、reshape/slice 约束等局部辅助；不是完整张量数学等价引擎。生产接入仍需将 JAX 的索引语义与 TVM 的整数语义对齐。

## 其他候选的范围核对

| 项目 | 证据 | 与本需求的关系 |
| --- | --- | --- |
| [Alive2](https://github.com/AliveToolkit/alive2) | 官方 README 与[论文](https://users.cs.utah.edu/~regehr/alive2-pldi21.pdf)，未本地构建 | LLVM 翻译验证与 refinement 检查，循环采用有界策略。可以证明低层变换，但需要额外 lowering，不能直接代替高层张量代数；单向 refinement 也不能直接当双向数学等价 |
| [Oryx matching](https://www.tensorflow.org/probability/oryx/api_docs/python/oryx/experimental/matching/jax_rewrite) | 官方 API 与[发布记录](https://github.com/jax-ml/oryx/releases)，未安装 | 可将 jaxpr 变成表达式并按用户规则改写；规则正确性由提供者负责。不是现成两程序等价/不等价判定器，当前 JAX 版本兼容性尚未验证 |
| [Herbie](https://herbie.uwplse.org/) | 官方项目与[报告定义](https://herbie.uwplse.org/doc/latest/report.html)，未安装 | 主目标是浮点精度优化，报告含采样误差；与本轮拓扑/数学等价目标不匹配 |
| [Daisy](https://github.com/malyzajko/daisy) | 官方 README，未安装 | 主目标是舍入误差分析和有限精度优化；数学范围分析可参考，但不作为本轮图等价主后端 |

没有因为这些工具未安装而断言它们有能力缺陷；排除主后端的依据是输入语言、输出契约与用户目标的差距。

## 真实 GLM 图覆盖清单

[清单脚本](../../../../../tools/experiments/equivalence_backends/glm_inventory.py)在禁止后端编译与 eager 数组运算的条件下追踪 glm/FP32 的 base 和十个改写版本：

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline \
  python -m tools.experiments.equivalence_backends.glm_inventory \
  --json /tmp/jax-reorder-equivalence-eval/glm-inventory.json
```

实际为 11 份图，base 递归包含 196 条方程；统计是嵌套图的语法出现次数，不是执行次数。全部版本合计 33 种 primitive：

```text
add and approx_top_k broadcast_in_dim convert_element_type custom_jvp_call
div dot_general dynamic_slice dynamic_update_slice exp gather ge gt iota jit
le lt max mul ne reduce_max reduce_sum rem reshape scan select_n sign slice
squeeze stop_gradient sub top_k
```

因此候选必须同时面对 contraction、分段条件、动态读取、循环区域和排序/选取的组合。`approx_top_k` 不能未经语义论证替换为精确 top-k；原输出还包含索引，不能只比较 attention 数值表达式。小型符号探针成功不能外推为完整 layer 已通过。

## 结论

本组优先保留 TVM 的索引约束能力作为备选辅助；MLIR/StableHLO 作为 IR/规则来源。Alive2、Oryx 不是当前数学等价瓶颈的直接替代方案，Herbie/Daisy 不进入本轮主路线。最终与其他三组的符号和张量结果一起决策。
