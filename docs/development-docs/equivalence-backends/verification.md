# 张量验证与 SMT 后端评估

评估日期：2026-10-15。对应[统一计划](../../exec-plans/completed/open-source-equivalence-backends.md)。本工作线没有运行 JAX layer、随机抽样或数值精度实验，也没有修改生产依赖。

## 结论与证据等级

| 候选 | 实际证据 | 对本仓库的具体价值 | 当前不能据此声称的能力 |
| --- | --- | --- | --- |
| Z3 5.3.0.0 | 隔离环境运行符号探针 | 实数多项式、条件分支、整数索引约束、精确反例模型 | 原生 JAX 图导入、任意大小张量归约、真实 exp 的完整自动证明 |
| cvc5 1.4.2 | 隔离环境运行符号探针 | 上述能力，加原生实数 exp；本次直接证明 E5 | 复杂超越函数组合一定可判定，或所有 SMT `sat` 都可直接当源程序反例 |
| TensorRight 0.1.0.0 | 固定提交源码核对；缺 Haskell 工具链，未构建 | 已有张量语义、索引对应义务、XLA/TASO 改写规则及规则验证框架 | 开箱即用的完整 GLM layer 等价判定器；本地规则运行通过 |
| Lean 4 / mathlib4 | 官方文档与声明核对；缺 Lean 工具链，未运行 | 实数 exp、有限求和、矩阵定理及可复核证明库 | 已有自动 JAX 后端或无需新增建模的完整图判定器 |

现有证据支持先把 SMT 看作**明示语义的局部证明及反例后端**。TensorRight 的复用内容更贴近张量，但还需审计语义和补齐算子，不能因其名字含 verification 就直接赋予源 JAX 结论。Lean/mathlib 适合长期维护可复核规则，接入成本包含表达式模型、翻译正确性和证明搜索。

## 可复现环境与命令

Python 为 3.12.3，依赖只放在 `/tmp/jax-reorder-equivalence-eval/verification/venv`。本次安装的确切版本是 `z3-solver==5.3.0.0`、`cvc5==1.4.2`。默认沙箱下载曾因 DNS 受限失败；经网络权限审批后安装成功。这不是求解能力失败。

```bash
UV_CACHE_DIR=/tmp/jax-reorder-equivalence-eval/verification/uv-cache uv venv --python 3.12 /tmp/jax-reorder-equivalence-eval/verification/venv
UV_CACHE_DIR=/tmp/jax-reorder-equivalence-eval/verification/uv-cache uv pip install --python /tmp/jax-reorder-equivalence-eval/verification/venv/bin/python z3-solver==5.3.0.0 cvc5==1.4.2
/tmp/jax-reorder-equivalence-eval/verification/venv/bin/python tools/experiments/equivalence_backends/verification_probe.py --timeout-ms 3000
```

可复现探针：[verification_probe.py](../../../tools/experiments/equivalence_backends/verification_probe.py)。运行输出临时保存在 `/tmp/jax-reorder-equivalence-eval/verification/results.json`，不提交生成日志。

脚本对已编码数学式提交“存在符合前提的输入使两输出不同”的查询。`unsat` 表示该编码及前提下等价；`sat` 提供反例；`unknown` 保留求解器原因。没有把待证恒等式注册为公理。脚本没有请求证明证书，因而这里的 `unsat` 信任所用求解器，不等于已接上仓库独立证明校验器。

## 符号探针结果

| 用例 | Z3 | cvc5 | 本次实际编码及限制 |
| --- | --- | --- | --- |
| E1 独立节点换序 | 未单测 | 未单测 | 现有结构比较职责，不为 SMT 重新建立调度 IR |
| E2 分配律 | `unsat`，等价 | `unsat`，等价 | 任意实数 `x,y,z`，无附加公理 |
| E3 收缩与缩放交换 | `unsat`，等价 | `unsat`，等价 | 一个任意输出元素、长度 3 的收缩；所有元素为任意符号实数 |
| E4 dot 与乘积归约 | `unsat`，等价 | `unsat`，等价 | 长度 3；按 dot 与广播乘积求和定义分别展开，第二侧倒序求和 |
| E5 `exp(x+y)=exp(x)exp(y)` | `unknown`，探针没有真实 exp 编码 | `unsat`，等价 | cvc5 原生 `Exponential`，未增加 exp 公理 |
| E6 正权重归一化引理 | `unsat`，等价 | `unsat`，等价 | 两个正权重、任意 mask、共同正缩放；全 mask 由显式分母零分支返回零 |
| E6 全 mask 分支 | `unsat`，等价 | `unsat`，等价 | 单独约束两 mask 均为假，证明输出为零 |
| E6 原生 exp 平移 | 未编码 | `unknown`，3 秒超时 | 两个实数 logits、任意实数平移、任意 mask、显式全 mask 防护 |
| N1 缺少缩放 | `sat`，不等价 | `sat`，不等价 | 长度 3，缩放值为正且不等于 1 |
| N2 错误归约轴 | `sat`，不等价 | `sat`，不等价 | 2×2 张量首个输出：行和与列和不同 |
| N3 删除 ReLU | `sat`，不等价 | `sat`，不等价 | 两者均找到 `x=-1` |
| N4 `< length` 改为 `<= length` | `sat`，不等价 | `sat`，不等价 | 符号整数索引、有效范围和动态 length，含 length=0 边界 |
| N5 错误 exp 加法，无附加约束 | 未支持真实 exp | `unknown`，3 秒超时 | 未将超时报为不等价 |
| N5 约束 `x=y=0` 的反例义务 | 未编码 | `sat`，不等价 | 原生 exp(0) 的精确符号模型，左侧 1、右侧 2；不是浮点抽样 |
| U1 资源预算 | `unknown`，资源上限 | `unknown`，RESOURCEOUT | 将单次求解资源预算设为 1，确认未知分支 |
| U1 把 exp 当未解释函数 | 原始查询 `sat`，源语义结果仍 `unknown` | 未单测 | 未解释函数模型不一定是真实 exp，禁止误报数学不等价 |

这些是数学符号验证，不是对模型的执行。但 E3/E4 使用了有限小维度展开，**不是任意收缩长度的归纳证明，也不是完整 GLM 改写已经通过**。E4 最关键的工作在于正确生成有限求和式；SMT 并没有自动发现 JAX dot 的含义。

E6 的正权重引理也没有自动证明 `exp(-inf)=0` 或解析源图的 `-inf`。本次人为给定数学转换：被 mask 的权重为零，未被 mask 的权重为正。若用于完整图，适配器必须证明这一转换及全 mask 分支与原图一致。原生 exp 平移案例表明：已支持 exp 且简单恒等式可证，并不意味着组合后的 softmax 义务也能在同一预算内完成。

Z3 的官方[算术说明](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/)明确区分不同算术片段，并说明除零在 SMT 中是总函数但结果未充分指定。因此适配器必须约束分母或显式表示未定义情形，不能把 SMT 的任意除零模型当作 JAX 的 NaN 语义。[cvc5 超越函数说明](https://cvc5.github.io/docs/cvc5-1.0.2/theories/transcendentals.html)提供实数 exp 等原语；所引文档为历史版语义说明，实际执行版本由上表明确记录。[Python API 的 Exponential](https://cvc5.github.io/docs/cvc5-1.1.1/api/python/pythonic/arith.html)也是历史版官方接口说明，当前 1.4.2 的可调用性已由探针验证。

## TensorRight：现成张量规则与接入缺口

源码来自官方仓库，克隆命令及固定提交：

```bash
git clone --depth=1 https://github.com/ADAPT-uiuc/TensorRight.git /tmp/jax-reorder-equivalence-eval/verification/TensorRight
git -C /tmp/jax-reorder-equivalence-eval/verification/TensorRight rev-parse HEAD
# b3e3a533ff67568db127783cc353ebe04142929f
```

[package.yaml](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/package.yaml) 声明版本 0.1.0.0、Apache-2.0，依赖 Grisette 0.11 系列与 SBV；[stack.yaml](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/stack.yaml) 使用 nightly-2025-01-06 和固定 Grisette 提交。当前 `shutil.which` 对 `stack`、`ghc` 均返回 `None`，因此没有执行 `stack build/test` 或 `make verify`。没有为此无界安装 Haskell 工具链。上游 README 的已验证规则数属于**上游报告，未本地复验**。

值得复用的现成内容：

- [matmul2D 规则](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/rules/taso/matmul2D/Main.hs) 已有矩阵结合律、点积结果与操作数间的标量缩放移动、转置和拼接规则；与 E3 接近。其 `matmulScalarLinear` 缩放的是右操作数，不能直接冒充本仓库缩放 Q 的完整测试已经完成。
- [XLA dot 规则](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/rules/xla/dot/Main.hs) 已有 dot 转为广播乘法再 reduce 的规则，与 E4 接近。输入输出轴及 contraction 索引对应仍需显式映射。
- 同一 matmul2D 文件明确把右分配律及分块拼接的部分规则留在源码而不从 `main` 调用，原因涉及 reduction binder 和 SI 对应能力。因此“存在规则函数”不等于“后端已经支持并验证它”。

算子和语义源码核对：

| GLM 所需内容 | TensorRight 此提交的实际范围 |
| --- | --- |
| `dot_general`、广播、求和 | DSL 有 Dot、Broadcast、Reduce；轴映射、batch 轴、求和对应需由适配器构造 |
| 比较、where、ReLU | 有 Compare、Select、逐元素 Max；可组合，但不是现成 JAX 转换器 |
| exp | `NumUnaryOp` 含 Exp，有限实数传至 Grisette 的 exp；仅“有原语”不能代替证明复杂组合 |
| `-inf` | 数值域有有限值、带符号 Inf、Unknown；含 `exp(-inf)=0` 分支，详见下文问题 |
| `reduce_max` | DSL 的 Reduce 是求和 binder；未发现通用 max 归约构造，逐元素 Max 不等于 reduce_max |
| `top_k`、`approx_top_k`、`gather` | 完整 Expr 构造列表中没有对应项；需要补语义，或用有证明的公共子图边界保留 |
| `dynamic_slice` / update | 有 DynamicSlice、DynamicUpdateSlice；仍需核对 JAX 的边界和参数约定 |
| reshape / transpose | 有 Relabel 与 ReshapeDegenerate；不是任意 JAX reshape 已适配 |
| `scan` / `while` / `jit` / custom JVP | Expr 构造列表没有相应控制流/调用区域；需外部区域处理及绑定证明 |
| dtype | 使用 Bool/Int/Real 数学域；不能未经说明抹掉 JAX cast、整数宽度与不同类型的语义 |

上述列表直接依据固定提交的 [Expr 定义](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/src/TensorRight/Internal/DSL/Expr.hs)、[DSL](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/src/TensorRight/Internal/DSL/DSL.hs)、[Typed 数值和归约操作](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/src/TensorRight/Internal/Core/Tensor/Typed.hs)。主工作线的真实 GLM inventory 为原图及十个改写共 33 种 primitive，原图递归计 196 条方程；本工作线未声称完成这些图的转换。

### 特殊值审计发现

[TensorInt.hs](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/src/TensorRight/Internal/Core/Tensor/TensorInt.hs) 的源码明确给出 `TensorReal = TensorNum SymAlgReal`；`Inf True` 是正无穷，`Inf False` 是负无穷。第 177–182 行的 `negate` 对 `Inf lv` 返回同一个 `Inf lv`，没有翻转符号。**已确认源码分支，尚未构建复现**；按通常扩展实数语义，这会使取负后的无穷符号错误。接入前应在 Haskell 中对 `negate negInf` 和 `negate posInf` 做回归，并核对下游减法。

同文件第 269–275 行明确实现负无穷指数为零；第 318–329 行的实数除法对非有限情形返回 Unknown，有限分支使用 `fdivOr 0`。后者的零分母 fallback 与源图未定义值是否一致，必须单独处理。这里关注的是数学定义域和显式特殊值语义，不能因为本轮不验证浮点误差就略过。

另一个待运行审计点：Typed.hs 的 `numUnaryOp` 会把操作应用于 `TensorElemSum` 内部并保留 Sum 标记；其求和绑定并不是普通数值。需专门检查 `exp(reduce_sum(x))` 与 `reduce_sum(exp(x))` 这类反例是否被正确拒绝。仅凭目前静态阅读，不宣称已复现错误证明。

### 判定结果不能直接转录

[Core.Verify](https://github.com/ADAPT-uiuc/TensorRight/blob/b3e3a533ff67568db127783cc353ebe04142929f/src/TensorRight/Internal/Core/Verify.hs) 区分求解器 Unsat、其他求解失败和返回模型，并在失败时打印索引及访问。模型反驳的可能是所给 SI 对应、shape/访问有效性或局部证明义务；它不必然构成原始完整张量输出不同的证据。适配器不能把统一的 FAIL 文本或任一中间 SAT 直接映射成“数学不等价”。

## Lean/mathlib：可复用定理，不是现成自动前端

证据范围为 2026-10-15 查阅的 mathlib4 官方文档和 master 文件；没有运行 Lean，因 `lean`、`lake` 均不在 PATH。查阅的 [lean-toolchain](https://github.com/leanprover-community/mathlib4/blob/master/lean-toolchain) 页面声明 `leanprover/lean4:v4.35.0`；这是该文档快照的工具链信息，不是本地已安装版本，也不是已固定完整 mathlib 依赖。

已有明确可复用内容：

- [Real.exp_add、Real.exp_zero](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Analysis/Complex/Exponential.html) 是现成定理，E5 无需自行从微积分重证。
- [矩阵乘法定义及定理](https://leanprover-community.github.io/mathlib4_docs/Mathlib/Data/Matrix/Mul.html) 包含有限求和定义、结合律和标量作用，适合表达 E3/E4 的任意有限维度版本。
- [官方教程](https://leanprover-community.github.io/mathematics_in_lean/C02_Basics.html) 展示 `rw`、`ring` 等自动化；它们依赖表达式已经建模到相应代数结构。`ring` 不是通用 exp、mask、top-k 或循环等价判定器。

实际缺口包括 JAX primitive 到 Lean 定义的可靠翻译、张量轴及索引约束、带 `-inf` 的扩展值域、top-k 的值与索引/并列规则、循环不变量、失败分类以及自动选取定理。Lean tactic 失败表示没找到证明；若要输出不等价，还要构造并证明一个有效反例。未发现现成直接接收当前 GLM jaxpr 的 mathlib 接口；此结论是本次检索范围内的评估，不声称不存在其他研究工程。

## 对下一步接入的约束

1. 保留当前结构比较；从已审计的张量子图生成数学义务，优先覆盖缩放、有限归约、分段谓词和索引边界。
2. 若使用 Z3/cvc5，先定义好有限求和与特殊值转换；小维度展开适合验证适配器，正式大图要保留符号索引或复用归约定理，不能以本探针替代可扩展设计。
3. 反例必须回到完整源语义校验：局部子图差异可能被下游操作消去，未解释函数模型可能不可实现，索引对应失败也可能只是证明方法不足。
4. TensorRight 可用于审计及复用既有张量规则，但应先完成上述特殊值和非线性归约反例复现，再决定其可信边界；完整 GLM 支持还缺多项实际算子/区域语义。
5. Lean/mathlib 可作为可复核规则库路线，暂没有本地运行证据支持把它列为最快的现成全图自动后端。
