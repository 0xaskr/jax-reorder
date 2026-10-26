# 可本地修改的 JAX 转换器、符号张量与公式资产

> 状态：历史研究与选型证据，候选版本、探针和建议按原时点保留，不作为当前已接入能力。生产选择见[当前代数设计](../../../../design-docs/current/algebraic-equivalence.md)。

日期：2026-10-16。本文记录 A 阶段的整体架构评估、原版探针和隔离 fork 改造；当时生产比较器与正式依赖未改。
执行入口：[更新后的迁移计划](../../trusted-sympy-backend.md)。

用户随后要求实施，生产 fork、默认 SymPy 路径和 12 组 Nerva 公式已接入；
实现与验收另见[当前设计](../../../../design-docs/current/algebraic-equivalence.md#2026-10-16-可信-sympy-默认数学后端)。
下文原版/改版数字和“下一步”保留为选型当时的证据，不代替生产回归结果。

## 结论与评价前提

建议优先推进 **本地维护 jax2sympy 转换前端 + SymPy 数学表示/判定 + nerva-jax/nerva-sympy 成套验证资产**。
这是基于本轮可复现改造的开发路线建议，不是已经完成的生产接入。

用户明确允许拉取并修改开源库。因此原版差异用于估计改造量，不作为淘汰依据。本轮仅修改 jax2sympy
的两个文件，保留其遍历、分派及大部分映射；在实际 nerva-jax 抽象图到 nerva-sympy 参考公式的
**10 组对应检查中全部得到恒等结论**。这比仅凭 README 判断其可复用性更有依据。

下一步的主要架构工作是：将逐元素表示扩展为可替换的符号张量构造，补充条件/来源元数据和区域接口。
这些任务可以围绕 fork 演进，并复用本仓库已经实现的索引和区域适配，不需要从零另起一套转换器。

本轮没有执行具体模型、训练循环或 JAX 数组数值比较。实际 tracing/转换期间阻断了 JAX 编译和 eager
primitive 执行；SymPy 数学结论按用户决定直接信任。没有尝试重新独立证明 SymPy 的内部算法。

## 候选与实际证据

| 项目/组件 | 证据等级 | 可以加速的工作 | 本地维护时的重点 |
| --- | --- | --- | --- |
| jax2sympy | 固定源码、原版和本地改版实跑 | jaxpr 遍历、primitive→SymPy 映射、逐元素表达、嵌套 jit | 保留主干；扩展符号张量、条件和区域接口 |
| nerva-jax + nerva-sympy | 两个固定源码、10 组对应检查实跑 | 可追踪 JAX 源码与现成符号公式的成套验证 | 抽取通用函数、明确布局和域、构造等价/错误改写 |
| SymPy 原生张量表示 | 1.14.0 实跑 | 符号矩阵、收缩、索引求和；无须枚举维度 | 映射 JAX 轴/坐标、保留条件；继续验证复杂组合 |
| tensorgrad | 固定源码、符号分配律实跑 | 命名轴、张量网络、共享表达式及符号化简 | 与 SymPy 表示比较；需要 JAX 导入及后端桥 |
| sympy2jax | 固定源码、公式往返与 tracing 实跑 | 将公式变为 JAX 测试程序 | 它的反向接口针对自己的 SymbolicModule |
| AutoHOOT | 固定源码、旧 JAX 导入接口检查 | JAX→einsum 图、张量图化简、局部 SymPy 桥 | 更新 tracing API，复用收缩映射和图操作 |
| Quax | 固定源码/文档 | primitive 重解释和多分派基础设施 | 数学映射仍需提供；拦截 materialise 执行回退 |
| PyTensor | 官方文档 | 张量图表示、图重写组织和变换资产 | 需 JAX/SymPy 桥；优化器不是三态判等接口 |
| Symbolics.jl | 官方数组和 SymPy 桥文档 | 惰性符号数组、数组元数据及官方 SymPy 转换 | 验证数组桥覆盖和跨语言适配成本 |
| egglog、TensorRight、其他 CAS/SMT/编译器 | 复用上一轮调查 | 搜索框架、张量定义、测试及专项转换 | 以具体新增收益决定是否采用，不默认增加数学主后端 |

未以 star 数、原版通过数量或实现语言决定优先级。后几项未运行完整集成，不能据文档宣称可直接替代前端。
历史覆盖 21 个项目的依据见[首轮评估](../open-source-equivalence-backends.md)。

## 源码版本与许可

| 源码 | 本轮固定 commit | 许可 |
| --- | --- | --- |
| [jax2sympy](https://github.com/johnviljoen/jax2sympy) | `2cb6a6568716b26af0abd542a941f87ae0580e25` | MIT |
| [nerva-sympy](https://github.com/wiegerw/nerva-sympy) | `981739f5d94f7d4703d75b981e76b30bcda81f22` | BSL-1.0 |
| [nerva-jax](https://github.com/wiegerw/nerva-jax) | `97b1c7934e0b4f7e57f69df2e8f6f6c1cd41785c` | BSL-1.0 |
| [tensorgrad](https://github.com/thomasahle/tensorgrad) | `59c85e4b34bc7f217383e9116d445c60e1b69ccf` | AGPL-3.0 |
| [sympy2jax](https://github.com/patrick-kidger/sympy2jax) | `4345d478accc49783b8da5a8ca7a3aad65cd0782` | Apache-2.0 |
| [Quax](https://github.com/nstarman/quax) | `c672e48893f5b8d4c26cc25b8b28d0f94615e311` | Apache-2.0 |
| [AutoHOOT](https://github.com/LinjianMa/AutoHOOT) | `aeca7a8cc4c1dc3ae6da81835170e115ae1538ac` | Apache-2.0 |

实验在 `/tmp/jax-reorder-architecture-eval/` 中保存原版和改版。未把这些库加入正式依赖。
许可按对应快照 LICENSE 记录；未来维护副本时保留来源、版权和许可证。本轮仓库只新增探针及实验补丁。

## jax2sympy：保留主干，逐步扩展表示

其主要转换逻辑集中在 `translate.py` 和 `primitive_mapping.py`：前者绑定 jaxpr 变量并遍历方程，
后者提供 primitive 映射。原版注册 57 个映射键，另外处理 jit/pjit；这不是 57 种语义已经完整验证。
输出是 NumPy 对象数组，每个输入元素对应一个 SymPy 符号；原用途是符号稀疏分析。
来源：[转换入口](https://github.com/johnviljoen/jax2sympy/blob/2cb6a6568716b26af0abd542a941f87ae0580e25/jax2sympy/translate.py)、
[primitive 映射](https://github.com/johnviljoen/jax2sympy/blob/2cb6a6568716b26af0abd542a941f87ae0580e25/jax2sympy/primitive_mapping.py)。

### 本地改造验证

[实验补丁](../../../../../tools/experiments/equivalence_backends/jax2sympy_architecture.patch)涉及两个文件，
43 行新增、35 行删除。这个数字仅说明本轮改造范围，不代表生产实现的预计总代码量。
补丁在全新副本上重放成功，两个源文件的字节内容与实际实验副本一致。

| 改造内容 | 原版与改版实验观察 | 架构含义 |
| --- | --- | --- |
| 精确 literal/捕获常量绑定 | 改版保留常量 1 和 2 的实际值，不再都变成 `c0_0` | 常量适配可以直接补入已有遍历 |
| 实数输入属性与计数器 | 改版的实数 log/exp 组合通过，调用间不共享默认字典 | 可保留符号构造入口并扩展元数据 |
| literal 输出和无输入函数 | 改版均可转换 | 输出处理复用同一读取接口 |
| select 分支顺序 | 改版与对应 Piecewise 公式一致 | 局部 primitive 适配即可完成 |
| dot_general 轴映射 | 普通矩阵与 batch 收缩改版均与参考符号表达一致 | 保留分派，替换收缩映射即可接入 |
| reduce_max/min、tile | 支持本轮 Nerva 对应公式的转换 | 通用算子扩展可服务多个公式 |
| 空 exp 输出 | 改版可处理空符号数组 | 小范围 API 适配；不表示所有空域已覆盖 |

通用探针共 23 项：改版 22 项构造出值表达式，`cond` 留为明确未支持。错误系数和返回值交换仍被区分。
其中 `x/x` 的输出值是 1，**只说明值表达式可构造**，不代表原始定义条件已经保存或源函数已判等。
没有将这些构造结果合计成“22 个完整数学证明”。

### 更值得投入的架构扩展

逐元素模式适合小图参照和公式测试，可以保留。为大 shape 增加惰性模式时，建议保留遍历/绑定/分派，
将“构造什么数学对象”抽成内部接口：

- 输入/常量构造：有序身份、shape、精确值或带来源的常量读取。
- 逐点构造：SymPy 函数组合；共享子表达式可复用。
- 张量构造：自由坐标、绑定求和、轴置换、广播和收缩，不先分配全体元素符号。
- 问题元数据：原始定义条件、输出来源和区域实参绑定，随表达式一起返回。
- 不支持或超预算：返回具体路径，保持上层比较器可继续报告 unknown。

这不是要求推翻现有库；它是 fork 从稀疏分析前端演进为数学比较前端的改造方向。
本仓库已有索引和区域适配可直接参与这一层。真正的恒等变换继续交给可信 SymPy。

本轮 `sum(x*x)` 的长度为 8、32、128、512 时，输出自由符号数依次为 8、32、128、512，
遍历树节点数为 25、97、385、1537。改版没有改变这一增长方式。
这说明下阶段要单独验证表示层演进；它不是拒绝复用遍历和映射的理由。
计时同时包括 tracing、构造、参考比较和 tracemalloc，不用它声称纯转换器加速倍数。

## Nerva 两个后端：有源码对应的验证资产

nerva-jax 使用 JAX 数组实现矩阵操作、激活、归一化、MLP 和手工梯度；nerva-sympy 提供对应符号函数。
两者的 softmax 模块采用同一组函数名和逐行布局，底层矩阵操作实现分别用 JAX 和 SymPy。
来源：[JAX 公式](https://github.com/wiegerw/nerva-jax/blob/97b1c7934e0b4f7e57f69df2e8f6f6c1cd41785c/src/nerva_jax/softmax_functions.py)、
[SymPy 公式](https://github.com/wiegerw/nerva-sympy/blob/981739f5d94f7d4703d75b981e76b30bcda81f22/src/nerva_sympy/softmax_functions.py)。

实际验证链如下，输入采用抽象 `(2,3)` 张量，矩阵乘法的第二输入为 `(3,2)`：

```text
实际 nerva-jax 函数
  → jax.make_jaxpr(ShapeDtypeStruct)
  → 本地改版 jax2sympy
  → SymPy 表达式
  ↔ 同一有序输入符号上的实际 nerva-sympy 函数
```

| 对应公式 | 原版转换器 | 本地改版转换器 |
| --- | --- | --- |
| softmax | 待补 tile | 恒等 |
| stable_softmax | 待补 reduce_max/tile | 恒等 |
| log_softmax | 待补 tile | 恒等 |
| stable_log_softmax | 待补 reduce_max/tile | 恒等 |
| sigmoid | 恒等 | 恒等 |
| sigmoid_gradient | 恒等 | 恒等 |
| tanh | 恒等 | 恒等 |
| ReLU | 恒等（使用 SymPy 条件规范化） | 恒等（同样查询路径） |
| Hadamard 乘法 | 恒等 | 恒等 |
| 矩阵乘法 | 需调整 dot 轴映射 | 恒等 |

这些是固定非退化 shape 的符号公式对应检查，不是数值抽样、完整模型或任意 shape 的生产判等。
本轮原版对应 5 项、改版对应 10 项。未修改 Nerva 的公式本体；新增的是本项目的抽象 tracing 和参考比较。

另外实际运行 nerva-sympy 的七个符号正例：归一化、整体平移、稳定 softmax、softmax Jacobian、
log-softmax Jacobian、Hadamard 定义、sigmoid 导数；故意删去 Jacobian 外积项的一个负例被区分。
这些用例使用有限实数符号、非空行，未覆盖 mask、无穷值或所有不可导边界。

本轮没有运行其整个上游测试套件。原测试工具无条件导入 Torch/TensorFlow；这些依赖服务多框架验证。
本地维护测试资产时可以拆出纯符号部分，本轮已直接导入相关函数运行，不需要安装完整训练生态。

架构价值是拿到**具体 JAX 实现与符号规格的对应关系**。后续可以在 JAX 侧构造通用等价/错误改写，
以符号规格核对转换器；生产判等仍读取 jaxpr，不能依赖已知名称查表或要求用户重写成 Nerva API。
手工梯度用例也只是验证输入，不替代本仓库对注册 JVP/VJP 的函数契约检查。

## 其他可复用项目的定位

### SymPy、tensorgrad 与张量表示

SymPy 1.14.0 的 `MatrixExpr.from_index_summation` 将二维符号求和转成矩阵乘法；本轮 ArrayContraction
表达式用符号 `(m,n,k)` 构造，仅 15 个遍历节点，不枚举元素。这里只验证表示和二维转换，
不声称其所有 batch/非线性互转都自动完成。来源：[SymPy 数组表达式](https://docs.sympy.org/latest/modules/tensor/array_expressions.html)。

tensorgrad 的符号命名维度探针将 `A@(B+C)` 和 `A@B+A@C` 化简为同构表示，保持输出轴 `(i,k)`。
可以借鉴或复用其张量构造与化简组织；其 `Tensor.__eq__` 调用图同构，普通 False 不是数学不恒等。
本轮只加载无需 Torch 的符号部分，没有运行编译器，没有实现 JAX→tensorgrad→SymPy 的完整桥。
来源：[Tensor API](https://github.com/thomasahle/tensorgrad/blob/59c85e4b34bc7f217383e9116d445c60e1b69ccf/tensorgrad/tensor.py)。

### AutoHOOT、Quax 与 PyTensor

AutoHOOT 确实有 JAX 前端，能将 dot_general 等映射为自己的 einsum 节点，图变换模块包含收缩融合和分配。
其 SymPy 桥处理局部表达式，使用字符串解析；可在 fork 中改成结构化接口。
快照的 tracing 入口使用旧 `jax.linear_util`，本轮在 JAX 0.11.2 实测该导入不可用。
这是需要更新的适配工作，没有据此否定其收缩映射和图变换资产。本轮未构建其整个旧依赖集合。
来源：[JAX 前端](https://github.com/LinjianMa/AutoHOOT/blob/aeca7a8cc4c1dc3ae6da81835170e115ae1538ac/autohoot/frontend/jax_frontend.py)、
[图化简](https://github.com/LinjianMa/AutoHOOT/blob/aeca7a8cc4c1dc3ae6da81835170e115ae1538ac/autohoot/graph_ops/graph_transformer.py)。

Quax 可以重解释已有 JAX 程序的 primitive，为自定义数组提供分派。它省的是解释器机制，而不是数学映射；
默认未匹配操作可能 materialise 回普通 JAX，符号对象可显式禁止该路径。本项目已有 jaxpr 获取和遍历，
因此应在确需重解释机制时引入它，不能仅因具备 JAX 兼容性就认为能省下整个前端。
来源：[Quax 实现说明](https://github.com/nstarman/quax/blob/c672e48893f5b8d4c26cc25b8b28d0f94615e311/docs/how-quaxify-works.md)。

PyTensor 提供节点重写、模式替换和重写数据库，可复用其变换资产与图组织。引入为主要 IR 时仍需要
JAX 导入、SymPy 转换及数学三态结果接口；本轮只核对官方文档，未实现这些桥。
来源：[PyTensor 图重写](https://pytensor.readthedocs.io/en/latest/extending/graph_rewriting.html)。

### Symbolics.jl 与 sympy2jax

Symbolics.jl 提供惰性符号数组、维度/形状元数据与 `@arrayop`，并有官方 `symbolics_to_sympy` 接口。
它可以作为跨语言表示方案评估；数组操作覆盖与桥接覆盖仍需实测，不因存在标量桥就宣称数组无损互转。
来源：[符号数组](https://symbolics.juliasymbolics.org/stable/manual/arrays/)、
[SymPy 桥](https://symbolics.juliasymbolics.org/v7.9/manual/external/)。

sympy2jax 本轮用 `SymbolicModule(..., make_array=False)` 将多项式和三角表达式生成可追踪的 JAX 程序，
再通过模块自身的 `.sympy()` 成功往返。它适合快速构造公式测试，不能把这个往返接口当成任意 JAX 图导入。
来源：[SymbolicModule 源码](https://github.com/patrick-kidger/sympy2jax/blob/4345d478accc49783b8da5a8ca7a3aad65cd0782/sympy2jax/sympy_module.py)。

## 三种架构路线的比较

| 路线 | 可复用的部分 | 额外工作 | 当前建议 |
| --- | --- | --- | --- |
| 本地 jax2sympy → SymPy | 上游遍历/分派/标量映射；本地索引/区域适配；Nerva 对应公式 | 扩展构造接口、元数据和惰性张量模式 | 优先推进，有实际小改造收益 |
| 现有 Lowerer → SymPy | 已有函数/区域/索引语义 | 完整数学对象构造与 API 迁移 | 作为可复用实现和对照，避免重复维护两套语义 |
| JAX → tensorgrad/PyTensor/Symbolics → SymPy | 额外张量操作或图变换资产 | 两端桥、不同 IR 的身份/条件/结果适配 | 有明确表示收益时采用，不先增加全套中间层 |

当前更适合把前两条结合：以可维护的社区转换器为基础，吸收已有本地语义适配，数学能力交给 SymPy。
暂未量化全部研发节省工期，也未完成符号张量模式；这些是后续里程碑，不凭几十行实验补丁承诺全部完成。

## 复现与验证

脚本：[architecture_reuse_probe.py](../../../../../tools/experiments/equivalence_backends/architecture_reuse_probe.py)。
源码/依赖在 `/tmp`；Python 3.12.3、JAX 0.11.2、SymPy 1.14.0、NumPy 2.5.3。
脚本验证源码 commit，报告 fork 文件 SHA256；每个项目在独立子进程内运行，45 秒超时后回收。
JSON 的 `reference_checks` 比较符号输出值，不表示自动完成源程序定义域检查。

在新的临时目录准备源码，分别 checkout 上表的 commit。脚本需要 jax2sympy、nerva-sympy、nerva-jax、
tensorgrad、sympy2jax 五个源码子目录；Quax/AutoHOOT 仅源码调查，不是运行探针的必需依赖。
额外依赖可安装到该目录的 `deps`，不触及项目环境：

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv pip install --target /tmp/jax-reorder-architecture-eval/deps --no-deps --only-binary :all: tqdm==4.70.1 networkx==3.2.1 equinox==0.13.8 jaxtyping==0.3.11 wadler-lindig==0.1.7
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tools/experiments/equivalence_backends/architecture_reuse_probe.py --sources-root /tmp/jax-reorder-architecture-eval --deps /tmp/jax-reorder-architecture-eval/deps --json /tmp/jax-reorder-architecture-eval/results.json
```

SymPy/JAX/NumPy 等来自仓库既有环境。实际探索时另装过 sympy2jax 和 Quax 的分派辅助依赖，完整列表在
JSON 的 `isolated_dependencies` 中；正式探针直接从固定 sympy2jax 源码导入，没有依赖 Quax。

在原版副本之外创建可修改目录并应用补丁，保留原版 LICENSE：

```bash
cp -R /tmp/jax-reorder-architecture-eval/jax2sympy /tmp/jax-reorder-architecture-eval/jax2sympy-fork
patch -d /tmp/jax-reorder-architecture-eval/jax2sympy-fork -p1 < tools/experiments/equivalence_backends/jax2sympy_architecture.patch
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tools/experiments/equivalence_backends/architecture_reuse_probe.py --sources-root /tmp/jax-reorder-architecture-eval --deps /tmp/jax-reorder-architecture-eval/deps --jax2sympy-fork /tmp/jax-reorder-architecture-eval/jax2sympy-fork --json /tmp/jax-reorder-architecture-eval/fork-results.json
```

创建副本步骤用于尚不存在的目标目录；已有本轮目录可直接运行 probe 命令。
最终验证从全新 `jax2sympy-replay-verified` 副本重放补丁后执行：两个改动文件 SHA256 与实验副本完全相同。
六个项目子进程均结束成功；其中不支持的转换和 unknown 作为报告内容保留，不能将退出码 0 当成全覆盖。
原版 Nerva 对应 5/10，改版对应 10/10；模型编译/eager 保护始终生效。

开发验证：探针 Ruff 与格式检查通过；补丁重放通过；文档本地链接检查覆盖 6 个文件、77 个目标，
`git diff --check` 通过；新增文件另行检查空白和差异。
没有修改受保护核心文档，没有运行生产全量 pytest，没有提交或推送。
