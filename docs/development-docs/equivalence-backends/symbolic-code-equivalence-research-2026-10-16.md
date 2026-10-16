# 用符号计算证明代码等价的开源研究

日期：2026-10-16。范围：符号计算、SMT、等式饱和、编译优化验证与张量程序等价性。

开源工具已经能够在明确的语言、数值域和前置条件下证明程序等价。最接近 SymPy 使用方式的是符号表达式建模与 Z3/egglog；直接比较编译前后 IR 可参考 Alive2 和 MLIR-TV；张量规则及 kernel 调度可参考 TensorRight、TASO 和 Exo。它们分别解决不同层级的问题，不能仅凭“验证通过”就互换结论。

对 `jax-reorder`，已有的可信 SymPy 默认数学后端与 JAX 符号转换是当前实现基础。本次 research 补充通用代码验证路线和论文入口；后端接入优先级应结合仓库已有探针和适配评估，不能仅按上游项目介绍排序。当前实现见[代数比较开发记录](../algebraic-equivalence.md)，已有实测见[开源后端评估](../open-source-equivalence-backends.md)和[架构复用报告](architecture-reuse.md)。

## 需要区分的等价性

| 目标 | 需要建立的事实 | 适合参考的路线 |
| --- | --- | --- |
| 表达式恒等 | 在指定数学域和定义条件下，两表达式表示同一函数 | SymPy、Z3、egglog |
| 程序行为等价 | 对合法输入，输出及要求观察的状态、异常和终止行为一致 | 符号执行、SMT、SAW；具体能力受建模范围限制 |
| 优化正确性 | 目标程序符合源程序允许的行为，必要时使用单向 refinement | Alive2、MLIR-TV |
| 张量规则正确性 | 保留 shape、轴、索引、归约及前置条件后，重写保持含义 | TensorRight、TASO |
| 调度变换正确性 | 循环、访存和配置变换满足依赖及合法性条件 | Exo |

SymPy 官方示例通过化简两表达式之差检查符号恒等；普通 `==` 比较的则是表达式结构。这类数学判断本身不建立 Python/JAX 源码到数学表达式的对应关系。定义域也必须保留，例如 `x/x` 化简为 `1` 后仍需记录原式的除零条件。[SymPy 相等性说明](https://docs.sympy.org/latest/explanation/gotchas.html#double-equals-signs)

数学实数、无限精度整数、定宽整数和浮点数是不同语义。Z3 的 `Int`/`Real` 不是机器整数或硬件浮点；定宽位运算需要对应的 bit-vector 模型。实数代数恒等不能直接作为浮点逐位一致的结论。[Z3 算术说明](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/)、[位向量说明](https://microsoft.github.io/z3guide/docs/theories/Bitvectors/)

## 开源项目对比

| 项目 | 输入或对象 | 核心方法 | 对本项目的价值与边界 |
| --- | --- | --- | --- |
| [SymPy](https://github.com/sympy/sympy) | 符号标量、矩阵、数组表达式、求和与条件 | 代数化简和数学查询 | 当前可信数学后端；继续完善源图转换、条件及符号张量表示 |
| [Z3 / Z3Py](https://github.com/Z3Prover/z3) | 显式建立的逻辑与算术约束 | SMT 求解 | 适合索引、分支条件、位运算及受限数学反例；不是现成 jaxpr 导入器 |
| [egglog Python](https://github.com/egraphs-good/egglog-python) | 自定义表达式语言与重写规则 | e-graph、等式饱和、Datalog | 可组织多步等价搜索；能力取决于规则和语义，不自动证明所注册规则正确 |
| [Alive2](https://github.com/AliveToolkit/alive2) | 优化前后的 LLVM IR | 符号执行与 SMT refinement 检查 | 可参考优化验证的语义和结果设计；循环验证有界，不是高层 JAX 数学比较后端 |
| [CrossHair](https://github.com/pschanely/CrossHair) | 带类型注解的 Python 函数 | 符号执行与 SMT 路径探索 | `diffbehavior` 适合寻找重构反例；没有找到差异不等于证明等价 |
| [TensorRight](https://github.com/ADAPT-uiuc/TensorRight) | 用 DSL 描述的张量重写及前置条件 | 张量指称语义、符号执行、SMT | 支持范围内可证明任意 rank/size 的规则；本仓库已有算子和特殊值适配审计，不能直接当成完整 layer 验证器 |
| [MLIR-TV](https://github.com/aqjune/mlir-tv) | 两个同签名 MLIR 函数 | SMT translation validation | 面向机器学习相关 dialect；接 JAX/StableHLO 前要核对 dialect、算子和版本覆盖 |
| [Exo](https://github.com/exo-lang/exo) | Exo 程序及调度操作 | 带 effect 分析和合法性检查的重写 | 可借鉴分步维护等价性的 kernel 优化流程；不是任意两份源码的通用判等器 |
| [SAW](https://github.com/GaloisInc/saw-script) | 支持的 LLVM/JVM/MIR 程序及规格 | 符号模型、SAT/SMT、验证契约 | 可参考实现与数学规格之间的验证链；常见用法是证明实现符合 Cryptol 规格 |
| [TASO](https://github.com/jiazhihao/TASO) | 张量计算图和替换规则 | 自动生成、验证规则及代价搜索 | 可复用张量公理和优化搜索思想；公理依赖与源图语义映射须单独核对 |

本表描述工具定位与接入判断。已有本地实验的版本、命令和结果保存在[后端评估](../open-source-equivalence-backends.md)；新增的 CrossHair、MLIR-TV、Exo 和 SAW 入口属于文献及文档层面的候选，不表示它们已接入或通过本仓库测试。

## 三种主要方法

### 符号执行与 SMT

对已经正确建模、在指定域内有定义的纯函数，可把等价性转为寻找反例：

```text
存在输入 x，使得 Pre(x) 且 f(x) != g(x)
```

`sat` 表示编码中存在满足条件的模型；`unsat` 表示在该编码和前提下不存在差异；`unknown` 表示没有得出结论。[Z3 Python 教程](https://microsoft.github.io/z3guide/programming/Z3%20Python/Introduction/)

如果编码使用了未解释函数或其他抽象，`sat` 模型可能无法对应真实程序输入，不能直接转成源程序不等价。带状态、异常或循环的程序还需明确可观察行为、定义条件和终止要求。本仓库已有这类抽象反例边界的实测，见[SMT 后端评估](verification.md)。

CrossHair 把这条路线用于 Python 函数比较，可通过 `crosshair diffbehavior module.f module.g` 寻找返回值或参数修改方面的差异。官方要求类型注解，并明确说明未找到差异不能保证等价。[diffbehavior 官方说明](https://crosshair.readthedocs.io/en/latest/diff_behavior.html)

SAW 更强调通过规格描述输入、内存和后置条件，证明实现满足规格；这是研究“源代码实现如何对应数学模型”的另一个入口。[SAW 官方介绍](https://tools.galois.com/saw)

### 等式饱和

egglog 将已知等价的表达式保存在同一 e-class，通过重写和同余闭包扩展可达等价关系。官方 Python 示例给出交换律、分配律与常量规则，然后检查两个表达式是否等价，并可提取一个优化表达式。[egglog Python 文档](https://egglog-python.readthedocs.io/latest/)

它适合处理“单步化简方向不好选择、需要组合多条规则”的问题。证明相对于注册规则及条件成立；两式未合并只说明当前规则和搜索未完成证明。分别提取出的表达式不同，也不能推出数学不等价。

本仓库已有 egglog 上游数组规则与手写索引规则的对照实验，结论是搜索框架本身不能代替张量定义和规则资产。是否引入它，应以对现有 SymPy 路线的实际新增覆盖为依据。[等式饱和评估](rewriting.md)

### 编译与调度验证

Alive2 检查 LLVM 源函数与目标函数的 refinement：目标行为必须符合源程序允许的行为，这比简单要求两端执行行为集合相同更适合处理未定义行为和优化。其论文采用有界循环展开，不能外推为任意循环次数的完整证明。[Alive2 论文](https://web.ist.utl.pt/nuno.lopes/pubs/alive2-pldi21.pdf)

MLIR-TV 将 translation validation 用于 MLIR，支持输入优化前后的两个同签名函数，重点覆盖机器学习编译相关 dialect。它可作为 IR 级比较的研究入口，但“接受 MLIR”不意味着支持所有 StableHLO 或 JAX lowering 结果。[MLIR-TV 项目说明](https://github.com/aqjune/mlir-tv)

Exo 从参考程序出发，通过受检查的调度原语构造优化程序；复杂调度可以组合这些原语。其实现使用 effect 分析等检查维护变换正确性，并跟踪过程间的等价关系。这种设计适合研究 kernel 调度，但需要 Exo 的程序表示及变换流程。[Exo 设计](https://github.com/exo-lang/exo/blob/main/docs/Design.md)、[实现结构](https://github.com/exo-lang/exo/blob/main/docs/System.md)

## 张量研究重点

TensorRight 的关键研究问题是如何证明一个重写规则对任意张量 rank 和维度大小成立。它使用 aggregated-axis 表示一组轴，结合指称语义推导有限组验证义务，再交给 SMT。POPL 2025 论文报告在所评估的 175 条 XLA 规则中证明了 115 条；这是论文实验口径，不与后续仓库 README 的规则数量混用。[TensorRight 论文](https://lsrcz.github.io/files/POPL25.pdf)

论文采用布尔、整数和实数张量语义，不能把这类结论直接当成硬件浮点逐位一致。它值得优先阅读的是轴、访问、归约及前置条件的建模方式。

对本仓库的接入判断还必须考虑已有固定提交审计：包括特殊值、非线性归约，以及 `reduce_max`、top-k、gather、控制流等覆盖问题。相关记录区分了源码发现与尚未构建复现的问题。因此 TensorRight 当前更适合作为张量语义和规则来源，而不是直接替换已接入的 SymPy 后端。[本地 TensorRight 评估](verification.md)

TASO 是较早的代表工作：自动生成并验证图替换规则，再搜索代价更低的计算图。其价值是将规则发现、规则验证和优化搜索组合起来。验证保证依赖所用公理；TensorRight 论文也指出 TASO 的基础公理验证使用小型具体形状，因此不能将所有层级的保证都表述为任意形状已完整验证。[TASO 项目](https://github.com/jiazhihao/TASO)、[TensorRight 对前作的讨论](https://lsrcz.github.io/files/POPL25.pdf)

## 对 jax-reorder 的建议

以下是结合仓库现状的接入判断，不是新增实现承诺。

1. **继续围绕可信 SymPy 路线完善转换。** 继续复用现有 SSA 遍历与 primitive 数学映射，重点处理张量轴、Indexed/Sum、定义条件、区域绑定和完整输出。本地已有公式对照比重新接入一个通用求解器更直接。[架构复用报告](architecture-reuse.md)
2. **把 SMT 作为有明确收益的专项候选。** 优先考虑索引范围、分支条件和定宽整数约束。已有 Z3/cvc5 探针可用于估计收益；不能将小维度展开结果外推到任意长度归约。[验证探针记录](verification.md)
3. **将 TensorRight/TASO 作为语义和规则来源。** 对候选规则核对形状、轴、定义域及特殊值，再判断是否可用于本项目的转换或测试资产。
4. **按目标决定是否引入 egglog 或 IR 验证。** 多步规则搜索覆盖不足时评估 egglog；需要验证 MLIR/LLVM 变换时评估 MLIR-TV/Alive2；需要构造受检查的 kernel 调度体系时参考 Exo。
5. **继续分别报告不同证据。** 结构不匹配、代数未判定、模型中的反例、源程序真实反例和浮点误差不是同一结果；浮点结合/交换改写不能仅靠实数恒等推导逐位一致。

## 论文与后续阅读

| 资料 | 阅读重点 |
| --- | --- |
| [Alive2 Bounded Translation Validation for LLVM，PLDI 2021](https://web.ist.utl.pt/nuno.lopes/pubs/alive2-pldi21.pdf) | LLVM 语义、refinement、符号执行与有界验证 |
| [TensorRight Automated Verification of Tensor Graph Rewrites，POPL 2025](https://lsrcz.github.io/files/POPL25.pdf) | 任意 rank/size、张量指称语义、归约和前置条件 |
| [SMT-Based Translation Validation for Machine Learning Compiler，CAV 2022](https://link.springer.com/chapter/10.1007/978-3-031-13188-2_19) | MLIR 机器学习编译变换的验证 |
| [Better Together Unifying Datalog and Equality Saturation](https://arxiv.org/abs/2304.04332) | egglog 的等式饱和与关系推理设计 |
| [Egglog Python A Pythonic Library for E-graphs](https://arxiv.org/abs/2305.04311) | 在 Python 中构造符号表达式与等价搜索 |
| [Exo 项目设计与论文入口](https://github.com/exo-lang/exo/blob/main/docs/Design.md) | 受检查的调度原语、变换组合和硬件目标抽象 |
| [TASO 项目与 SOSP 2019 论文入口](https://github.com/jiazhihao/TASO#publication) | 自动生成图替换规则、验证和优化搜索 |

继续检索时可使用 `program equivalence checking`、`translation validation`、`relational symbolic execution`、`equality saturation`、`tensor graph rewrite verification` 和 `verified scheduling`。
