# 可信 SymPy 后端：开源组件复用与架构迁移计划

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

状态：2026-10-16，**F0–F7 实施与验收完成，已归档；全量 2166 passed，1 个上游弃用警告**。
本版替代先前按具体模型安排的 P0–P8 草案。本轮按用户要求实施 F0–F7；验收后用户明确授权提交远端，目标为 `origin/main`。

## 1. 用户决定与本轮目标

以下决定是执行前提，不再重新征求确认：

- 仓库比较两份 JAX 代码的计算图和数学表达；目标是函数/拓扑等价，不是数值输出接近。
- 已有结构比较保留。第二阶段完全信任 SymPy 的数学结论，不再要求自写证明器独立批准。
- 先从整体架构评估社区复用，暂不展开 GLM52 等具体业务细节。
- **允许将开源库拉到本地修改。评估其作为开发基础的价值，不以原版存在错误或缺少某个算子淘汰项目。**
- 将 jax2sympy、nerva-sympy 和新增的 nerva-jax 一并纳入；其他项目按可替代的职责比较。
- 前一轮已完成研究、隔离源码改造和通用符号探针；用户随后明确要求开始实施，生产迁移已恢复。

已交付候选复用矩阵、可复现的原版/本地改版实验及路线建议。本版将其落实为源码维护方式、模块迁移表、
任务依赖和验收条件。评价依据仍是保留多少上游能力、补多少适配、是否减少长期维护，而非给项目做缺陷排名。

## 2. 整体职责与不可丢失的比较契约

```text
两份 JAX 函数 + 输入/静态参数/追踪环境
  ├─ 现有结构比较 → 结构等价 / 结构不同 / 无法比较
  └─ 数学转换前端（以本地维护的 jax2sympy fork 为基础）
       ├─ 输入/输出身份、常量、轴、索引和区域绑定
       ├─ SymPy 数学对象 + 原始定义条件 + 来源
       └─ 可信 SymPy 查询 → 数学等价 / 不恒等 / unknown

验证资产：nerva-jax 源程序 ↔ nerva-sympy 参考公式
          sympy2jax 可从公式生成额外的 JAX 测试程序
```

这里的职责不是要求新增五层 IR。先复用库已有表示，确有信息缺口才增加最小的适配元数据。
Nerva 组合是验证资产，生产比较器仍接收任意受支持的 JAX 图，不按已知函数名称直接给答案。

### 数学问题与结果

- 按已追踪签名比较全部有序输出，保留输入/output pytree、shape、dtype 解释、常量和静态参数。
- 张量乘法、逐元素乘法、归约轴、自由轴/绑定轴、广播、索引和区域实参不得混淆。
- 采用声明的理想数学解释；不承诺浮点逐位一致，不执行模型来判等。
- 输出值和原始定义条件共同构成数学问题。保存条件是转换职责，条件推理交给 SymPy。
- `equivalent` 需要完整范围上的恒等；`inequivalent` 表示函数不恒等，而不是所有输入都不同。
- 一个最终输出已被确认不恒等即可给出负结论；中间表达式不同不能直接推断最终输出不同。
- `different` 仅表示结构/函数契约不同。转换缺失、CAS 未判定、超时、异常分别报告。
- SymPy 的适用恒等查询返回 True/False/None 按其实际含义采用；表达式结构不等或化简残差不为字面零，不能直接当成不恒等。
- 不透明子图若抽象成泛化变量，必须标明查询范围；泛化问题的 False 不能自动提升成原程序的负结论。
- 数学判定可以使用可信 SymPy 自身的启发式；本项目不另行抽样执行 JAX 数组。

## 3. 可本地修改前提下的候选路线

候选证据和精确版本写入[架构复用评估](research/equivalence-backends/architecture-reuse.md)。
此前 21 个项目的报告继续作为已知资料使用，不重复安装所有 CAS/SMT/编译器。

| 候选 | 可保留的能力 | 主要需要改造/验证 | 角色与优先级 |
| --- | --- | --- | --- |
| jax2sympy 本地 fork | jaxpr 遍历、primitive 分派、逐元素 SymPy 表达及嵌套 jit 转换 | 完整语义元数据、惰性张量表示、区域与资源接口 | 第一优先的前端开发基础 |
| nerva-jax + nerva-sympy | 同一套矩阵/激活/归一化公式的 JAX 与符号版本 | 抽象 tracing 适配、布局契约、正反改写、剥离训练测试依赖 | 第一优先的成套验证资产 |
| SymPy 原生 MatrixExpr/ArrayExpr/Indexed/Sum | 符号形状、收缩/置换、有限求和与数学运算 | 为 JAX 建立完整轴/坐标映射，验证非线性与条件组合 | 第一优先的目标表示和判定后端 |
| sympy2jax | 符号公式生成 JAX Callable，自己的表达式可往返 | 测试公式筛选、假设/常量与输入绑定 | 扩大通用测试集 |
| tensorgrad | 命名维度、张量网络、符号归约与化简 | JAX/SymPy 桥、结果接口和本地维护方案 | 重点比较的张量表示备选 |
| AutoHOOT | 已有 JAX 前端、einsum 图变换和局部 SymPy 桥 | 更新 JAX API、扩大算子/输出范围、避免字符串身份 | 张量前端/转换代码来源 |
| Quax / JAX 官方解释器 | primitive 重解释、分派和 tracing 框架 | 提供数学映射、禁止 materialise 执行回退、完整区域问题 | 需要替换解释器框架时使用 |
| PyTensor | 符号图、图重写组织、成熟的张量变换 | JAX 导入与 SymPy 导出、严格比较契约 | 图组织/改写资产参考 |
| Symbolics.jl | 惰性符号数组和官方 SymPy 桥 | JAX 前端、数组桥覆盖、跨语言部署 | 跨语言表示备选 |
| egglog / TensorRight / CAS / SMT 等既有候选 | 搜索框架、张量定义、公式和测试 | 只有明确新增收益时单独验证 | 不默认增加另一套数学主后端 |

### 复用选择的依据

1. **架构契合度**：能接在哪一层，是否保留我们需要的信息，而不是当前覆盖了多少具体模型算子。
2. **保留比例**：能够继续使用的遍历、映射、张量操作、测试和工具；不以代码行数单独衡量价值。
3. **改造类型**：局部修正、算子扩展、接口扩展和表示层改造分别估计。发现错误用于界定改造量。
4. **可组合性**：标量、张量、条件与区域是否能在同一输出问题中组合；不能只展示独立公式。
5. **规模与维护**：是否必须逐元素枚举、是否重复建设已有功能、依赖成本、源码许可和版本升级成本。

原版直接可用并不是门槛。小补丁即可适配、需要新增张量表示、需要整体重写是三种不同的复用成本。

## 4. 已完成的复用评估与证据

### A0：范围、版本与基线

- [x] 读取权限、工作区及历史评估，确认生产迁移暂停。
- [x] 固定已下载源码的 commit 和许可证；保存原版副本。
- [x] 保留 `tests/sympy_equivalence.py` 及用户已有文档改动。
- [x] 更新评价前提：可以 fork，可以修改；不以缺陷数量作选型依据。
- [x] 新增 nerva-jax，与 nerva-sympy 按成套资产评估。

基线：main 的 HEAD 为 `ee59e9d9f841855db9546fe30d565395abd1023d`；当前 JAX 0.11.2、
SymPy 1.14.0、Python 3.12。历史全量测试为 2095 passed，不计为本轮重跑结果。

### A1：原版探针，识别可以直接保留的部分

- [x] 实际调用 jax2sympy 公共转换 API，记录标量组合、常量、输出绑定、广播/归约、转置、矩阵和 jit。
- [x] 运行 nerva-sympy 符号函数，验证归一化、平移、导数公式及一个错误公式反例。
- [x] 运行 SymPy 原生符号矩阵/数组和 tensorgrad 的符号维度探针。
- [x] 运行 sympy2jax 的公式往返及抽象 tracing，明确它不是任意 jaxpr 的反向转换器。
- [x] 核对 AutoHOOT、Quax、PyTensor、Symbolics.jl 的接口与表示，复用历史后端调查。

验收：每个结论标明是原版实跑、源码核对还是文档核对。成功构造表达式与完成判等分别记录。

### A2：本地改造验证，评估开发基础的价值

- [x] 在 jax2sympy 隔离副本中做最小语义适配，保持原遍历与分派主干，保存可重复应用的补丁。
- [x] 对同一通用探针比较原版与改版，标明保留内容、新增内容和仍需架构改造的部分。
- [x] 用实际 nerva-jax 函数进行抽象 tracing，转换后与对应 nerva-sympy 函数比较。
- [x] 将成对公式验证与任意两程序自动判等区分；不得按 Nerva 函数名称硬编码比较结果。
- [x] 记录实际改动文件和验证收益；不要将小范围补丁说成已经完成通用前端。

### A3：区分局部扩展与表示层改造

- [x] 测量逐元素路径的递增 shape：符号数、表达式节点、时间和 Python 分配峰值。
- [x] 验证原生 SymPy 与 tensorgrad 可用符号维度表达收缩，不枚举张量元素。
- [x] 形成 fork 下一步的候选表示接口：保留小图逐元素模式，在 F3 验证 ArrayExpr 或 Indexed/Sum。
- [x] 明确逐点映射、遍历/分派的复用与归约/收缩/布局操作的惰性构造改造范围。
- [x] 将定义条件、输出来源和区域绑定列入适配元数据设计；实际实现属于 F1–F4。

A 阶段只确定重用边界、候选接口和后续验收。未证明规模收益不写性能承诺。

### A4：形成可执行的复用方案

- [x] 给出“本地 fork 为基础 / 部分代码复用 / 测试资产 / 架构参考”的逐项结论。
- [x] 比较三条路线：改造 jax2sympy、现有 Lowerer 接 SymPy、引入额外张量 IR；标明共同可复用部分。
- [x] 将推荐路线落到下一节任务，不预设必须新增某个自研模块。
- [x] 完成实验脚本、补丁和报告的 lint、格式、链接和差异检查。

A 阶段完成定义是有证据的架构建议和后续计划。生产接入与迁移验收属于下方 F 阶段。

实际结果：改版保留原转换主干，仅两个上游文件的 43 行新增/35 行删除；通用 23 项探针中
22 项构造值表达式、cond 明确未支持，定义域诊断不计作源函数已判等。Nerva 对应公式由原版
5/10 提升到改版 10/10；七个符号公式正例和一个错误公式反例符合预期。
SymPy、tensorgrad 和 sympy2jax 的各自表示/生成探针通过。全部六组子进程完成，无模型执行。
实验补丁已在全新副本重放并核对字节；Ruff、格式、Markdown 本地链接和差异检查通过。
这些是 A 阶段的历史结果；F0–F7 的本轮实施与验收另见下文。

## 5. 已收敛的实施选择

### 本地维护与依赖边界

以下路径已按本轮授权落地；源码与测试资产随仓库维护。

| 组件 | 计划落点 | 维护方式 |
| --- | --- | --- |
| jax2sympy fork | `src/jax_reorder/_vendor/jax2sympy/` | 随本仓库维护的内部源码副本；保留上游文件结构、许可证、固定 revision 和修改记录，不另发包 |
| nerva-jax / nerva-sympy | `tests/fixtures/nerva_jax/`、`tests/fixtures/nerva_sympy/` | 按现有 fixture 的 manifest/来源/hash 模式保存所需源码及许可证；测试适配和手写改写与原始快照分开 |
| SymPy | 现有正式依赖 | 使用原生表达式和查询；先沿用当前版本约束，升级单独验证 |
| sympy2jax 及其他候选 | 测试生成或隔离实验 | 初期不进入运行时依赖；仅在明确减少工作量时提取适用代码或增加依赖 |

选择仓库内维护，是为了能同时修改转换器与调用方，并在同一版本中复现；不依赖 `/tmp` 副本、可编辑安装、
嵌套 Git 仓库或运行时下载。使用评估报告中的固定 commit 起步，实验补丁经迁入后的测试重新验收。
不整体引入 Nerva 的训练栈；其通用函数和符号参考是测试资产，生产比较器不依赖 Nerva。

### 现有代码的去向

| 现有部分 | 迁移决定 | 实际改造点 |
| --- | --- | --- |
| `functions.py`、tracing、pytree/函数契约、结构比较 | 保留 | 数学结果适配；成功与输出差异定位不再依赖旧 proof 对象存在 |
| `algebra.py`、`AlgebraComparison` | 保留公共入口并调整编排 | 接入转换问题与 SymPy 结果，统一多输出和 unknown 汇总 |
| `algebra_lower.py`、`algebra_ir.py` | 拆分复用 | 保留可用的绑定、来源和索引信息；新数学路径逐步转到 fork，不在两端重复解释相同 primitive |
| `algebra_indices.py`、`algebra_calls.py`、`algebra_jit.py`、`algebra_scan.py` | 复用语义适配 | 将轴、实参、返回口和 carry 映射接到 fork；不连带迁入旧恒等证明规则 |
| `algebra_scalar.py`、`algebra_domains.py`、`algebra_piecewise.py`、`algebra_finiteness.py` | 按职责迁移 | 保留构造问题所需的语义与原始条件；条件推理和公式化简交给 SymPy |
| `algebra_normalize.py`、现有 `algebra_sympy.py` 候选链 | 退出新数学主路径 | 完整公式送入 SymPy，不先用旧多项式项表完成判等 |
| `algebra_proof.py`、`algebra_inequivalence.py` 及旧证据类型 | 保留旧 API 的兼容用途 | 只验证已有旧格式证据，不作为新 SymPy 结论的批准条件 |

迁移单位是“职责和算子”，不是整文件搬运。每个新路径的 primitive 只保留一个语义实现；
旧兼容路径的保留范围单独标注。复用索引/绑定代码时不能把旧证明器通过间接调用带回新判定链。

### 最小数据契约

沿用现有结果类型，按需补字段；不先建一套通用编译器 IR。转换输出至少能表达：

- **输入问题**：有序输入身份、常量实际值、追踪签名、输出 pytree 及已声明的数学解释/假设。
- **转换产物**：SymPy 输出对象、shape/轴和绑定信息、原始定义条件、来源路径及完整转换或抽象转换的范围。
- **后端结果**：采用的查询及其适用范围、SymPy 版本、True/False/None、最终输出位置、资源使用或失败原因。

`proof=None` 是新可信后端的合法结果。后端记录用于解释和复现，不伪装成独立证书。
unknown 至少区分不支持转换、条件未判定、恒等查询未判定、预算耗尽、超时和内部错误。
一个完整转换的最终输出被确认为不恒等时可返回 `inequivalent`；全部输出及适用条件确认后才能返回
`equivalent`；其余未决情况返回 `unknown`。函数契约/结构差异仍按现有结构结果单独报告。

## 6. 恢复生产迁移后的执行路线

下列任务随实施记录进展。阶段完成条件是技术检查点，不是重复请求用户批准的关卡。
恢复后的第一步是 F0，将实验资产变为仓库内可复现的维护基础；不重新进行广泛候选调研。

```text
F0 源码与基线
  ├─ F1 问题/结果接口和资源边界 → F2 标量 → F3 张量 → F4 区域
  └─ F5 成套测试资产 ─────────── 随 F2/F3/F4 逐步验收
F1–F5 完成 → F6 默认路径切换与兼容 → F7 全量验证与归档
```

F5 先准备已有的十组参考和正反例，不能等到最后才补测试。它与 F1 可以分开推进，
但公共字段与测试入口由 F1 统一；修改同一个转换分派/结果模块的任务按依赖串行集成。
每阶段先通过定向测试再推进，未完成项保留未勾选，并记录具体缺口。

### F0：固定本地维护方式与组件边界

- [x] 按第 5 节落点导入固定快照和许可证，记录文件 hash、上游 revision、来源与本地变更。
- [x] 将已验证的 jax2sympy 两文件补丁迁入内部副本，调整导入命名空间，保留遍历与分派主干。
- [x] 保留相关上游测试与原版实验记录；将现有 23 项探针和 Nerva 十组对应检查整理成可重复的基线。
- [x] 明确各 primitive 在 fork/本地适配器中的唯一实现来源，以及未迁移旧路径的兼容用途。
- [x] 通过离线 sdist/wheel 构建和临时目录独立导入，确认不依赖历史 `/tmp` 源码副本、网络或训练框架；Nerva 测试按 manifest 从仓库快照提取。

完成条件：源码、许可证、基线和导入路径完整；迁入后的结果与评估一致。
保留的逐元素参照接口仍不转换 `cond`；生产 builder 接口已接入区域和原始定义条件，两者范围分别测试。

### F1：转换问题、结果接口与资源边界

- [x] 先定义第 5 节的最小契约与多输出汇总，保留现有公共比较入口和结构快路径。
- [x] 调整 `AlgebraComparison.equivalence_kind` 及 `functions.py` 的输出差异定位，允许成功结果没有旧 proof。
- [x] 新后端记录 backend/version、实际查询、结果范围、最终输出位置及失败原因；不通过解析字符串决定状态。
- [x] 建立可终止的 worker 边界，将可能无界的符号构造、化简和查询纳入预算；明确超时回收方式。
- [x] 覆盖全部输出相等、有效输出不恒等、部分未决、异常/超时及之后查询恢复。

完成条件：不依赖旧证书即可承载新结论；三态数学结果和操作失败有稳定接口。
本阶段只搭接口与生命周期，不用伪造的 SymPy 成功记录代替 F2–F4 的真实能力验收。

### F2：完整标量组合与域信息

- [x] 复用 jax2sympy 中经验证的逐点映射，将加减乘、exp/log、三角函数、sqrt/abs 和有理式组合交给 SymPy。
- [x] 统一精确常量、变量假设、共享表达式与作用域；在自动约分前保留原始定义条件。
- [x] 由 SymPy 处理条件及数学变换，不为每个恒等式新增本地证明规则。
- [x] 通过公共 JAX 比较入口覆盖 `sin²+cos²`、`log(exp(x))`、`sqrt(x*x)=abs(x)`、错误系数、输入/输出位置和多输出。
- [x] 使用 `x/x` 与 `1`、`sqrt(x*x)` 与 `x` 检查带/不带前提时提交的问题，确认假设来自输入契约而非临时补入。
- [x] 在测试中阻断旧 normalizer/verifier 的调用，确认新路径仍能给出结果。

完成条件：有真实 JAX→完整 SymPy 公式→恒等查询的等价、不恒等和未决用例；
至少包含现有受限候选链之外的非线性组合。未化简为零不等于不恒等。

### F3：张量表示的可替换构造层

- [x] 在 fork 中将 jaxpr 遍历/绑定与数学对象构造分离，保留逐元素模式用于小图参照，共用 primitive 语义。
- [x] 统一以 Indexed/Sum 表示坐标与归约；收缩查询尝试 SymPy 原生 ArrayExpr，混合表达保留标量查询。无需另建 MatrixExpr 前端。
- [x] 复用已有本地索引适配与上游逐点操作，保留自由轴与绑定轴；已有映射足够，未额外复制 AutoHOOT 代码。
- [x] 覆盖一般 dot_general、batch 轴、广播、transpose、reshape、slice、reduce_sum、空域和单位轴。
- [x] 在小 shape 上对照逐元素与惰性表示对应的符号公式，同时运行两份 JAX 的完整正反例。
- [x] 对相同图使用 8、32、128、512 等递增维度，记录元素符号数、唯一节点数、时间与峰值内存，和已有基线对照。
- [x] 验证 `sum(exp(x))`、`exp(sum(x))` 等非线性边界；分别报告“能表示”和“能判等”。

完成条件：无巨大常量嵌入的通用收缩/布局/归约图，其构造不再无条件枚举全部输入元素；
通用正反例可靠且规模数据可复现。JAX tracing 仍针对给定签名，不因内部采用符号索引就宣称支持任意动态 shape。
只有原生 SymPy 表示出现可复现的组合缺口时，才针对该缺口验证 tensorgrad 等替代；不重启全库选型。

### F4：条件、选择与区域组合

- [x] 将 Piecewise、分支顺序、比较边界、特殊值与原始条件传到同一个数学问题；复用 SymPy 的条件处理。
- [x] 先接入 jit/call 的实参、常量与返回口，再接入 cond 的绑定和分支组合。
- [x] 迁移已支持的 scan/while 绑定与受限循环能力；无法表达的循环明确 unknown，不承诺一般循环判等。
- [x] 明确动态索引、离散选择、不透明算子及 effects 的覆盖；保留所有依赖，不凭抽象中间变量不同断言整函数不恒等。
- [x] 运行中间量变化被下游消去、共享子图、分支边界、carry 和嵌套区域的通用组合测试。

完成条件：支持范围和拒绝路径都可定位，区域内部数学处理走 SymPy；
原有受限支持有对应回归或明确迁移缺口，未适配区域不冒充整函数负结论。

### F5：成套测试资产与通用反例（从 F0 后开始）

- [x] 登记已实跑的十组对应：softmax、stable_softmax、log_softmax、stable_log_softmax、sigmoid、sigmoid_gradient、tanh、ReLU、Hadamard 积、矩阵乘法。
- [x] 每组记录两端源文件/revision、输入域、轴布局、输出契约和必要适配；不按同名函数推定公式对应。
- [x] 建立两层测试：JAX→参考公式检查转换忠实性；原始 JAX↔手写改写检查公共判等接口。
- [x] 为用例组加入等价改写、系数/轴/输入或输出位置错误，以及未支持/预算不足；期望来自公式含义而非被测转换器。
- [x] 随 F2–F4 扩展矩阵操作、激活、归一化、损失与梯度公式；不引入训练数据、随机初始化或完整训练循环。
- [x] 保留 A 阶段 sympy2jax 生成实验及来源；本轮生产回归采用手写 JAX 和独立 Nerva 符号参考，不新增可选生成依赖。
- [x] 在抽象 tracing 后的验证链阻断 JAX 数值执行，确认结论来自符号问题而非采样误差。

完成条件：两层测试分别统计结果；既有 10/10 仅作为第一层的隔离实验基线。
生产比较器不读取 Nerva 用例登记表，也不根据函数名称提前给结论。

### F6：默认路径切换与兼容性

- [x] `equivalence="auto"` 保持结构快路径，数学主路径接可信 SymPy；旧引擎退出新结果的批准链。
- [x] 保留旧证据 API 的兼容职责，更新示例/报告对 proof 字段的假设；不长期维持两套数学主引擎。
- [x] 通过真实符号构造/查询检查 F1 的预算、超时回收和下一查询恢复；默认限额依据实测确定，不预先宣称性能。
- [x] 初版不依赖跨查询缓存完成任务；确需缓存时，将符号身份、假设、条件、后端版本和查询范围纳入键并单独验证。
- [x] 核对旧能力迁移表并修复定向回归中的退化；剩余能力边界记入 ALG-003/004/005，未用旧后端兜底。全量结果按 F7 单独验收并通过。

完成条件：禁用旧 normalizer/verifier 后新默认路径工作；失败可恢复，结果来源明确。
旧引擎的证据兼容测试与新默认路径测试分开，避免把兼容代码仍存在误当作新结果仍依赖它。

### F7：最终验证与归档

- [x] 完成通用验收矩阵、受影响既有测试与一次全量测试；结构比较和求导契约不退化。
- [x] 最后运行现有模型的 jaxpr 回归；具体业务细节不反向决定通用架构，不新增模型数值判等门槛。
- [x] 审查默认调用链、原始条件、完整输出定位、unknown 原因、资源释放和随包源码来源。
- [x] 更新允许修改的开发文档、技术债和计划索引，记录实际通过项和未覆盖项后归档。
- [x] 若需同步受保护架构文档，先列出具体改动范围按保护规则处理；提交/推送依实际用户授权。

完成条件：默认数学判定完全以可信 SymPy 为后端，转换与验证资产可离线复现；
已有结构能力保留，未决情况可解释，没有用小图实验或旧全量结果替代实际迁移验收。

## 7. 通用验收矩阵

每类能力至少核对以下两条路径，结果分开记录：

1. **转换检查**：单份真实 JAX 的 jaxpr 转换后与独立符号参考比较，验证源程序是否被准确表达。
2. **完整比较**：两份手写 JAX 通过公共 API，验证结构快路径、数学三态、输出汇总和失败定位。

两条路径都使用符号结论；不以数组采样或数值接近作为数学等价依据。

| 类别 | 必需能力 | 反例或拒绝路径 |
| --- | --- | --- |
| 标量组合 | 分配律、指数与三角恒等、log/exp、sqrt/abs | 错误系数、无来源假设、约分丢域 |
| 身份与常量 | 多输入/多输出、捕获常量、相同元素复用 | 输出交换、常量变化、作用域碰撞 |
| 张量 | 矩阵结合、收缩/广播归约、布局等价 | 错轴、错索引、Hadamard/矩阵混淆 |
| 条件与区域 | Piecewise、边界点、同绑定区域组合 | 错分支、carry 变化、未支持 effects |
| 数学负结论 | 合法最终输出不恒等 | 仅中间节点不同、泛化变量 False |
| 未知与资源 | None、异常、超时分别报告并可恢复 | 不以没化简成功推断不等价 |
| 来源与兼容 | upstream/fork/本地适配分别标注 | 小图结论冒充全覆盖、旧证书冒充新结果 |

## 8. 决策、进展与待验证选择

### 实施与验收记录

- 已固定仓库内 jax2sympy 两文件副本、许可证及 revision/hash；Nerva 两端原始源码快照与 hash 检查只在测试侧使用。
- 生产路径使用 fork 的 builder 接口，构造 Indexed/Sum、Piecewise 和原始定义条件；完整公式及 ArrayExpr 转换由 SymPy 判定。
- 默认入口已经切换。旧比较/证据生成编排已移到测试 helper，兼容辅助模块仍随包保留；公开旧证据验证 API 不变，新的数学结果使用 `backend_result`，不调用旧 normalizer/verifier。
- 已接入可终止 worker、10 秒默认墙钟上限、预算、失败分类和输出坐标记录；超时、预算和异常后的恢复均有定向测试。
- Nerva 已扩展为 12 组独立参考（原十组加平方损失及梯度），另有手写 sigmoid、ReLU、矩阵乘法、损失和梯度改写；保留禁止 JAX 数值执行的检查。
- 迁移复用了已有索引、源图有限性及区域端口适配。非有限操作和离散操作保留完整定义与有序端口，只使用符号同余；它们的抽象 False 不作为源函数不等价结论。
- 回归发现并修复了共享 JIT 端口覆盖父图输入、Lambda 绑定捕获、特殊值过早消项等转换问题。原式中的未用部分计算以条件保留；不把删除计算当成自动合法。
- 大签名缩放回归确认 SymPy `simplify` 默认 `doit` 会隐式展开 Sum；改为预算控制后显式求值、其余 `doit=False`，保留惰性形式。嵌套求和展开预估按乘法增长计费。
- 首轮全量是诊断运行（2012 通过、91 失败、36 错误），期间代码仍在调整，不能作为最终验收。后续定向结果及最终全量另行记录。

F7 已完成：GLM 三组签名均保持 5 个正例已证明、5 个负例 condition_undecided；新判定链阻断 JAX 数值执行。低精度 VJP、单位归约及嵌套收缩通过全量回归。剩余通用能力边界记录于 ALG-003/004/005，不是未完成的迁移步骤。

| 项目 | 当前决定或状态 | 后续落点 |
| --- | --- | --- |
| 可信基础 | 完全采用 SymPy 数学结论；本地负责准确构造问题 | F1/F2，不新增独立批准器 |
| 转换前端 | 用 jax2sympy 本地 fork 演进，保留已有主干并吸收本仓库适配 | F0–F4，不平行重写第二套前端 |
| 成套资产 | nerva-jax 与 nerva-sympy 放在测试侧，分别核对转换与完整判等 | F5 提前启动 |
| 评估进展 | A0–A4 和 F0–F7 完成 | 已归档，按后续用户授权提交到 origin/main |
| 张量组合表示 | Indexed/Sum + Piecewise，适用时转 ArrayExpr 查询 | F3 已验证；未引入另一套数学 IR |
| 资源默认值 | worker 默认 10 秒、预算 10,000；超时 kill/wait 并回收 | tracing/父进程结构检查不在该时间内；无 RSS 硬上限 |
| 既有能力迁移 | 索引/区域/有限性适配已复用，新路径独立通过数学查询 | 旧证据仅兼容；剩余回归在 F6/F7 核对 |

2026-10-16 本轮实施：默认数学链路切换到 SymPy；补齐嵌套求和词法绑定、部分函数条件、
固定宽度整数反例范围、killable worker 与旧证据兼容回归。后端记录是审计信息，不是独立证明证书。

## 9. 验证与文件边界

已完成评估的脚本入口：[architecture_reuse_probe.py](../../../tools/experiments/equivalence_backends/architecture_reuse_probe.py)。
下载、依赖、原版与改版 JSON 位于 `/tmp/jax-reorder-architecture-eval/`；正式依赖和 `uv.lock` 不变。
复现命令、补丁及实际结果见[评估报告](research/equivalence-backends/architecture-reuse.md)。

评估脚本的历史检查命令：

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline ruff check tools/experiments/equivalence_backends/architecture_reuse_probe.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline ruff format --check tools/experiments/equivalence_backends/architecture_reuse_probe.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tools/check_docs.py --changed
git diff --check
```

生产实施的定向记录：

- 249 项回归通过（234.93 秒）：新后端、Nerva、上游 23 项基线、矩阵、特殊值、有限性和 JIT 组合。
- 独立双重求和转换/矩阵结合子集 20 项通过；DeepSeek 剩余两个 bfloat16 完整上下文 VJP 均通过。
- 补充的整数 dtype 负结论范围和循环谓词原始条件 3 项通过。
- 规模脚本输出已记录在[当前设计](history/algebraic-equivalence-validation.md#离线复现与规模记录)。
- 修复惰性求和后，标量索引与新后端 109 项通过（90.76 秒）；GLM 大签名缩放正例在 float32/bfloat16 均通过。
- Ruff、格式检查（186 文件）、tracing 示例、新 SymPy 示例通过。离线 sdist/wheel 构建、许可证/manifest 入包、临时目录独立导入及数学比较通过。
- 稳定版本全量：**2166 passed，1 warning，1403.69 秒**。唯一 warning 是既有 SGLang fixture 的 Pallas reciprocal 弃用提示，不是本轮引入的后端。
- 本轮新增 71 项测试（41 新后端、29 Nerva、1 项包含 23 场景的上游基线）；旧证据兼容、结构比较和一阶求导回归均通过。
- `tests/algebraic_equivalence.py` 的 forward/JVP/VJP 与已有 CPU 编译检查通过；新 `tests/sympy_equivalence.py` 只抽象追踪和符号比较。
- 文档链接、空白检查及归档引用检查通过；未修改受保护文档、正式依赖或锁文件；后续用户已授权提交远端。

本轮主要命令（已有锁定依赖的本地环境）：

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline pytest -q --tb=short
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline ruff check .
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline ruff format --check .
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tests/inspect_jaxpr.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tests/sympy_equivalence.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tests/algebraic_equivalence.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tools/experiments/equivalence_backends/trusted_sympy_scaling.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv build --offline --out-dir /tmp/jax-reorder-trusted-wheel
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline python tools/check_docs.py --changed
git diff --check
```

受保护的 README/AGENTS/CLAUDE/ARCHITECTURE、docs/index.md、docs/PLANS.md 及 design-docs 保持只读。
设计与进展写在 development-docs 和 exec-plans。现有 [algebraic-equivalence.md](../../design-docs/current/algebraic-equivalence.md)
的顶部描述新默认实现，后半部分明确保留为旧候选/独立证明协议的历史记录。
