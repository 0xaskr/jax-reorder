# 历史：目标架构

> 状态：历史设计依据：部分原则延续，旧接口草案及阶段范围不作为当前契约；具体替代关系见[设计索引](../../index.md#历史设计与替代关系)。

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](../../../exec-plans/completed/pallas-history/index.md)查找。

## 3. 目标架构

### 3.1 分层与职责

```text
JAX ───── core/adapters/jax ─────┐
                                 ├─ ProgramView（源事实，唯一解析）
Pallas ── core/adapters/pallas ──┘        │
                                          ├─ Graph（派生规范比较表示）
                                          ├─ analysis（地址域/存储/通信/依赖）
                                          └─ math projection（公式与定义条件）→ SymPy
```

核心边界是**源语义只解释一次**：JAX/Pallas 对象、参数 schema、区域绑定和 effects 只在适配器中解释；
`ProgramView` 保存尚未规范编号的完整语义事实与来源位置；`Graph`、分析产物和数学投影都从视图派生，
不再读取 `GridMapping`、`FlatTree`、`jit` 参数等源内部对象补充含义。`SourceHandle` 保留原始对象，
用于定位和重建，但不作为重新解释源语义的入口。

Pallas 适配器复用 JAX 的遍历和普通原语处理，补充 grid、Ref、设备、DMA、同步等语义；两条比较路径
消费同一份完整语义表示。代数比较同时用于同类图和跨 JAX / Pallas 图，后者优先级仍最后。

建议初版采用"原始 jaxpr 加带类型的只读语义视图"：视图保存计算、区域、存储和通信事实，分析阶段
推导全局地址、内存依赖、初始化及传输/等待关系。图算法保留原始结构，或使用经过条件检查的派生视图；
数学层完成准入检查后才生成公式。多对一转换另记来源和条件，不能冒充原始图双射。保留源 jaxpr 便于
重排后执行、定位和复核；结构算法和数学转换读取统一视图。不能让 `algebra` 经由源对象重新判断
`GridMapping`、`Ref` 或 JIT 参数的含义。数学准入仍是独立步骤，例如前端解释布局和分片，数学层决定
该调用是否可展开。

"只适配一次"指一次比较上下文内，每个源区域及其绑定的语义由同一处解释，求导规则也只按指定策略
物化一次。图搜索、对应复核和数学转换可多次遍历视图。缓存必须区分版本、环境、区域绑定和求导策略，
不能仅按一个 jaxpr 对象的身份全局缓存；forward、JVP、VJP 是分别追踪的程序，各自适配。

### 3.2 核心决定

| 议题 | 决定 | 理由与边界 |
| --- | --- | --- |
| 表示关系 | `ProgramView` 是唯一源事实，`Graph` 由它派生；允许同时存在视图、派生 Graph、分析产物和数学投影，但只有一处拥有源格式解释权。 | 现有 `Graph` 包含规范编号和搜索状态，不适合作为可分析语义模型；让适配器独占解释权可消除重复解释。 |
| correspondence | matcher 生成候选对应；独立 verifier 读取两侧**未规范化视图**复核结构证据，读取派生视图复核 normalized 的终端匹配。原图到派生图使用规则证据和多对一来源关系，不复用原图双射。 | 对应检查独立于规范化算法，但源适配是共同信任基础，不能宣称独立验证了适配本身。 |
| 结果证据 | `StructuralEvidence`、`NormalizedEvidence`、`AlgebraicEvidence` 三类；`normalized` 是等价证据种类，不新增结果状态。 | 结构等价需要原图双射；多对一改写不能伪装成原图双射；A↔A 返回 structural，A↔B 经规则链和派生匹配后返回 normalized。 |
| 入口 | 扩展既有 `compare_functions` / `compare_algebraic_jaxprs`，不新增公开 Pallas API。`equivalence="auto"` 顺序改为 原图结构 → 已验证规范化 → 数学。 | 适配必须在共同入口完成，不能靠捕获旧结构路径的拒绝异常绕过语义检查。Pallas 仅前向；请求 Pallas 求导明确拒绝。 |
| interpret 政策 | 首批比较 `interpret=False` 的原生源图，另生成验证副本；`dma_execution_mode` 属于验证配置，`debug`/`name` 属于诊断来源；不得无条件忽略 `InterpretParams` effects。 | 锁定版本中 `skip_floating_point_ops=True` 会改变计算结果，TPU interpreter 还会引入 IO effects；简单例子的 inner jaxpr 相同不足以支持直接忽略。 |
| 函数与包装 | M1 接受已审查的 `jit` / `shard_map` 包装，要求唯一静态 `pallas_call`、相同 mesh 和分片契约；包装及设备绑定不能删除。跨分块规范化先限于各 shard 独立、无通信。 | 单个 `pallas_call` 的限制不删除外层 `shard_map`、mesh、分片和设备映射。非空 `input_output_aliases` 在 M1 规范化中拒绝，但完整编码后可以做结构自比较。 |

### 3.3 数据模型（草案）

#### 3.3.1 通用源事实

| 对象 | 必需字段与不变量 |
| --- | --- |
| `ProgramView` | 根区域、抽象输入输出契约、JAX 版本、执行环境及语义策略。函数入口另附 pytree / 静态参数；裸 jaxpr 无法恢复这些信息。 |
| `RegionView` | 有序形参、常量绑定、操作和返回值；父子区域实参与形参的绑定、分支顺序、循环 carry / 反馈关系、区域执行上下文。 |
| `OperationView` | 区域内稳定 ID、经审查的原语身份、有序操作数和结果、完整参数、区域绑定、执行语义及来源。未使用的操作同样保留。 |
| `ValueDef` / `Operand` | 区分数组、Ref、token，及结果引用、字面量和捕获常量。保存 dtype、shape、weak type、布局、分片等适用属性，不能只留 shape / dtype。 |
| `ParameterView` | 已解析的不可变参数；参数里的变量和子图使用有类型的引用，允许规范化时按绑定重命名。不是 `dict[str, Any]`，也不把变量编码成不可解析的字符串。 |
| `Origin` | 原区域、方程、端口和参数位置；派生节点记录规则与所有来源节点。来源编号用于定位，不参与语义相等判断。 |

`PrimitiveKey` 由固定版本的受审查登记表生成，登记时校验真实 Primitive 身份。名称只用于展示和分派，
不能通过伪造同名 primitive 获得支持。已知纯原语可保留精确操作应用，不要求源适配立即将它展开成
SymPy 公式。不认识的语义字段、effect 或对象仍明确失败；完整视图不能依赖后续删除这些节点来合法化。

原始可执行对象放在独立的 `SourceHandle` 中，由适配/重建入口持有。视图不能仅用 `frozen=True` 包住
仍可变的参数字典：参数容器需要冻结，常量内容需要稳定快照或受控的不可变存储，避免源对象变化后两条
路径读到不同语义。无须复制两套完整常量数组，也不在视图层创建 SymPy 对象。

#### 3.3.2 Pallas 执行语义

普通纯 JAX 操作使用值计算部分，Pallas 再组合下列描述。JAX 官方状态或通信原语也可以使用相应描述；
不能按输入来自 JAX 就默认它无状态。

| 描述 | 必须表达的信息 |
| --- | --- |
| `KernelDomain` | grid 轴与范围、顺序/并行语义、设备/core 坐标、BlockMapping、动态标量参数和 kernel 区域。 |
| `Storage` / `RefView` | 存储分配位置与实例作用域、设备、memory space、生命周期、输入输出 alias；视图到存储地址的变换、切片和形状。 |
| `Read` / `Write` / `Atomic` | 所访问 Ref、索引关系、mask、默认值、写入值以及原子操作语义。原子更新不能按普通 store 处理。 |
| `DmaStart` | 源/目的 Ref 与设备映射、传输范围、发送/接收完成信号及完整传输属性。 |
| `DmaWait` | 发送或接收等待、作用的 Ref / 信号量、完成条件、等待量及其计量规则。 |
| `SemaphoreOp` | 信号量存储与初态来源、增加/消费的数量和单位、目标设备、实例作用域与复用位置。 |

`Storage` 的分配位置与动态实例分开：同一 scratch 分配在不同设备、循环或 kernel 调用实例中不能自动
视为同一缓冲区，也不能默认每次 grid 迭代都重新初始化。别名由显式绑定和视图变换描述；尚不能确定
的地址重叠关系留作分析任务。

索引先保留精确整数运算或指向已适配索引区域的引用，不预设所有映射都是仿射。例如 blocked 模式下
`i = 128*p + r`，是由 grid、BlockMapping 和块内坐标组合得到的关系；尾块 mask、padding、整数溢出/
类型转换及将 program_id 用作数据都不能被抹掉。全局读写关系是分析结果，可以按需构造，不必在源适配
时展开每个元素。

DMA 发起和完成等待先分别记录。**不在适配阶段把 wait 绑定到唯一 start**：一个等待可能消费多个传输
的累计完成量，信号量也可能跨迭代复用。实例需要包含设备、区域、迭代和分支条件；配对关系由后续分析
建立。发送完成与接收完成分别约束源缓冲区复用和目的缓冲区可读性。分析用的读/写完成事件是语义描述，
不伪装成源 jaxpr 中原本存在的额外节点。

区域中保留循环与分支，不能为了生成一张 DAG 就任意展开循环或删除反馈边。

#### 3.3.3 分析产物

分析产物独立于原始语义事实，按需要生成：

| 产物 | 含义 |
| --- | --- |
| `DependencyAnalysis` | 数据、读后写/写后读/写写、原子操作、同步和跨设备进展依赖；包含适用域与区域实例。 |
| `MemoryAnalysis` | 读到的数据版本、初始化、写入覆盖、重叠、生命周期和 alias 条件。不能假设并发写入天然存在唯一的"最新版本"。 |
| `CommunicationAnalysis` | 传输与等待关系、信号量计数及消费、发送/接收完成和跨设备顺序条件。 |
| `RewriteEvidence` | 规则、作用位置、已核实前提和源/派生节点对应；多对一转换与原图双射分别记录。 |

条件分别记录已确认、被反例否定或尚未完成，附来源和原因；它们是内部分析结果，不是新增公开比较状态。
语义解析成功不代表已经证明无竞争、无未初始化读取或必然终止。分析未完成时，某条改写可不应用；
保留源视图仍可尝试保守结构比较。若最终结论所需的条件缺失，则走现有分类异常通道，不能返回数学不等价。

### 3.4 存储版本与数学投影

`get`/`swap` 是 Pallas kernel 的核心状态语义，必须在进入数学层前消除。建议在 core 内实现一个小的
存储符号解释器（例如 `core/memory.py` 的 `functionalize_region(region, bindings, domain)`），只消费
已适配视图：

- 维护 `ValueId → ValueExpr` 与 `StorageId → MemoryVersion` 两套环境；`get` 读当前版本，`swap`
  返回旧版本上的值并产生新版本。
- 切片和 mask 表达为带 guard 的区域更新，不能把部分写入当作整块替换。
- 输出初态标为未初始化，不能填零；被丢弃的 `swap` 旧值不应导致普通 store 被误拒绝；实际使用
  未初始化值必须失败。
- 版本模型从第一天存在；首批只接受全片覆盖、相同地址读写及已证明不相交的更新。

`MemoryProjection` 保存纯值 DAG、索引关系、定义条件和转换证据，**不包含 SymPy 对象**；数学层通过
投影适配入口复用 `SymbolicBuilder` 的运算规则，不能假设现有接口无需调整。`discharge_state` 可作为
受限用例的交叉验证参照，但不作为生产语义入口：它会生成新的 slice/gather/update 图，之后再适配会
增加来源对应和语义所有权问题。

未知源语义在适配阶段拒绝；已识别但没有存储消除或数学规则的原语，在对应阶段按路径拒绝。结构比较
可能支持某原语，不代表数学投影也支持它。

### 3.5 消费接口（草案）

以下是职责签名，类型名对应上文模型，尚非可调用的实现或新增公共 API：

```python
@dataclass(frozen=True)
class AdaptedProgram:
    view: ProgramView
    source: SourceHandle


def adapt(source: SourceJaxpr, *, context: AdaptationContext) -> AdaptedProgram: ...


def canonicalize_view(view: ProgramView) -> StructuralResult: ...


def verify_view_correspondence(
    left: ProgramView, right: ProgramView, witness: GraphCorrespondence
) -> None: ...


def analyze(view: ProgramView, *, budget: AnalysisBudget) -> ExecutionAnalysis: ...


def normalize_view(view: ProgramView, analysis: ExecutionAnalysis) -> NormalizationResult: ...


def lower_math(view: ProgramView, *, budget: MathBudget) -> MathProjection: ...
```

`AdaptationContext` 包含锁定版本、环境与 JAX 求导规则策略。`StructuralResult` 保存规范表示、端口/
区域映射和重建指令，适配入口用 `SourceHandle` 生成现有返回值中的 jaxpr。纯重排可复用源方程；
多对一语义转换不得承诺仍可直接重建原有 `Canonicalized.jaxpr`，需要另外记录派生视图及证据。
原图的独立对应检查读取未规范化的语义视图，不能只重新比较同一份规范结果。

`normalize_view` 返回新视图和转换依据；没有适用规则时返回原视图及原因。不能修改源 jaxpr，也不能
将已推导的全局坐标覆盖到原始 BlockMapping 上使来源丢失。

`lower_math` 负责数学准入，必要时请求执行分析和受审查转换。其结果包含有序输出数学图、定义条件、
数学假设、浮点警告和执行转换依据。未解决的内存/通信条件不能变成一个无人检查的 `pending` 字段后
送入 worker。未使用计算的定义条件也要保留，沿用现有完整图契约。执行可观察行为若仍需单独比较，
必须在顶层完成相应比较后才能给出完整等价结论；仅有相同输出公式不能越过这一步。

`MathProjection` 是语义经过检查后的数学输入，按需要扩展现有 `SymbolicGraph` 接口，再由现有
`SymbolicBuilder` 和可信 SymPy 判定。它不承载全部 Pallas 状态，也不要求另建通用张量编译器 IR。
共同调用复用、非有限值处理和 worker 传输仍在数学层。完整相同的已审查纯操作可以共享函数符号；
不透明符号残差不直接证明源程序不等价。

### 3.6 SymPy 直译路线的定位

问题：能否把 JAX / Pallas 直接翻译成 SymPy 比较，是否更简单？2026-10-18 第三轮讨论与实测的结论：

- **JAX 纯值子集已经在直译。** 现有数学链路不是裸 jaxpr 直接交给 CAS，而是
  `结构快路径 → freeze（效果/Ref/token 检查）→ SymbolicBuilder → 可信 worker`；它保留完整输出与
  定义条件、共享调用身份、区域限制和预算/超时边界。它不覆盖 effects、Ref、token 和全部原语。
  这是数学第二阶段/消费者，不是比较主路径；用户已确认不采用"整个程序直接转 SymPy"作为架构。
- **Pallas 直线式片段可以直译，但需要前置状态功能化。** 实测 `discharge_state(kernel,
  should_discharge=True, strip_memory_space=True)` 可把"完整写入、无 kernel 上下文原语"的 kernel
  变成纯值图（`y = x + 1` 得到 `add`）；现有 `freeze` 与 `SymbolicBuilder` 随后能得到 SymPy 公式
  （`x[i] + 1`）。这使"直接转 SymPy"适合作为 M1 的快速原型或数学消费者实现。
- **但它不能替代视图、结构证据与地址/存储分析：**
  - kernel jaxpr 本身是 effectful 的，现有 `freeze` 直接拒绝；必须先功能化。`discharge_state`
    不处理 `program_id`（实测 `AssertionError`）和 DMA/semaphore（实测 `dma_sem` dtype 失败）。
  - 部分写入会退化为对“初始输出值”的 `dynamic_update_slice`；现有符号层只能把它当不透明函数。
    mask、尾块、未初始化读取的语义不能只靠 SymPy 表达，必须由存储版本/初始化分析给出。
  - 跨分块比较的核心是"每个输出元素由哪个实例写、覆盖/唯一/初始化是否成立"的域性质，不是表达式
    化简；SymPy 只在公式对齐后做最后判等。
  - A↔A 的结构等价必须不依赖 CAS；否则 identical 程序也会受预算、超时或未决影响，且无法产出原图
    correspondence 证据。
- **定位**：SymPy 是数学判定后端/数学消费者，不是完整比较架构。独立直译探针至多作为 M1 原型或
  小域 oracle；不另建只读源对象、直接生成 SymPy 的长期旁路。若做 2–3 天 spike，仍须完成最小状态
  功能化、index map 求值和覆盖/唯一写入证明，并显式拒绝 `program_id`、mask/部分写、循环、DMA、
  非空 alias；如果只能靠忽略 effects、假定初始化或大域枚举通过，则该路线应被否定，把可用部分并入
  既定 M1。

可以直接写进后续设计的结论：

> 对已适配的纯值 JAX 子集，以及静态 blocked 索引、无通信和输入输出 alias、初始化与唯一写入
> 已证明的 Pallas M1 子集，可将纯值投影交给 SymPy 比较；SymPy 不替代源语义视图、结构证据、
> 地址与存储分析或通信进展检查，不支持及未决情况继续通过分类异常报告。

### 3.7 中间层与开源组件选型

结论：**没有可直接替换 `ProgramView` 的完整开源语义中间层。** 现有开源 IR 要么只覆盖纯值张量
（StableHLO/HLO），要么是通用编译器 IR 但不含 JAX/Pallas 语义（MLIR/TVM），要么是 JAX 版本锁定的
私有 lowering 表示（HiJAX/LoJAX）。中间层本体仍需自研为"薄适配 + 类型化事实"；下列开源组件可以
复用，避免从零造轮子：

| 组件 | 可复用内容 | 局限 | 建议 |
| --- | --- | --- | --- |
| JAX HiJAX/LoJAX（`jax._src.hijax`、`Primitive.to_lojax`、`GridMapping.to_lojax`、`jax._src.state`、`discharge_state`、`GridMapping`/`BlockMapping`/`index_map_jaxpr`） | 与锁定版本完全一致的源事实、Ref/effects、状态功能化、Pallas 参数与索引映射。 | 私有 API、版本锁定；锁定版本中 `dma_start`/`dma_wait` 尚未实现 LoJAX 转换；不提供网格、地址、存储或通信分析。 | 作为**源事实提取基础**，包在自研 `ProgramView` 内；字段与版本用 adapter inventory 审计，不把 LoJAX 当公共稳定 IR。 |
| StableHLO（OpenXLA） | 开放 IR 规范、verifier、参考解释器；JAX 的 `jax.export` 可导出。 | 纯值张量语义，不含 Ref、effects、DMA、semaphore、TPU 进展；Pallas TPU 内核走 Mosaic 而不是 StableHLO。 | 作为纯值/区域的独立 oracle 或规范化交叉检查，不作为中间层本体。 |
| MLIR Affine/Presburger、ISL（`islpy`） | 整数集与关系、覆盖/交叠判定、映射求逆、集合差；`IntegerRelation` 是成熟实现。 | MLIR 依赖重（C++ 工具链）；`islpy` 的平台可用性需验证（Windows 通常需本地编译或 conda 包）。 | 通用整数域证明组件。M1 的 blocked 仿射映射可先用符号+区间+小域 oracle；需要通用覆盖证明时优先在 Linux CI/WSL 接 ISL。 |
| Z3 / cvc5 | 线性整数/多态约束、条件推理、精确反例；Z3 有跨平台 PyPI wheel。 | 不是 IR；量化公式和性能需专门编码，证明可解释性较弱。 | 跨平台 SMT 备用；先用 M1 覆盖/交叠问题比较 ISL 与 Z3 的编码、性能和可审计性。 |
| TVM Relax / TIR | 图 IR、`structural_equal`、索引范围与 pass 基础设施。 | 把 JAX/Pallas 搬进 TVM 成本高；没有 Pallas TPU/通信语义。 | 参考其结构比较与 pass 设计，不作为主 IR。 |
| xDSL | Python 原生 MLIR 方言/IR 定义框架。 | 生态较小，仍需自写方言、分析和验证。 | 可选原型框架；若坚持 Python-only 的 MLIR 风格可评估。 |
| egg / egglog | e-graph、等式饱和、重写 explanation。 | 不是语义模型；规则覆盖与可信度要自备。 | normalized 规则链和证据重放的后期候选。 |
| Mosaic interpreter、StableHLO reference interpreter、IREE | 执行语义 oracle、编译/运行验证。 | 不能替代静态等价结论；模拟宽松行为不代表硬件语义。 | 测试与反例 oracle，不参与结果判定。 |

本项目此前已评估过 MLIR、StableHLO、TVM、Z3、cvc5、egglog 等作为**数学后端/局部助手**的边界
（见[开源数学后端评估](../../../exec-plans/completed/research/open-source-equivalence-backends.md)），结论同样是"可局部复用，不能直接补齐
完整判定"。本轮的区别是把它们放到**语义中间层**的位置重新评估：仍然没有现成整体方案。

推荐组合：

```text
JAX 内部事实（HiJAX/LoJAX + GridMapping/BlockMapping + state/effects + discharge_state）
        ↓ 自研薄 ProgramView（唯一源语义解释权）
    ├─ Graph（自研规范比较表示，可参考 structural_equal 类设计）
    ├─ Domain/Analysis（M1 自研符号+区间；通用覆盖证明接 ISL 或 Z3）
    └─ Math projection（现有 SymbolicBuilder + 可信 SymPy）
验证 oracle：Mosaic interpreter、StableHLO reference interpreter、CPU interpret
```

约束：不引入完整 MLIR/IREE/TVM 作为主 IR；不把 JAX 私有 LoJAX 当公共稳定接口；新增依赖（如
`islpy`/`z3-solver`）须先过平台可用性（Windows + WSL）、许可证、CI 和 `uv.lock` 审查，且只能作为
分析组件，不进入源事实定义。

接入决策（当前阶段）：

| 组件 | 现在接入？ | 形式与时机 |
| --- | --- | --- |
| JAX HiJAX/LoJAX、`GridMapping`/`BlockMapping`、`state`/effects、`discharge_state` | **是（无新依赖）** | 现版本 JAX 已包含；只作为源事实提取基础，包在 `ProgramView` 后面，字段纳入 inventory 审计。 |
| SymPy | **是（已有依赖）** | 数学消费者，保持现状。 |
| Mosaic interpreter / CPU interpret | **是（无新依赖）** | 只作测试 oracle，不参与结果判定。 |
| StableHLO | 否 | 测试时用 `jax.export` 导出作独立 oracle；不为它增加生产依赖。 |
| MLIR | 否（保留可映射边界） | 见下节"MLIR 的折中接入"；不进入 M1 主链路。 |
| ISL/islpy、Z3/cvc5 | 否 | 通用覆盖/交叠证明需要时，二选一作为 optional extra；先用 spike 比较平台可用性、编码、性能和可审计性。M1 不依赖它们。 |
| TVM、IREE、xDSL | 否 | 只作设计参考或执行验证；不进入生产依赖。 |
| egg/egglog | 否 | `normalized` 规则链需要引擎时再评估，作为 dev/optional 依赖。 |

依赖策略：

- 核心依赖保持 `jax[cpu]`、`numpy`、`sympy` 不变；**M1 不新增任何运行时依赖**。
- 分析工具（ISL 或 Z3）只作为 optional extra 引入，并明确平台限制；两者不同时接入，先用 M1
  覆盖/交叠案例比较编码、性能和可审计性后再二选一。
- 测试 oracle 使用 JAX 自带能力（`jax.export` 的 StableHLO 导出、Mosaic/CPU interpret），不列为
  生产依赖；StableHLO reference interpreter 需要 LLVM 工具链，暂不作为必需 oracle。
- 每个新增依赖必须通过：Windows/WSL/CI 可用性、许可证、`uv.lock` 锁定、维护活跃度与版本锁定策略；
  原生/C++ 依赖只在 Linux 专用 CI job 使用，Windows 开发路径保持纯 Python 可运行。

现在不接入其他组件的直接理由：StableHLO 的价值可由 `jax.export` 在测试中获得且不含 Pallas 语义；
M1 的仿射域不需要 ISL/Z3；TVM/xDSL 成本高且不提供 Pallas TPU 语义；egglog 不是语义模型；IREE 是
执行/编译验证工具而不是判定器。等 M1 的实际反例说明需要更强域工具时，再按上表引入，避免提前扩大
依赖面和实现范围。

**MLIR 的折中接入。** 如果项目长期希望靠向 MLIR，建议把 MLIR 定位为**分析后端/序列化目标**，而不是
把 `ProgramView` 改写成 C++ dialect：`ProgramView` 保持 Python 类型化事实，另提供可导出的
operation/region/attribute/effect 投影，需要时生成 MLIR 文本或通过 MLIR Python bindings 做域分析。

- 收益：复用 MLIR 的 SSA/区域/verifier 框架、Affine/Presburger 整数关系分析和 pass 基础设施，
  为未来 Pallas/Mosaic 互操作留接口。
- 成本：完整 MLIR 作为主 IR 意味着方言设计、构建/打包（官方没有稳定的 PyPI wheel；conda-forge
  `mlir` 的 Windows 支持不完整）、CI 与 Windows 开发链路改造；Pallas TPU 的 Mosaic dialect 是 XLA
  内部实现，不会因为接入 MLIR 就自动获得语义。
- 折中路径：在 M1 之后用 3–5 天 spike，把 A/B 的视图投影成 MLIR 文本（或经 IREE Python 包使用
  MLIR Python API），只验证整数域分析能否复用；若收益明显，再把 MLIR 作为**可选分析后端**接入
  （Linux CI/WSL），而不是核心依赖。Windows 开发保留纯 Python 路径。
- 现在要做的准备：`ProgramView` 的字段、区域/操作/属性/effect 建模和稳定 ID 设计保持 MLIR 可映射，
  不让证据对象依赖 Python 对象身份；这样以后接不接 MLIR 都不需要重写源事实层。

**最终决定（2026-10-18 用户确认）**：中间抽象层**由本项目自研**；不把 JAX/Pallas 直接翻译成 SymPy
作为比较主路径（SymPy 继续作为数学判定后端）；MLIR 虽然长期方向看好，但现阶段接入成本过大，**暂不
接入**，只保留以后作为可选分析后端/序列化目标。开始 M1 不需要接入任何新的开源组件；其他组件按需
以 optional/dev 形式引入。避免为了"用 MLIR"而把实现成本和语义范围扩大，也避免同时接入 ISL 与 Z3。
