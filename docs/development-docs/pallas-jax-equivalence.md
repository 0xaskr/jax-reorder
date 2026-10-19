# JAX / Pallas 等价比较：设计与验证

更新：2026-10-18（第二轮 review 后重构）。代码基线：`f8625181b17efded8c22c648946e3539da8f2fbd` 及未提交文档修改。
环境：Python 3.12，JAX / jaxlib 0.11.2，CPU tracing；尚无真实 TPU 编译或执行证据。

本文是 JAX / Pallas 等价比较设计的唯一记录，按"需求 → 现状证据 → 架构 → 比较语义 → 通信 →
覆盖与审计 → 验证 → 计划"组织。已开始实现 §8 的内部链路及有限分片/DMA 片段，当前代码与验证见 §9.3–9.5；
尚未接入公开 Pallas 比较入口，也未完成通信与 JAX 消费者迁移。未落地部分的类型、接口和文件名仍为建议。
现有结果契约见[比较结果](../references/equivalence-results.md)，
数学后端见[代数开发记录](algebraic-equivalence.md)。

## 摘要

- **目标**：在锁定的 JAX/jaxlib 版本上，比较两份 JAX 或 Pallas 程序是否等价。比较依据是程序图、
  执行语义和数学含义，不是数值抽样；不能完成判定时走分类异常，不返回"未知"结果。
- **架构决定**：源语义只适配一次（`ProgramView` 是唯一源事实），`Graph` 由视图派生；图逻辑比较、
  存储/通信分析和数学投影共用同一份事实；只有适配器拥有源格式解释权。**中间层自研**，不把
  JAX/Pallas 直接翻译成 SymPy 作为比较主路径；MLIR 虽然长期有价值，但现阶段接入成本过大，暂不接入。
- **结果决定**：保留"计算图等价 / 数学等价 / 数学不等价 + 异常"契约；新增
  `structural` / `normalized` / `algebraic` 三类等价证据，`normalized` 不是新结果状态。
- **当前最大风险**：同一 mesh 下不同通信协议/路由/拆分合并的等价判定，以及一般整数索引域的
 覆盖与交叠证明；目前只有小规模 CPU tracing 证据。
- **实施顺序**：Pallas 垂直 slice（第 8 节）已建立内部链路；继续补齐 M1，随后推进通信与 JAX 迁移。

## 1. 已确认的要求

| 事项 | 用户确认的要求 |
| --- | --- |
| 比较方向 | 先支持 Pallas ↔ Pallas、JAX ↔ JAX；Pallas ↔ JAX 也必须支持，优先级最后。前两个方向之间未指定先后。 |
| Pallas 第一阶段 | 必须支持改变 block 大小、grid 和索引映射，不能把"相同分块"作为第一阶段的范围边界。 |
| JAX 覆盖 | 以支持所有官方原语为目标，不限于现有 CPU lax 清单；不支持用户自定义 Primitive、FFI 和 callback。 |
| Pallas 覆盖 | 面向完整计算图及其组合，不按 matmul、softmax、attention 等业务算子名称限制。 |
| kernel 边界 | 每侧限定一个 `pallas_call`；不要求多个 kernel 的融合与拆分。 |
| 后端 | 面向 TPU，必须覆盖多 TPU 通信。具体 TPU 代际、设备数量和 mesh 属于待绑定的执行环境。 |
| mesh 范围 | 当前只支持两侧相同的 mesh；暂不比较不同 mesh 的程序。 |
| 通信改写 | 允许单个 `pallas_call` 内拆分/合并 DMA 传输、改变传输路由和通信协议。改写仍须保持调用边界的可观察结果、内存依赖及进展条件。 |
| 求导 | JAX 层支持前向、JVP/VJP；Pallas 层只考虑前向，不要求 kernel 的 JVP/VJP。 |
| 浮点契约 | 沿用现有区分：数学等价可以通过，但明确报告浮点重排，不保证逐位相同。 |
| 架构方向 | 统一源语义适配，抽取中间语义层，分别用于图逻辑比较和数学转换；优先小改动。具体视图模型、接口及目录名称仍待细化。 |
| 中间层实现方式 | **自研中间抽象层**；不把 JAX/Pallas 直接翻译成 SymPy 作为比较主路径（SymPy 仍作为数学判定后端）。MLIR 虽长期看好，但现阶段接入成本过大，暂不作为主 IR；保留以后作为可选分析后端/序列化目标。 |
| 讨论记录 | 以本地开发文档保存共识及未决事项。 |

"先选择少数业务算子""第一阶段只接受相同 grid / BlockSpec"均不作为本次需求。具体算子可以用于
验收，但不能成为受支持计算图的白名单。相同 mesh 不限制 kernel 内部的 block、grid 和索引映射改写；
传输的拆分/合并也不扩大为多个 kernel 的融合/拆分。外部输入输出分片继续作为显式契约记录，
其全局对应关系不能仅由 mesh 相同推出。

## 2. 现状与证据

### 2.1 当前代码已具备什么

| 层次 | 当前能力与缺口 | 入口 |
| --- | --- | --- |
| JAX 结构比较 | 比较完整图、操作数位置、参数、常量、抽象值和已适配的嵌套区域；不是只比较输出可达子图。 | [core/canonical.py](../../src/jax_reorder/core/canonical.py) |
| 原语识别 | 按 Primitive 对象身份识别公开 lax、已加载 lax 内部原语及特定包装原语；参数和 effects 仍需通过语义检查。尚不等于所有 JAX 原语。 | [core/support.py](../../src/jax_reorder/core/support.py) |
| JAX 数学比较 | 结构快路径后，将支持的完整输出公式及定义条件交给可信 SymPy；不保证任意图都能转换或判定。 | [algebra/compare.py](../../src/jax_reorder/algebra/compare.py)、[algebra/trusted.py](../../src/jax_reorder/algebra/trusted.py) |
| 坐标与公式 | 已有精确整数坐标、Indexed/Sum 和区域转换；没有完整的 Pallas 全局写入域和 TPU 同步语义适配。 | [algebra/indices.py](../../src/jax_reorder/algebra/indices.py)、[algebra/symbolic.py](../../src/jax_reorder/algebra/symbolic.py)、[algebra/regions.py](../../src/jax_reorder/algebra/regions.py) |
| 函数契约 | 记录 pytree、静态参数、抽象输入输出、追踪环境；前向、JVP、VJP 分别检查。 | [functions.py](../../src/jax_reorder/functions.py) |
| Pallas | 当前 `pallas_call` 明确抛出不支持异常，包括图与自身比较；CPU interpret 执行不等于比较器已支持。 | [test_full_attention.py](../../tests/test_full_attention.py) |
| 结果与错误 | 完整比较区分结构等价、数学等价、数学不等价；结构阶段的 `different` 不能推出数学不等价，未完成判定抛异常。 | [status.py](../../src/jax_reorder/status.py)、[errors.py](../../src/jax_reorder/errors.py) |

[CPU lax 清单](lax-coverage.md)由 [lax_inventory.py](../../tests/lax_inventory.py) 维护，记录的是固定版本
公开 lax 原语及测试绑定，不能当作全部 JAX / Pallas / TPU 原语清单。

### 2.2 CPU tracing 实测事实（JAX 0.11.2）

以下为本轮在 CPU 上实际追踪到的事实，用于判断视图字段与适配边界是否可落地；它们不是 TPU 执行证据。

- `pallas_call` traced 参数共 11 个：`out_avals, jaxpr, debug, interpret, grid_mapping, mesh,
  input_output_aliases, compiler_params, cost_estimate, metadata, name`；外层没有 BlockSpec callable。
- `GridMapping` 保存 `grid`、`grid_names`、`block_mappings`、`index_map_tree`、`index_map_avals`、
  `vmapped_dims`、`scratch_avals`、`num_index_operands`、`num_inputs`、`num_outputs` 等字段；
  `BlockMapping` 保存 `block_shape`、`transformed_block_aval`（`AbstractRef`）、
  `index_map_jaxpr`（已 trace 的 `ClosedJaxpr`）、`index_map_out_tree`、`array_aval`、`origin`、
  `transforms`、`pipeline_mode`。
- 最小探针 A（128 分块、正序）的 index map jaxpr 是恒等；B（256 分块、反向）是 `lambda p: 3 - p`，
  两者都可用 `jax.core.eval_jaxpr` 求值。index map 已 trace **不等于**它一定纯、静态或仿射，
  仍需逐项审查。
- kernel 区域中 `get`/`swap` 的参数只有 `tree=PyTreeDef(NDIndexer(Slice(0,128,1), (128,), ()))`；
  Ref 的 aval 是 `AbstractRef(inner_aval=ShapedArray, memory_space=...)`；effects 精确到 Ref 端口；
  `program_id[axis=0]` 是普通原语，可以作为数据参与计算。
- DMA 在 CPU tracing 下即可出现：`dma_start` 参数为 `tree`（src/dst/dst_sem/src_sem/device）、
  `device_id_type`、`priority`、`add`；`dma_wait` 参数为 `tree`、`device_id_type`、
  `insert_dummy_device`、`is_wait_send`；effects 覆盖 Ref 与 semaphore 端口。
- `interpret=True/False` 的 kernel jaxpr 文本相同，只差外层 `interpret` 参数；
  `compiler_params.dimension_semantics` 承载顺序/并行语义，其余 `CompilerParams` 字段混合了
  同步、数值、检查、编译和资源语义，不能整体忽略。

本地最小探针（2026-10-18，CPU）：两份 `y = x + 1` 的 Pallas 程序。

| 项目 | A | B |
| --- | --- | --- |
| 全局输入输出 | `float32[1024]` | `float32[1024]` |
| block 大小 | 128 | 256 |
| grid | `(8,)` | `(4,)` |
| 输入与输出 index_map | `p -> (p,)` | `q -> (3 - q,)` |
| kernel 内部原语 | `get`, `add`, `swap` | `get`, `add`, `swap` |
| 内部 effects | `ReadEffect`, `WriteEffect` | `ReadEffect`, `WriteEffect` |
| 引用抽象值 | `Ref<default>{float32[128]}` | `Ref<default>{float32[256]}` |

两份图的外层原语均为 `pallas_call`。A↔A、B↔B、A↔B 的当前结构比较均抛出：

```text
UnsupportedJaxprError:
structural:left.jaxpr.eqns[0] [unsupported_semantics]: unsupported primitive pallas_call
```

两份函数以 `interpret=True` 在 `cpu:0` 执行；对唯一输入 `arange(1024, dtype=float32)`，输出均与
NumPy 的 `x + 1` 精确相同。这只是一次抽样执行证据，不是比较器已经证明等价，更不是 TPU 编译、执行
或并发验证。可复现的构造如下；使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python` 运行：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def build(block, reverse):
    groups = 1024 // block

    def index_map(p):
        return (groups - 1 - p if reverse else p,)

    spec = pl.BlockSpec((block,), index_map)

    def kernel(x_ref, y_ref):
        y_ref[:] = x_ref[:] + 1.0

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
        grid=(groups,),
        in_specs=(spec,),
        out_specs=spec,
        interpret=True,
    )


a, b = build(128, False), build(256, True)
```

### 2.3 本轮 review 确认的缺口

| 事项 | 结论 | 依据 |
| --- | --- | --- |
| 源语义解释次数 | core 结构路径与 algebra 适配各解释一次；`freeze` 仍解析 scan FlatTree、reduce computation、cast sharding、default_jit、共同调用和非有限子图。Pallas 若照此实现会成为第三次。 | [core/params.py](../../src/jax_reorder/core/params.py)、[core/semantics.py](../../src/jax_reorder/core/semantics.py)、[algebra/adapter/transport.py](../../src/jax_reorder/algebra/adapter/transport.py) |
| `ProgramView` 与 `Graph` | 职责重叠：两者都保存操作数位置、参数、常量、区域和输入输出。`Graph` 是类型化比较键，但字段仍绑定 JAX 对象（`Frozen` 实际为 `Any`），不能直接承担可分析语义模型。 | [core/canonical.py](../../src/jax_reorder/core/canonical.py) |
| effects 与 Ref | canonical 只接受 `NamedAxisEffect` / `NoDCEEffect`；algebra 拒绝任何 effectful 图和非 `ShapedArray` 值。kernel 的 `ReadEffect` / `WriteEffect` 与 `AbstractRef` 被两层拒绝。 | `check_effects`、`freeze` |
| 原语与参数审计 | 身份扫描只覆盖 lax/pjit/shard_map/custom primitives，不覆盖 Mosaic Pallas 原语；adapter inventory 也没有 Pallas 参数 schema。 | [core/support.py](../../src/jax_reorder/core/support.py)、[adapter_inventory.py](../../tests/adapter_inventory.py) |
| 文档与 CI | Ruff 0.17 会格式化本文件内嵌 Python 代码块；原样提交会使 `ruff format --check .` 失败。README/ARCHITECTURE 仍是 UNKNOWN/无代数模式的旧描述（受保护，需授权更新）。 | CI 配置、技术债 DOC-002 |

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
（见[开源数学后端评估](open-source-equivalence-backends.md)），结论同样是"可局部复用，不能直接补齐
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

## 4. 比较语义与可判定范围

### 4.1 结果状态与证据

正常结果保持三类，失败走分类异常：

| 结果 | 含义 | 证据 |
| --- | --- | --- |
| 计算图等价 | 规范化后算子、参数、常量、抽象值、操作数位置、依赖和受支持嵌套语义一致；Pallas 程序还要求域事实、Ref、effects 等全部匹配。 | `StructuralEvidence`（原图双射 + 独立复核） |
| 数学等价、计算图不同 | 第一阶段的规则应用与派生匹配已完成，且在声明数学域与定义条件下输出公式等价。 | `NormalizedEvidence`（规则链、条件、来源、终端匹配）或 `AlgebraicEvidence`（数学后端结果） |
| 数学不等价 | 在声明范围内，可信数学判断确认不是同一个函数；必须有完整输出或明确规则证据，不能用局部差异代替。 | 数学后端结果或规则否定证据 |

`normalized` 是等价**证据种类**，不增加第四种结果状态。结构阶段仍只返回 `equivalent` / `different`
诊断；无法完成判定、条件未决、预算不足、不支持语义都抛现有分类异常。`normalized` 也不代表逐位
一致；凡实际使用浮点重排规则，都必须记录适用数学域和警告。Pallas 只比较前向；请求 Pallas 求导时
明确报告不支持，不静默改为对数学投影求导。

### 4.2 M1：第一个可判定 fragment

M1 是内部原型验收范围，不替代用户已确认的首阶段交付要求。它定义"第一批代码能判定的问题"，
余下语义明确拒绝并计为覆盖缺口。

| 项目 | M1 范围 |
| --- | --- |
| grid、shape | 静态、有限；允许两侧不同 grid 和 block 大小；先验证单设备。 |
| `index_map_jaxpr` | 整数常量、程序坐标、`add/sub/neg`、乘常量；正常量除数的 `div/rem` 仅在已证明非负被除数时接受；转换必须保持数值；检查中间运算不溢出。 |
| 索引依赖 | 可以依赖静态坐标，不能依赖输入张量内容、scalar-prefetch Ref 或不透明 callable。 |
| kernel | 直线式计算、已审查的逐元素纯原语、普通读写、坐标决定的 mask；保留 dtype、参数和 `other`。 |
| 存储 | 输入只读、输出每个有效位置恰好写一次；没有输入输出 alias、跨 program 状态或 scratch 复用。 |
| 循环、累加、atomic、DMA | 均不进入 M1，分别返回明确的不支持原因。`ARBITRARY` 顺序轴上的无状态、互不重叠写入仍可分析；被排除的是尚未表达的累加状态。 |

判定过程：

1. 从适配视图构造每个有效执行位置 `d=(program, lane)` 的读地址、写地址和符号值。
2. 检查地址域：有效读取合法，且对每个输出位置证明存在唯一写入者
   `∀i ∈ O, ∃!d: W(d) = i`（符号证明；小域枚举只作 oracle）。
3. 按全局输出坐标比较符号表达式；输入叶子是"输入编号 + 全局坐标"，不是采样数值。
4. 全图语义、mask、定义条件和未使用节点仍须检查；不支持的节点不能因其没有输出贡献而忽略。

首批生产规则支持普通 blocked 映射及反向遍历。对 `i = Bπ(p) + r`、`p = π⁻¹(⌊i/B⌋)`、`r = i mod B`
验证逆映射、范围、尾块 guard 和整数不溢出，即可证明覆盖与唯一写入；区间加 `div/mod` 化简不能解决
一般仿射映射的覆盖、交叠问题，更一般的关系后续接 Presburger/SMT 判定器。**读取集合相同也不够**：
必须比较"每个输出坐标对应哪些读取及值表达式"，例如正序和倒序读取可以具有相同读取集合。

| 例子 | 是否进入 M1 | 预期 |
| --- | --- | --- |
| `x + 1`：128/256 分块，含反向 grid 遍历 | 进入 | 相同全局读写和值表达式，返回 normalized 等价。 |
| `program_id` 作数据 | 进入 | 必须保留；例如 `y[i]=program_id` 在两种分块下于 `i=128` 得到 1 与 0，必须发现差异。 |
| 非整除尾块 mask | 进入 | 逐项检查读、写、mask 和 `other`；漏写、重复写、未保护的越界访问不能通过。 |
| `ARBITRARY` 顺序累加 | 不进入 | 需要初始化、跨迭代存储版本和归约顺序模型；不能当成普通并行写。 |

资源与规模要求：小域符号证明须与独立枚举 oracle 一致；大静态 shape（例如 `N = 2**24 + 13`）只做
抽象 tracing、不分配大数组；大域案例禁止调用枚举，表示大小不能随元素数量展开；不支持的映射明确失败，
不能偷偷退回大规模枚举。预算不足走分类异常，不返回不等价。这一条用于防止 M1 变成"只能跑小例子"的
死胡同；枚举只作原型验证和回归 oracle，不作为生产默认算法。

### 4.3 改变分块后比较全局含义

把两侧的程序坐标、块内坐标与索引映射组合成全局读写关系。最小探针（普通 blocked 索引模式）：

```text
A: i = 128*p + r,       0 <= p < 8, 0 <= r < 128
B: i = 256*(3-q) + s,   0 <= q < 4, 0 <= s < 256
```

两侧都恰好覆盖 `0 <= i < 1024`，每个位置写入一次，读取同一输入位置，并写入 `x[i] + 1`。这个例子说明
目标不是要求两侧 block / grid / index_map 相同，而是比较组合后的语义；一般情况还需要处理尾块、mask、
不同索引模式、轴交换、grid 维度拆分/合并、循环与累加。若 kernel 将 `program_id` 用作计算值，也必须
代入实际映射后比较，不能删除。浮点数学解释不允许放松整数下标、位运算、类型转换或精度参数的语义。

BlockSpec 的索引模式影响块索引到元素索引的转换，尾块输入填充值也不能默认视为零。CPU interpret 的
grid 调度不能直接代表 TPU 的并发语义。见 [Grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)。

全局语义至少保留：

- 输入输出及设备/内存空间中的存储身份，alias、scratch、缓冲区版本与生命周期。
- 程序和循环的迭代域、索引映射、读写地址、mask、默认值、写入覆盖及初始化条件。
- 数据依赖、内存读写依赖、控制流、同步事件及跨设备通信关系。
- 实际值公式、归约域、dtype、累加类型、精度参数及数学定义条件。

不能只比较最后的值公式而漏掉未初始化读取、写入遗漏或同步错误；完整图中的未知语义也不能通过删除
无用节点绕过检查。全局域分析应尽量使用符号表示，不把逐元素枚举作为大张量比较的默认实现。现有整数
坐标模块可以复用，但不能单独承担任意域的覆盖、重叠和循环依赖判断；是否需要额外的整数域分析工具
仍待验证。

### 4.4 TPU 执行约束

TPU 不能一律按"所有 program 相互独立"处理。官方文档允许在满足顺序和连续性约束时，多个 program 对
同一输出窗口累加；同时，指定并行 grid 维度又允许跨 core 调度。因此"每个输出只允许一个 program 写入"
不能成为整个需求的限制。见 [TPU 的 grid 与多核执行语义](https://docs.jax.dev/en/latest/pallas/tpu/details.html)。

内存空间、scratch 的跨迭代状态和多缓冲流水线也必须保留，不能把每次 invocation 都当作从零初始化的
独立函数。相关机制见 [TPU Pipelining](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)。

## 5. 通信与进展

用户已确认允许单个 `pallas_call` 内拆分/合并 DMA、改变传输路由和通信协议；比较仍须保持调用边界的
可观察结果、内存依赖及进展条件。讨论后拆成两个 milestone：先做受限异步同步化，再做协议/路由/拆分
合并比较。

### 5.1 受限异步同步化

用户提出：将发起异步搬运的 trigger/start 移到 wait 的上一步，就等价于同步。这里按"将 start 延后到
匹配的完成等待之前"理解。该思路可作为规范化规则，但需要验证移动区间中的依赖。以下充分条件和处理
方式仍属于需要技术验证的算法建议。

可同步化的情形（语义伪代码，`sync_copy` 不指定具体公开 API）：

```text
原图：                         规范化后：
start_copy(src, dst, event)    independent_compute()
independent_compute()         sync_copy(src, dst)
wait_complete(event)          consume(dst)
consume(dst)
```

这里消除的是搬运与计算重叠的调度差异，保留搬运本身及数据依赖。性能不属于等价结论。建议的充分条件：

1. 能识别 start、完成等待及搬运实例，包含所在循环迭代、分支、设备和传输范围。不能仅凭信号量变量名
   配对；复用信号量或等待累计传输量时需要额外分析。
2. 移动不改变源数据版本、目的地写入、地址/索引计算、别名关系、缓冲区存活期，也不改变任何与该内存
   范围有关的可观察操作。只读的重叠读取不必一律禁止。
3. 区间内没有依赖该传输提前启动或完成的同步、进度查询、远端动作和外部 effect；等待本身的信号量
   消费及退出状态也得到保留或可证明地消去。
4. 移动不跨越不兼容的控制流边界，不改变是否执行和执行次数，不引入新的等待环。只证明输出公式相同
   不足以证明一个原本能完成的程序仍能完成。

### 5.2 协议、路由、拆分合并

第二种比较不能要求内部事件逐一对应，也不能只要求输出公式相同。建议的有限 fragment 充分条件：

1. **事件可有限实例化**：同一固定 mesh，传输数量、设备目标、范围和等待量可确定；数据相关循环、
   轮询和动态路由暂不进入。
2. **存储关系确定**：源数据已初始化；发送完成前源区域保持稳定；接收完成前目的区域不被读取；
   重叠访问受到已证明的顺序保护。
3. **计数消费明确**：信号量初态有依据；每次等待对应明确的一组生产事件；累计等待可以包含多个 DMA，
   但必须核对总量、单位和消费次数。
4. **复用可区分**：缓冲区和信号量有实例、代次；不存在前后两代信号混入同一次等待、计数溢出或竞争
   消费者。
5. **不存在未建模的阻塞来源**：不读取 semaphore 数值决定控制流；资源限制、隐式 barrier 和所依赖的
   DMA 进展保证进入执行契约。

建议的数据结构至少包括：

```text
StorageInstance：分配位置、设备、调用/迭代实例、存储版本
Transfer：源/目的区域、数据版本、传输量、发送/接收完成事件
Wait：信号量实例、消费量、单位、已证明的生产事件集合
EventGraph：程序顺序、数据依赖、内存冲突、完成与等待依赖
```

**事件图无环本身不够。** 在上述受限模型中，等待没有竞争性消费、所有前驱条件都是明确的"全部完成"，
才可以检查计数和初态、构造完整依赖图、做拓扑检查、验证所有等待可满足且退出状态合法。在明确的
"已发起 DMA 最终完成"等执行假设下，这才支持进展结论。不能仅找到一个成功调度就声称所有允许调度都
安全。超出这个模型的情况先拒绝相关改写，不能用不完整 DAG 代替协议状态分析。

两个 milestone 分别验收：

- **异步同步化**：证明 start 跨过的操作与它可交换，并重新检查全局等待依赖。
- **协议/路由/拆分合并**：两侧分别证明存储安全和进展，再比较边界可观察的数据流。拆分需证明范围
  恰好分割；转发需证明数据版本沿路径保持，中间缓冲区不引入额外可观察行为。

### 5.3 不能无条件移动的反例

以下依据发送/接收语义构造的双设备协议示意，不是已经在 TPU 上运行的实验。两次传输使用独立缓冲区
和匹配的完成信号，排除数据竞争干扰：

```text
设备 A：                     设备 B：
start_send(x -> B)           wait_recv(x <- A)
wait_recv(y <- B)            start_send(y -> A)
wait_send(x)                 wait_send(y)
```

原协议的进展顺序是 A 发出 x，B 收到 x 后发送 y，A 收到 y，然后两侧完成等待。如果把 A 的
`start_send(x)` 延后到它自己的 `wait_send(x)` 前，A 就先等待 y；B 又必须收到 x 才发送 y，形成等待环。
改变的不只是执行时间，还包括程序能否完成。

因此建议的处理是：**先识别数据、内存和同步依赖，再将满足条件的异步区域同步化。** 无法应用该规则时
保留异步事件和等待关系，继续尝试图比较；不能应用规则不代表两图不等价。缺少必要语义或最终未能判定时，
继续使用现有分类异常通道。异步同步化也不是所有异步等价改写的唯一规则。

该反例已用两个 CPU 模拟设备重放：正常协议可完成；延后发送版本完成模拟图编译后，在调用模拟执行时
未返回，由父进程的 20 秒子进程预算终止。超时现象与等待环分析一致；超时本身不作为一般程序死锁的证明。

### 5.4 进展契约与 DMA 参数

官方文档区分发送完成、接收完成与信号量计数消费，不能把任意 wait 当作同一个事件。本地锁定版本的
`jax/_src/pallas/mosaic/primitives.py::AsyncCopyDescriptor` 将 `start`、`wait_send`、`wait_recv` 分开
实现；远端 `.wait()` 会依次等待发送和接收。见
[TPU 分布式通信与 DMA 信号量](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html)。

进展契约按四类记录，不能把未证明的程序条件塞进 assumptions：

| 类别 | 可报告内容 |
| --- | --- |
| 官方操作语义 | 发送/接收完成的含义、等待的计数消费、退出时 semaphore 状态要求、barrier 的作用。 |
| 平台模型假设 | 无设备/链路故障；可运行参与者不会永久饥饿；满足合法性和资源条件的已发起 DMA 最终完成。 |
| 必须检查的程序条件 | 目标设备与路由合法、存储生命周期、初始化、计数守恒、等待来源、复用代次，以及所需入口同步。 |
| 首版拒绝的情况 | 未知初态、竞争性消费无法消解、依赖原始 semaphore 数值的控制流、未建模阻塞资源、未证明安全的 barrier 删除。 |

官方文档给出了操作和故障语义，但没有找到支持"任意已发起 DMA 无条件最终完成"的保证；该条只能作为
平台模型假设报告，不能标成官方活性定理。每条记录建议包含 `id、适用环境、依据类别、检查结果、来源`；
报告分别列出"模型内进展证明""CPU 模拟结果""TPU 实机验证状态"。

DMA 参与比较的参数需要逐字段处理：`add` 影响数据更新，`device_id_type` 影响设备解释，
`is_wait_send` 影响完成语义；不能只比较 effects 的类型集合。

## 6. 原语覆盖、参数分类与审计

建议将覆盖范围固定到项目锁定的 JAX / jaxlib 版本并维护清单，升级时审查差异。清单应包含官方原语、
变换与包装产生的内部原语、嵌套区域，以及 TPU Pallas 的状态、同步和通信原语，不能只扫描
`jax.lax` 的公开导出。用户已明确排除第三方/用户自定义 Primitive、FFI 与 callback；这些入口应明确
拒绝，不列入官方原语覆盖完成度。不能仅因名字相同或抽象 shape 相同就视为同一操作。

每个原语分别记录三个方面，避免把"认识名字"当成"支持语义"：

| 方面 | 必须回答的问题 |
| --- | --- |
| 结构语义 | 完整参数、抽象值、输入输出位置、effects、区域绑定与执行环境能否保真表示？ |
| 数学语义 | 可以展开为哪种公式，适用的 dtype、定义条件和改写规则是什么？未展开时能否保留经过审查的精确算子应用？ |
| 验证证据 | 是否有合法参数变化、独立重排、语义反例与组合测试？追踪、比较、编译和数值执行分别验证了什么？ |

不要求每个原语都先展开成加乘公式才能做结构比较。已识别且语义完整的纯原语，可以在相同输入、参数
和环境下按同一函数应用比较；跨分块使用时还需要其索引、切片或归约语义，不能假设任意原语都逐点执行。
不透明函数的符号残差不能直接作为源程序数学不等价的依据。

effects、随机状态、通信和外部调用不能套用纯函数规则。尚未适配的语义应明确失败，并计为覆盖缺口；
这样的失败不能算作"已完成所有原语支持"。原语覆盖完整也不意味着任意循环、输入相关控制流或任意图
变换都有可完成的数学判定。

inventory 应覆盖三个层次：

1. **身份清单**：普通 JAX、state、Pallas、Mosaic TPU 的真实 Primitive 对象；同名伪造对象拒绝。
2. **参数结构**：参数键、类型、枚举、默认值、嵌套描述符、Ref/区域绑定和 effect 端口角色。
3. **能力状态**：分别记录"能解析""能结构比较""能规范化""能数学转换"，不能因为出现在清单里就
   计为完整支持。

每个 schema 都应有合法变体测试、语义参数变化反例，以及未知字段、未知描述符类型、同名假 primitive
的拒绝测试。除已列字段外，还需审查 `BlockMapping.index_map_out_tree`、`GridMapping` 的索引操作数、
scratch、环境 hooks 以及 Ref transforms。

### 6.1 `pallas_call` 参数分类

| 参数 | 分类与处理 |
| --- | --- |
| `jaxpr` / `out_avals` / `grid_mapping` / `mesh` / `input_output_aliases` | 语义事实，完整解释。 |
| `interpret` | 执行模式。简单例子的 inner jaxpr 相同，不足以支持直接忽略；特殊 InterpretParams 还可能改变计算行为。比较对象和验证运行模式分别记录。 |
| `debug` / `name` | 通常属于诊断和来源信息；不要与 kernel 内实际的调试 effect 混淆。 |
| `cost_estimate` | 成本/执行策略信息，校验 schema 并保留，不当作计算公式。 |
| `metadata` | 通常为诊断信息，但要保留映射、验证保留键及合法性；锁定源码中已有 `mesh_axes` 的特殊处理。 |

锁定版本 `CompilerParams` 的字段建议分组：

| 字段 | 初始处理原则 |
| --- | --- |
| `dimension_semantics` | 调度与顺序语义；默认值也须结合上下文解释。 |
| `collective_id`、`has_side_effects`、`skip_device_barrier`、`allow_collective_id_without_custom_barrier` | 同步、外部 effect 或执行契约；不能直接忽略；隐式 barrier 也须进入分析。 |
| `shape_invariant_numerics` | 数值行为相关，必须保留并审查。 |
| `disable_bounds_checks`、`disable_semaphore_checks` | 影响检查行为；只有相应有效性条件已证明时，才讨论消除差异。 |
| `flags` | 逐键审计，不能把任意配置字典当成无语义。 |
| `allow_input_fusion`、`vmem_limit_bytes`、`internal_scratch_in_bytes`、`serialization_format`、`use_tc_tiling_on_sc`、`needs_layout_passes`、`fuse_transposed_lhs_in_matmul`、`opt_level` | 编译、资源和执行策略字段；初期保留，是否允许差异逐项建立版本化规则，不能仅凭名称断定不影响数值或可执行性。 |

## 7. 验证设计

### 7.1 证据层级

结构/数学结论、IR 合法性、CPU 模拟、TPU 编译执行、数值抽样、性能数据分别记录，不能互相替代：

- 结构/数学结论来自规范化的精确比较与可信数学后端，不依赖运行结果。
- IR 合法性校验只说明图能通过本项目检查，不证明数值等价或硬件可执行。
- CPU interpret 适合调查重排、地址与同步错误，是调试/测试工具；不能替代静态等价结论。
- TPU 编译/执行、多 host、layout、精度和实际并发行为需要单独实测。
- 数值抽样不能作为等价证明；样本相等或不等都不能单独改变结构/数学结论。
- 性能数据不属于等价结论。

### 7.2 验收矩阵

验收按语义能力及其组合组织，不按业务算子名称设支持边界。

| 维度 | 正例 | 反例或应明确失败的情况 |
| --- | --- | --- |
| 原语与嵌套区域 | 每个原语的自比较、合法参数组合、独立操作换序、嵌套调用 | 参数/操作数/输出交换，未解释的 effect 或区域，伪装成同名原语 |
| 分块与 grid | block 大小变化、grid 轴拆合/置换、等价索引映射 | 错误输入坐标、写入遗漏、重复贡献、将 program_id 差异忽略 |
| 边界与状态 | 尾块 mask、显式初始化、正确的窗口累加和 scratch 复用 | 错误 mask/默认值、读取未初始化数据、错误 alias 或生命周期 |
| 数学变换 | 归约分组、块内/块间计算重新组织；报告所用数学前提与浮点警告 | 系数/归约域/精度语义改变；查询未决不得返回不等价 |
| 异步同步化 | 本地独立搬运与同步版、多个独立搬运、可审查的流水线重排 | 不匹配的等待、可观察的进度变化、跨循环实例错误配对、5.3 的双设备等待环 |
| 通信实现改写 | 相同 mesh 下 DMA 传输拆分/合并、等价路由与通信协议变化 | 漏传/重复贡献、错误目标或范围、过早复用缓冲区、破坏进展条件；不同 mesh 明确报告契约不匹配 |
| TPU 组合 | 顺序轴累加、并行轴分区、DMA 与计算/通信组合 | 破坏依赖的重排、错误传输目标/范围、缺失同步、无法解释的硬件属性 |
| 函数与资源边界 | 多输出与 pytree、输入签名、确定性诊断和资源恢复 | 契约冲突、预算耗尽、超时、尚未审查的导数模式 |

结构/数学结论、IR 合法性、CPU 模拟、TPU 编译执行、数值抽样、性能数据分别断言。模拟发现的问题可以
帮助定位，模拟通过不能替代静态等价结论或 TPU 实测。正常结果继续沿用现有三类结论；结构不同、同步化
规则不适用或域分析未决都不能直接推出数学不等价。

### 7.3 通信 CPU 验收环境

每个用例在独立子进程中运行，避免故意构造的等待环留在调用者进程：

- Python 3.12，锁定 JAX/jaxlib 0.11.2；导入 JAX 前设置 `JAX_PLATFORMS=cpu` 与
  `XLA_FLAGS=--xla_force_host_platform_device_count=2`。
- 使用两个逻辑 CPU 设备的 `Mesh(..., ("d",))`，通过 `shard_map` 包装唯一调用；中继测试另开三设备进程。
- 通信使用 `pltpu.InterpretParams`（不是普通 `interpret=True`），分别运行 eager / on_wait。
- 固定 `detect_races=True`、`skip_floating_point_ops=False`、越界读取报错、未初始化填 NaN、
  logical bounds、禁止额外 HBM 分配；首批每设备一个模拟 core。
- 完成输出等待及 effects 同步后，检查数值和竞争检测结果。锁定版本不能假定竞争检测会自动抛异常。

进程预算建议：启动/导入/编译直到发送 `READY` 为 90 秒；正例执行 30 秒；已有静态等待环证据的负例
执行观察 5 秒；单进程总上限 125 秒。父进程按阶段计时，超时后终止整个进程组、2 秒后强杀并回收；
编译超时与执行停滞分别报告。超时只作模拟观察，不能单独证明死锁。

建议文件为 `tests/test_pallas_communication.py` 与 worker；静态协议检查进入普通 pytest，动态模拟
标记为 `pallas_communication`，普通 CI 排除，由独立 job 串行执行。

### 7.4 历史技术验证记录

以下为已执行的历史验证；本轮交互式重构不重复运行探针或生产算法测试。

首轮使用临时探针 `pallas_semantics_probe.py`，在独立子进程中配置 CPU 设备、追踪源图并运行官方 TPU
interpreter。每个进程使用固定版本，父进程负责超时和回收；故意构造的等待环不会留在调用者进程中，
子进程预算设为 20 秒。仓库当前未保留该脚本，后续实施需要将这些用例固化为回归测试。

每个 DMA 实验先以 `interpret=False` 追踪原始图，确认递归区域中恰有一个 `pallas_call`；再单独构造
`InterpretParams` 版本供模拟。模拟执行使用的内部 callback 不属于被比较源程序，也不改变"排除用户
callback/FFI"的需求。先完成 CPU 模拟图编译，再打印执行开始标记，以区分编译未完成和执行等待。

| 探针 | 次数/配置 | 观察 |
| --- | --- | --- |
| 独立本地搬运，start 位于独立计算之前 | eager、on_wait 各一次 | 数值样本与 `x + a + 1` 精确一致，动态 race detector 未报告竞争。 |
| 同一搬运，start 延后到 wait 前 | eager、on_wait 各一次 | 同上；可作为有条件异步同步化的正例。 |
| 两设备相互交换数据 | eager、on_wait 各一次 | 输出与 NumPy 的分片交换一致，未报告竞争。 |
| 两设备请求—回复 | eager、on_wait 各一次 | A 先发起发送，B 收到后再回复；输出正确，未报告竞争。 |
| 请求—回复中将 A 的 start 延后到等待回复之后 | on_wait | 已完成模拟图编译，进入执行后不返回；父进程的 20 秒子进程预算终止该进程。 |
| 状态消去及 AD 边界 | 单独一次调查 | 见下；确认局部状态转换可以复用，也确认原生 Pallas AD 的缺口。 |

共 10 个调查配置符合预期。其中 8 个是正常 DMA 模拟、1 个是故意等待的反例、1 个是状态/AD 调查。
"符合预期"不表示 10 个程序都等价或都成功执行。每个正常配置只执行一个固定数值输入，不代表对全部
输入或所有调度完成验证。

### 7.5 官方 interpreter 能复用到什么程度

锁定版本的 `InterpretParams` 支持 `dma_execution_mode="eager"` 和 `"on_wait"`，并有动态 race detector。
这适合调查重排、地址与同步错误；它是调试/测试工具，不能替代静态等价结论。见
[官方 InterpretParams 文档](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.pallas.tpu.InterpretParams.html)。

本地源码 `jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py` 中，DMA 分成发起、读出源数据并
通知发送完成、写入目标并通知接收完成；`on_wait` 推迟实际读写，但发起操作仍需先登记传输。因此
"DMA 在等待时完成"不等于"可将任何 start 移到当前设备的 wait 前"。

建议将官方 interpreter 用作组合反例和数值对照工具，静态适配仍保留独立的设备/地址/信号量语义。之后
TPU 代际、layout、实际精度和硬件并发行为需要单独实测。CPU 模拟通过不能消除平台假设；例如解释器
甚至允许启用真实 TPU 不支持的 HBM 分配方式，验证配置必须限制在目标硬件语义内。

### 7.6 状态转换与求导调查

使用 `float32[128]`、完整初始化输出的 `y = x * x` kernel：

- 原始 kernel 为 Ref 输入上的 `get` / `mul` / `swap`，包含读写 effects。
- `discharge_state` 可将引用更新转为显式值；默认保留 `float32<default>` memory space。
- 在这个独立局部例子中显式移除 memory-space 标记后，纯值图只剩 `mul`，没有 effects。前向、JVP 和
  VJP 的一个样本分别与 `x*x`、`2*x*t`、`2*x*c` 一致。
- 对原始 `pallas_call`，JVP 追踪成功；VJP 报
  `Linearization failed to produce known values for all output primals`。
- 将整个调用交给通用 HLO interpreter 后会出现 `while`；对这个表示直接做 VJP，又遇到 JAX 不支持该
  while 反向求导的错误。

局部 `discharge_state` 不解释完整 grid、跨设备 DMA 或共享存储。示例中零初始化输出不影响结果，
是因为每个位置都完整写入；不能据此为任意 kernel 补零或删除 memory-space 语义。建议将它作为经过
前提检查的局部转换组件，不用整个 HLO interpreter 的结果充当通用 TPU 语义表示。

Pallas 仅比较前向；JAX 层保留 JVP/VJP。上述 AD 调查记录用于解释设计取舍，不为本次范围新增 kernel
自动求导后端。

### 7.7 技术把握与证据边界

需求边界和职责结构已足够明确，可以推进原型验证；尚不能据此声称全部等价算法已经成立。共享语义图
能表达执行过程，与比较器能自动证明两个不同过程等价，是两个分别需要验证的目标。

| 技术项 | 当前把握与证据 | 尚未确定的部分 |
| --- | --- | --- |
| 单次源适配与共享视图 | 源码依赖核对支持职责拆分；现有结构算法和数学后端可复用。 | 新视图尚未实现；区域绑定、求导规则缓存、源图重建和迁移兼容性尚未原型验证。保留目录不意味着语义改造量一定很小。 |
| 跨分块与全局索引 | 对 `x + 1` 的两种分块/反向遍历已有数学推导及单个 CPU 样本。 | 动态索引、非整除尾块、重叠写入、循环累加等组合的域表示、判定算法与工具仍待验证；批量示例只作 oracle。 |
| 不同通信协议的等价 | 已调查本地搬运、两设备交换/请求—回复及延后发送反例，证明仅看 primitive 清单或最终公式不够。 | **最大的算法不确定性**：尚无适用于传输拆合、路由与协议变化的比较算法。需要全局数据流、信号量消费、缓冲区复用和进展条件，不能要求事件一一配对。 |
| 内存状态到数学公式 | 局部、完整初始化的 Ref kernel 可通过 state discharge 变成纯值图。 | 有 alias、流水线、多设备写入和原子操作时，如何建立正确的数据版本与转换条件尚未完成。不能把局部探针外推为通用 Pallas 数学转换。 |
| 官方原语及 SymPy 覆盖 | 现有结构/数学链路可复用；源码明确保留不支持、定义条件未决、CAS 未决和超时异常。 | 完整官方原语清单尚未建立；所有原语的语义适配与所有图变换的可判定性分别是覆盖问题。公式能构造出来，也不保证当前查询方法能在预算内判断。 |
| 新结论与证据接口 | 已确认原图双射与多对一语义转换需要区分，并选定三类证据。 | 如何在现有结果对象中保存派生视图、规则条件与后续匹配，尚无实现验证。 |
| TPU 硬件语义 | 历史实验来自 CPU 模拟和有限数值样本。 | 尚无真实 TPU 编译/执行证据，也未验证多 host、特定代际、layout 和实际并发行为。 |
| 真实规模与资源 | 现有后端有预算与进程超时机制。 | 尚未测量新语义分析在真实 kernel、长流水线和嵌套区域上的时间、内存及符号规模。 |

"所有官方原语都能被正确解释"不等于"任意两张由它们构成的图都能被自动判等"。对任意输入相关循环、
动态访问、通信过程或数学恒等式，不能承诺一套有限预算流程始终给出完备结论；未完成时保留分类异常，
而不是误报等价或不等价。

成功结论也有明确信任边界：源适配、规范化规则与实现、JAX 语义模型和可信 SymPy。SymPy 查询记录用于
审计，非独立形式化证明证书；原图对应检查共享源语义解释，也不能单独排除前端解释错误。本项目尚无
端到端形式化验证，不能声称实现已经达到零误判保证。

优先用同一 mesh 下"不同协议但应等价"的正例，以及只改变等待/缓冲区复用就应拒绝的近似反例验证模型，
再扩展索引和状态组合。需要获得分析依据及失败诊断，只有模拟输出相同不足以完成这项验证；相关实验
用例后续需固化为可重放的回归测试。

## 8. 开发计划

### 8.1 文件级顺序

先做 Pallas 垂直 slice，暂不迁移全部 JAX 消费者；不先搭完整框架，也不要求先创建所有空模块。

用户已确认：中间抽象层由本项目**自研**；不把 JAX/Pallas 直接翻译成 SymPy 作为比较主路径；
MLIR 虽长期看好但现阶段接入成本过大，**暂不接入**。因此下列文件全部用 Python 实现，不需要新的
开源依赖；`core/model.py` 的字段设计保持以后可映射到 MLIR，但不等待 MLIR。

现有位置的调整要求（目录名可先保留）：

| 现有位置 | 核对结论与调整要求 |
| --- | --- |
| [core/params.py](../../src/jax_reorder/core/params.py)、[core/semantics.py](../../src/jax_reorder/core/semantics.py) | 已集中部分参数与语义编码，可逐项提取为共同源适配；不能只给返回字典换一个名称。 |
| [core/canonical.py](../../src/jax_reorder/core/canonical.py) | `Graph` 是规范结果，`semantic_views` 是局部参数缓存。将算法读取改接视图，保留原始端口、来源及重建能力。 |
| [algebra/adapter/transport.py](../../src/jax_reorder/algebra/adapter/transport.py) | 仍解析 JAX 参数、scan/JIT、求导包装和 abstract value。源格式判断移到适配器，数学准入和 worker 冻结留在数学层。 |
| [algebra/adapter/calls.py](../../src/jax_reorder/algebra/adapter/calls.py)、[jit.py](../../src/jax_reorder/algebra/adapter/jit.py) | 共同调用复用留在 algebra，改为读取完整视图、契约和对应关系。 |
| [algebra/finiteness.py](../../src/jax_reorder/algebra/finiteness.py) | `nonfinite_graph` 也直接读取源子图和参数，需随数学消费者一起接到视图，不能只整理 adapter 目录。 |
| [algebra/_typing.py](../../src/jax_reorder/algebra/_typing.py) | 当前 `SymbolicGraph` 只描述纯数组运算，继续用作数学投影接口，不能将它当作完整源语义层。 |

`core/adapters`、结构算法、公共分析和 `algebra` 都可以依赖 `core/model`；`model` 不反向导入它们，
`core` 不反向依赖 `algebra` 或 `functions`。`algebra/adapter` 现名可暂时保留，消除源解析比先更名
更重要。具体文件名可以按实现调整，下表是建议顺序：

| 顺序 | 文件（相对 `src/jax_reorder/`） | 最小内容 |
| --- | --- | --- |
| 1 | `core/model.py` | Program/Region/Operation/Value、抽象类型、Origin、Ref、绑定 effects；独立 `SourceHandle`。 |
| 2 | `core/index_domain.py` | `IndexExpr`、`Domain`、`AccessRelation`；从已适配 index-map 区域求值，blocked 逆映射检查。 |
| 3 | `core/adapters/jax.py`、`core/adapters/pallas/call.py` | 共用遍历与最小纯原语处理，Pallas 参数与 Ref 解码。 |
| 4 | `core/canonical.py`、`core/correspondence.py` | 视图消费入口，先支持本 slice 的结构匹配与复核。 |
| 5 | `core/memory.py` | 存储版本与纯值投影。 |
| 6 | `core/normalization.py` | 跨分块规则、前提检查、normalized 证据。 |

首个 slice 就包含 `SourceHandle`、`Origin`、Ref、effects；完整可执行重建和 DMA 协议模型可以晚做。
可以晚做 DMA 协议模型、循环分析、复杂 alias 和硬件布局分析；遇到相关语义必须明确拒绝，不能暂时
丢弃字段。

### 8.2 三天 slice 验收

最小实现 + `tests/test_pallas_semantic_slice.py`：

- `test_self_comparison_has_structural_witness`
- `test_retile_reverse_grid_symbolically`
- `test_large_shape_does_not_enumerate`
- `test_tail_mask_gap_is_rejected`
- `test_program_id_data_is_preserved`
- `test_store_then_get_uses_current_version`

另用 `tests/pallas_domain_oracle.py` 对小域交叉验证。支持一维 blocked、反向遍历、128/256 跨分块和
受限尾块；只建立内部可运行比较链路。

可以把 SymPy 接入这个 slice 作为数学消费者：用 `discharge_state` + `strip_memory_space` 处理直线式、
完整写入的 kernel，复用现有 `SymbolicBuilder` 和可信 worker；但 `program_id`、mask/部分写、循环、
DMA 和非空 alias 必须显式拒绝，不能为了演示简洁性跳过状态与地址域证明。若这条直译探针无法在
2–3 天内完成上述最小状态功能化和覆盖证明，则停止旁路，把可用部分并入视图 + 分析实现。

三天内不做：通信、循环/atomic、非空 alias 的规范化、完整 JAX 迁移、公共 API 接入；也不宣称已经
完成整个 M1。

### 8.3 后续 milestone

1. **M1 静态跨分块**：完成符号地址域、覆盖/唯一写入证明、normalized 证据；支持 128/256 与反向
   grid、受限尾块；拒绝循环、累加、atomic、DMA、非空 alias。
2. **M2 有限通信**：先本地 DMA 同步化，再同 mesh 的拆分合并与转发；验收覆盖本地 start 提前/延后
   正例、源过早改写/目的过早读取反例、双设备请求—回复与等待环反例、整块与两段传输、三个逻辑设备
   的直接发送与中继发送。
3. **M3 JAX 迁移与入口接入**：两条现有路径消费共享视图，接入证据类型和 inventory；JIT、scan/while、
   区域绑定、常量、JVP/VJP、共同调用、非有限值和 correspondence 的现有回归保持结果及异常分类。
4. **后续**：循环、原子操作、scratch/alias、一般整数域判定器、DMA 协议/路由改写、TPU 代际与多 host
   实测、完整官方原语清单。M1 完成后安排 3–5 天 MLIR 分析后端 spike（见 3.7），再决定是否以
   optional 形式接入；在此之前保持纯 Python 视图与域实现。

第一步可能推翻字段设计和索引表示；第二步最可能推翻"现有图加几条顺序边就够用"的设想。因此在这两步
之前，不完成大范围 JAX 迁移。

### 8.4 待细化的实现机制

| 议题 | 尚待确定的细节 | 当前建议 |
| --- | --- | --- |
| 分块、动态索引与控制流 | 如何表达带 mask 的访问域、动态 grid、索引数组、循环携带状态，并证明覆盖/重叠关系？ | 视图保留精确整数运算、区域和符号域；M1 用有限 fragment + 小域 oracle；选择整数域工具前用跨分块、非整除尾块、动态索引和循环反例检验。 |
| 状态和同步改写的条件 | 如何建立 alias、初始化、缓冲区版本、累计等待及跨设备进展依赖？输入值前提从哪里来？ | 首先使用签名、原图和已证明的事实。未证明的索引范围或通信条件不能默认为真；条件不满足时不应用规则，最终所需条件未决走分类异常。 |
| 规范化后的结论如何报告 | 多对一改写后的匹配如何表示？ | 保留三类结果状态，用 `StructuralEvidence`/`NormalizedEvidence`/`AlgebraicEvidence` 记录依据；规则链保存规则版本、作用位置、转换前后引用、已检查前提和来源对应。 |
| 数值和特殊原语的精确语义 | 浮点重排约定之外，整数、转换、复数、非有限值、随机及原子更新如何解释？ | 沿用既有浮点约定；不扩展为忽略整数位宽、溢出、mask、精度或转换语义的许可。确定性 PRNG 按同一 key 和输入的结果比较，不默认采用同分布等价。 |

### 8.5 未决事项

- 通信的 Presburger/SMT 判定器、动态索引与 alias 的一般算法仍是研究项；首版只覆盖有限 fragment
  并显式拒绝其余语义。
- `normalized` 证据字段、规则链格式、以及 `AlgebraComparison` 从"correspondence 是否存在"改为显式
  证据分派的具体接口尚未实现验证。
- 整数域工具（ISL/islpy、Z3、MLIR Presburger）尚未做平台与性能 spike；新增依赖前需先验证
  Windows/WSL/CI 可用性，并用 M1 覆盖/交叠案例比较编码、性能和可审计性。
- README/ARCHITECTURE 与部分历史文档仍有过时描述；受保护文档的更新需要用户对具体文件的明确授权。
- 全量 pytest 基线需要在本机正常终端重新确认；本轮定向回归通过（见下节），但 agent 环境中的全量
  运行长时间停滞，不能据此更新全量数字。

## 9. 验证与讨论记录

### 9.1 本轮检查

- 定向 pytest：`tests/test_package_layout.py`、`tests/test_canonical.py`、`tests/test_semantics.py`
  共 130 passed；`tests/test_correspondence.py`、`tests/test_algebra_structural.py` 共 203 passed。
- 全量 pytest：两次后台运行在 agent 环境中长时间停在早期用例后终止；未完成，不报告为通过。
- `ruff check .` 通过；`ruff format --check .` 为 193 files already formatted；
  `tools/check_docs.py --changed` 通过；`git diff --check` 通过。
- 本轮生产代码未变；新增/修改仅为本文及开发文档索引已有的入口。

### 9.2 讨论与文档历史

- 2026-10-18：本文合并此前 5 份相关 Markdown 为唯一记录，保留已确认范围、结构方案、接口草案、
  异步反例和验证结论；删除重复记录并更新开发文档索引。
- 2026-10-18 第二轮：独立 review 当前仓库与本文计划，并与 Codex 讨论架构（模型 `gpt-6-astra`、
  `model_reasoning_effort="max"`）。用户指定的原线程 `01a14dbd-7d91-7662-9d27-a7dcd47e4145` 被
  Codex 桌面端占用（active writer），讨论从其 fork `01a14f08-e360-7851-a555-72b987247922` 继续，
  最终按讨论结果重构本文；受保护文档未改动。
- 2026-10-18 第三轮：确认"直接转 SymPy"的定位（见 3.6），并用四个最小探针核实 `discharge_state`
  的可用边界：直线式完整写 kernel 可转成纯值图并复用现有符号层；`program_id` 触发 `AssertionError`；
  DMA/semaphore kernel 因 `dma_sem` dtype 失败；部分写入退化为对初始输出值的 `dynamic_update_slice`
  并被符号层当作不透明函数。结论是 SymPy 只作数学消费者/原型，不替代视图、结构证据和地址/存储分析。
- 2026-10-18 第四轮：确认保留中间层（`ProgramView`），并梳理中间层可复用的开源组件（见 3.7）。
  结论是没有可直接替换的完整开源语义中间层；采用"JAX 内部事实 + 自研薄视图 + 开源域/判定/重写
  组件"的组合，MLIR/StableHLO/TVM 只作局部组件或 oracle。
- 2026-10-18 第五轮：确认 M1 不接入新的开源组件。MLIR 保留为以后的可选分析后端/序列化目标，
  不把 `ProgramView` 改写成 C++ dialect；StableHLO 只作测试 oracle，ISL/Z3 等按需以 optional 形式
  引入。依赖策略、接入决策与 MLIR 折中见 3.7。
- 2026-10-18 第六轮：用户最终确认：(1) 不把 JAX/Pallas 直接翻译成 SymPy 作为比较主路径，必须有
  中间抽象层；(2) 中间层自研，MLIR 虽然长期看好，但现阶段接入成本过大，暂不接入，只保留未来作为
  可选分析后端/序列化目标的可能。结论固化在 §1、§3.7 和 §8.1。
- 2026-10-18 第七轮：用户要求把确认结论写入文档、暂不实现代码。本轮只保留设计文档与执行计划
  （[active 计划](../exec-plans/active/pallas-semantic-layer.md)），未保留任何实现文件。

目录调查、首轮可复现实验和接口草案已完成；完整生产迁移尚未完成。实施进展维护在
[active 计划](../exec-plans/active/pallas-semantic-layer.md)，当前可运行边界如下。

### 9.3 内部垂直 slice 实施（2026-10-18）

用户已要求开始实现。本节是当前代码状态，前述实验/讨论记录保留其历史时点含义；
slice 通过不代表 M1、通信或整个目标完成。

| 实现 | 当前行为 |
| --- | --- |
| [core/model.py](../../src/jax_reorder/core/model.py) | 不可变 Program/Region/Operation/Value/Ref/effect、来源和独立 SourceHandle；视图内没有 JAX 或 SymPy 对象。 |
| [core/adapters/jax.py](../../src/jax_reorder/core/adapters/jax.py)、[Pallas call](../../src/jax_reorder/core/adapters/pallas/call.py) | 锁定版本和 Primitive 身份，校验 SSA、参数 schema、抽象值、上下文、Ref effects 与区域绑定；完整适配后才交给消费者。 |
| [core/index_domain.py](../../src/jax_reorder/core/index_domain.py) | 精确整数表达式、逐操作溢出检查、blocked 正向/反向双射及逆映射证明；生产路径不枚举程序或元素。 |
| [core/memory.py](../../src/jax_reorder/core/memory.py) | 输入只读、输出未初始化、swap 旧值、当前存储版本；尾块读写 guard 和有效域完整覆盖；共享纯值 DAG 不含 CAS 对象。 |
| [core/view_graph.py](../../src/jax_reorder/core/view_graph.py) | 内部结构快路径和原视图复核；目前保留源方程顺序，不替代旧 JAX BFS/歧义搜索。 |
| [core/normalization.py](../../src/jax_reorder/core/normalization.py) | `compare_program_views` 先结构匹配，再重分块规则与终端纯值匹配；规则证据可重放。公式不同通常抛数学未决，仅输入无关整数输出的精确坐标反例可判不等价。 |

已验证的正例包括 `x + 1`、组合逐元素非线性计算的 128/256 分块与反向 grid、非整除尾块，
以及写入一个输出后读取该新版本并生成另一个输出。mask 在所有有效输出坐标为真、在越界尾部为假
时才可消除；`other` 保留在源事实中并检查操作数，不能用补零代替输出初始化。
纯值投影只改写地址与 Ref 状态，保持浮点操作和操作数顺序；目前没有浮点交换/结合规则。

初始 slice 接受直接或默认 JIT 包装下的单个 `interpret=False` Pallas 调用、一维静态 blocked grid、同形输入输出、
块内整片或 lane 索引、默认 Ref 内存空间/布局/分片/上下文。不同读取映射即使覆盖同一集合也不会直接
判等。默认 JIT 的全部参数、区域实参/结果绑定和调用边界进入视图及规则证据；非默认 JIT 明确拒绝。
后续扩展已保留 alias/CompilerParams 源事实，并支持有限 shard_map 与单 program 本地 DMA/scratch
（§9.4–9.5）。循环/atomic、一般索引/部分切片、不完整 mask、未知字段/原语/effect 仍明确拒绝；
这些都是后续覆盖缺口，未缩减 §1 的目标。
结构自比较不承诺存储初始化和无竞争，只有规范化路径具有对应的域/存储证明。

源适配和规则实现是共同信任基础。原视图对应复核独立于 matcher；规则重放共享规则实现，不是另一套
形式化证明器。图不匹配不能推出数学不等价。`compare_functions` / `compare_algebraic_jaxprs` 仍按原先
行为拒绝 Pallas，接入留在 M3；不增加公开 Pallas API。

验证入口：

- [test_pallas_semantic_slice.py](../../tests/test_pallas_semantic_slice.py)：§8.2 的六个命名验收、近似反例、
  证据篡改、同名假 Primitive、完整图读取和 DAG 规模回归。大形状 `2**24 + 13` 只用 ShapeDtypeStruct；
  比较期间禁止 `eval_jaxpr` 且分析预算上限 256。
- [pallas_domain_oracle.py](../../tests/pallas_domain_oracle.py)：独立枚举小域；覆盖/漏写 1,280 个案例；
  pytest 另将生产逆映射证明与 oracle 对照。生产代码不导入 oracle。
- [adapter_inventory.py](../../tests/adapter_inventory.py)：Pallas call、grid、block、Ref、indexer 和读写参数
  纳入原有锁定版本快照；未知版本/字段仍须重新审查。
- CPU interpret 对完整块和尾块做独立数值对照；这是有限样本证据，未提供 TPU 编译或执行证据。

初始 slice 验证：Pallas slice 与包分层检查合计 **57 passed**，较早的结构/语义/对应关系组合回归 **382 passed**；
Ruff、格式检查和 basedpyright 通过。
后续扩展和全量回归的命令、失败及结果以 active 计划为准。

### 9.4 M1：分片与执行参数扩展

已增加类型化设备排列、mesh 轴、抽象分片和 `shard_map` 契约；原始/局部形状与输入输出 spec 的
对应在适配器内按锁定 JAX 规则校验。恢复手动轴环境后再适配子区域，保留 `check_vma` 与设备顺序。
`jit(shard_map(...))` 和 `shard_map(jit(...))` 均保留调用边界。不同 mesh 排列、不同分片绑定不能
只因局部输出公式相同而获得 normalized 结论。

Pallas 在 `check_vma=True` 时清除 kernel Ref 的 manual-axis variation；现在分别保留 BlockMapping
中的 Ref 类型、kernel 的形参类型及该绑定政策，不能要求两者原样相同，也不能把变化性整体丢弃。
对各 shard 独立的 kernel，根据纯值 DAG 的实际输入依赖证明输出在未装配轴上不变；关闭 `check_vma`
也不豁免此检查。分片输入产生不同输出却声明 replicated 的近似反例明确拒绝。

TPU `CompilerParams` 的全部 17 个字段纳入 schema 和视图。默认 grid 顺序按锁定 Mosaic lowering 的
`arbitrary` 语义记录。只有完成无跨 program 状态、完整覆盖及唯一写入证明后，才允许默认/显式
`arbitrary`/`parallel` 调度之间的跨分块规范化；其它数值、barrier、side-effect、资源策略保持契约，
非默认策略尚不应用跨分块规则。未知 flags 逐键审查前拒绝，core/subcore 调度仍需硬件绑定证明。
非空 input/output alias 已完整编码并可结构自比较，进入存储规范化时仍明确拒绝。

验证见 [test_pallas_sharding.py](../../tests/test_pallas_sharding.py)：两 CPU 设备的符号比较与独立 interpret
样本、设备换序/分片变化/错误复制声明反例、编译参数策略及证据篡改。当前与既有 slice、分层合计
**83 passed**；不作为 TPU 执行证据。

### 9.5 M2 本地 DMA：有限直线片段

已实现无远端设备、无 alias 的整 Ref 直线式本地 DMA，显式建模 start 与 wait，适配阶段不配对事件。
当前分析限单个 program：scratch DMA semaphore 在调用入口的零初态及退出时归零作为可核对规则，
避免未经证明地把共享 scratch 当成每个 grid program 私有。本节记录首次整块实现；累计等待与本地
拆分合并已在 §9.6 扩展，多 program 与远端协议仍待实现，不能因本地片段通过宣称 M2 完成。

存储分析跟踪发起时的源版本、目的缓冲区和信号量；源被提前修改、目的被提前读写、等待缺少已发起传输、
退出仍有未完成 DMA 都应失败。完成等待后才能发布目标新版本，并记录传输量与信号量消费。对无依赖的
算术允许 start 提前/延后，终端纯值比较只在内存安全与有限进展前提完成之后进行。反例与官方 CPU
interpreter 对照分别报告；不能用一次模拟成功替代静态证明。

[communication.py](../../src/jax_reorder/core/communication.py) 跟踪每个信号量的一次未完成传输、
源版本、目标 Ref、传输范围、start/wait 来源和复用代次。等待消耗完整传输后，信号量可以再次使用。
不同信号量可同时传输到互不重叠的目的 Ref；共享只读源允许，等待完成顺序可以不同。
`logical_bytes` 仅记录逻辑数据量，不解释为特定硬件代际的信号量计数编码。
normalized 结果明确携带“运行时最终完成有效本地 DMA”的平台进展假设；缺少 start/wait、未初始化源、
提前读写与未消费计数是必须检查的程序条件，不能放入假设来绕过。
计数语义参见 [JAX TPU 分布式文档](https://docs.jax.dev/en/latest/pallas/tpu/distributed.html#dma-semaphores)；
锁定版本的 `mosaic/interpret/shared_memory.py::Semaphore` 使用零初态，wait 完整消费完成量。

同一规则已扩展到 call 生命周期的 VMEM 数组 scratch：入口未初始化，完整块写入后产生新版本，
允许串行覆写和多次复用；每次 DMA 的源版本须保持到完成等待。仍只允许单个 program，以免把调用级
scratch 隐含改成每个 grid program 私有；输出继续要求一次完整写入。验证覆盖直接复制与经 scratch
中转的相同值、等待后的合法复用，以及未等待就覆盖源或读取目标的反例。

VMEM 的 `MemoryRef` 构造器使用空的默认抽象 mesh，即使处于 shard_map 内也是如此；这不代表
多个设备共享同一临时缓冲区。视图同时保留原始 scratch 类型和绑定到当前内核局部 mesh 后的数组
类型。只允许默认空描述符或已经匹配的局部 mesh，显式指向其它 mesh 的 scratch 拒绝，不能直接
丢弃分片字段。该路径分别验证 `check_vma=True/False`，输出仍需通过实际输入依赖的复制性检查。

验证入口为 [test_pallas_dma.py](../../tests/test_pallas_dma.py)。官方 CPU interpreter 在 eager/on_wait
模式下对串行复制、独立计算重排与 scratch 复用做数值对照；适配器仍要求原生 `interpret=False`
源图，模拟模式不作为待比较的执行契约。符号测试另检查多个在途读取共享源时，一个 wait 不能解除
其它传输的保护；100 次 scratch 版本更新在预算 512 内比较，不进行调度搜索。

当前 Pallas/分片/DMA/包分层与既有结构语义组合回归 **454 passed**；Ruff、格式、类型检查、
adapter/lax inventory、运行示例和小域 oracle 通过。全量回归仍有 SymPy worker 10 秒超时，
不能报告全量通过；具体失败与命令见 [active 计划](../exec-plans/active/pallas-semantic-layer.md)。

### 9.6 DMA 拆分合并：区间版本与累计等待

本阶段扩展单 program 本地传输到静态连续 Ref 切片。源视图保留 source/destination 的精确起点、
长度和步长；首个规则仅接受正长度、单位步长、合法边界，未知变换与动态切片仍拒绝。
存储以半开区间分段保存值、初始化状态和写入版本，按源图中出现的边界切分，不逐元素枚举。
两个不交叠目标区间可以独立写入；最终每个输出位置仍须有唯一写入。复制到不同起点时，输入坐标
须同步平移，不能只比较读集合。相邻同值公式区间可合并；其它分段保留为精确选择表达式。

一个信号量可以有多个未完成传输。wait 是计数消费，不把第一个 wait 强行配给第一个 start。
同一代允许一次总等待或多次部分等待；在总消费量达到该代已发起传输量前，所有源/目的的在途区间
继续受保护。最后一次等待后才发布全部目标版本，完整归零后才能开启下一代。部分消费后新增 start
暂拒绝，防止前后代混入。等待量过大、退出有剩余计数、重复等待、重叠目标、提前覆盖/读取都拒绝。
跨多个传输的计数合成先限定相同 dtype、匹配存储对、128 元素对齐的连续区间，并检查逻辑字节总量
不超过有符号 32 位上界；不把逻辑字节记录宣称为所有 TPU 代际的物理寄存器编码。

验证矩阵包括整块与两段/多段、不同发起顺序、总等待与分次消费、scratch 经切片中转、移位/交换
两个分段、未初始化间隙、共享计数器的错误局部完成推断、代次复用、预算与证据篡改。
CPU eager/on_wait 的独立模拟按相同矩阵验证正常路径；静态拒绝不能由一次成功模拟替代。

本地区间、分片、原 slice 与分层测试 **147 passed**；独立进程官方 interpreter **16 passed**，
两种模式均显式检查数值与 race detector。进程启动/编译预算 90 秒，READY 后执行预算 30 秒；
超时清理、错误退出和孤立子进程的回归 **4 passed**。CI 通过 `pallas_communication` marker
单独串行运行模拟，普通 job 保留静态检查。DMA 完整/切片 PyTree 与 TransformedRef 字段已纳入
锁定版本 inventory；这些仍是 CPU 证据，未进行 TPU 编译执行。

### 9.7 有限远端协议：进展与访问区间

已建立不依赖 JAX 对象的有限设备事件分析并接入源适配，全局值投影仍待接入。每台设备保留程序顺序，
每个 start 派生独立的发送完成、接收完成事件；只加入 start→发送完成→接收完成以及实际 wait 的
生产依赖，不能把 start 自动推迟到 send wait 前。首个远端计数规则要求每个 semaphore 实例仅一代、
一次完整消费，允许多个同类 producer 累加；部分消费、竞争消费者与混用发送/接收计数先拒绝。

计数零初态、总量相等与上界检查在拓扑分析之前完成。源读取保护区间从 start 到发送完成，目的写入
保护区间从 start 到接收完成；有交叠的冲突访问必须由真实依赖证明完全先后，不能使用某个拓扑排序
恰好给出的顺序。模型内无环只在这些计数条件以及平台进展假设下说明有限事件可完成。
该分析本身不产生等价结论；实际等价仍须补充分配生命周期、初始化及全局值公式的证明。

验证先覆盖双设备交换、请求—回复、延后发送的等待环、直接/中继三设备路由、计数不守恒及源/目的
提前访问；小协议另用独立计数状态机枚举所有执行交错核对进展，生产分析只访问事件与区间边界。

2026-10-19 实施进展：

- [core/protocol.py](../../src/jax_reorder/core/protocol.py) 实现设备/Ref 实例、发送与接收完成节点、
  单代累计计数及完整消费、拓扑进展和基于可达关系的冲突区间检查；不会把某个拓扑排序当成所有执行
  交错都成立的先后关系。独立计数状态机覆盖双设备六条命令的 **36 种设备内排列**及其所有交错。
- [core/remote.py](../../src/jax_reorder/core/remote.py) 从视图实例化单 program、一维 concrete mesh
  的有限协议；设备坐标、分支选择与中间整数运算须静态可求且不溢出。只接受默认 compiler 策略，
  其中 `skip_device_barrier=False` 且没有自定义 barrier；入口同步依据默认 TPU barrier 契约。
  其它同步策略拒绝，不作为额外平台假设。DMA 区间要求 128 元素对齐，每个计数器保持相同 dtype。
- 三设备 `lax.switch` 会生成 `clamp`；该原语按锁定身份和空参数 schema 适配，协议控制只求值三个
  静态整数且上下界有序的情形，并检查结果位宽。一般符号 clamp 不在整数域规则内，不能按名称删去。
- 适配器保留 `NamedAxisEffect`、axis_index、条件区域实参/结果与 effects 绑定；远端 DMA 记录原始
  源/目的/两个 semaphore 和 mesh 坐标 tuple，包括 wait_send 的源/目的交换。所有分支都先适配，
  未执行分支也不能隐藏未知语义。当前不接受 dict 路由、动态索引或控制、显式 barrier、多个计数消费代。
- `RemoteProtocolEvidence` 显式列出尚未证明的生命周期、输入只读、初始化/覆盖与值对应条件；
  平台无故障/无永久饥饿/有效 DMA 最终完成单独记录。证据可按源视图重放，不能代替全局内存和值证明。
  `normalize_view` 仍拒绝远端 DMA，公共 API 未接入，M2 尚未完成。
- 远端官方 interpreter **5 passed**：交换与请求—回复各在 eager/on_wait 完成、数值正确且无竞争；
  延后发送版本先静态检出等待环，再在 READY 后执行观察 5 秒超时，子进程组已回收。超时不是死锁证明。
- 三设备直接发送/中继的实际源图及官方模拟另 **4 passed**（两种执行模式）；与本地及双设备批次
  合计 **25 个隔离通信用例**通过。三设备最初因 switch 生成的 clamp 被拒绝，补齐静态控制规则后
  通过。最新 Pallas 与既有结构组合回归 **502 passed**；不包含未完成的全量 SymPy 回归。
