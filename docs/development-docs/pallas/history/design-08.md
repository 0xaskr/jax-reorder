# 历史：开发计划

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

## 8. 开发计划

### 8.1 文件级顺序

先做 Pallas 垂直 slice，暂不迁移全部 JAX 消费者；不先搭完整框架，也不要求先创建所有空模块。

用户已确认：中间抽象层由本项目**自研**；不把 JAX/Pallas 直接翻译成 SymPy 作为比较主路径；
MLIR 虽长期看好但现阶段接入成本过大，**暂不接入**。因此下列文件全部用 Python 实现，不需要新的
开源依赖；`core/model.py` 的字段设计保持以后可映射到 MLIR，但不等待 MLIR。

现有位置的调整要求（目录名可先保留）：

| 现有位置 | 核对结论与调整要求 |
| --- | --- |
| [core/params.py](../../../../src/jax_reorder/core/params.py)、[core/semantics.py](../../../../src/jax_reorder/core/semantics.py) | 已集中部分参数与语义编码，可逐项提取为共同源适配；不能只给返回字典换一个名称。 |
| [core/canonical.py](../../../../src/jax_reorder/core/canonical.py) | `Graph` 是规范结果，`semantic_views` 是局部参数缓存。将算法读取改接视图，保留原始端口、来源及重建能力。 |
| [algebra/adapter/transport.py](../../../../src/jax_reorder/algebra/adapter/transport.py) | 仍解析 JAX 参数、scan/JIT、求导包装和 abstract value。源格式判断移到适配器，数学准入和 worker 冻结留在数学层。 |
| [algebra/adapter/calls.py](../../../../src/jax_reorder/algebra/adapter/calls.py)、[jit.py](../../../../src/jax_reorder/algebra/adapter/jit.py) | 共同调用复用留在 algebra，改为读取完整视图、契约和对应关系。 |
| [algebra/finiteness.py](../../../../src/jax_reorder/algebra/finiteness.py) | `nonfinite_graph` 也直接读取源子图和参数，需随数学消费者一起接到视图，不能只整理 adapter 目录。 |
| [algebra/_typing.py](../../../../src/jax_reorder/algebra/_typing.py) | 当前 `SymbolicGraph` 只描述纯数组运算，继续用作数学投影接口，不能将它当作完整源语义层。 |

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
