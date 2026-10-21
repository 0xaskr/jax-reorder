# 历史：9.14 多维 grid 与访存（2026-10-19）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.14 多维 grid 与访存（2026-10-19）

用户明确要求在语义层组合 grid、index_map 和 kernel 局部坐标，不将地址解释交给代数系统。
按 [JAX grid/BlockSpec 契约](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)，
Blocked 的每轴全局地址是 `index_map(program)[axis] * block_shape[axis] + local[axis]`。
源适配保留完整轴序、类型、索引、切片、mask 和副作用；core 证明 program 域到块域的映射，
再组合逐轴访存，检查覆盖、越界、初始化与冲突。SymPy 仅消费已消去状态的完整值公式。

本轮已扩展多个正静态 grid 轴及多维 Blocked。规则域支持轴对应、换序和反向，
允许长度为 1 的退化轴；未使用的非平凡 grid 轴会造成重复 writer，必须拒绝证明。
grid 与数组的秩不必相同：将块坐标按混合进制线性化，验证每个 program 轴的系数、
偏移、总块数及每轴边界后，建立可逆对应，支持一维 grid 拆为二维/三维块坐标及相反合并。
例如 `grid=(8,)`、`index_map(p)=(p // 4, p % 4)`、`block=(8,128)` 覆盖 `(16,512)`；
实际地址是 `(8*(p//4)+r, 128*(p%4)+c)`。core 证明其与 `grid=(2,4)`、
`index_map(i,j)=(i,j)` 的逐元素地址对应，不调用 SymPy。

index map 的纯 JIT 包装保留参数、类型和运算。Python 整除的符号修正若不能在整个域
直接确定，则按每轴零点/内部/末点作至多三段的预算内划分，要求所有覆盖矩形得到同一
整数表达式；不能得到统一证明时明确拒绝，不枚举 program。
内存区域用半开矩形表示，静态切片按源边界分割，不枚举元素或 program。
可验证的坐标变换用于值表达式，不能将不连续矩形误认为连续线性区间。
普通本地资源保留逐 program 初始化、归零和 DMA 完成要求；不能因增加维度而删除原有检查。

#### 已实现范围与边界

| 项目 | 已实现 | 仍明确拒绝的证明范围 |
| --- | --- | --- |
| 地址与覆盖 | 多维正静态 grid/Blocked、轴换序/反向/退化轴、混合进制拆合；输入允许重复读取和广播，不要求输入唯一 writer | 动态 grid、一般非线性映射、重复 writer、未覆盖输出轴、Element/BoundedSlice、非默认布局 |
| 局部访存 | 多维整块、单位步长矩形、静态标量索引与降秩、全块 identity lane 索引；逐区域初始化、版本、源证据重放 | 一般高级/数据索引、动态切片、非单位步长；多输出目前要求相同输出形状及 writer 坐标 |
| 值与坐标 | 逐元素广播、broadcast/transpose/无置换 reshape；外层矩阵/行向量纯计算和精确数组常量；program_id 与具名设备轴依赖保留 | 未审查纯原语、kernel 内直接捕获数组常量；浮点 iota 不当作整数坐标；一般 dot/reduce 的张量值规则仍需扩展 |
| 尾块 | 对齐输入输出 map 的全块 identity 访问，证明 mask 在全部有效域为真、每个 padding slab 为假 | 有效域内留洞、越界尾部无 guard、一般部分/动态 mask、padded scratch/alias 的生命周期组合 |
| 本地资源 | VMEM scratch、同坐标输入输出 alias、整块及连续矩形 DMA、稳定读数/顺序普通 counter；独立多 program，包括 parallel scratch/DMA | 非连续 DMA、远端多维 DMA 语义投影、错位 alias、跨 program 状态；parallel 普通信号量缺 core 绑定仍拒绝 |
| 控制流 | 静态 scan/fori 与有限 carry while；动态分支逐支检查，合并矩形版本和值 | 数据依赖终止未证明、分支遗留在途 DMA 或不同计数、无初始化分支 |

DMA 的传输区域必须在线性布局连续；普通矩形访问的在途冲突检查使用保守的线性包围区间，
可能拒绝实际上分离的列区域，不会据此把未知冲突判成安全。多维远端协议/全局组装没有
随本轮自动扩展；已有单轴规则继续保留。局部结构相同与资源证明成功仍分别报告。

实现：[tensor_domain.py](../../../../src/jax_reorder/core/tensor_domain.py) 证明域及逆映射；
[tensor_storage.py](../../../../src/jax_reorder/core/tensor_storage.py) 保存矩形分区；
[tensor_values.py](../../../../src/jax_reorder/core/tensor_values.py) 组合值坐标；
[tensor_memory.py](../../../../src/jax_reorder/core/tensor_memory.py) 检查状态、控制流与资源。
适配器只提取源事实；代数桥接只接收已证明的多维输入坐标和输出值。源图仍保留全部操作，
本轮没有实施结构层的纯死代码删减。

后续串行 grid 扩展以 kernel 的实际代码为准：`dimension_semantics` 给出迭代顺序约束，
不能仅凭 `arbitrary` 把状态更新解释成求和。`y[...] = f(...)` 是覆盖，
`y[...] += f(...)` 是加法更新，`y[...] = g(y[...], x[...])` 则可能是一般递推。
core 应结合实际地址、初值、分支和读写版本建立跨迭代状态传递；即使是 `+=`，也需检查
右侧对旧状态的依赖及中间值的可观察用途。只有满足转换条件的加法累加才可投影成求和，
再由 SymPy 比较数学表达式；其它递推保留原有依赖或明确报告尚不支持。
这项工作的准确范围是“串行 grid 状态语义”，目前尚未实现，不能由调度参数直接替代。

#### 验证结果

[新增测试](../../../../tests/test_pallas_tensor_memory.py) 覆盖二维/三维、分块变化、换轴/反向、
拆合维度、转置/广播、矩形分区、状态与控制流组合。负例保留输入映射差异、program_id
作为数据的差异、重复/漏写、越界、未初始化、在途 DMA、缺 core 绑定、未知动态索引等。
`MemoryVersion.rectangle` 的篡改会被证据重放拒绝；带 mesh 的局部设备坐标保留轴范围。
最后审查补充浮点 iota 反例：完整结构自比较仍通过，整数坐标投影明确拒绝。
`2**30` 个输出元素及大型混合进制域在固定分析预算内完成，未枚举元素或 grid 实例。

JAX/jaxlib **0.11.2**、Python **3.12**；本轮最终批次：

| 检查 | 结果 | 证据含义 |
| --- | --- | --- |
| 多维新增测试完整批次 | **62 passed，26.92s** | 含 core normalized、公共代数桥接、拒绝路径、预算与 CPU 对照 |
| 既有 Pallas 主回归 | **428 passed，58 deselected，218.72s** | 结构、远端/本地协议与值、分片、数学、局部入口及分层；排除通信执行 marker |
| 既有局部 CPU 专项 | **11 passed，59 deselected，22.29s** | 原局部控制/资源及两 core scratch/DMA/alias 全部实际执行成功 |

新增完整批次中的 12 项 TPU 专用 CPU interpreter 检查，**11 项实际执行成功**，
包括矩形读写、本地 DMA、scratch、counter、alias、循环/分支及两 core parallel scratch/DMA。
尾块 masked load 的一项在上游编译阶段报 `NotImplementedError: masked load_p`，测试明确
断言该拒绝，不将它报告成执行成功。另有一项通用 CPU `interpret=True` 尾块数值对照实际通过。
没有真实 TPU 编译/执行证据；CPU 数值、资源检查和 core 精确证明分别报告。

复现命令（可按环境设置 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`）：

```bash
uv run pytest tests/test_pallas_tensor_memory.py -q --tb=short
uv run pytest tests/test_pallas_structural.py tests/test_pallas_remote.py \
  tests/test_pallas_remote_values.py tests/test_pallas_dma.py \
  tests/test_pallas_dma_regions.py tests/test_pallas_sharding.py \
  tests/test_pallas_semantic_slice.py tests/test_pallas_protocol.py \
  tests/test_pallas_algebra.py tests/test_pallas_coverage.py \
  tests/test_pallas_communication_protocols.py tests/test_pallas_local_entry.py \
  tests/test_pallas_local_scope.py tests/test_package_layout.py \
  -m 'not pallas_communication' -q --tb=short
uv run pytest tests/test_pallas_local_entry.py tests/test_pallas_local_scope.py \
  -m pallas_communication -q --tb=short
```

临时日志 `/tmp/jax-reorder-tensor-reviewed.log`、`/tmp/jax-reorder-tensor-regression.log`
和 `/tmp/jax-reorder-tensor-existing-cpu.log`
记录对应批次；可能被清理，持久复现入口是上述测试与命令。此前增量检查与最终批次重叠，
不相加为独立用例数，也不将定向回归报告为全仓库 pytest 全量结果。
Ruff、格式检查、basedpyright、tracing 示例、锁定版本 adapter inventory、文档链接及
`git diff --check` 均通过；受保护的核心文档保持原样。
