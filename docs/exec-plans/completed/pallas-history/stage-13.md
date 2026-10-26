# 历史：9.13 局部入口与覆盖补齐规则（2026-10-19）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.13 局部入口与覆盖补齐规则（2026-10-19）

本节落实 §9.12 定向验证后用户要求补齐的三项：调用环境、有限控制流和多 program 资源。
§9.12 的拒绝结果是修改前快照；本节列出的支持与边界为实施后的状态。

先建立独立的局部结构契约：抽取唯一静态 `pallas_call`，保留完整 kernel、grid、Ref/alias、
compiler 选项和输入来源表达式，按调用/具名轴绑定恢复环境。忽略不影响局部计算的物理设备排列，
但设备坐标的轴身份、大小及它参与的运算/路由均保留。多个调用或调用处于未解释的外层控制流中
明确拒绝。局部结构匹配不宣称外层程序或通信有效；通信检查单列，证据须从原始输入重放。

源 adapter 增加 scalar prefetch、标量整数索引、scan/while 的精确 schema 与区域端口；
不因某分支未执行而略过其结构校验。有限值解释在分析预算内展开已知次数的 scan 或能用
静态 carry 逐步判定的 while；终止条件依赖数据且没有边界证明时拒绝。动态条件枚举所有
分支并合并值表达式，各分支独立证明初始化、资源有效和退出状态；不可用某一合法路径代替全部路径。

本地普通信号量由单一顺序流维护计数，read 仅在观察值确定时转为控制值；DMA 在完整等待后
发布目标版本并解除在途保护。动态分支合并要求 DMA 静止、计数状态一致；未完成 DMA 或
计数依赖分支时明确拒绝。远端协议继续使用已有必要因果证明，不能把本地顺序计数规则套到远端。

多 program 采用独立分块证明：各 program 的所有 scratch 读取由本 program 的写入初始化，
每次退出 DMA 静止、counter 归零，故不依赖上个 program 留下的内容；全局输出分块无重叠。
支持的 alias 必须有相同输入/输出分块映射；禁止跨块读取已被别的 program 改写的数据。
锁定实现中的 `input_output_aliases` 复用外层 HBM 缓冲区，默认 blocked kernel 的输入/输出
仍是分别 copy-in/copy-out 的局部 staging Ref，不能据此合并它们或假定局部输出已经初始化。
真正通过区域实参传递的同一 Ref 则共用版本；输出的先后写入保留顺序，不能当作无序写。
顺序更新保留每个版本；未知布局、跨 program carry 与 core/subcore 特殊调度仍须专用证明。

上游依据：[TPU grid 与流水线](https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html)
区分 arbitrary 的顺序语义与 parallel 的独立迭代要求；锁定 JAX 0.11.2 的官方 interpreter
按 core 分配 VMEM/scratch 并在 grid 循环间复用。这里通过初始化/归零义务证明迭代独立，
不把 scratch 错当成每次 grid 迭代自动重新分配。使用单/多 core CPU 对照及跨迭代负例检查。

外层绑定进入投影时，常量数组保留精确内容与读取索引，纯运算保留原始操作、类型与参数；
不能把捕获常量变成任意新输入。验证至少包含：常量值突变、父输入换序、同大小不同轴、
循环次数突变、分支缺 wait、读数随调度变化、scratch 先读后写、未完成 DMA 跨迭代和 alias 错位。

#### 已落地接口与覆盖

公开入口和证据重放位于 [core/local.py](../../../../src/jax_reorder/core/local.py)：

```python
from jax_reorder import compare_pallas_kernels, verify_pallas_kernel_comparison

# left/right 是包含调用及其父环境的 jaxpr，不是丢失轴绑定的裸 kernel。
result = compare_pallas_kernels(left, right, check_protocol=True)
assert result.scope == "device_local"
assert result.protocol_checked
# 仅对 EQUIVALENT 结果重放源绑定及请求过的资源证明。
verify_pallas_kernel_comparison(left, right, result)
```

默认 `check_protocol=False`，结构匹配只表示设备局部图及必要环境一致。
`left_protocol`、`right_protocol` 分别记录每侧的可选证明；请求证明但遇到不支持的条件时
抛异常，不返回“已检查”。`DIFFERENT` 仅表示结构不同，不能据此断言数学不等价。
重放重新提取和运行相同规则，用于检验证据与源图绑定；它不是另一个独立证明算法。

局部入口先复用现有完整图规范化，故独立纯运算换序仍能匹配；随后从类型化视图提取调用。
kernel 中未使用纯运算与全部副作用继续保留。外层无关计算不属于局部结论，调用参数依赖的
常量、输入端口、运算及相关逻辑轴则进入绑定。共享表达式用 DAG 处理，避免重复展开；
证据不会仅凭哈希相同判等。物理设备排列和未参与局部语义的轴不进入局部键；
远端坐标保留全部 mesh 轴的身份与顺序，不能只按大小对应。

| 项目 | 已实现 | 必须保留的边界 |
| --- | --- | --- |
| 外层计算 | Python/NumPy 标量、精确数组常量、JAX 纯运算实参；父输入端口和内容突变反例 | 一个静态调用；未知包装或外层控制流拒绝；全函数投影不接受任意后处理/分片组装 |
| 调用环境 | 多维 mesh 的局部结构；具名轴经 scalar prefetch 传入；0 维或一维 prefetch Ref、静态有界标量读取 | 缺失轴绑定、嵌套 shard 环境、动态 prefetch index map 需要专用证明；计算过的全局分片实参尚不能做局部值证明 |
| 有限控制流 | 静态长度 scan/fori、反向 scan 与标量序列历史、静态 carry 决定终止的 while；动态局部分支逐支检查并合并值 | 数据依赖 while 终止未证明则拒绝；无限循环耗尽分析预算；动态分支不能留下不同 counter 或在途 DMA |
| 稳定读数 | 本地普通信号量计数、稳定 read 的分支依赖；已完成 DMA 的稳定零值 | 在途 DMA read 不能按某次调度结果固定；远端 read/控制仍使用已有有限协议规则 |
| 多 program | 一维完整独立分块的 scratch、本地 DMA、顺序普通信号量；逐 program 初始化、归零和完成义务；大域符号证明 | 跨迭代 carry、残余计数、未完成 DMA 拒绝；parallel 普通信号量缺 core 目标绑定时拒绝 |
| alias | 同坐标 BlockSpec 的外层输入/输出缓冲区复用，局部 staging Ref 分离；区域同一 Ref 保持身份 | 输出仍需本 program 初始化；错位块映射、重复父实参、一般布局和跨块依赖拒绝 |
| Python 取模路由 | `(axis_index("d") + 1) % 2` 产生的受审查纯 JIT 包装进入有限远端协议及值传播 | 不将包装作为不透明值，不扩展为一般动态路由或多轴全局协议 |

新增实现分工：[调用绑定](../../../../src/jax_reorder/core/call_bindings.py)、
[循环源适配](../../../../src/jax_reorder/core/adapters/control_flow.py)、
[局部资源与值解释](../../../../src/jax_reorder/core/local_memory.py)。局部常量/控制/状态的完整输出
已接回原有数学入口，仍由 SymPy 判断公式关系；地址、初始化、DMA 和计数证明留在 core。
本阶段尚不宣称任意合法 Pallas 语法都可规范化；后续 §9.14 已补静态多维地址域。
atomic、一般 alias/动态循环、数组/变换 semaphore、非默认执行策略及真实 TPU 验证继续跟踪。

#### 并行普通信号量的实际限制

两 core 的官方 CPU interpreter 对照发现：parallel grid 中未指定 core 目标的普通
`semaphore_signal` 可能让另一个 core 的 wait 一直无法满足。测试由隔离执行器超时回收。
核对锁定的 JAX 0.11.2 源文件
`jax/_src/pallas/mosaic/interpret/interpret_pallas_call.py`：signal 在缺失目标 core 时取 0，
wait 使用执行所在 core。native lowering 保留缺省 core，尚无本项目的硬件绑定证明；
不能把这次 CPU 现象直接外推为所有 TPU 执行都会死锁。

因此当前静态规则拒绝多 program 的 parallel 普通信号量，报需要显式 core 目标绑定；
不会将缺省 signal 当成每个 core 私有的本地操作。该例固定为拒绝回归，不在成功对照中重复执行。
parallel scratch、DMA 和安全 alias 已由两 core CPU 对照覆盖；默认 arbitrary 顺序 grid
的普通信号量另有成功对照。CPU interpreter 仍缺 semaphore_read lowering，两个相关用例
只验证预期的上游编译拒绝；静态推理不能替代真实 TPU 运行证据。

#### 可复现验证

新增 [test_pallas_local_entry.py](../../../../tests/test_pallas_local_entry.py) 验证局部/全函数 scope
分离、逻辑轴与父实参、纯运算换序、证据篡改、所有分支的资源义务、有限/无限 while、
scan 历史、scalar prefetch、稳定/不稳定 read、跨 program carry 和 alias 反例。
已有 [局部边界测试](../../../../tests/test_pallas_local_scope.py) 的受支持用例由预期拒绝更新为
完整投影断言；错误目标、缺失信号和循环等待仍保留拒绝路径。`2**24` 元素的大域用例
在固定分析预算内证明 scratch/DMA/普通信号量三个独立分块模式，不枚举 grid 或元素。

CPU 对照单独使用隔离进程和官方 `InterpretParams`，包括 race detector、越界检查、
未初始化值检查，以及 DMA 的 eager/on_wait 两种完成方式。两 core 并行对照覆盖 scratch、
DMA 和 alias；不执行已知不能满足等待的静态负例。

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest \
  tests/test_pallas_structural.py tests/test_pallas_remote.py tests/test_pallas_remote_values.py \
  tests/test_pallas_dma.py tests/test_pallas_dma_regions.py tests/test_pallas_sharding.py \
  tests/test_pallas_semantic_slice.py tests/test_pallas_protocol.py tests/test_pallas_algebra.py \
  tests/test_pallas_coverage.py tests/test_pallas_communication_protocols.py \
  tests/test_pallas_local_entry.py tests/test_pallas_local_scope.py tests/test_package_layout.py \
  -m 'not pallas_communication' -q --tb=short
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest \
  tests/test_pallas_local_entry.py tests/test_pallas_local_scope.py \
  tests/test_pallas_communication_protocols.py -m pallas_communication -q --tb=short
```

验证结果（Python 3.12.3，JAX/jaxlib 0.11.2，CPU）：

| 批次 | 结果 | 含义 |
| --- | --- | --- |
| 上述主回归 | **427 passed，58 deselected，248.05s** | 结构、规范化、协议、代数、旧尾块及局部入口 |
| 最后 alias 审查后的局部入口复查 | **35 passed，4 deselected，13.27s** | 比主回归多一个重复父实参 alias 拒绝用例，其余与主回归重叠 |
| CPU 独立对照批次 | **27 passed，71 deselected，53.20s** | 25 项实际成功执行，2 项预期 semaphore_read lowering 拒绝 |

不将重叠批次相加为独立用例数；这些是定向检查，不是全仓库 pytest 全量结果。
原始日志为 `/tmp/jax-reorder-pallas-release-check.log`、
`/tmp/jax-reorder-local-final-alias-check.log` 和 `/tmp/jax-reorder-local-oracles-final.log`。
临时日志可能被清理，测试代码与命令是持久的复现入口。
Ruff、格式检查、basedpyright、tracing 示例、锁定版本 adapter inventory、文档链接检查及
`git diff --check` 均通过；核心受保护文档未修改。没有真实 TPU 编译或执行证据。
