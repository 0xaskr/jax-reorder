# 历史：9.12 设备局部视角的定向验证（2026-10-19）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.12 设备局部视角的定向验证（2026-10-19）

本轮按用户要求继续验证并保存本地证据，检查外层环境、动态通信和多 program 资源三类
不确定性。新增 [test_pallas_local_scope.py](../../../../tests/test_pallas_local_scope.py) 的 31 项
回归，未修改生产算法或实现新的设备局部入口。现有公共结构比较、类型化视图适配和
语义规范化分别调用，避免把其中一层的通过或拒绝当作其他层的结论。

基线是 `f8effb03c6a768490ef07e792fb64a3ced3a34d6` 加当前工作区的未提交实现；
环境为 Python 3.12.3、JAX/jaxlib 0.11.2、pytest 9.1.1。真实 native tracing 产生源图；
默认两个 CPU 设备，2×2 mesh 用四设备独立进程。没有真实 TPU 或多主机验证。

**外层计算与环境绑定。** “在 CPU 上计算”不决定它是否成为外层 jaxpr 方程。
下面四类静态预处理均在被追踪函数内构造，结果用途不同，边界也不同：

| 来源及用法 | 源图或前端结果 | 当前规范化与独立对照 |
| --- | --- | --- |
| Python `sum((1, 2))`，kernel 中加此标量 | 外层只有 `pallas_call`，标量进入 kernel 字面量 | 规范化通过；值投影及 CPU `interpret=True` 都得到 `x + 3` |
| NumPy 对静态数组求和，kernel 中加所得标量 | 同上，无外层常量槽位 | 同上；改变常量到 4 会产生结构差异 |
| NumPy `full((128,), 3)`，数组作为显式参数传给 kernel | 外层只有 `pallas_call`，但有一个数组常量槽位 | 结构比较、视图适配和 CPU 执行通过；规范化拒绝外层常量绑定 |
| 同一 NumPy 数组由 kernel 闭包直接捕获 | JAX 前端报 `captures constants ... You should pass them as inputs` | 尚未进入本库比较；按 JAX 要求显式传参后归入上一行 |

因此不能写成“`pallas_call` 前有计算就拒绝”。对被追踪的 `function(x + 1)`，外层有
`add` 方程，当前规范化拒绝；对静态标量预处理则通过。数组参数用例即使没有外层算术
方程仍被拒绝，属于本库覆盖缺口，不是该 Pallas 程序不合法。
当前 [normalization.py](../../../../src/jax_reorder/core/normalization.py) 的 `_call` 只接受没有
外层常量、只有一条调用、输入输出按序直连的已审查包装，尚未提取任意父图中的调用及绑定。

| 定向用例 | 结构与来源证据 | 规范化边界及含义 |
| --- | --- | --- |
| 外层 `axis_index("d") + offset` 经 scalar prefetch 传入 | 自比较通过，改变 offset 被识别为结构差异 | 类型化 adapter 首先拒绝 scalar prefetch/dynamic index；尚未到环境推理阶段 |
| 反转 mesh 物理设备排列，kernel 自身不使用轴 | 完整父图不同，抽出的纯 kernel 图相同 | 说明局部图与完整环境契约有区别，不代表可以删除所有 mesh 信息 |
| 抽出使用 `axis_index("d")` 的裸 kernel | 父图结构比较通过；裸 kernel 报 `InvalidJaxprError: unbound axis name: d` | 新局部入口需要保留/恢复具名轴上下文，不能只取子 jaxpr |
| 2×2 mesh，路由 `(row, (col + 1) % 2)` | 两轴及大小均保留；将 col 换成同范围的 row 被识别为结构差异 | 现有全局协议规范化要求单个 concrete mesh 轴；这不应成为未来局部结构入口的限制 |
| `axis_index("d") == 0` 决定平衡的 signal/wait 及输出分支 | 翻转条件产生结构差异 | 当前有限设备域内可规范化，完整输出投影与 NumPy 预期一致 |
| 输入值 `x[0] > 0` 决定相同形式的分支 | 保留条件及两分支，翻转条件产生结构差异 | 当前第一个拒绝点是标量 Ref 索引 `unsupported index operand`；不能把它误报为已经进入动态协议分析 |
| `signal(1); read()==1; wait(1)` 的稳定读数决定分支 | 结构与视图适配通过，翻转条件产生结构差异 | 协议展开把 read 结果作为未知控制量，报 `control/route must be a static integer`；稳定读数到控制流尚未接通 |
| 有限两次 `fori_loop` / `while_loop`，循环内 signal/wait 与更新输出 | 自比较通过；改成三次产生结构差异 | 类型化 adapter 分别拒绝 `scan` / `while`，尚不能据此给出值投影 |
| `(d + 1) % 2` 与漏掉 modulo 的路由 | 两图结构不同；错误图自比较仍可通过 | 合法路由的交换值投影正确；错误路由因末端目标越界拒绝 |
| 缺少 signal 或请求—回复循环等待 | 两个相同错误图仍能结构匹配 | 规范化分别报告计数不守恒或等待依赖环；结构相同不证明协议有效 |
| 普通两 program 分块，以及输入输出 alias | 普通 128/256 分块规范化可比较；alias 保留并产生结构差异 | alias 源事实可适配，但其执行契约仍不能规范化 |
| scratch、本地 DMA、普通 semaphore，各一/二 program | 六类源图均可结构比较 | 单 program 通过；多 program 的 scratch/DMA 缺生命周期证明，semaphore 缺多 program 协议规则 |

上述 `DIFFERENT` 仅指结构比较未匹配，不据此断言数学不等价。反过来，错误协议自比较
通过也是结构契约的正常行为，通信有效性必须单独检查。当前已知错误路由/等待环和
合法但尚未支持的构造都使用 `UnsupportedJaxprError`、`unsupported_semantics`，靠具体
原因区分；未来局部接口还需区分“已确认错误”与“证明能力不足”，不能把同一分类当作相同结论。

**官方 CPU 对照。** 新增七项隔离进程测试覆盖输入条件的两条分支、两类有限循环、
多 program scratch/DMA/普通 semaphore，其中 DMA 分别执行 `on_wait` 与 `eager`。
全部数组输出符合独立 NumPy 预期，开启 race、逻辑边界及未初始化内存检查后未报告 race。
因此这些具体合法用例的规范化拒绝属于当前支持范围不足。解释器设为一个 core/thread，
不能把该结果推广为任意并发调度或 TPU 硬件正确性。已知缺 signal、等待环及越界路由仅
做静态检查，不执行可能阻塞的通信。

联合通信测试的 23 项中，21 项实际执行成功，另外两项明确检查本地/远端 `semaphore_read`
的 CPU MLIR lowering 缺失；不能报告为 23 项全部可执行。稳定和不稳定观察值的既有源图、
规范化及证据篡改回归也重跑通过，但这不替代缺失的 read 后端执行证据。

**复现与结果。** 以下三批互不重复，共 **273 passed**；本轮未运行全仓库 pytest。

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest \
  tests/test_pallas_local_scope.py -m 'not pallas_communication' -q --tb=short
# 24 passed，7.57 秒；其中三项还执行无通信的 CPU 数值对照。

UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest \
  tests/test_pallas_local_scope.py tests/test_pallas_communication_protocols.py \
  -m pallas_communication -q --tb=short
# 23 passed，43.80 秒；七项新增，十六项既有。

UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest \
  tests/test_pallas_structural.py tests/test_pallas_remote.py \
  tests/test_pallas_dma.py tests/test_pallas_dma_regions.py \
  tests/test_pallas_sharding.py tests/test_pallas_semantic_slice.py \
  tests/test_pallas_protocol.py \
  tests/test_pallas_communication_protocols.py::test_counter_reads_are_not_treated_as_pure_ignored_or_constant_values \
  tests/test_pallas_communication_protocols.py::test_counter_evidence_and_witness_are_replayed_from_source \
  -m 'not pallas_communication' -q --tb=short
# 226 passed，17.13 秒。
```

本次原始 pytest 日志分别保存为 `/tmp/jax-reorder-local-scope-static.log`、
`/tmp/jax-reorder-local-scope-cpu.log` 和 `/tmp/jax-reorder-local-scope-regression.log`；
临时日志可能被清理，长期可复现证据为上述仓库测试与命令。
全仓库 Ruff、格式检查及 `git diff --check` 通过；`tools/check_docs.py --changed`
检查 11 个文档、213 个本地链接通过。

下一步仍按已定范围先补局部入口和覆盖：提取调用及外层常量/实参/轴绑定，补齐 scalar
prefetch 与标量索引；随后处理有限控制流及稳定 read 的控制依赖、多 program 资源生命周期
和 alias。按具体通信检查取得必要范围及对端约束，不因已实现的全局规范化只有单轴而
限制局部结构入口。一般符号协议、任意布局和真实 TPU 验证仍未完成，不交由 SymPy 推断。
