# 历史：9.15 开源 kernel 驱动的测试与修复（2026-10-20，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.15 开源 kernel 驱动的测试与修复（2026-10-20，完成）

本阶段于 2026-10-19 开始，2026-10-20 完成。用户要求先用真实开源 case，再按失败证据实施修复。新增 JAX 0.11.2 官方 matmul 与
FlashAttention 原调用，GitHub 固定 revision 为 `32544801e26115ac1794926d027148abf3baf009`；
从 GitHub contents API 取得完整字节，与已安装模块逐字节核对，blob/SHA-256 保存在
[来源清单](../../../../tests/fixtures/pallas_kernels/jax_state_manifest.json)。另复用既有固定 Tokamax
RoPE/量化 RoPE 和 vLLM 单步 Attention 原函数。官方调用保留源码、输入映射和执行策略；Tokamax 的旧调用包装错误单独修正，见下文。

首轮真实源追踪发现：普通结构入口的五类自比较通过，但局部入口分别受整块 `ref[...]`
的空 transform、合法空 grid（一次 kernel 调用）、外层 `custom_vjp_call` 与未登记的
dot/reduce/gather/tile schema 阻挡。这些首先是源适配覆盖问题，尚未到串行状态证明。
修复须完整保留端口、参数和区域；自定义求导包装只在已声明的 forward 契约下进入，
复用现有精确 schema 校验，不改变全函数求导比较契约。新纯原语进入视图不代表它的
值投影或数学等价已经实现，未支持的后续分析仍须明确失败。

[源案例与独立 CPU 对照](../../../../tests/pallas_upstream_state_cases.py) 使用官方
`force_tpu_interpret_mode`，保持原 kernel/grid，启用 race、越界和未初始化检查。
matmul 保留多个输出块及两个 K 块；FlashAttention 保留两个 KV 块，覆盖实际跨迭代状态。
另从原 matmul 函数的 AST 分别产生“每轮重置”“覆盖而非加法”“旧状态乘二”等 AI 反例；
原函数不被替换，两个随机种子提供独立数值反例。测试与结果持续记录在
[回归入口](../../../../tests/test_pallas_upstream_state.py)，静态结论、数值执行、拒绝路径分别报告。

串行状态实现以官方 matmul/FlashAttention 的实际后缀轴为首个规则：输出窗口由 grid 前缀
唯一确定，窗口不随连续的 `arbitrary` 后缀变化。覆盖和地址仍用符号域证明；状态解释器
在分析预算内按后缀字典序执行源 body，携带输出和 scratch 版本，每次把输入读取绑定到
该迭代的坐标。既不预设加法，也不把重复写入直接合并；覆盖、一般递推、首轮初始化和
末轮才输出均按源代码处理。大量串行状态步骤超过预算时先拒绝，不展开巨大迭代域；
输出空间及并行前缀不枚举。串行步骤间的通信资源须另有生命周期证明，首轮规则不放宽它。

dot/reduce 以带绑定坐标的纯张量操作节点进入值图，保存输入形状、dtype、收缩/归约参数及
结果读取坐标；core 不将 dot 改写为乘加或证明求和恒等式。代数桥接恢复这些张量操作并
复用既有 SymPy 转换，绑定坐标与输出/设备坐标隔离，避免嵌套操作的变量捕获。

Tokamax 的 lane gather 另设有限、精确的地址规则：只接收末轴、静态形状、
`PROMISE_IN_BOUNDS` 的逐 lane gather，索引必须完全由 lane 坐标和整数/布尔纯计算
构成。按预算穷尽该固定 lane 域，检查每一步整数表示范围和每个目标地址；它不是
随机数值采样，也不枚举 grid 或全局输出空间。数据依赖的索引、其他 gather 布局及
越界承诺明确拒绝。计算结果保存为确定的坐标查表，供后续地址替换与值投影使用。


**本轮发现并修复的问题。**

1. 源适配遗漏合法空 grid、整块空 transform、custom JVP/VJP forward 包装、
   `CostEstimate.remote_bytes_transferred`、动态切片起点及 dot/reduce/gather/tile 参数。
   新字段保留原始含义；动态切片只在有限循环已确定起点且矩形边界成立时解释。
2. 原来的输出唯一 writer 规则拒绝 matmul 和 FlashAttention 的串行更新。新规则证明
   唯一拥有输出窗口的 grid 前缀，再重放连续串行后缀。删初始化和并行重复 writer 均拒绝。
3. RoPE 的旧测试包装把上游原地函数绑定到输入 staging Ref，没有写输出 Ref。通用
   `interpret=True` 没暴露这个问题；TPU 专用解释器输出全 NaN。现在先将输入写到输出
   staging Ref，再调用未修改的上游 `_rope_body` 原地更新输出。外部 HBM alias 不等于
   两个局部 staging Ref 相同，不能用 alias 声明代替初始化。
4. 量化 RoPE 的两个输出形状不同（8×128 与 8×1）。两者分别证明写入覆盖和地址逆映射，
   共享相同窗口生命周期，不再要求输出形状完全相同。lane gather 穷尽检查地址范围，
   整数 bitwise 与布尔操作按 dtype 区分，并增加 XOR 越界承诺反例。
5. 分块 dot 的 SymPy 反例原先被两个完整求和的展开上限挡住。现在在相同已建立定义域、
   有限浮点输出下先构造完整输出差，交 SymPy 消去重叠前缀，再按原预算展开剩余部分。
   False 仍要求具体合法输出坐标和合法输入元素符号，不能从局部中间差异推出不等价。
6. 全量回归发现单 program 的 `x[0]` 读取在坐标组合后留下冗余块坐标，导致旧条件分支
   正例未决。修复在 core 中使用已证明的单块域直接绑定 program=0、lane=global；多
   program 仍保留块坐标。原失败的四类外层/分支/循环数学回归已复测通过。

**具体串行 case。** 官方 matmul 使用两个 K 块，首次将 accumulator 置零，然后每轮
执行 `acc = acc + dot(tile_x, tile_y)`。把两个块对某输出元素的贡献记为 3 和 5：

| 更新代码 | 最终值 | 本轮结果 |
| --- | --- | --- |
| 首轮置零，每轮 `acc += partial` | 8 | K=128、K=256 及普通 JAX matmul 由 SymPy 判等价 |
| 每轮先置零，再相加 | 5 | 完整输出反例，SymPy 判不等价；实际 CPU 执行也有反例 |
| 每轮 `acc = partial` | 5 | 同上 |
| 每轮 `acc = 2 * acc + partial` | 11 | 同上 |
| 删除初始化 | 未初始化读取 | core 明确拒绝 |
| 将重复输出的 K 维声明为 parallel | 无独立 writer 证明 | core 明确拒绝 |

3 和 5 是说明更新规律的例子；实际回归使用原始 16×256 与 256×256 输入、8×128
输出块和随机种子 7/23，没有把上游矩阵内核换成标量算法。源循环包含多少次覆盖、
累加或一般递推，必须由实际读写版本决定，不能仅看 grid 维度名。

**验证层次。** 五个原例（官方 matmul、官方 FlashAttention、Tokamax RoPE/量化 RoPE、
vLLM 单步 Attention）均通过两个结构入口、可重放资源/内存投影，以及独立 NumPy
投影求值对照。后者在完整输出的首点、末点和中点验证两个随机种子，只是独立数值
检查；精确覆盖、初始化和顺序结论来自 core。已有 17 个 AI 变体也全部通过该投影路径，
正例数值一致，负例取得数值差异。原例的双边资源检查及重放使用 `analysis_budget=30000`，
以容纳 RoPE 的逐 lane 地址证明；单边投影使用默认 10000，预算不足仍报分类异常。新的独立求值器见
[pallas_projection_oracle.py](../../../../tests/pallas_projection_oracle.py)。

数学证明单列：本轮完成 matmul 的两个正例和三个负例。另外，官方 FlashAttention
保持每次 128 个 KV 元素的计算不变，将两次迭代从 grid 的串行维移到一个 256 宽窗口内的
kernel 循环，core 可直接证明两个规范化值图相同，证据重放通过；第二次切片起点实际
为 128，专用 CPU 解释器亦执行通过。这不等于任意改变归约分组都已证明。一般跨 KV
归约分块改写、量化 cast 等复杂数学结论不由数值结果代替。

CPU 专用解释器主批次 79 passed，随后新增 FlashAttention 宽窗口执行也通过；80 项中
3 项明确断言上游缺 `semaphore_read`/masked load lowering，1 项确认错误通信协议在
READY 后停滞，其余 76 项完成预期数值/资源执行。新增的十项原始调用/状态反例全部
实际执行。没有真实 TPU 执行证据。Pallas 静态主回归 **539 passed，79 deselected，378.63s**；随后四项 bitwise/布尔 lane
边界与一项 FlashAttention 跨 grid/循环正例分别验证通过。按当前 collection 去重后为
**544 项非通信-marker 测试 + 80 项通信-marker 测试 = 624 项全部通过**，无 skip/xfail。
其中“测试通过”包含上述明确断言的拒绝和停滞，不等于所有错误程序执行成功。

复现当前完整矩阵：

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest tests/test_pallas_*.py \
  -m 'not pallas_communication' -q --tb=short
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest tests/test_pallas_*.py \
  -m pallas_communication -q --tb=short
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest tests/test_pallas_upstream_state.py -q --tb=short
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest \
  --ignore-glob='tests/test_pallas_*.py' -m 'not pallas_communication' -q --tb=short
```

本轮原始日志：`/tmp/jax-reorder-upstream-baseline.log`（修改前 499 passed、70 deselected）、
`/tmp/jax-reorder-pallas-final.log`、`/tmp/jax-reorder-pallas-cpu-final.log`、
`/tmp/jax-reorder-upstream-semantic-final.log`（40 项）、
`/tmp/jax-reorder-upstream-lanes-final.log`（28 项重叠专项，含四项新增边界）、
`/tmp/jax-reorder-upstream-attention-replay.log`（3 项重叠专项，含两个新增用例）。
重叠批次不相加。临时日志可能被清理，持久证据为源码、清单和上述回归入口。
共享 JAX/SymPy 路径的补充回归已完成。首次非 Pallas 批次通过 362 项后主动中断，
仅将未执行的 1599 个 node ID 分为四组，分别 **400、400、400、399 passed**；
与已完成的 362 项去重后为 **1961 项全部通过**。四个分组各报告同一个上游
`pallas.reciprocal` 迁移弃用警告，没有失败或跳过。日志为
`/tmp/jax-reorder-nonpallas-regression.log` 和 `/tmp/jax-reorder-rest-{0,1,2,3}.log`；
去重清单保存在 `/tmp/jax-reorder-regression-shards.json`。

按最终 pytest collection 核对，Pallas 624 项与其余 1961 项无重叠，完整覆盖当前
**2585 个测试节点，分批全部通过**。这是同一份最终代码上的分批验证结果，未声称
单次全仓库 pytest 运行。Ruff、261 个文件的格式检查、basedpyright（0 error/0 warning）、
tracing 示例、锁定版本 adapter inventory、文档链接与 `git diff --check` 均通过。
本阶段已无未解决的测试失败；串行通信/alias、非后缀窗口复用、一般动态索引和复杂
归约分组等仍是明确拒绝或尚未证明的范围，见 [PALLAS-001](../../index.md#待处理)。
