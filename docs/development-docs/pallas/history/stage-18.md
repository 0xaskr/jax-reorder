# 历史：9.18 多输出独立 index_map 窗口（2026-10-20）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.18 多输出独立 index_map 窗口（2026-10-20）

状态：完成。

用户要求补齐不同输出的窗口寿命。具体例子是 `grid=(M,K)`、末轴串行，
两个输出的 BlockSpec 分别使用 `y(i,j)=(i,j)` 和 `s(i,j)=(i,)`：
每次写 `y=2*x`，首步初始化 `s=0`，随后按代码更新 `s+=x`。
两者输入输出存储独立。`y` 每步换块，`s` 跨 K 步保留；这是各自 index_map
确定的 staging 生命周期，输出形状不同和累加公式本身不能代替该证明。

具体可运行的 BlockSpec 设置见
[两输出案例](../../../../tests/pallas_multi_output_cases.py)，其全局形状分别为
`(M*8, K*128)` 和 `(M*8, 128)`：

```python
grid = (M, K)
in_specs = [pl.BlockSpec((8, 128), lambda i, j: (i, j))]
out_specs = (
    pl.BlockSpec((8, 128), lambda i, j: (i, j)),  # 每个 j 一个结果块
    pl.BlockSpec((8, 128), lambda i, j: (i, 0)),  # K 步共享一个汇总块
)
compiler_params = pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))
```

修改前两个输出顺序均复现 `outputs require the same serial window lifetime`。
原来的 `state_domains` 要求所有输出缺失的 grid 轴相同，`tensor_memory` 也将
所有输出一起封存并失效。已改为：

1. 每个输出独立证明完整覆盖、唯一拥有者及连续的 arbitrary 后缀，保存各自
   `serial_axes`；所有输出后缀的并集构成实际串行重放域。持久数组 scratch
   仍按既有规则扩大到完整串行后缀，parallel 前缀保持符号且预算先于展开。
2. 每个输出根据自身拥有者轴判断窗口切换。仅对该输出检查完整初始化、
   在途 DMA 读写冲突、封存值公式并使 staging 失效；其他输出及 scratch
   的实际内容保留。最终每个输出分别按它的地址逆映射和窗口矩形组装。
3. ANY 输出保留全数组显式访问语义；输入 staging 的跨步 DMA 检查继续适用。
   同一输出块的非连续访问、parallel 重复 writer、未初始化读取、未完成
   DMA 和预算不足仍明确拒绝。输入输出共享 HBM 仍在用户排除范围内。
4. 一维多输出入口保留原独立 program 证明，仅在已知覆盖/窗口限制处进入
   tensor 状态证明，避免把独立大 grid 无条件展开。

同一输出块仍须连续写入，依据
[官方 TPU BlockSpec 契约](https://docs.jax.dev/en/latest/pallas/tpu/details.html#blockspecs-and-grid-iteration)。
本轮接入的是能分别证明拥有者前缀与串行后缀的静态规则块映射；不会把未知
地址函数、任意重复 writer 或一般动态 index_map 直接当作合法窗口。

已完成的 [39 项回归](../../../../tests/test_pallas_multi_output.py) 包括：

| 案例 | 结果 |
| --- | --- |
| `y(i,j)=(i,j)`、`s(i,j)=(i,0)` | 每步换 `y`，保留 `s` 累加；两种输出端口顺序、正/反向 `y` map、把运行中 `s` 写入 `y` 的前缀和均通过完整投影与重放。 |
| 三输出：`(p,3*i+k)`、`(p,i)`、`(p,0)` | 分别在每步、每个 k 循环结束、全部 i/k 结束时封存，内部块及边界坐标正确。 |
| 各输出局部块形状也不同 | `y` 为 8×128，汇总为 8×1；保持 reduce 的输入范围和汇总窗口。 |
| 把 j 移入 kernel 内 `fori_loop` | 两个不同 grid/BlockSpec 的 kernel 返回 normalized equivalent，证据重放通过；公开代数入口也等价。 |
| 上游 `test_hoisted_scratch_space` 派生 | 保持锁定函数体不变，包装增加每步 history 输出；原 final 输出只在末轮写入。两者分别组装为完整历史和最终状态。 |
| 仅修改汇总：每步清零、覆盖、输出 map 写到另一行 | 主输出仍正确，公共代数入口返回 inequivalent，并给出完整输出坐标反例；CPU 对照覆盖每步清零与错 map。 |
| 缺汇总初始化、短窗口漏写、短窗口读取上一块值 | 分别在真实读取或该输出的窗口结束处拒绝，不能借另一输出较长的寿命保留旧值。 |
| DMA 使用长寿命输出作为源或目标 | 在下一步等待可通过；另一输出正常切块。DMA 使用正在切换的短寿命源/目标、读前漏 wait 均拒绝。 |
| 一维 Ref、无控制流覆盖更新、多输出大 parallel 前缀 | 公共局部入口和规范化入口均接入；`2**24` 个独立前缀在预算 256 下仍保持符号。 |
| 把重复 writer 改为 parallel、非连续输出复访、超大串行后缀 | 保留原拒绝与预算异常，不通过枚举整个输出空间规避证明。 |

上游来源沿用 §9.17 的锁定 JAX 0.11.2 fixture/manifest，无新下载或修改上游源码。
派生例和自建例均明确区分。NumPy 投影对照使用两个种子、每个输出块的首尾
坐标；完整等价由值公式及证据重放给出。11 项官方 CPU 用例检查所有输出
元素，开启 race、逻辑 bounds、NaN 未初始化填充、`dma_execution_mode="on_wait"`；
这些数值/竞态结果不替代静态证明，也不是 TPU 实机验证。

CPU 首次运行的 10 项曾在编译阶段统一报原生 Pallas 不支持 CPU，原因是测试
先构造 `pallas_call`、后进入强制解释器上下文；JAX 在构造时读取该配置。
修正为在上下文内构造调用后，10 项及新增 loop 项全部通过。没有替换通用
`interpret=True`、修改源语义或放宽失败断言。

既有回归还发现一处入口兼容性问题：把所有一维多输出统一转入 local 路径，
虽然数学结论仍等价，却改变了普通同尺寸多输出的既有证明规则标签。
现按输出全局形状/局部块形状是否不同选择扩展入口；同尺寸简单多输出继续
使用原路径，控制流和 scratch 等既有扩展条件不变。修复后原断言直接通过，
没有修改旧测试期望或扩大数学证明范围。

验证结果（命令前缀均为 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run`）：

- 新增静态/数学：首批 26 passed（15.57 秒），补充内循环等价与不同块形状
  2 passed（5.30 秒）；28 项静态全部通过。
- 相关 CPU：新文件、grid-state、pipeline、upstream-state、tensor-memory 五文件
  50 passed（110.89 秒），包含新增 10 项；新增 loop CPU 单测 1 passed
  （4.95 秒）。共 11 项新增、40 项既有 CPU 用例通过。
- 八文件既有静态/代数回归：`test_pallas_grid_state.py`、`test_pallas_local_entry.py`、
  `test_pallas_local_scope.py`、`test_pallas_pipeline.py`、`test_pallas_tensor_memory.py`、
  `test_pallas_upstream_state.py`、`test_pallas_algebra.py`、`test_pallas_semantic_slice.py`，
  使用 `pytest <上述文件> -m 'not pallas_communication' -q --tb=short`：
  首批 295 passed、1 failed、51 deselected（484.19 秒），唯一失败为上述规则标签。
- 最终复测 `pytest tests/test_pallas_multi_output.py tests/test_pallas_algebra.py
  tests/test_pallas_local_entry.py tests/test_pallas_semantic_slice.py
  -m 'not pallas_communication' -q --tb=short`：
  **144 passed、15 deselected（53.84 秒）**，包含最终新增的全部 28 项静态用例
  和受入口调整影响的既有断言。
- Ruff、269 文件格式检查、basedpyright（0 error/0 warning）、tracing 示例、
  文档本地链接与 `git diff --check` 通过。

按上述批次去重，本轮验证 **375 项**：39 项新增、336 项既有；其中 324 项
静态/数学、51 项官方 CPU 用例。没有把重叠批次重复相加，也不是全仓库重跑。
本子阶段无未解决的测试失败。一般动态地址/布局、不定长循环、仅信号量
跨窗口组合、远端流水和真实 TPU 仍在后续范围；输入输出共享 HBM 继续排除。

**后续覆盖边界核实（2026-10-20）。** 同一 grid 轴内部的分组窗口仍未接入。
具体为 `grid=(4,)`，输入和 y 的全局形状 8×512，汇总 s 为 8×256，
三个局部块均为 8×128；输入/y 使用 `index_map(j)=(0,j)`，
s 使用 `index_map(j)=(0,j//2)`。kernel 在偶数 j 将 s 清零，每步写
`y=2*x` 并执行 `s+=x`，因此 s 的输出块依次为 `0,0,1,1`。
官方 TPU CPU interpreter 在 race、逻辑 bounds、NaN 未初始化填充和
on_wait 设置下检查完整输出正确，且未报告竞态；当前规范化仍拒绝
`output block map lacks a proven axis bijection`。现有规则通过完全缺失的
grid 轴识别串行后缀，尚未分解 `j//2` 中的拥有者坐标与组内步骤。
这是已确认的静态地址/窗口覆盖缺口，应先在 core 补证明，不交给 CAS
推测；它不属于一般数据依赖索引，也不是已完成的整轴窗口回归失败。
