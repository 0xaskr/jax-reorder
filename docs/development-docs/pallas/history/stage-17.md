# 历史：9.17 跨输出块状态复用（2026-10-20，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.17 跨输出块状态复用（2026-10-20，完成）

范围：输入与输出存储独立，关注切换到其他输出块后继续使用先前状态的合法程序。
首先核对 [TPU BlockSpec 执行契约](https://docs.jax.dev/en/latest/pallas/tpu/details.html)：
自动 staging 对同一输出块的多次写入要求连续；把 K 维移到输出块维度之前，不能
自动推导旧输出窗口会从 HBM 恢复。原先“非后缀窗口复用”的待办需据此区分合法
显式状态与不满足源契约的调度，不能仅删除串行后缀限制。

本阶段先用锁定 JAX 源码中显式 HBM/DMA 和持久 scratch 案例，保留原始函数作为
基线；再构造跨块调度、状态槽位、初始化及等待的 AI 变体。生产规则应区分输入
staging、输出 staging、scratch 和显式 ANY Ref 的生命周期，按实际读写版本证明。
CPU 数值/竞态检查与语义规范化结论分开报告，未知动态地址、共享 HBM alias 和
真实 TPU 执行不包含在本阶段完成声明中。

已复现：锁定 JAX `test_hoisted_scratch_space` 原函数通过；仅把最后一次输出改成
每次输出、输出映射改为 `(i, 0)` 后，当前分析把 program 轴保持为符号，错误地
无法证明第一步 scratch 初始化。这与自动输出块重复访问不同：每块输出只写一次，
跨 program 复用的是独立 scratch。

实现设计：有 call-lifetime 数组 scratch 时，把 parallel 前缀之后的 arbitrary
后缀按预算展开，包含改变输出块的轴；parallel 前缀继续保持符号。原有输出 ownership
和连续写入检查不变。每个输出窗口结束时检查完整初始化并封存公式；切换窗口先
检查未完成 DMA 对 staging Ref 的读写冲突，再使旧输出 staging 内容失效，scratch
和 semaphore 状态继续按实际顺序演化。输出公式按所属矩形窗口选择；窗口内坐标由
全局坐标减去窗口起点得到，完整覆盖来自已有 ownership 证明。禁止依赖 parallel
顺序、arbitrary/parallel 交错的状态生命周期、
预算外展开和共享 HBM alias。显式 ANY 输出仍按真实 DMA 读写版本复用，不经过
自动 staging 的旧值恢复。反例包含缺失初始化、重置状态、错误槽位、旧输出读取、
跨窗口未等待 DMA；数值验证使用官方 TPU CPU interpreter 的 race/bounds 检查。

一维数组 scratch 先尝试原来的独立 program 证明；只有它在覆盖、窗口对齐或初始化
处无法证明独立性时，才尝试有预算的串行状态证明，其余拒绝原因原样保留。大规模
独立 grid 仍保持符号分析，`2**24` 元素的既有预算 256 回归通过。原来统一拒绝的
串行 scratch carry 现在验证为正例；未完成 DMA、残留计数和 parallel 携带状态仍拒绝。
整数坐标反例分派根据实际投影域判断，避免把一维 tensor 的 `output:0` 当作旧路径
的 `global`；未实现的数学判定仍显式未决。
JAX `%` 的 helper 会比较两次整数比较所得的布尔符号；有限重放期间两者
均已确定时，`eq/ne` 直接折叠为布尔值，保留动态槽位拒绝，没有放宽一般整数算术。
独立 NumPy 投影解释器同时修复多元 `and/or` 的求值，覆盖一维与三维矩形谓词。

来源固定为 JAX 0.11.2 commit `32544801e26115ac1794926d027148abf3baf009` 的
[`tests/pallas/tpu_pallas_test.py`](https://github.com/jax-ml/jax/blob/32544801e26115ac1794926d027148abf3baf009/tests/pallas/tpu_pallas_test.py)：
完整文件保存在 [fixture](../../../../tests/fixtures/pallas_kernels/jax_tpu_state_test.py.txt)，
重新下载后逐字节相同，SHA-256/Git blob 见
[manifest](../../../../tests/fixtures/pallas_kernels/jax_grid_state_manifest.json)。原函数通过 AST
抽取且不改函数体；AI 变体单独生成新函数，不修改安装的 JAX 模块。

| 案例 | 结果与证据 |
| --- | --- |
| 上游 `test_hoisted_scratch_space` | 3 次更新同一 scratch，最后输出；原函数资源投影、重放与 CPU 执行通过。 |
| AI：每次写入新的输出块 | 输出依次为 `x+1`、`(x+1)+1`、`((x+1)+1)+1`；从原先未初始化拒绝修复为完整输出证明，公开代数入口与纯 JAX concat 参考式一致。 |
| AI：A、B、A、B 两个 scratch 槽位 | 输出历史为 `A+1, B+1, (A+1)+1, (B+1)+1`。每轮重置状态返回 inequivalent；漏初始化、把两次初始化写入槽 0 均拒绝。 |
| 上游 `test_hbm_hbm_grid_dma` | 2 个 program 分别复制两个完整 HBM 区域；原函数 2 次传输、资源证明和 CPU 通过。 |
| AI：HBM 输出 A、B 初始化后，再更新 A、B | 通过显式 DMA 读回输出并乘 2；6 次实际传输，输出不与输入共享 HBM。A/B 更新换序得到相同规范化输出并可重放，公开代数入口与 `stack([2*x[0], 2*x[1]])` 一致。 |
| AI：DMA 更新写错目标 | 资源有效但完整输出不同；公共代数返回 inequivalent，两个随机种子的实际 CPU 输出也不同。漏输入 wait 则在读缓冲区时明确拒绝。 |
| 自建边界：一维 Ref、parallel 前缀 + 两个串行轴、反向输出映射、带归约后缀 | 核对每个窗口首尾坐标、每个独立 prefix 的状态；仅在归约后缀完成后提交输出窗口。 |
| 自建边界：窗口切换时有 pending DMA | 仅 scratch→scratch 在途传输可跨窗口，并在后续 program 等待；占用输出 staging 的源/目标 DMA 在切块处拒绝。新窗口直接读取旧输出也拒绝。 |
| 自建边界：不连续自动输出复访、parallel 携带状态、串并行交错、预算不足 | 分别拒绝源执行契约、无法证明初始化、未知生命周期、预算超限；未用数值抽样替代这些证明。 |

已知代数边界：同一个 DMA 正例直接比较纯 JAX `x * 2` 时，SymPy 不能把两块固定
输入读取合并为 `input[output:0]`，返回 `MathematicalUndecidedError`；专门回归保留
该失败关闭行为。这是完整值公式的分段恒等式问题，资源/地址证明已经完成，不是
把计算不同或证明未知当成程序不等价。上表的 `stack` 参考式仍是完整输出证明。

CPU 项均在隔离进程内运行官方 TPU interpreter，开启 race、NaN 未初始化填充与
logical bounds 检查，使用两个种子检查所有输出元素；覆盖原始两个函数、四个合法
变体、两个值错误变体，共 8 项。它们不能替代资源证据，也不代表真实 TPU 验证。

本阶段新增 **38 项**，另将既有串行 scratch carry 的拒绝测试更新为合法状态复用
正例；parallel 与未完成资源的拒绝不变。最终 Pallas collection 为 **703 项**。
验证命令均使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run`：

| 验证范围 | 结果 |
| --- | --- |
| Pallas 静态/数学全批 `pytest tests/test_pallas*.py -m 'not pallas_communication' -q` | 603 passed、98 deselected，854.01 秒；启动时新文件为 36 项，后来补充的一维 2 项已包含在下行最终 30 项复测中。 |
| 新文件 `pytest tests/test_pallas_grid_state.py -m 'not pallas_communication' -q --tb=short` | 最终 30 passed、8 deselected，65.91 秒；CPU 8 项另批通过。 |
| 官方 CPU 全批 `pytest tests/test_pallas*.py -m pallas_communication -q` | 98 passed、603 deselected，399.51 秒；已含新文件 8 项。 |
| 最终一维入口/局部状态/DMA/semantic-slice 四文件回归 | 147 passed、17 deselected，44.32 秒；另 13 项定向检查核对大规模独立 grid、串行 carry 和一维 Ref。 |
| 共享适配/区域与包分层七文件回归 | `test_algebra_array_conversion.py`、`test_package_layout.py`、`test_comparison_errors.py`、`test_adapter_guards.py`、`test_semantics.py`、`test_region_bindings.py`、`test_algebra_regions.py`：232 passed，326.42 秒。 |
| 静态检查与工具 | Ruff、267 文件格式检查、basedpyright（0 error/0 warning）、tracing 示例、JAX/jaxlib 0.11.2 adapter inventory、文档链接及 `git diff --check` 通过。 |

实施期间的一维直接分派曾破坏已有独立 program 的坐标表达及大 grid 预算，复测
发现 3 个失败：坐标 oracle、旧的统一拒绝断言、大规模预算。现已改为保留独立证明
再尝试受限串行证明；上述 147 项最终回归覆盖这三处，未通过改 oracle 或扩大预算
掩盖退化。各批次计数有重叠，不能直接相加。按最终 collection 的 605 项静态/数学
与 98 项 CPU marker 去重，当前 703 项 Pallas 测试已分批验证通过，无未解决的测试
失败；已知数学未决例是显式边界断言，不是等价结论。该子阶段完成。

本阶段结束时仍不支持多输出各自不同的窗口生命周期，后续 §9.18 已补齐静态规则映射。
其他边界包括一般动态数据地址/布局和不定长循环、
仅信号量的跨窗口组合、远端流水协议；输入输出共享 HBM 继续按用户决定排除。
自动 staging 的非连续同块写入属于源执行契约限制。该子阶段不会据此宣称支持任意
合法 Pallas，整体语义层计划继续 active。
