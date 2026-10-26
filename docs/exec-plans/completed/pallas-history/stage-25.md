# 历史：9.25 并行 owner 的 ANY 尾块恢复（2026-10-21，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.25 并行 owner 的 ANY 尾块恢复（2026-10-21，完成）

具体拒绝来自恢复函数要求固定有效矩形：输出 `(16,384)`、块 `(8,256)`，
grid 的 `p` 并行、`t` 串行，`index_map(p,t)=(t%2,p)`。每个 p 都执行
A→B→A，但 p=0 应恢复 8×256，p=1 只恢复 8×128；剩余 VMEM padding
没有对应的已写回 HBM 值。保留 p 为符号而只计算一个 Box 会混淆两种范围。

本轮在 core 将具有不同有效区域的并行域分开解释，分别保留串行历史，再按
原输出坐标组装。拆分仅是证明分支，不增加实际 barrier、不同 owner 的顺序
或通信；必须沿用全局唯一 writer/owner 证明。预算按拆分、重放及表达式工作
计费，不按所有 tensor 元素枚举，也不将地址/初始化问题交给 SymPy。

验证包括完整输出与独立参考值、部分更新及固定输出、多个尾轴、映射变换，
以及越过有效恢复范围读取旧 padding、首访未初始化、缺 DMA wait、跨 owner
重叠的拒绝。原生离线编译、CPU 数值检查和语义证明分别报告，生产源码在
主集成回归期间冻结；受保护文档、依赖及此前改动保持不变。

**实现。** [并行域划分](../../../../src/jax_reorder/core/tensor_owner_partition.py)
在完整 `GridPlan` 已证明 writer/owner 后收集所有 padded ANY 输出、所有串行
窗口的有效区要求。只拆分完整块与尾块混合的整数区间；各输出/窗口的分类
全部相同且相邻的矩形在重放前合并。每个矩形仍可代表大量独立 program，
范围分析、分区、合并和重放共同使用原有分析预算。

[tensor 状态解释器](../../../../src/jax_reorder/core/tensor_memory.py)复用原计划和
输入/device 绑定，分别保存各分区中各输出的串行历史。输出组装使用每个
`OutputWindow.inverse`，不能误用 expanded plan 顶层未证明的 inverse。
版本、DMA completion 和 wait 的来源携带 `owners[...]` 范围，完整证据可
由原图重新生成；这些记录表示互相独立的证明，不能将展平的记录误作跨 owner
的执行时序。恢复只写已提交的有效 HBM 区域，未提交 padding 仍未初始化。

**用例与独立审查。** [真实 Pallas 源码](../../../../tests/pallas_parallel_tail_cases.py)
及[回归](../../../../tests/test_pallas_parallel_tail.py)覆盖完整块/尾块的部分及完整
更新、反向映射、两个尾轴、mixed-radix、多输出的不同 tail 与生命周期、
local DMA、公共比较入口和证据重放。第二个固定输出可以在连续生命周期内
保留已经初始化的 VMEM padding；切换窗口后的 ANY 恢复不会保留该 padding。
重访时错误重置有完整 NumPy 反例；core 表达式不同仍只报告需数学判定，
不从结构差异推出不等价。首访缺初始化、旧 padding 泄漏、跨 owner 重叠或
交换及缺 DMA wait 均明确拒绝。

[分区回归](../../../../tests/test_tensor_owner_partition.py)验证精确覆盖、范围签名
和预算。8 个与百万 program 的完整投影分别消耗 258、338 分析步，均保存
8 条带分区范围的版本记录，无需逐 program 枚举；独立审查中十亿个
单调 owner 只产生两个区域，分区花费 150 预算。24 组不同 grid/映射/窗口
历史通过 72 次随机完整 NumPy 输出检查，共 6,336 个坐标；另 500 组小整数域
逐点检查 13,966 个点，恰好覆盖一次且分类一致。DMA 证明的四种篡改（删
transfer、抹 transfer/wait 的 owner 范围、伪造恢复矩形）均被证据重放拒绝。
外层 `shard_map` 的 device 坐标参与输入计算时，公共协议检查和重放通过，
两个 device 坐标下的全部 12,288 个输出值与独立参考一致。

9 份固定源码通过离线 TPU AOT 的 StableHLO 校验并生成 Executable，涵盖
部分/完整更新、反向 owner、异 tail 多输出、双尾轴、mixed-radix 和 local DMA。
环境仍为 JAX/jaxlib 0.11.2、libtpu 0.0.48.1，离线 `v5:2x2x1` megacore；
没有执行 TPU kernel，编译成功与 CPU 数值检查均不替代语义证明。

**保留边界。** 非单调映射可能产生大量不相邻区域，仍按预算明确拒绝，
不承诺任意合法映射均在默认预算内完成。显式 `program_point` / core observer
路径沿用原行为，不重复其外部事件或推断新的跨 program 顺序。本轮没有扩大
普通非 ANY 的 padded 访问或 masked 未初始化值语义；真实完整更新用 owner
条件与精确静态切片表达。远端 grid、未知路由/终止、物理布局和此前排除项
未变化，没有修改 SymPy、默认预算或 60 秒时限。

**最终集成验证。** 生产代码合入后冻结，再运行既有完整 Pallas 集合及分区
单测；随后合入的 30 个新场景独立运行，没有在回归期间修改生产代码。
原始记录保存于 `/tmp/jax-reorder-parallel-tail-dwz8eocj/`。

| 批次 | 最终结果 | 原始记录（上述临时目录） |
| --- | --- | --- |
| Pallas 静态与分区单测 | **1,353 passed，809.77 秒**；含 6 个新增分区测试 | `integrated-static.log`、逐项 JSONL |
| Pallas CPU 通信/解释器 | **224 passed，396.02 秒** | `integrated-cpu.log`、逐项 JSONL |
| 新增并行尾块场景 | **30 passed，36.88 秒**；含完整 NumPy/CPU 对照、公共入口与证据重放 | `integrated-new.log`、逐项 JSONL |
| 共享结构与包分层 | **368 passed，38.27 秒** | `integrated-core.log`、逐项 JSONL |
| 离线 TPU AOT | **9/9** IR 校验并生成 Executable；无实机执行 | `cases-aot.json`、`cases-aot.log` |
| 质量门 | Ruff、332 文件格式检查、basedpyright（0 errors/0 warnings）、tracing、adapter/lax inventory、文档链接与差异检查通过 | `integrated-*.log` |

最终 collection 中 **1,601 个 Pallas ID + 6 个分区 ID** 全部覆盖，加上共享
核心，合计 **1,975 个不同 pytest ID** 通过，其中 36 项新增。分支定向 152 项
及分支新场景与集成结果重叠，不重复相加；独立探针和 AOT 不计入 pytest 数。
覆盖核对保存在 `integrated-coverage.json`。最终补丁为
`integrated-current.patch`，SHA 清单为 `integrated-final-manifest.json`；两份
生产文件与独立审查版本一致，本轮仅修改两份生产文件、增加三份测试文件和
更新五份允许修改的开发/计划文档。受保护文档、依赖、配置及此前改动保持
不变，没有提交或推送。本子阶段完成，整体计划的其他边界继续单列跟踪。
