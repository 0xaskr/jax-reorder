# 历史：9.24 本地 ANY 恢复与多维远端原存储（2026-10-21，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.24 本地 ANY 恢复与多维远端原存储（2026-10-21，完成）

用户确认两项都应在语义层继续补齐。`RevisitMode.ANY` 与内存空间 ANY
分别建模；只依据锁定源实现已定义的恢复契约准入。以 A→B→A 为例，
第三次访问 A 读取的是 A 上次写回后的版本；暂存块切换不能丢弃该依赖。
首访仍无旧输出可读，部分写入仅定义实际覆盖区域，多输出分别维护生命周期。
串行 grid 展开给出执行顺序，parallel 维度不因此获得额外顺序或同步。

多维远端原存储在 core 维持 shape/坐标与传输足迹的一致映射。线性协议
地址只是同一真实存储的表示，不能把有洞矩形变成包围区间，不能改变 lane
次序或一次源 DMA 的完成计数。二维、三维及 reshape/transpose/index
组合须由源 trace、独立值参照、错坐标/缺同步反例和证据重放共同限定。

实现分别在同一工作区基线的隔离副本推进；交叉审查重点是隐式恢复与显式
DMA 的版本传播、不同输出窗口，以及多维原存储的初始化/冲突检查。
CPU 数值验证、图证明与离线 TPU AOT 分别记录，不要求实机，也不把后端
拒绝计入原生编译成功。默认/IMMEDIATE 重访与双缓冲 ANY 等源契约保持原边界。

**本地恢复实现。** grid 展开按完整 index_map 和并行 owner 识别同一窗口，
重复访问不重复计算覆盖面积。每个输出分别保存最后已写回版本，A→A 继续
使用暂存内容，A→B→A 通过正常 storage write 恢复 A 并生成新版本；最终输出
取各窗口最新值。首访没有保存值，部分未初始化和未完成 DMA 继续拒绝。
ANY 不为不同 parallel owner 补顺序，重复 parallel writer 和 owner 交换
均有固定拒绝用例。本地入口将需要恢复的调用接入 tensor 执行器。

恢复区域是 HBM 逻辑输出与实际暂存 Ref 的交集：padding 不写回，也不能
在切换后恢复。源码在 B 暂存 padding 写入 7 后，B→B 可以继续读取并得到 8；
B→A→B 再从该 padding 读取则未初始化。两者均能原生 AOT，但语义准入
结果不同。静态串行尾块、部分更新及固定 accumulator 与切换输出并存已补齐；
符号 parallel owner 同时跨完整块和尾块、无法固定有效 Box 时仍明确拒绝。

**多维远端实现。** 原 Ref 保留二维/三维 shape，协议使用行优先线性地址，
`remote_coordinates` 在真实 shape 与逻辑视图之间提供双向坐标映射。
矩形窗口、reshape/transpose/index 的实际集合展开为不交叠 spans；连续尾维
先合并，整张巨大连续 tensor 只需一个区间，有洞列块按行段计费，不枚举每个
lane。普通 Access 与 DMA 使用同样的精确足迹；一次源 DMA 仍只有原来的
完成事件和 byte count。staging 逐段保存 local↔HBM 映射，尾块只发布有效区。
单 mesh 轴可以分片 tensor 的任意一个轴，全局输出按真实多维坐标拼接；
输入只读、冲突、初始化、源/目标版本与全部必要等待仍独立校验。

**真实组合与审查。** [组合源码](../../../../tests/pallas_restore_nd_cases.py)及
[回归](../../../../tests/test_pallas_restore_nd_integration.py)在两分支合并后
**12 passed，28.88 秒**。包含 `(8,768)` 输入、`(8,384)` 尾块输出加固定
`(8,128)` accumulator 的 A/B/A 更新；`P(None,"d")` 列分片、二维原 Ref
每行 128 列的远端 DMA、ANY 部分恢复、错列与漏 peer 同步；等待换序证据
重放及伪造恢复标记拒绝。普通本地与 self-target 远端路径逐设备对照同一
完整 NumPy 输出，不宣称新增本地 shard_map 全局组装规则。上述本地尾块
正例及错值源码的 generic CPU interpreter 完整输出也与 NumPy 一致。

独立审查通过 24 条窗口历史的 196,608 个输出坐标、14 项本地边界、
1,000 条视图变换链的 5,809 个 lane、7,688 项协议集合冲突检查，以及
120 组多维 staging 的 816 个窗口和 4,898 对坐标。审查发现并修复了
普通 Ref 变换的冗余完整 index 误拒绝，以及无重访 ANY 的窗口维度错误。
这些是独立有限验证，不计入 pytest 条数，也不替代比较证明。

新增混合 DMA 反例还确认：local DMA 与远端接收共用计数器，远端到达的额度
可能提前满足 local wait，无法证明本地源已就绪，必须拒绝。测试正例使用
独立 local semaphore；原错误源码保留为拒绝用例，没有放宽协议证明。

验证日志、独立探针、冻结补丁和分支报告保存在
`/tmp/jax-reorder-local-any-nd-f2q10uj5/`。完整 Pallas 分批集成回归已通过；
本轮不修改代数规则、默认 60 秒时限或依赖，不连接实机，不提交/推送。

**编译、预算与保留边界。** 本轮离线 TPU AOT 共 23 份源码完成 IR 校验并生成
Executable：本地分支 6、远端分支 10、主线程组合 7；另 1 份
`TransposeTransform(permutation=(1,0,2))` 被锁定 native 后端明确拒绝。
AOT 集合包含能编译但语义层因未初始化 padding 拒绝的反例，不能将编译成功
写成内存正确性证明。环境仍为 JAX/jaxlib 0.11.2、libtpu 0.0.48.1，离线
`v5:2x2x1` megacore；没有执行 TPU kernel。远端官方 CPU interpreter
14 个 pytest 用例及 4 个独立 eager/on_wait 执行均检查完整输出和 race；
后四项分别是 local DMA→remote 与真实二维尾块，不计入 pytest 总数。

公共比较在 `algebra_budget=100000` 下，对二维列 tile 等待重排返回
EQUIVALENT，对完整二维交换的错误 self-target 返回 INEQUIVALENT。
两侧 16×256 有洞列 tile 的 core 足迹/版本/冲突工作可能耗尽默认 10000
预算，需要调用方显式增加分析预算；该预算不足发生在 SymPy 之前。
另一个 16 行多分片错误源列用例已有完整投影和 NumPy 不同输出证据，但
下游复杂 Piecewise 数学比较仍超过 60 秒而未决。这是本轮新增查询的
实测限制，没有将超时报为不等价或提高时限。

语义层仍限已审查的同步 ANY、正静态原 Ref、有限一维串行远端 grid、
单具体 mesh 轴和可解析路由；DMA 每段保持 128 元素对齐要求。
masked 普通远端访问、未知数据路由/终止、任意物理布局及已约定排除项
未扩大支持。本阶段保留的本地符号并行 owner 跨完整块/尾块恢复拒绝，
随后由 §9.25 补齐。

**最终集成验证。** 两条分支先合并，再冻结生产源码完成下表；全批完成后
仅新增两个真实远端尾块测试，并单独运行，没有在回归期间更改生产代码。

| 批次 | 最终结果 | 原始记录（上述临时目录） |
| --- | --- | --- |
| 完整 Pallas 静态 | **1,345 passed，671.18 秒** | `integrated-pallas-static.log`、逐项 JSONL |
| 完整 Pallas CPU 通信/解释器 | **224 passed，332.49 秒**；含原有预期拒绝测试 | `integrated-pallas-cpu.log`、逐项 JSONL |
| 最后新增远端尾块 | **2 passed，2.84 秒**；完整 6,144 个输出值与错误 padding 读取拒绝 | `integrated-padding-final.log` |
| 交叉用例 | **12 passed，28.88 秒**，也已包含于完整静态批次 | `integrated-cross.log` |
| 共享结构与包分层 | **368 passed，46.75 秒** | `integrated-core.log` |
| 质量门 | Ruff、328 文件格式检查、basedpyright（0 errors/0 warnings）、tracing、adapter/lax inventory、文档链接与差异检查通过 | `integrated-*.log` |

最终 collection **1,571 个 Pallas ID**，由 1,345 静态、224 CPU 和最后 2 项
覆盖，无遗漏；其他交叉/分支批次存在重叠，不能重复相加。原生编译与独立
探针数量按前文单列，不算 pytest 条数。本轮相对工作区快照的补丁及最终
SHA 清单为 `integrated-current.patch`、`integrated-manifest-final.json`；
11 个生产文件与独立审查版本一致，受保护文档、配置、依赖和此前用户改动
保持不变。本子阶段完成，剩余明确边界继续由整体 active 计划和技术债跟踪。
