# 历史：9.27 远端多维 grid 与并行事件流（2026-10-21，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.27 远端多维 grid 与并行事件流（2026-10-21，完成）

源 grid 的各轴坐标同时绑定 program_id、num_programs 与 index_map；有限
串行轴按源网格次序展开，parallel owner 保持独立命令流。协议的节点保留
device 和 stream，只有同流命令顺序及实际 DMA/信号量依赖生成边，不把
任选的拓扑排序当作所有运行都具备的先后关系。共享 Ref 不按 owner 克隆；
计数器仍要求唯一有序消费流，跨流消费缺少证明时明确拒绝。

I/O 窗口、ANY 恢复、实际 scratch、DMA 版本和最终输出组装共同使用同一
grid 计划。纯串行多维与一维混合进制改写得出相同完整值；并行写相同
远端窗口、缺 wait、错误索引或路由不能因展开而被接受。共享 staging 和
scratch 不按 owner 私有化。跨 program 的 start 保持在途，后续 program
的真实 wait 才消费完成计数，不插入每 program barrier 或强制清空 DMA。

**执行条件与两份协议证明。** 单 core 并不自动意味着所有 device 采用相同
并行网格次序：设备 A 先执行 owner 0、设备 B 先执行 owner 1，均等待对端
对应 owner 时，单 core 无法切到另一个 owner 解锁。因此并行远端路径要求
调用者提供已有的 `CoreExecutionBinding`，仅接受一个 core 的完整、无重复、
合法网格点序列，并明确约定所有设备执行该同一序列。只传
`execution_cores=1` 不足以证明并行通信。该参数是附条件证明的外部前提，
不会配置编译器、自动推断 core，亦不会让普通 pallas_call 自动满足它。

第一份 `protocol` 保持并行 owner 独立，用于必要因果、共享资源冲突和
存储版本投影；第二份 `ordered_protocol` 按显式绑定的完整调用顺序证明
单 core 实际能够推进，其额外顺序不能用于放过第一份证明中的冲突。真实
三维正例覆盖全部八种串并轴组合。独立反例 `grid=(2,2)`、语义
`(arbitrary,parallel)` 中，设备 A 的 `(0,0)` 等待 B 的 `(1,0)`，B 的
`(0,1)` 等待 A 的 `(0,1)`；即使每个计数器生产/消费都属于同一 owner，
多流模型仍可错误见证进展，实际单 core 顺序则死锁。第二份证明准确拒绝。
另有 owner 0 先 signal、owner 1 后 wait 的安全正例，避免一概拒绝跨 owner。

`normalize_view(..., execution=binding)` 保留单源条件；
`compare_program_views(..., execution=(left_binding, right_binding))` 允许二维
与一维展平分别绑定不同坐标流；公共 `compare_pallas_kernels` 的现有 execution
参数也接通远端检查。条件及实际顺序证明随证据重放，遗漏、错序、缺点、
重复、错 grid、布尔伪造及删除条件均有反例。输出中的分段坐标表达式复用
已有精确域分区比较，解决不同 scratch bank 形状导致的模表达式差异；
不交给 SymPy，不把数值采样当作等价证据。

**验证范围。** 新增真实源码包括二维/三维、全串行、全并行、交错轴、
展平改写、真实 banked scratch/信号量、跨迭代 DMA，以及二维自动输出
staging 与同步 ANY 恢复。错误 index_map/peer 用独立 NumPy 输出区分；
错误等待、共享 scratch、重叠远端写、初始化、预算和证据篡改明确拒绝。
新用例静态 **99 项通过**；独立协议多流 **30 项通过**。官方 CPU worker
**20 项通过**，其中 **18 项完整数值及 race 检查**，另 2 项只确认官方
interpreter 对交错串并轴的 prefix 限制，不能计作数值成功。

六份原始源码在离线 `v5:2x2x1` megacore 环境完成 trace、lower、IR verify
并生成 `Executable`：二维串行、flat、串并混合、交错轴、全并行、三维混合。
交错轴的 native AOT 成功与 CPU interpreter 限制分别记录；无 TPU 实机执行。
脚本为 [pallas_remote_grid_aot.py](../../../../tests/pallas_aot.py)，
原始结果 `/tmp/jax-reorder-remote-grid-aot.json`；主验证工件目录为
`/tmp/jax-reorder-remote-grid-ekhm4ft4/`。

**保留边界。** 自动多 core 或未知设备间 program 次序仍缺绑定；实际共享
自动 staging 若在独立 owner 间冲突，仍拒绝，不能用私有复制或新增 barrier
绕过。当前并行正例使用实际分组 scratch/信号量及完整 HBM Ref。一个 semaphore
的 Wait/Read 仍要求唯一有序消费流。多 mesh 轴全局组装、未知数据路由/终止、
远端条件初始化、非默认物理布局等旧边界不在本轮扩展范围；core_map/mpmd_map
及输入输出共享 HBM 按用户决定继续不支持。未修改 SymPy、60 秒时限、依赖
或受保护文档，也不提交/推送。

**最终回归。** 分批去重 **1,183 项通过**：

| 批次 | 结果 | 原始记录 |
| --- | --- | --- |
| 新 grid、绑定、独立审查、二维 staging/ANY 静态用例 | 99 passed，66.89 秒 | `final-new.log` |
| 既有远端、协议、本地 grid/core、DMA 组合、mask 集成（含新增协议多流 30 项） | 507 passed，308.83 秒 | `final-regression.log` |
| 共享 core、函数契约、区域绑定及代数直通 | 557 passed，214.04 秒 | `core-regression.log` |
| 官方 CPU worker | 20 passed；18 项数值/race，2 项 interpreter 限制路径 | worker 测试及 `verification.json` |

表内四组测试文件/标记不重叠；此前 215 项初步远端回归、最终错误证据分类
修正后的 47 项重放复测均通过，不重复计数。不声称这是单次完整 pytest。
首轮新用例暴露静态整数 floor-division helper 的 sign 缺口，以及 DMA Ref
降维后空 sharding partitions 的 rank 差异，均在保留类型/分片元数据的前提下
修复；全并行与展平形式的分段坐标差异由精确域分区解决，没有放宽比较语义。

Ruff、355 文件格式检查、basedpyright（0 错误/警告）、tracing、adapter/lax/
Pallas inventory、Markdown 链接及 diff 检查全部通过。最终 7 份生产文件、
8 份测试/fixture 文件和 5 份允许修改的文档已写入工件目录的最终哈希清单；
实际源码、验证日志及 AOT 结果可本地复查。此子阶段完成，整体语义层计划
继续跟踪上文明确保留的范围。
