# 通信协议扩展：普通信号量、复用、部分等待与混合 DMA

创建与完成：2026-10-19。状态：已完成。用户目标：开 subagent 调研通信协议的问题，然后根据调研结果解决。

## 目标与完成条件

覆盖上轮明确列出的四个通信缺口，保持 core 证明通信、代数层只比较完整纯值图的分工：

1. 普通信号量的本地/远端 signal、wait、静态数量和目标绑定进入语义视图及协议证明；可确定的 counter read 保留观察值，无法确定时拒绝。
2. 同一 semaphore 跨次消费/复用可证明，不再限定单代单个 consumer；缓冲区复用需要原有依赖建立顺序，不能自行补同步。
3. 部分 wait 支持累计消费，不能据 descriptor 或某次执行猜测 producer 身份；达到完整计数或其他原有依赖时才发布完成事实。
4. 本地与远端 DMA 可在同一内核/协议里组合，计数及源/目标存储版本统一检查。

完成需有源适配快照、公开规范化/代数入口、证据重放、正例/反例、独立小域状态机与官方 interpreter 对照。
沿用已有有限静态单轴/每设备单程序的通信分析域；多轴、多程序和一般动态控制流属于上一轮另列的覆盖工作，不将其混入本任务。
但不能以只完成上述某一项代替四项完成。受保护文档不修改，不提交/推送。

## 设计与取舍

- 调研由 `communication_protocol_research` 子代理执行；核对锁定 JAX 0.11.2 源码及官方上游。
  设计依据写入 [Pallas 设计 §9.11](pallas-history/stage-11.md)。
- DMA semaphore 是进度计数，可逐步增加。部分计数不能解释为整笔传输完成。
- 每个 semaphore 的 wait 必须属于其设备的单一顺序流。用累计产量和累计消费阈值建模；
  启用条件单调，可用有限事件的饱和固定点证明进展，不枚举执行调度。
- 必要依赖迭代：只能排除已证明在 wait 之后才发起的 producer；耗尽剩余候选总量时，才推出
  全部候选完整完成先于 wait。普通 signal 的必要贡献最多推出 signal 发起先于 wait，不隐式完成 DMA。
- 固定点生成的合法调度只用于求值；竞争排除必须使用原有顺序及已证明的必要依赖可达性。
- 信号量单位/类型、初始值、最终归零、计数范围、生命周期和读写版本仍显式检查。
  锁定 TPU backend 不支持 `decrement=False`，必须明确拒绝；未知计数观察不替换为某次调度结果。

## 工作项

- [x] 子代理调研定稿与实现规则核对。
- [x] typed semaphore 源事实、effects、静态目标及 schema/inventory。
- [x] 累计阈值、必要依赖固定点、进展及计数观察证据。
- [x] 复用/部分等待/混合 DMA 接入版本传播与公共入口。
- [x] 独立状态机、源反例及官方 CPU 对照；回归与静态检查。
- [x] 按四项完成条件审计，记录真实边界并归档。

## 进展与验证

- 初始核对：现有协议只允许单代一次全量 wait，remote.py 明确拒绝混合本地/远端；
  typed adapter 仅接受 DMA semaphore，公共结构适配器已保留普通 signal/wait/read。
- 调研已确认：JAX 锁定提交 `32544801e26115ac1794926d027148abf3baf009`；
  TPU DMA wait lowering 不按 start 身份配对；普通 signal 不会隐式 join 已发起 DMA。
- 新增本地/远端 regular signal、wait、确定的 read，以及 mixed local/remote DMA；
  语义证据包括必要边、实际满足计数的 witness、版本和 observations，可从原源视图重放。
- 发现并修复公共结构规范化遗漏 `semaphore_read` 的资源顺序：空 effects 不代表纯函数，
  否则原先 wait 后的零读数可能移到 wait 前。新增结构反例和公开代数回归。
- `adapter_inventory` 仅增加实际追踪到的 semaphore section，其余旧快照逐项保持相同。
- CPU interpreter 的四个初始失败均定位到缺少 `semaphore_read` CPU lowering。
  保留两类明确的拒绝回归，并用不含读取的独立 signal/wait 内核执行官方正例；
  计数观察仍由独立逐单位机器、源图、公开比较与证据重放验证。
- CPU interpreter 不代替真实 TPU 或一般并发证明，尤其其超额计数
  happens-before 已有上游 TODO，必须辅以独立状态机反例。

## 验证结果与完成审计

使用锁定 `.venv` 的 Python 3.12 / JAX 0.11.2；未修改依赖、未提交或推送。

| 命令/范围 | 结果 |
| --- | --- |
| `pytest -q tests/test_pallas_protocol.py tests/test_pallas_communication_protocols.py tests/test_pallas_remote.py tests/test_pallas_remote_values.py -m 'not pallas_communication'` | 80 passed，43.95s |
| 最终协议/新入口复核 `pytest -q tests/test_pallas_protocol.py tests/test_pallas_communication_protocols.py -m 'not pallas_communication'` | 39 passed，18.73s；含新增的 48 种部分等待排列及逐单位独立状态检查 |
| structural / dma / dma_regions / semantic_slice / sharding / storage / package_layout / adapter_guards | 201 passed，19.26s；通信 marker 排除 |
| protocol / coverage / algebra | 115 passed，163.23s；protocol 后补一项的最终结果见上述 39 项复核 |
| 新官方通信案例 ack / mixed / shared / partial / signal，各 eager/on_wait | 10 passed；与初始读取缺 lowering 的失败同轮运行 |
| 普通本地/远端 signal/wait，两种模式，以及两类 read 上游拒绝固定回归 | 6 passed，14.05s，其中 2 项确认预期编译拒绝 |
| 既有本地/远端官方通信回归 | 31 passed，68.26s，包含已有的等待环预期超时检查 |
| `ruff check .` / `ruff format --check .` | 通过，244 个 Python 文件格式通过 |
| `basedpyright` | 0 errors / warnings / notes |
| `python -m tests.adapter_inventory` / `python tests/inspect_jaxpr.py` | 快照与原图追踪比较通过 |
| `python tools/check_docs.py --changed` / `git diff --check` | 归档后通过，11 份 Markdown、202 个本地链接目标 |

四项完成条件逐项对应：

1. regular 源类型、effects、scalar/tuple target、signal/wait/read 进入 typed adapter；确定读数
   经公开 CAS 与证据重放验证，不确定读数及 `decrement=False` 明确拒绝。
2. ack 建立必要因果后，同一 DMA semaphore 与 scratch 可复用；缺 ack/提前下一代拒绝。
   regular 多代消费也已覆盖，累计数量超过单次上限但存活计数安全的协议可通过。
3. 部分等待以累计阈值证明；512+1024 等不同大小贡献的反例用同比例逐单位模型检查，
   必需部分贡献不推出整笔 DMA 完成，不能据 descriptor 选择 producer。
4. 本地→远端串接、共享 DMA counter 的不交叠 local/remote 写均经过值投影、公开数学入口
   和官方解释器；提前中继及未完成 DMA 就 signal 的不安全读取被拒绝。

调研子代理另在内存脚本中审计了逐单位 DMA 与 regular signal 的排列；持久回归使用
独立的 [计数状态机](../../../tests/protocol_counter_oracle.py)，不把临时调研脚本计为 CI 测试。

## 剩余边界

四项均已在事先声明的有限静态域完成。多轴/多程序、动态控制流、read 驱动路由、变换/数组
semaphore、barrier、自定义执行和外部 alias 仍需独立规则；必要因果规则是充分条件，
未证明的安全协议也可能明确拒绝。单笔 self-target remote DMA 的 send/receive semaphore
仍要求互异。CPU interpreter 缺少 read lowering、实际 TPU/多主机未验证的边界见设计 §9.11。
这些事项记录在 [PALLAS-001](../index.md#待处理)。
