# JAX / Pallas 语义层：当前计划

创建：2026-10-18。状态：**实施中**；最近功能阶段 2026-10-21 已完成远端多维 grid。
默认只读本页和[开发入口](../../development-docs/pallas-jax-equivalence.md)，按任务选择一个专题。
已完成工作与原始命令移至[历史快照](../completed/context-history/pallas-semantic-layer-2026-10-21.md)，
快照中的旧缺口不代表当前状态。

## 目标与完成条件

比较分布式程序中单个 device 的一个 Pallas kernel，mesh 可以存在；保留必要设备/轴/实参与通信依赖。
完整结构比较、可选资源/协议证明与纯值数学判定分开。每次扩展须有正例、反例、拒绝路径和源码证据。
grid/地址/初始化/同步留在 core；SymPy 消费已证明合法的值投影，默认 60 秒。实机不阻塞图/AOT 验证。

## 当前待办

| 优先级 | 工作项 | 完成条件 / 约束 |
| --- | --- | --- |
| 下一覆盖候选 | 远端条件初始化 | 同数据 mask 写后读的安全模式可证明；漏初始化/错误 predicate/在途 DMA 仍拒绝 |
| 后续覆盖 | 并行共享自动 staging | 按真实存储和必要同步证明复用；不私有复制 Ref 或插入 barrier 绕过冲突 |
| 后续覆盖 | 一般索引、数据路由/终止、非默认布局 | 按源事实逐类增加确定规则与拒绝反例；不以采样或 CAS 替代地址/通信证明 |
| 持续 | 大图与 CAS 预算 | 只优化已确认瓶颈，保留预算收费及未决结论；一般复杂查询不承诺 60 秒内完成 |
| 独立后续 | JAX 全消费者统一视图、多轴全局协议/组装 | 不作为设备局部第一版的额外准入条件；保留各自契约 |

具体已知边界与源后端限制见[验证专题](../../development-docs/pallas/validation.md)。
`core_map`、`mpmd_map` 与输入输出共享 HBM 按用户决定暂不支持；MLIR/ISL/Z3 不作为当前依赖。

## 已确定的执行条件

- 不从 program_id 推测 core；`CoreExecutionBinding` 是调用者提供并保证的条件，不配置实际运行。
- 远端并行要求完整单 core 点流及各设备同序，独立流冲突与实际序列进展分别证明。
- 保留 scratch/staging 的真实共享身份、跨 program 在途 DMA 和各输出生命周期。
- 结构匹配不能替代通信正确性；未证明条件明确拒绝，不能据此报告程序不等价。

算法入口按[契约/API](../../development-docs/pallas/contracts.md)、[grid/内存](../../development-docs/pallas/grid-memory.md)、
[通信/绑定](../../development-docs/pallas/communication.md)路由，避免读全阶段历史。

## 最近完成与证据

| 子阶段 | 结果 | 原始记录 |
| --- | --- | --- |
| 远端多维 grid 与并行流 | 分批去重 1,183 项通过、6 份离线 AOT；CPU 的 2 项解释器限制不计数值成功 | [阶段 27](../../development-docs/pallas/history/stage-27.md) |
| 普通尾块与 masked 访存 | 分批去重 2,167 项；最终 423 受影响 + 368 core；后端 masked-load 拒绝独立保留 | [阶段 26](../../development-docs/pallas/history/stage-26.md) |
| 并行 owner ANY 尾块 | 1,975 不同测试 ID、9 AOT，独立有限域/完整输出 oracle | [阶段 25](../../development-docs/pallas/history/stage-25.md) |

不将跨阶段结果相加。新的功能阶段单独保存准确命令、源码状态、失败及修复证据，完成后只在本页保留一行入口。
