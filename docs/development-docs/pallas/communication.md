# Pallas 通信与执行绑定

从[任务入口](../pallas-jax-equivalence.md)到此；协议、可见性与进展由 core 证明，SymPy 不补这些语义。

## 两种独立问题

完整结构比较保留 DMA start/wait、signal/wait/read、设备/Ref/semaphore 绑定、计数和效果顺序。
它不模拟耗时，不要求 start/wait 相邻，也不以最终计数相等合并操作。
`signal(1); signal(1); wait(2)` 与 `signal(2); wait(2)` 的原始结构不同。

请求语义/协议检查后，才额外证明路由、足迹、计数生产消费、必要因果、资源冲突、初始化和版本。
两个相同的错误图不能因此报告通信正确；缺对端上下文或范围时明确拒绝所请求的检查。
设备局部结论只携带相关环境/接口条件，不升级为一般全分布式协议证明。

## DMA、计数与存储

- start 保留传输源版本、目标真实存储、足迹与原始完成计数；实际 wait 才消费相应完成证据。
- 传输拆分/合并、部分等待、跨代复用与普通信号量的受限规则保留必要依赖，不靠某次任选调度判等。
- 正在读取的 DMA 源、尚未完成的目标不能被未同步访问；普通 load/store 使用与值投影一致的坐标。
  数据 mask 的协议足迹允许保守扩大，值执行必须验证实际足迹被其覆盖。
- read 只能在证明稳定时作为控制值；不能把在途计数按某次执行结果固定。
- 跨 program start/wait 继续传递在途状态，不按每个 program 隐含清空 DMA。
- 一个 semaphore 的 Wait/Read 当前仍要求唯一有序消费流；跨流消费缺证明时拒绝。

源码：[protocol.py](../../../src/jax_reorder/core/protocol.py)、
[remote.py](../../../src/jax_reorder/core/remote.py)、
[remote_memory.py](../../../src/jax_reorder/core/remote_memory.py)。

## 多维 Grid 与并行条件

多维串行 grid 按源坐标顺序执行；parallel owner 保持独立流。
协议节点同时保留 device 和 stream；边只来自同流源顺序与真实同步关系。
共享 scratch/staging 不按 owner 私有复制，重叠远端写或缺 wait 不因展平而安全。

`CoreExecutionBinding.programs[c]` 是调用者保证的 core c 完整 program 坐标序列。
它是外部前提，**不是编译设置**；普通 `pallas_call` 不自动满足，也不会自动推断 core。
绑定须覆盖全部 grid 点且无重复，符合坐标范围、源字典序和 owner 分配约束。

远端 parallel 当前要求一个 core 的完整点流，并明确所有设备采用相同序列。
仅 `execution_cores=1` 不能排除设备 A/B 选择相反 owner 次序造成的等待死锁。

| 证明 | 保留什么 | 不能用于什么 |
| --- | --- | --- |
| `protocol` | 并行 owner 独立、必要因果、共享资源冲突及存储版本 | 不能从任选拓扑排序推出所有执行都有该顺序 |
| `ordered_protocol` | 按绑定的实际完整点流，证明单 core 能够推进 | 额外顺序不能放过第一份证明中的冲突 |

`normalize_view(..., execution=binding)` 保存单侧条件；
`compare_program_views(..., execution=(left_binding, right_binding))` 可为二维/展平图分别绑定。
公共 `compare_pallas_kernels` 及证据重放同样保留条件；遗漏/伪造/错序绑定都必须拒绝。
源码：[core_execution.py](../../../src/jax_reorder/core/core_execution.py)。

## 当前限制

当前并行成功案例使用实际分组的 scratch/信号量及完整 HBM Ref。
共享自动 staging 若在独立 owner 间冲突仍拒绝，不能新增 barrier 或复制缓冲来绕过。
自动多 core/未知设备间点流、任意数据路由与终止、一般多轴全局组装没有通用证明。
远端向量索引仅覆盖已审核形态，数据 mask 条件初始化仍保守；完整列表见[验证与边界](validation.md)。

## 按需证据

- 必要依赖、普通信号量、跨代/部分等待及独立计数状态机：[阶段 11](history/stage-11.md)。
- 局部逻辑轴/设备路由与等待错误：[阶段 12](history/stage-12.md)、[阶段 13](history/stage-13.md)。
- 多维原 Ref、非连续 footprint 与源后端限制：[阶段 24](history/stage-24.md)。
- Mask 足迹与在途 DMA 冲突：[阶段 26](history/stage-26.md)。
- 多维/三维全部串并组合、双协议证明及单 core 死锁反例：[阶段 27](history/stage-27.md)。
