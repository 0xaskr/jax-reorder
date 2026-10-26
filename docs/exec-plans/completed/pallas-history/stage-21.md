# 历史：9.21 剩余覆盖专项与大图预算（2026-10-20，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.21 剩余覆盖专项与大图预算（2026-10-20，完成）

本轮在 §9.20 的完整实现上继续推进远端自动缓冲流水、交错 grid/core 绑定、
动态访存与部分 Ref/布局，以及真实大图预算。三条隔离分支先写具体源语义、
支持范围和反例，主线程处理结构接口搜索并集成。输入输出共享 HBM 不纳入，
grid/访存/通信证明保持在 core，原生编译继续使用离线 libtpu。

自动缓冲必须显式解释源程序真实的 staging、writeback 和窗口失效；跨 program
的在途传输不能丢失。交错 grid 与多 core 需要调度事实或可复核调用前提，
不能把程序坐标猜作物理 core。动态访存和 Ref 视图必须保留原存储地址、别名、
初始化与 mask 坐标；只有有限约束足以精确覆盖时才展开。

大图接口先用同步端口置换下保持不变的源标签和连接细化分组，再对真正无法
区分的组做完整搜索。外部端口顺序、反馈、常量、效果和嵌套语义不变；不能用
首个成功候选代替规范形式，不以提高预算或删除未使用节点掩盖阶乘搜索问题。
验证包含真实 GLM5 原图与独立副本、合法端口换序、错误反馈/元数据和剩余对称
图的预算拒绝，并复核规范化幂等与原图对应证据。

**已实现的源语义。**

- 远端自动 IO：新增 `RemoteIOPlan`，为 HBM 与 kernel staging 分配不同资源身份。
  在首 program / 输入 index_map 变化时加入真实 reload，在各输出独立窗口结束时
  capture、失效旧 staging 并提交 HBM。所有事件参与原有必要依赖、竞态检查和
  值重放；固定窗口内允许跨 program start/wait，不能在窗口变化处假造同步。
  当前覆盖静态有限 rank-one 串行 grid、单个 concrete device mesh 轴。
  输出重访、覆盖缺口、未知路由和无因果顺序的窗口交接明确拒绝。
- padding：取消 scratch/DMA 的 blanket 拒绝，改为在 DMA start 检查真实源矩形。
  未定义输入尾部不能传入计算；显式初始化或原版本已存在的 scratch 尾值可保留。
  256 block / 384 logical size 的末尾有效 128 片段可搬运；多输出分别封存自己的窗口。
  先填 7 再搬有效尾片段的正例与漏初始化、漏 wait、旧尾值参与 reduce 的反例，
  都按具体值和原存储生命周期检查。
- 交错 grid：独立 program 及连续串行后缀可直接投影；有界枚举并行 owner 时，
  用真实 scratch 写区域两两不相交、每轮 DMA 完成及自身初始化证明 bank 隔离。
  **跨 owner 间隔保持状态的 bank 规则要求调用者提供 `execution_cores=1`。**
  锁定 Python lowering 只转发调度声明，AOT 通过不能证明 owner 永远落在同一 core；
  默认路径因此继续拒绝缺失这一事实的持久状态。
- 多 core 普通信号量：公开 `CoreExecutionBinding(programs=...)` 指定每个 core 的
  有序 grid 点流，经 `compare_pallas_kernels(..., check_protocol=True, execution=...)`
  验证完整覆盖、无重复、串行顺序、真实 target 和全协议等待拓扑。
  结果和证据重放明确携带调用者前提，普通 `pallas_call` 不会因该参数而改变编译调度。
  每个 program 的内存必须自行初始化并完成 DMA，信号量计数则跨完整 core 流保留。
  新多 core 路径要求显式 `core_index`；不借用解释器的默认 core 0 解释原生硬件。
- 动态地址与控制：源 program 域、显式 clamp/min/max 及受限算术/选择表达式给出
  有限范围后，按预算分叉并执行剩余区域。相同动态标量的写后读保留同一路径绑定；
  每条分支独立检查初始化、边界、DMA、计数，未初始化的分支不能被其他分支补全。
  越界地址先明确拒绝，再计费枚举。算术/选择式有限 while 复用同一状态重放。
- 部分 Ref：新增静态 index 变换，与 reshape/transpose 组合为原 storage 坐标映射；
  预算内逐点计算精确 footprint，再合并相邻矩形，不能用包围矩形填平 stride 洞。
  读取、写入和在途 DMA 冲突都使用相同原存储身份。动态变换索引、带 mask/padding
  的变换视图及变换后的 DMA endpoint 继续拒绝；手造 IR 混用 view/base 坐标也拒绝。
- 布局：只放开已审核的 TensorCore `needs_layout_passes=False`，其他选项仍须为
  受审查默认值。完整 compiler 参数保留，默认布局与该配置仍是不同契约；
  物理 array layout、SparseCore 布局、融合及安全检查选项没有被一起放开。
- 大图：内部无根输出以类型和反馈输入消费者原语多重集细化，再在剩余组中精确搜索。
  GLM5 absorbed Attention / Decoder 的 prefill、decode 四个真实配置全部在
  `search_budget=1` 下通过完整结构比较和对应复核，约 71–93 秒/例。
  小图同时覆盖合法换序、错误反馈和真正对称组的预算拒绝；未删除原始计算。

**交叉审查。** 调度审查动态 Ref 时，对 180 条 reshape/transpose/static-index 链
做了 1116 次独立 NumPy 坐标/footprint 对照，无差异。远端审查调度时发现
banked 状态缺少稳定 core 归属依据，按上述显式单 core 前提收紧；主线程还消除
`tensor_memory` / `grid_execution` 导入环，改为传入投影回调，包分层守卫保持启用。
接口细化另经 80 个合法随机置换复核（jit/cond/scan/while、嵌套调用、有根/无根），
均通过对应证据、幂等和 JAX IR 检查；12 个错误反馈/分支变体判为不同。
最小完整预算探针与大预算结果一致，预算不足继续明确拒绝。
远端的混合 buffered/ANY 输入额外做了 512 元素独立投影对照；最终窗口只初始化
一半有效输出的反例在真实 output capture 处拒绝。这些投影证据没有计作硬件执行。

**离线 TPU 编译。** 独立环境仍为 JAX/jaxlib 0.11.2、libtpu 0.0.48.1、
`JAX_PLATFORMS=cpu`，compile-only `v5:2x2x1` / megacore。没有执行 TPU kernel。

| 源码组 | 原生后端结果 | 与图证明的区别 |
| --- | --- | --- |
| 自动远端窗口、多输出、有效尾块、搬错块、本地尾片段/旧尾值/reduce | 7/7 生成 Executable | 搬错块能编译，仍须由图/值反例识别 |
| 交错独立/连续后缀/banked/DMA、两个显式 core 路由候选、真实 TensorCoreMesh 源绑定参考 | 8/8 生成 Executable | 编译不建立调用者 core 绑定；TensorCoreMesh 参考没有被算作自动导入绑定的证明 |
| 有限动态写/切片/while、路径初始化、DMA、部分 reshape 读写、布局开关 | 11/11 生成 Executable | 只对应这些受审查具体源码 |
| `fuse_transposed_lhs_in_matmul=True` 审计探针 | 1 份生成 Executable | 未准入 core 布局例外 |
| stride Ref 链、transpose Ref 链，各读/写 | 4 份 trace 成功、TPU lowering 拒绝 | core/通用 CPU 视图语义不能替代锁定后端支持 |
| `pp.load(mask=...)` 的 3 个 padding 源例 | TPU lowering 明确拒绝 | 官方 TPU CPU interpreter 同样拒绝，静态 mask/other 证据单列 |

本轮成功编译共 27 份具体源码，未将失败改记为成功，未将审计探针全部算作本库准入。
报告与原始日志保存在 `/tmp/jax-reorder-remaining-y9dyxemv/`：
`scheduling-aot.json`、`staging-native-aot-aligned.json`、
`staging-native-aot.json`、`dynamic/dynamic-aot.json`。
调度组可独立重放：

```bash
JAX_PLATFORMS=cpu /tmp/jax-reorder-tpu-aot/bin/python \
  -m tests.pallas_scheduling_aot --output /tmp/pallas-scheduling-aot.json
```

**范围确认（2026-10-20）。** 按用户决定，`core_map` 和 `mpmd_map` 暂不支持，
其入口适配与自动 core 绑定不纳入当前阶段补齐项。已有 `pallas_call` 的显式
`CoreExecutionBinding` 与 `execution_cores=1` 条件证明保持上述支持范围；
TensorCoreMesh 源绑定参考的离线编译结果不表示已支持这两个入口。

**明确剩余范围。** 任意动态路由/终止、未知 core 分配下跨 program 私有状态、
反复重访的远端自动输出窗口、多轴 mesh 全局组装、动态 Ref 变换和一般物理布局
仍不具备完整证明。`where(mask, raw_padded_load, other)` 先读未定义尾部再消除的
形式也没有 poison/liveness 证明。上述情况不能用编译成功、部分数值样本或
增大 SymPy 时限替代语义事实；输入输出共享 HBM 继续按用户决定排除。

**主工作区集成验收。**

| 检查 | 最终证据 |
| --- | --- |
| Pallas collection | 1141 项，较本轮基线增加 174 项；最终代码全部已有分批通过结果 |
| 全部非通信 marker 首轮 | 964 passed、1 failed、176 deselected，613.60 秒；唯一失败为预算报错位置断言，见下段 |
| 全部官方 CPU 通信 marker | 176 passed、965 deselected，346.08 秒 |
| 修正断言后的新增功能与包分层组合 | 152 passed、26 deselected，32.09 秒，包含全部新增静态用例与包分层守卫 |
| 共享结构、区域绑定、代数区域、深度/规模 | 167 + 91 + 54 = 312 项通过，三个文件集合互不重叠，含 8 项新接口细化回归 |
| 真实 GLM5 absorbed Attention/Decoder | 4 passed，接口预算 1，322.26 秒；CPU 原正常 lowering 仍按预期拒绝 |
| 原生 TPU 编译 | 本轮 27 份源码生成 Executable；7 份明确后端拒绝另列，未执行 kernel |
| Ruff、format、basedpyright、tracing、adapter inventory | 通过，类型检查 0 errors / 0 warnings |

集成后动态地址规则会先于 bank owner 枚举耗尽同一分析预算，原测试只允许旧
`interleaved_banks.owners` 位置，故首轮失败。没有放宽预算或修改生产逻辑：
入口继续断言预算拒绝，并直接调用 bank helper，要求 owner 计费失败发生在投影
回调执行之前。该单例 1 passed，随后上述 152 项最终组合全部通过。
分批数量有交叉，不能将所有表格行相加作为总覆盖数；Pallas 总数以最终 collection 为准。

主工作区日志保存在同一临时目录的 `integrated-collection.log`、
`integrated-static.log`、`integrated-cpu.log`、`integrated-new-final.log`、
`tracing.log` 和 `adapter-inventory.log`。受保护文档及项目依赖未修改，
本子阶段完成；更广的语义层计划继续 active，未证明范围同步到技术债记录。
