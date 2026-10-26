# 历史：9.20 并行补齐语义覆盖与验证（2026-10-20，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.20 并行补齐语义覆盖与验证（2026-10-20，完成）

用户要求多个 subagent 同时推进 §9.19 后的剩余问题。三条隔离实现分支分别
负责 grid/索引域、通信资源、动态语法；主线程负责图规范化、环境验证与集成。
每条分支必须先固定具体案例和支持条件，再修改算法；所有规则仍以源视图
为依据，不能丢失副作用、地址别名、目标 core 或并发顺序以获得匹配。

- Grid 的尾块证明区分有效输出域与填充域；不规则静态映射可以在预算内
  穷尽坐标作精确证明，但不得用抽样代替覆盖。大规模独立轴继续保留符号。
- 通信中的数组元素、设备和 core 是不同的资源坐标；跨 program/迭代保留
  DMA 代次、存储版本和实际等待。多 core 必须有可核对的执行绑定。
- 动态语法只在静态约束或有界精确解释能够证明地址、终止和 effect 时接入。
  atomic 不能无条件降成普通 load/store，布局不能仅凭逻辑 shape 被忽略。
- 图规范化只增加不改变计算次序和副作用的精确规则；整数到浮点转换及
  浮点表达式仍保留其操作语义，不用代数近似抹平差异。
- 验证分别记录静态证明、实际 CPU 执行、TPU AOT lowering 和最终后端编译。
  用户先指定 falcon 2x2x1，随后明确无需实机；不连接或启动该环境。

结果和剩余条件记录如下；这些闭合子集不代表任意 Pallas 语法已支持。

主线程的规范化规则按输出坐标建立完整矩形分区：收集整数除法/余数和
单轴仿射谓词的边界，预算检查后覆盖每个区间的笛卡尔积。在每个区域内，
只利用整数范围简化地址和选择分支，并比较保留类型与运算次序的值 DAG。
这不是取代表点抽样；所有区域的全部坐标均须得到相同表达式。常量整数
转 float32/float64 仅在宿主中间 double 能精确表示该整数时按目标 dtype
舍入，浮点加法不交换、不结合。源规范化证据验证时重放同一个完整分区。

按用户最新指示，硬件验收改为离线 AOT。初次探测只发现 CPU、缺少 libtpu；
此结果不再作为任务阻塞。先在 CPU 主机通过抽象 TPU 环境执行
`jax.jit(fn).trace(ShapeDtypeStruct(...)).lower(lowering_platforms=("tpu",))`，
生成含 TPU custom call 的 StableHLO 并通过 IR verifier。最终 `.compile()`
初次尝试失败于 Host 没有 `tpu_custom_call` 实现；该失败不能推出需要 TPU 实机。
按用户建议安装编译后端后的成功结果见下文。可复现入口为
[AOT 检查脚本](../../../../tools/pallas_aot_check.py)，默认不执行 kernel。

**本轮已实现的闭合子集。**

| 范围 | 精确规则与反例 |
| --- | --- |
| 静态 index_map | 对决定分支的有限坐标建立 lookup，支持 select/cond、有限 scan/while；穷尽检查唯一写入、覆盖并构造逆映射。独立大轴仍为符号，耦合大域受预算限制；碰撞和缺失覆盖拒绝。 |
| 分组尾块 | 按每个输出 Ref 的有效全局域独立证明 mask，允许最后一组不满；masked load 的 `other` 必须保留，错误 mask/填充值有独立反例。 |
| 信号量数组 | 普通与 DMA 信号量以设备、分配、数组元素为资源身份；有限索引、计数、read、scope/call 清零分别检查。不同槽位不能合并。 |
| 远端循环与 grid | protocol 与值重放同步展开有限 scan/while。跨 program 仅接入静态一维串行 grid、显式 ANY/HBM 完整 Ref、恒定零 index_map；该子集不存在被忽略的自动 staging/writeback。 |
| 远端切片 | 可绑定动态切片保留精确连续区间；修复 TensorIndex 被错误退化为整块的路径。数据依赖未知地址仍拒绝。 |
| grid 与内部循环 | 完整坐标分区使分组大小 2/3 的真实 grid 与内部循环直接得到 normalized equivalent，不调用 SymPy；错误更新/映射不能因此匹配。未知 device 自由坐标保持未证明。 |

交叉审查用 2440 组有符号整数分区、负除法/余数与 lookup 组合进行独立有限域
差分检查，其中 1522 次正证明未发现假阳性。该检查用于发现实现错误，不替代
完整分区算法的证明条件。另验证分组输出窗口与信号量数组组合：正常槽位通过，
错误槽位等待拒绝。

**必须区分上游语法和本库证明范围。** 锁定 JAX 的 TPU 原生 lowering 接受
交错 parallel/arbitrary 维度，但专用 TPU CPU interpreter 要求并行维度为前缀。
因此不能把交错维度宣称为非法语法；本库暂未证明其原生调度下的共享状态生命周期，
继续明确拒绝。`core_parallel` 同样可以 AOT lowering，但未给出普通 pallas_call
的 program 到物理 core 绑定。同一个 signal-to-core1/wait 协议在两种绑定下
分别可完成和无法完成，不能从 grid 坐标猜测绑定。保留 `execution_cores=1`
调用前提，任意多 core 尚不支持。

原生 TPU masked load 也被锁定 lowering 拒绝；对应 padding/reduction 用例仅有
通用 `interpret=True` 数值证据。另建 full-block input load＋masked output store
的原生兼容尾块用例，分别验证偶数组与不完整末组。填充域 scratch/DMA、外部
HBM alias、远端自动缓冲块多 program 和一般数据依赖路由仍未纳入。

**语法和存储。** 静态正步长用精确离散矩形集合读写，未被选中的洞仍未初始化，
DMA 冲突仍使用原 storage 区间。有限 program 坐标决定的动态读取使用保守地址
包围域检查资源义务、精确坐标表达输出；一般动态写继续拒绝。完整静态 Ref
reshape/transpose 链由双向坐标映射解释，不创建新 storage，不允许通过改 shape
逃逸在途 DMA 检查；部分/动态视图和非默认物理布局仍需独立证明。

有限 while 可以由 program 有限域或循环前整数 min/max/clamp 的显式有限域驱动。
逐个情况执行原条件/body，确认退出，检查 DMA quiescent、计数一致和初始化，再以
真实选择条件合并值。超预算、无界终止不放行。交叉审查修复互斥分支的重复 transfer ID：
每个 case 保留独立 origin 和全局唯一 ID；generation 按该分支真实状态开始，
wait 的 producers 仍精确引用。communication 证据是各互斥情况的记录集合，
不能把它解释为一条同时执行的动态轨迹。

锁定后端的真实 AOT 拒绝也有回归：`atomic_rmw`/`addupdate` 没有 TPU TensorCore
lowering；直接 `lax.reshape(dimensions=...)` 与 Ref transpose 的原生 lowering
未实现。完整 Ref reshape 原生 lowering 成功，但专用 TPU CPU interpreter 仍
将 transform 当作 NDIndexer，报 AttributeError；改用官方通用 Pallas interpreter
验证 reshape 读/写/DMA-safe 三个数值用例，没有修改上游执行器。Ref transpose 的
通用解释器同样未实现，只有源图双射语义和明确后端拒绝证据。

**集成交叉审查修复。** 整数索引压掉单例轴时，即使访问区域等于完整 Ref，
mask 仍位于结果坐标系。旧证明会将结果 `local:0` 错当成 Ref 的单例轴，使
`shape=(1,128)`、`mask=arange(128)<64` 的部分 store 被当作全写。现仅在
mask 证明入口应用 `to_value`，将 mask 转到 Ref 坐标；masked load 的最终选择
继续使用结果坐标。永久回归覆盖首/中单例轴、正确尾 mask、`other=7` 经 reduce
保留，以及部分 load/store 的拒绝；官方通用 CPU 同时确认反例的实际输出。
Ref view、步长与 DMA 的额外探针检查了拒绝路径及不相交写的完整输出。

完整集成回归另外发现官方四缓冲 matmul 超过既有 100,000 分析预算。按相同
输入逐类计费对照，本轮基线为 97,248，合入后为 100,640，差值恰为新增的
3,392 次 `access_regions`：单矩形访问已经按操作和 storage 检查计费，却再次
按笛卡尔积大小收费。修复只对额外生成的矩形预收费，不增加测试预算，也不
跳过证据重放。新增步长/多矩形扩张仍保留分配前预算检查。

**真正的离线后端编译。** 独立 `/tmp` 虚拟环境安装 `jax[tpu]==0.11.2`，
得到 JAX/jaxlib 0.11.2 与 libtpu 0.0.48.1，未修改项目依赖和 lockfile。
使用 `get_topology_desc(platform="tpu", topology_name="v5:2x2x1",
chip_config_name="megacore", chips_per_host_bounds=(2,2,1), num_slices=1,
wrap=(False,False,False))` 构造 4 个 compile-only devices。与
[MaxText 官方离线编译实现](https://github.com/AI-Hypercomputer/maxtext/blob/main/src/maxtext/trainers/pre_train/train_compile.py)
相同，目标拓扑由编译客户端提供，不要求运行时设备。

本地 kernel 显式置于 replicated `shard_map` 中，避免 Mosaic 不支持自动 SPMD
partitioning 的真实限制；远端两个 participant 的原用例使用前两个 compile-only
devices，保持 `peer=1-device_id` 原代码。`JAX_PLATFORMS=cpu` 关闭运行时 TPU
后端发现；没有连接 falcon 或执行 kernel。插件初始化有非致命的缺少运行时 worker
元数据日志，不影响下面编译结果。

共 **29 个具体用例**完成 trace → TPU lowering → StableHLO verifier →
`.compile()`，均得到 `Executable`：

- 4 个基础用例：分组大小 2/3、grid DMA、固定官方双缓冲 copy。
- 18 个扩展用例：静态 stride/Ref 索引链、值 transpose+reshape、动态 read/slice、
  program/输入有界 while、完整 Ref reshape 读/写/DMA-safe、8 个 grid 映射/尾块用例。
- 7 个通信用例：普通数组 read、scope、serial、远端 DMA/普通数组、跨 program HBM
  DMA，以及 core_parallel 的合法源码编译。

交错 grid 和 core_parallel 编译成功仍不证明任意执行绑定的资源正确性。
语义拒绝边界保留。原始 JSON/探针保存在本轮
`/tmp/jax-reorder-pallas-gaps-f0i_dq77/` 的 `aot-compile-report.json`、
`aot-extra-report.json` 和 `communication-native-aot.json`；基础复现命令：

```bash
uv venv /tmp/jax-reorder-tpu-aot --python 3.12
uv pip install --python /tmp/jax-reorder-tpu-aot/bin/python 'jax[tpu]==0.11.2'
JAX_PLATFORMS=cpu /tmp/jax-reorder-tpu-aot/bin/python \
  -m tools.pallas_aot_check --compile --output /tmp/pallas-aot.json
```

**集成验收。**

| 检查 | 结果 |
| --- | --- |
| 最终 Pallas collection | 967 项，较本轮开始增加 182 项；按最终用例去重，均已有分批通过结果 |
| 首批全部非通信 marker | 808 passed、1 failed、147 deselected；唯一失败为上述四缓冲 matmul 的重复预算计费，已修复 |
| 首批全部通信/CPU marker | 147 passed、809 deselected；后来新增 mask 的 3 项通用 CPU 对照也通过 |
| 最后 mask 修复的受影响组合 | 260 passed，包含 tensor memory、grid、syntax、Ref views、多输出及其 CPU 对照 |
| 最后预算修复的定向组合 | 112 passed，含原失败用例；恢复原 work=97,248，另保留多矩形预收费反例 |
| 预算修复后完整 pipeline 非通信集 | 31 passed、10 deselected，包含全部真实流水正例、坏调度反例、代数入口与证据重放 |
| 共享 core/适配/数组转换 | 78 passed；tracing 示例和 adapter inventory 通过 |
| 原生 TPU 离线编译 | 29 个具体用例均生成 Executable；没有执行 kernel、没有精度/性能声明 |
| Ruff、format、basedpyright | 全部通过，类型检查 0 errors / 0 warnings |

上述批次有重叠，不能简单相加作为覆盖总数。最初静态批次完成前曾按后来变化的
collection 顺序误把失败定位到 worker 超时测试；最终 traceback 已确认是预算回归，
不是偶发超时，记录以最终根因为准。完整原始日志保存在本轮 `/tmp` 目录。
受保护文档、项目依赖和 lockfile 未修改；整体语义层计划继续 active，
未证明的调度、布局及一般动态语义已同步到技术债记录。
