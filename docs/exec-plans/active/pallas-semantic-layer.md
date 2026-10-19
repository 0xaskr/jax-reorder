# JAX / Pallas 语义中间层：设计记录与后续计划

创建：2026-10-18。状态：**实施中**（用户已要求实现设计）。
设计依据：[Pallas / JAX 等价比较设计](../../development-docs/pallas-jax-equivalence.md)
（尤其 §3、§4.2、§8）。用户已确认：不把 JAX/Pallas 直接翻译成 SymPy 作为比较主路径；
中间抽象层由本项目自研；MLIR 虽长期看好但现阶段接入成本过大，暂不接入。

## 目标与完成条件

- 自研 Python 类型化源语义视图（`ProgramView`：值/区域/操作/Ref/effects/来源），只由
  `core/adapters` 解释 JAX/Pallas 源对象；`Graph`、分析与数学投影从视图派生。
- 完整目标仍为设计 §1、§8：同 mesh 的单 `pallas_call` 比较、重分块、内存与多 TPU 通信，
  官方 JAX 原语与 JVP/VJP 消费者迁移，最后接入 Pallas↔JAX；不能把内部 slice 当作完成条件。
- 当前交付阶段为 A↔A 结构证据与 A↔B 跨分块 normalized 内部链路，加上受限的 M1 分片和
  M2 本地 DMA/VMEM scratch；公共 API、远端通信及现有 JAX 消费者迁移仍待实现。
- 不新增运行时依赖；暂不接入 MLIR、StableHLO、ISL/Z3 等外部组件。

## 设计与取舍

- `ProgramView` 是唯一源事实；适配器独占 JAX/Pallas 对象解释权；视图不含比较算法、不构造 SymPy 对象。
- 源事实从 JAX 内部对象提取：`pallas_call` 参数、`GridMapping`/`BlockMapping`、已 trace 的
  `index_map_jaxpr`、kernel jaxpr、Ref/effects。私有 API 字段纳入 adapter inventory 审计。
- M1 生产索引域采用 blocked 正向/反向逆映射的符号证明，不枚举程序或元素；枚举仅用于独立小域 oracle。
- 结果契约保持三类状态 + 分类异常；`normalized` 已作为内部证据种类实现。
- 当前适配的已知边界（显式拒绝）：kernel 捕获常量、非默认布局、动态 grid、dynamic index
  操作数、vmapped dims、Element/BoundedSlice 块维度、普通读写的部分块切片、非默认 JIT、
  未知 effect/原语。同 mesh 的 shard_map、完整 CompilerParams、alias 源事实已保留；
  非默认执行策略与 alias 仍不可进入规范化。单 program 的本地 DMA 与 VMEM scratch 已接入。
  远端 DMA/设备条件分支已有源适配及有限进展/冲突检查，但全局初始化和值投影尚未接入规范化。

## 工作项

先交付内部垂直 slice，继续按设计推进 M1、M2、M3；slice 通过不代表整个目标完成。

- [x] `core/model.py`：Program/Region/Operation/Value/Literal/Ref/effects/Origin/SourceHandle，
      Pallas `GridAxis`/`BlockMappingView`/`KernelDomainView`/`PallasCallView`。
- [x] `core/index_domain.py`：整数索引表达式、index-map 区域求值、blocked 覆盖/唯一写入检查。
- [x] `core/adapters/jax.py`：本 slice 的受审查原语注册表和区域适配（SSA 检查、类型、effects、JIT 绑定）。
- [x] `core/adapters/pallas/call.py`：解码单 `pallas_call` 的 grid、BlockMapping、index map、Ref、
      effects、interpret/alias/compiler 参数；显式拒绝未审查 schema。
- [x] `tests/test_pallas_semantic_slice.py`：A/B fact 保留、program_id、覆盖/尾块、部分写拒绝。
- [x] `core/memory.py` 存储版本与纯值投影、`core/normalization.py` 规则与
      `NormalizedEvidence`；`core/view_graph.py` 的内部结构快路径与原视图复核。
- [x] adapter inventory 扩展：Pallas/Ref/indexer 字段和原语参数已加入现有版本快照。
- [ ] 文档同步：设计文档与执行记录；受保护文档（README/ARCHITECTURE/PLANS）待单独授权。
- [x] M1 扩展片段：JIT/shard_map、相同 mesh/分片契约、受限尾块与完整 CompilerParams inventory。
- [x] M2 本地片段：start/wait 源事实、单 program 完整块 DMA、信号量代次、VMEM scratch
      初始化/版本/复用、读写冲突检查、证据重放与官方 interpreter 对照。
- [x] M2 本地区间：静态连续切片、拆分合并、区间版本、累计消费、独立进程 oracle 与 CI job。
- [x] M2 远端分析组件：mesh tuple 路由、axis_index/条件区域、独立发送/接收完成、单代全量计数消费、
      依赖环与访问区间检查、源图证据重放；该组件不产生等价结论。
- [ ] M2：本地 DMA、拆分合并、直接/中继、请求—回复、等待环和缓冲区复用反例。
- [ ] M3：现有 JAX 结构/数学消费者统一适配，公共入口证据分派，JVP/VJP 回归。
- [ ] 后续覆盖：循环/atomic、scratch/alias、一般整数域、通信协议、TPU 实测与完整官方原语清单。

## 决策与进展

- 2026-10-18 实施启动：核对工作区仅有设计文档和执行计划改动，未发现已有实现。
  已读取 protected-docs 配置；核心文档保持只读。按设计 §4.2 修正旧计划中枚举程序实例的建议。
  锁定环境实际为 JAX 0.11.2；`BlockMapping.index_map_jaxpr` 实际可为 `Jaxpr`，effects 绑定 Var；
  公共 `pl.load/store` 已移除，尾块探针需使用锁定版本内部 masked primitives。
- 已实现内部链路，生产域分析不枚举程序或元素；`2**24 + 13` abstract tracing 通过预算 256
  的跨分块证明。支持正向/反向 blocked、坐标 mask、128/256 跨分块和非负 div/rem 的逆映射恒等式。
- Ref 解释器先检查全部操作，再按当前版本读写；不会把未初始化输出填零。完整有效域只有一个写入者；
  普通 store 的未使用旧值可丢弃，但实际使用旧值明确拒绝。读集合相同不代表地址对应相同。
- 结构证据目前保持源方程顺序，独立复核未规范化视图；normalized 证据重放规则前提并检查终端值 DAG。
  默认 JIT 的参数/区域绑定和调用边界进入证据；非默认 JIT 契约明确拒绝。
- 首轮失败暴露了 `mul.out_dtype` 与超越函数 `accuracy` 参数遗漏，按实测 schema 修复；
  JAX tracing 会消除未使用 get，完整图读取反例改为显式插入保留的源方程。
- DAG 投影与相等检查使用共享遍历，80 次重复平方不会展开成指数规模树；整数索引各中间运算检查溢出。
- M1 扩展：恢复 shard_map 的手动轴环境，保存 mesh/设备排列/端口分片；Pallas 清除 Ref variation
  的绑定政策单独保留。输出复制性由实际值依赖证明，不能依赖关闭后的 check_vma 声明。
  Mosaic 17 个 CompilerParams 字段逐项保存；未知 flags 拒绝，空 FrozenDict 单独编码。
- M2 本地片段：DMA 适配不预配对，分析只在 start 已发起、源版本稳定和完整 wait 匹配时发布目标；
  每个 semaphore 同时最多一次传输，完整消费后可复用；不同 semaphore 允许共享只读源。
  VMEM scratch 不初始化为零，整块写产生新版本；源/目的在途保护使用索引，不枚举执行调度。
  结论显式携带有效本地 DMA 最终完成的平台假设。程序自身缺少等待/提前覆盖不是可豁免假设。
- shard_map + scratch 组合检查确认：VMEM 构造器的空抽象 mesh 并非环境外构造特有现象，
  环境内构造也相同。保留原描述符，并显式记录内核局部 mesh 绑定；其它已命名 mesh 不自动重绑。
- M2 本地区间扩展替代此前每信号量单次未完成传输限制。相同存储对/dtype 的对齐切片可以共同计数；
  部分 wait 只消费数量，不指定已完成 producer，消费整代总量后才解除全部在途保护并发布目标版本。
  区间版本证书允许同一缓冲区的不交叠区域独立更新，源/目的坐标差进入值公式；生产路径按边界切分。
  静态 stride/OOB/缺口/重叠/计数不足或过量/代次混用均有拒绝路径；大形状不分配数据。
- 官方 interpreter 移至逐例隔离进程：启动/编译 90 秒，READY 后执行 30 秒，超时终止进程组；
  增加故意挂起和持有管道的子进程测试，验证阶段报告与清理。CI 单独串行运行通信 marker。
- 2026-10-19：远端分析现从完整适配视图实例化固定设备的有限命令；条件分支只由静态设备坐标决定。
  start→发送完成→接收完成保持分离；一代一次完整消费建立全部 producer 前驱，计数不守恒先拒绝。
  交叠冲突由依赖可达关系检查，不能以某个安全拓扑排序代替所有调度的证明。双设备请求—回复保留
  提前 start，延后到等待回复之后会检出环。默认 TPU device barrier 的源参数进入准入，跳过 barrier 拒绝。
  远端证据单列未完成的生命周期、初始化/覆盖及全局值条件；只有模型内进展与冲突次序，不返回等价。

- 早期讨论阶段曾只记录结论、暂不实现；后续用户明确要求实现本文，现按完整目标继续推进。

## 验证结果

- 最终扩展组合：上述 Pallas/分层文件加 canonical/semantics/correspondence/algebra_structural，
  `-m 'not pallas_communication' -q --tb=short`：**502 passed，21 deselected，41.25s**；
  随后追加三设备 switch/clamp 与对应拒绝路径，见下方专项记录。
- 最后 Pallas/协议/清理/分层静态专项（含 clamp 上下界与中间溢出反例）：
  **175 passed，25 deselected，22.18s**。该批次覆盖全部当前 Pallas 文件，未重复全量 SymPy 测试。
- 本地与双设备全部隔离通信模拟重跑：**21 passed，166.55s**；三设备直接发送/中继（两种模式）
  另 **4 passed，37.65s**，合计 **25 个通信用例**。三设备 worker 同时核对原生源图的协议证据。
  三设备初次使用 lax.switch 时 **4 failed**，原因是源图新增 clamp 尚未适配；已补充受审查 schema
  和静态整数控制求值后通过，没有删除 clamp 或仅以模拟结果绕过源适配。
- Ruff、格式检查（224 文件）、basedpyright、adapter inventory、文档链接及差异空白检查通过；
  tracing 示例与 **1,280** 个小域 oracle 通过。受保护文档和权限配置无 staged/unstaged 修改。
- 本地区间与既有结构组合回归（上文六文件、oracle 清理测试及 canonical/semantics/correspondence/
  algebra_structural）：**480 passed，16 deselected，37.17s**，这是远端适配扩展前的批次。
- 远端源图/事件/清理定向检查：**25 passed，5 deselected，8.98s**；包括独立计数状态机与证据篡改。
- `uv run pytest tests/test_pallas_remote.py -m pallas_communication -q --tb=short`：
  **5 passed，46.75s**；四个正常模拟与一个已有静态等待环证据的执行超时观察，按阶段回收进程组。
- 隔离 runner 清理逻辑改进后重跑本地区间 eager 模拟：**5 passed，42.45s**。
- 本地区间最终静态批次：`uv run pytest tests/test_pallas_dma.py tests/test_pallas_dma_regions.py
  tests/test_pallas_storage.py tests/test_pallas_semantic_slice.py tests/test_pallas_sharding.py
  tests/test_package_layout.py -m 'not pallas_communication' -q --tb=short`：**147 passed，19.60s**。
- 独立进程 CPU oracle：上述 DMA 两文件加 `-m pallas_communication`：**16 passed，122.75s**；
  包含串行/重排/复用、拆分总等待/分次等待、半块交换、中继、不交叠版本更新，两种 DMA 模式均通过。
- `uv run pytest tests/test_pallas_oracle.py -q --tb=short`：**4 passed，4.21s**，包含编译前/执行阶段
  超时、非零退出、错误结果，以及父进程已退出但子进程忽略 SIGTERM 的清理路径。
- 适配 inventory 新增完整/切片 DMA PyTree 静态元数据与 `TransformedRef` 字段，重新生成锁定快照后通过；
  Ruff 与 basedpyright 已通过本地区间生产代码检查。之前 uv editable 重建受 DNS 限制失败，依赖构建已恢复，
  失败时未执行的测试已重新运行，未把构建失败计为测试结果。
- `uv run pytest tests/test_pallas_dma.py tests/test_pallas_sharding.py
  tests/test_pallas_semantic_slice.py tests/test_package_layout.py -q --tb=short`：最终 **126 passed，17.82s**
  （session `16795`）。包含共享源保护、100 次版本预算、shard_map + scratch、不同 mesh 拒绝路径；
  最后的输入无关整数差异回归确认不等价证据也携带 DMA 完成假设。
- 上述四个文件加 `test_canonical.py test_semantics.py test_correspondence.py test_algebra_structural.py`：
  最新 **454 passed，33.22s**（session `65486`）；此前 VMEM 扩展前批次 **441 passed**。
- DMA 专项 **40 passed** 后补充不同 scratch mesh 反例，已纳入上述 454 项；eager/on_wait
  数值对照均显式断言官方 race detector 未报告冲突。
- `uv run pytest tests/test_pallas_semantic_slice.py tests/test_package_layout.py tests/test_canonical.py
  tests/test_semantics.py tests/test_correspondence.py tests/test_algebra_structural.py -q`：**382 passed**
  （默认 JIT 扩展前的回归批次；扩展后新增路径已包含在上述 57 项中）。
- `uv run python tests/pallas_domain_oracle.py`：**1,280** 个小域覆盖/漏写案例通过；pytest 另交叉检查生产域证明。
- `uv run python tests/adapter_inventory.py`、`uv run python tests/lax_inventory.py`：通过。
- `uv run python examples/inspect_jaxpr.py`、`uv run python examples/bfs_reordering.py`：通过。
- Ruff、格式检查（212 文件）、basedpyright、`tools/check_docs.py --changed`、`git diff --check` 已通过；
  adapter/lax inventory、两个运行示例和 1,280 个小域 oracle 已在当前扩展后重跑通过。
- 原位置的全量 `uv run pytest -q --tb=short` 出现多项失败后中止（session `35056` 已返回 exit 130），
  输出 `/tmp/jax-reorder-pallas-pytest.log`，不报告通过。
- 首个失败 `test_elementwise_polynomial_rules[float32-pair6]` 单独复现为 SymPy worker **10 秒超时**
  （session `40680`，1 failed，29.44s）。HEAD 源码复制到 `/tmp` 后同一测试 **1 passed，18.01s**；
  当前源码复制到 `/tmp` 后 **1 passed，15.85s**。逐文件核对：既有实现模块内容均未改动，新实现模块 9 个。
  这些证据与源码位置/文件系统开销影响超时一致，尚不能宣称全部全量失败的根因已确认。
- 先前 slice 的全量验证使用当时同内容源码副本（不包含后续 sharding/CompilerParams/DMA 扩展）：
  `PYTHONPATH=/tmp/jax-reorder-pallas-current-rgxyl_23/src .venv/bin/python -m pytest -x -q --tb=short`。
  session **23664** 已结束：**1 failed, 116 passed，1067.60s**；失败为
  `test_absolute_matches_both_half_axis_spellings[float32-<lambda>0]` 的 SymPy worker 10 秒超时。
  输出 `/tmp/jax-reorder-pallas-pytest-local-src.log`；未修改 worker 时间阈值，不报告全量通过。
- DMA 独立数值对照使用官方 `pltpu.InterpretParams` 的 eager/on_wait 两种模式及 race detector；
  直接复制、串行中转、独立算术重排、scratch 复用成功。反例只做静态拒绝，避免执行潜在死锁 kernel。
- 所有证据均为 CPU tracing/测试；没有真实 TPU 编译、多设备 TPU 执行或并发证明。

## 未解决问题

- M1 尚缺非默认 CompilerParams 的重写规则和更一般地址/状态组合。
  现有符号规则只证明一维正向/反向 blocked；非线性/一般映射不能偷偷退回枚举。
- 普通读写的部分写/mask、循环、atomic、alias、远端通信的全局初始化/值投影仍未实现；本地 DMA/VMEM scratch
  只支持单 program，DMA 已允许静态连续切片与累计等待，不能推广为每个 program 私有 scratch。
- 远端路径目前只返回协议分析证据；下一步按事件依赖与区间版本接通全局内存和值投影，再比较直接/中继
  的边界值。计数跨代复用、部分远端等待、混合本地/远端协议和一般同步原语仍须新增规则。
- 公共比较 API 和现有 JAX 全部消费者尚未迁移；结构快路径未接通既有 BFS/歧义搜索。
  M2、M3 及后续范围都仍属于原始目标，不能以 slice 已通过替代完成。
- MLIR 分析后端、ISL/Z3 选型是 M1 之后的 spike，不阻塞当前设计。
