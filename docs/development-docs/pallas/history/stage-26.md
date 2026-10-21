# 历史：9.26 普通尾块与 masked 访存（2026-10-21，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.26 普通尾块与 masked 访存（2026-10-21，完成）

用户要求补齐普通尾块/mask 覆盖：修改前本地路径对非 ANY padded 访问要求
完整对齐 block，并在应用 mask/other 前检查被选择存储的初始化；远端协议
直接拒绝普通 masked load/store。这些限制会拒绝安全静态子切片、只读取
已初始化 lane 的 masked load，以及后续明确选择掉无效结果的源码。

本轮按原 Ref 坐标、访问结果坐标和 mask 共同确定实际足迹；静态几何条件
在 core 中按矩形和预算分析，不枚举完整 tensor，不借助 SymPy。masked store
只覆盖 active lane，其他 lane 保留原版本和初始化状态；masked load 只对
active lane 施加实际读取与 DMA/协议冲突义务。尾部 HBM 有效域、已分配
VMEM padding 和未初始化 VMEM 值分别处理，不能将 padding 默认初始化。

数据依赖 mask 若不能化成静态区域，允许在原访问范围已证明安全的前提下
使用保守足迹检查 DMA/协议，并用条件表达式保留实际值和未写入部分。
同一 predicate 的后续 where 可以通过结构性条件假设消除无效分支；不同
数据条件的关系不擅自推断，无法证明的越界、初始化或冲突仍拒绝。

没有 other 的 masked load 保留无效结果的未定义性；后续 select/where 只有
在已证明覆盖相同无效区域时才可消除该义务。未定义值若进入有效输出、
索引、条件或未屏蔽的计算结果必须拒绝，不能当作零或任意共同值获得匹配。
远端协议与值投影共用同一坐标映射和足迹构造：静态 mask 的区域一致，数据
mask 的协议足迹可以保守扩大，值执行逐段验证真实足迹被其覆盖，并保留对应
协议事件、DMA 完成、等待、初始化及版本关系。静态空 mask 不产生虚假访问，
但不豁免真实 DMA wait；协议阶段尚未知为空的数据 mask 仍重放原保守事件。

测试覆盖非 ANY 安全切片、多次 masked store 的完整覆盖、masked load 的
显式 other 与后续 where、变换 Ref/多维尾块，以及远端普通访存与正在进行
DMA 的冲突/不相交正反例。每个正常输出对照完整独立参考值，负例包含漏写、
错误 mask、无效值泄漏和缺等待；补证据重放、预算、CPU 与适用的离线 AOT。
源后端拒绝与本项目拒绝分别记录。实现与独立审查完成后冻结源码，集成结果见下。

**实现与消费义务。** [共享 mask 分析](../../../../src/jax_reorder/core/masked_access.py)
保留实际 active 矩形、非局部 owner 条件以及数据 mask 的保守包络；相邻证明
分区会合并，碎片化条件耗尽预算时明确失败。本地与远端复用该规则，将访问
结果坐标映射回原 Ref，再检查初始化、在途 DMA 冲突和版本。masked store
保留 inactive 原值；消费 masked swap 返回值时，inactive 部分同样视为未定义，
不假设不同后端会返回相同的旧值。

`undefined` 标记沿 reshape/transpose 和纯计算传播；select/where 按消费域
检查可能被选择的分支，只有证明不可到达后才能消去标记。reduce/dot 会检查
全部参与计算的输入坐标，归约之后才 mask 不能掩盖已经参与归约的 padding。
`0 * undefined` 不能当作定义值；mask、索引、分支和 while 条件也不能消费
未定义值。DMA start 对实际源矩形检查定义性，即使 DMA 结果随后被完整覆盖
也不能省略该义务。显式 other 保留原表达式，不能被随意换成零。

已分配的自动 input VMEM padding 可以先读取、再由正确 where 屏蔽；未分配
HBM 或普通 scratch/output 的 active 未初始化读取仍拒绝。普通输出只要求
最终有效 HBM 区域有定义，不强求不提交的 padding 初始化。原一维完整尾块
快速规则仍优先使用；扩展路径复用同一个分析预算并重新证明，不捕获预算
异常，也不通过重置预算掩盖失败。`defer_padding=True` 的原代数消费契约保留。

分布式 kernel 的原始 `jnp.where(mask, value, scalar)` 会在内部 JIT 中产生
`reshard` 和带显式 sharding 的 `broadcast_in_dim`。离线 native 编译确认该
形式合法；锁定 Mosaic 的 `reshard` lowering 返回原值。新增
[受限适配](../../../../src/jax_reorder/core/adapters/pallas/reshard.py)只在 Pallas
kernel 内接受：目标必须是当前绑定的全 Manual abstract mesh，输入输出均
为 replicated、无 concrete mesh 或设备重排，dtype、weak type、值元数据
及 memory kind 不变。广播额外遵守原维度映射；原操作和参数完整保留，
[core 规则](../../../../src/jax_reorder/core/local_reshard.py)再次验证后才能作局部
值映射。普通外层 JAX reshard、真实分区变化、目标 mesh 不一致以及证据
参数篡改仍拒绝；kernel 上下文通过 finally 恢复，不泄漏到外层调用。
真实 where 的完整值、公共比较入口、源码证据与四组双设备 CPU 运行均验证；
新增源 schema 已写入 adapter inventory。该规则没有将 mesh 当作无关信息删除。

集成回归发现真实四缓冲 matmul 的原预算仅余 640 步：四次规范化原本合计
99,360/100,000，新路径对无 mask 的每次访问也做 predicate 分区，额外收费
1,088 步。已对严格的 Python `True` 直接复用此前证明过的精确 boxes，不再
重复分区和合并；实际 mask、定义性和 DMA 检查均保留。预算恢复到 99,360，
没有增加上限；修改前版本对照、完整 pipeline 回归和独立有限 oracle 均通过。

**独立验证。** [本地场景](../../../../tests/test_pallas_masked_access.py)、
[远端场景](../../../../tests/test_pallas_remote_mask.py)和
[交叉集成](../../../../tests/test_pallas_mask_access_integration.py)检查完整输出，
覆盖安全切片、部分和多次写入、有洞 mask、显式 other、无 other 后续清理、
数据条件、变换坐标、双设备接收值、空访问与证据重放。错误 mask、多读一列、
漏写、未定义 selector、DMA 源泄漏、仅一 lane 相交和缺 wait 均有拒绝反例。
旧 4 个“部分 mask 必须覆盖完整 block”的拒绝用例转为完整值正例，同时新增
真正缺初始化的负例。独立有限枚举共 1,200 组、79,131 个点，检查 active
区域与定义性；32 层共享表达式不重复展开，十亿 lane 的规则区域保持符号化。

**源后端边界。** 锁定 JAX/jaxlib 0.11.2 的 TPU lowering 及 TPU 专用 CPU
interpreter 仍拒绝 `load_p(mask=...)`；本轮不修改源 kernel 来假装通过 native
编译。通用 CPU Pallas interpreter 可以做相应完整数值对照。对齐 `(8,256)`
的本地 10 份 AOT 源码中 4 份成功，其余 6 份为 masked load 拒绝；远端
15 份中 5 份成功，其余 10 份同类拒绝。成功包含安全切片、普通读取后 where、
数据 masked store，以及与 DMA 不相交的 masked store。环境为离线
`v5:2x2x1` megacore、libtpu 0.0.48.1；未执行 TPU kernel。

官方 TPU CPU interpreter 的 swap race checker 按整个 view 检查 write，未按
mask 裁剪。因此远端 DMA 读取低半块、masked store 更新高半块的源码在其
race checker 下被保守拒绝；该例仅关闭这项检测完成数值对照，不能将结果
称为解释器的无竞争证明。其他远端 CPU 样例保留 race 检测；该例的语义足迹
证明、IR 校验及 native AOT 结果分别记录。

**保留边界。** 碎片化 mask 和有限动态地址仍受分析预算控制；未知数据 mask
按保守范围检查通信冲突，不推断不同数据 predicate 的代数关系。远端数据
masked store 当前要求可能被保留的 inactive 存储已初始化；静态 mask 可以
只写部分未初始化存储，再仅读取已写区域。远端多维 grid、未知路由/终止、
物理布局、core_map/mpmd_map 和输入输出共享 HBM 不在本轮扩展范围。
没有修改 SymPy、默认预算或 60 秒时限，也未提交或推送。


**最终集成结果。** 最终 collection 的 **1,793 个 Pallas ID + 6 个分区 ID**
均有通过记录；加上最终共享 core 的 368 项，去重合计 **2,167 项通过**，比
上一轮净增 192 项。不是将重叠批次相加；逐项覆盖核对记录为
`/tmp/jax-reorder-mask-access-f46cey_k/root-final-coverage.json`。

| 批次 | 结果 | 原始记录（上述临时目录） |
| --- | --- | --- |
| 初次完整静态集合 | 1,498 passed、2 failed，1,068.25 秒；旧初始化错误文案和四缓冲预算回归随后修复 | `root-static.log/jsonl` |
| 完整 CPU 解释器集合 | **244 passed，524.57 秒**；与静态入口证明分别统计 | `root-cpu.log/jsonl` |
| 原始 where 的受限适配与独立审查 | **55 passed，13.76 秒**，含 4 项双设备 CPU 检查 | `root-reshard.log/jsonl` |
| 最终源码定向集成 | **423 passed，281.92 秒**，覆盖全部新 mask/where、完整 pipeline、core 执行和受影响尾块；初次两项失败均关闭 | `root-final-regression.log/jsonl` |
| 最终共享核心 | **368 passed，49.57 秒** | `root-final-core.log/jsonl` |
| 离线 TPU AOT | 9 份固定场景成功，16 份含 masked load 的源码被锁定后端拒绝；另原始 where 与 lax.select 对照 2 份编译成功，无实机执行 | `local-native-aot.json`、`remote-aot.json`、`probe-reshard-plain.json`、`probe-reshard-control.json` |
| 质量门 | Ruff、346 文件格式检查、basedpyright（0 errors / warnings）、tracing、adapter / lax inventory、文档链接及差异检查通过 | `root-final-*.log` |

完整静态/CPU 批次基于第一轮冻结的 mask 实现；补齐受限 where 入口、更新旧
错误文案断言和修复恒真访问预算后，最后冻结的 **12 份生产文件**完成独立
源码复核、423 项受影响集成及 368 项共享核心回归。没有将早期失败隐去，也
没有把不同源码批次声称为同一次完整回归。最终文件哈希与独立审查一致，
有限 oracle 在最终 helper 上再次全部通过；源码、测试和范围记录可逐项重放。

最终相对本轮初始工作区的补丁为 `root-final.patch`，文件清单为
`root-final-manifest.json`，源码冻结清单为 `root-final-source-manifest.json`。
本轮变更 12 份生产文件、19 份测试/fixture 文件和 5 份允许修改的开发/计划
文档；受保护文档、依赖、配置及此前改动保持不变。此子阶段完成，整体语义
层计划的其余边界继续跟踪。
