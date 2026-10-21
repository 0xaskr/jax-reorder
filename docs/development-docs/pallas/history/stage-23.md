# 历史：9.23 四类剩余问题并行补齐（2026-10-21，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.23 四类剩余问题并行补齐（2026-10-21，完成）

本轮按用户要求分别推进组合覆盖、自动缓冲源契约、动态行为/布局与代数性能。
以 §9.22 的完整工作区为基线；各分支先记录具体源 case、当前拒绝/超时和必要
前提，再实现与独立审查，最终由主线程合并共享模型和解释器。

1. Ref/DMA 组合在 core 保留原存储及逻辑元素次序。有洞足迹不能合并成包围矩形；
   远端完成计数、覆盖、冲突与版本必须对应真实传输集合。padded Ref 的变换同时
   组合值地址和初始化义务，不能因为改变 shape 而丢弃未定义 lane。
2. 自动输出重访只依据实际源流水策略和写回/恢复依赖准入。默认策略与显式恢复
   策略分别核对，不能从 AOT 成功推出隐式 DMA 已按某顺序完成。
3. 动态索引/路由/循环按现有 SSA 与已知有限域扩展规则。布局仅接入有源描述、
   可保留或证明坐标语义的选项；未给定的域、进展和物理映射不由比较器猜测。
4. 数学层先处理已固定的真实尺寸失败：mask=383/385、mean 错分母和 256-lane Max。
   保留独立初始化与数学定义域；允许预算受控的区间/分片公式规范化，但完整
   判等或不等结论仍须来自可信数学检查，不能用数值样本替代。默认时限仍为 60 秒。

正例须与独立参考值/图对照；反例包括错误地址/目标、漏同步、错误 mask、非法
前缀消项和不满足动态范围。CPU、原生离线编译与图/代数证明分别报告。
core_map/mpmd_map、输入输出共享 HBM、多轴 mesh 全局证明和实机仍不纳入。

**实际实现与源契约（2026-10-21）。** 四条分支在隔离副本实现，均相对同一
工作区基线生成补丁；主线程逐份检查后合入，未覆盖此前用户改动。

- Ref/DMA：padded 输入的 reshape/transpose/index 同时传递值坐标与初始化
  条件；行优先 reshape 直接保留两侧相同的尾维，只重排前缀。真实有洞 DMA
  使用互不相交的 boxes/spans，源版本、目标发布、冲突和 byte count 都按
  实际集合计算；洞内访问不会被包围矩形误判。一次源传输仍只有一组完成事件，
  不拆成多个信号量生产者。联集冲突按片段对计费，伪造空、负长、交叠、乱序
  或越界足迹会拒绝。远端映射携带 destination→source 坐标，原 storage
  仍限既有的一维向量；多个连续 index 链已有原生可执行源码。
- 自动缓冲：锁定 JAX 0.11.2 的 `Buffered(1, RevisitMode.ANY)` 有明确的
  输出恢复契约，原生 XLA emitter 支持同步单缓冲，并明确拒绝双缓冲 ANY。
  官方固定快照中的 `TPUPipelineModeRevisitTest` 非连续块归约是源实例。
  在现有单串行 grid、rank-one 远端路径中，已写回的 A 窗口在 A→B→A 前
  产生 restore_read/restore_write，参加同一协议和存储版本重放；连续重复
  不重复恢复，首访仍未初始化。默认/IMMEDIATE 非连续重访继续拒绝。
  尚未实现恢复的本地值执行器明确拒绝输出 ANY，不能默默忽略 pipeline_mode。
- 动态范围：整数位掩码、有符号 Python `%`、受限整除及有界循环 carry 已
  接入。真假分支条件依据完整表达式对应收窄范围，保留 sign/rem 的关联和
  原始 ValueExpr；相同区间的不同输入不能混为一个值。每次范围查询限制
  20,000 步、深度 48、每个 selector 至多 8 个上下文，耗尽时整次查询退回
  dtype 范围；具体地址/循环枚举另按原 AnalysisBudget 计费。未证明终止、
  零次迭代的缺初始化、未完成 DMA 和缺少设备间约束的数据路由仍拒绝。
  调用既有 IndexExpr bounds 前，按实际递归展开计费；共享 DAG 不能绕过预算
  触发指数展开。独立随机检查 1,600 个表达式、78,400 组输入点，区间均包含
  精确求值结果；这项验证不代替生产规则的证明条件。
  `fuse_transposed_lhs_in_matmul=True` 只作为已审查的源代码生成提示准入，
  保留实际 compiler options 和 matmul 语义；没有放开安全开关或任意物理布局。
- 代数：初始化 Product 在原区间上独立分解，整数仿射 guard 使用精确有理
  临界点；同一被加项的有限 Sum 按重叠端点消项，不增加定义域、不提高
  256 项展开上限。大 Max/Min 保留全部不同操作数，用相同集合的同余证明
  正例；集合不同不能直接判不等，负结论须有满足条件的精确有理见证。
  额外符号/查询假设会禁用不适用的见证捷径。周期或未覆盖的 Boolean
  Piecewise 无法转成 ITE 时返回 MathematicalUndecidedError，不补默认分支。

代数独立审查发现并修复了本轮首版的大极值抽象误判：令 `m(k)` 为第 k 行
40 个输入的 Max，不能将它当成常数，使前两行与后两行的 `Sum(m(k))` 匹配。
每行所有元素取行号时，两者精确为 1 和 5。最终规则禁止将依赖 Sum/Product
绑定变量的极值替换为常数；Sum/Product × Max/Min 均有固定反例，公共 JAX
入口不得返回等价。共享 helper 的 Derivative、Integral、Limit、Subs 也各有
复现反例，连同 Lambda 参数绑定一起保守拒绝该捷径。独立复核 26 项通过。
这是真实的健全性修复；相关表达式允许保持数学未决，不能以性能捷径牺牲正确性。

真实 `(8,384)` 输入、256-lane block 的固定查询已得到下列公共 API 结论，
默认 worker 时限保持 60 秒：mask=383 的 sum/max 为 INEQUIVALENT；mask=385
因完整输出初始化无法成立而拒绝；错误 mean 分母为 INEQUIVALENT；填充值
0、7、-3 的 Max 正例均为 EQUIVALENT。实测这些定向查询约 5～10 秒，
只是本机观测，不构成任意 CAS 查询的时限承诺。log、倒数、sqrt 的区间反例
确认没有通过前缀消项扩展偏函数定义域。

**交叉审查与组合用例。** 主线程新增
[组合测试](../../../../tests/test_pallas_gap_integration.py)，四分支合并前的三线
组合已 **6 passed，13.12 秒**：同步 ANY 恢复加远端 index 链保留未覆盖半块，
错误源半块保留值差异，漏 peer 同步拒绝；`(16,384)` 输入、512 列 padded
Ref、`reshape((2,8,512))` 的 mask=383/384/385 分别得到不等价、等价和
初始化拒绝。该 reshape 最初触发冗余 `Mod(k,512)` 的 CAS 内部错误，已用
上述共同尾轴地址规则及独立 Boolean 异常分类闭合，而非隐藏失败或加入假设。
组合 ANY/index 源码另生成离线 TPU Executable。

原生与解释器的边界单列：一维原 Ref reshape 的远端示例被锁定 TPU lowering
明确拒绝；它们的语义/NumPy 正例不算原生执行证据。官方 TPU CPU interpreter
尚不识别相关 ReshapeTransform 的 race range，也未实现 ANY 输出重访；
通用 CPU interpreter 的 TransposeTransform 同样有限制。这些预期拒绝与
真正完成的 CPU 数值对照分别计数。没有连接 falcon 或执行 TPU kernel。
离线 AOT 使用 JAX/jaxlib 0.11.2、libtpu 0.0.48.1 和 `v5:2x2x1` 编译拓扑；
四组共 21 份源码完成 IR 校验并生成 Executable：Ref/DMA 4、缓冲恢复 6、
动态索引/循环/布局 10、主线程组合 1。4 份一维 Ref reshape 源码和 1 份
双缓冲 ANY 按原生后端限制拒绝，单列为 5 项源拒绝，未计入编译成功。

完整 Pallas 集成批次通过：静态比较 1,269 项（859.54 秒），CPU 通信 210 项
（503.09 秒）；共享结构回归 368 项通过（68.43 秒）。这三批使用两项最终
安全修正之前的冻结源码；修正后的定向回归单独记录，不能混称一次全仓库运行。
原始日志、各分支报告、补丁、AOT JSON 和逐测试完成记录保存在
`/tmp/jax-reorder-four-gaps-38rh92cg/`。会话中断曾终止两组长回归：
通用代数旧段完成 329 passed 和 1 项已在最终实现修复的有理临界点 oracle
失败；依据保存的 collection ID 从第 331 项续跑剩余 435 项，全部通过
（674.61 秒）。765 个原 collection ID 没有遗漏或重复，失败 oracle 已在
最终实现单独通过；这是分段验证，不是一次连续 765 项全绿。动态旧回归
停在进度中途，不作完成声明；由合并后的完整 Pallas 批次覆盖。已完成的
49 项动态新增测试、其他定向测试和 AOT 记录保持有效。

最终两项安全修正合入后，主树定向回归 **362 passed、19 deselected，160.16 秒**，
覆盖动态范围/视图、Ref 组合、极值与区间、尾块初始化/值/资源、大型 CAS、
三线组合、可信 SymPy 和包分层；完整 Pallas collection 为 **1,481 项**。
Ruff、319 个 Python 文件格式检查及 basedpyright 通过（0 errors、0 warnings）；
tracing 示例、锁定 adapter/lax inventory、文档链接和 `git diff --check` 均通过。
各批测试有重叠，数量不能相加作为独立用例总数。结果按实际来源分列：

| 验证 | 结果与证据 |
| --- | --- |
| Pallas 静态比较 | 1,269 passed；`integrated-pallas-static.log` |
| Pallas CPU 通信测试 | 210 passed；`integrated-pallas-cpu.log`；含预期解释器拒绝断言，不全部等于数值执行 |
| 共享结构回归 | 368 passed；`integrated-core.log` |
| 最终修正后的定向回归 | 362 passed；`integrated-final-targeted.log`、逐测试 JSONL |
| 通用代数 | 原 765 个 ID 分段覆盖，329 通过、1 个已修 oracle、续跑 435 通过；见 `algebra-report.md` |
| 独立审查 | 78,400 个区间包含点与 26 个极值检查通过；最终主树与审查版本一致 |
| 原生离线 TPU AOT | 21 个 Executable、5 个明确源拒绝；各分支 `*-aot*.json` 与 `integrated-aot.json` |

最终相对本轮工作区基线的补丁及 SHA 清单为上述临时目录中的
`integrated-current.patch` 和 `integrated-manifest-final.json`。本子阶段完成，
整体计划继续跟踪下列范围；未修改受保护文档或依赖，未提交/推送，未使用实机。

剩余边界：本地输出 ANY 的恢复、多维原 storage 的远端值组装、缺少设备间
约束的数据路由、一般终止证明、任意物理布局，以及不满足现有同余/区间规则
的大型非线性 CAS 查询仍不保证支持。默认/IMMEDIATE 非连续重访按源契约拒绝；
`core_map`/`mpmd_map`、输入输出共享 HBM 和多轴 mesh 全局证明继续排除。
这些边界不影响本轮固定正例、值差异反例和资源错误拒绝路径的完成。
