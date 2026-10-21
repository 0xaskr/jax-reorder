# Pallas 验证与当前边界

从[任务入口](../pallas-jax-equivalence.md)到此；截至 2026-10-21，未声明任意合法 Pallas 语法全覆盖。

## 当前缺口与排除项

| 范围 | 当前限制 | 处理位置/证据 |
| --- | --- | --- |
| 远端条件初始化 | 数据 mask 写入仍要求可能保留的 inactive 存储已初始化；相同 mask 后读取的安全场景可能被保守拒绝 | core；[阶段 26](history/stage-26.md) |
| 并行自动 staging | 独立 owner 间实际共享缓冲发生冲突则拒绝；不能把共享 Ref 私有化 | core；[阶段 27](history/stage-27.md) |
| 执行条件 | 普通调用不自动提供 core 绑定；远端并行要求完整单 core 点流与各设备同序 | [执行绑定](communication.md) |
| 索引/控制 | 远端非恒等向量索引、未证明动态地址、数据决定路由或终止仍非普遍支持 | core；[当前计划](../../exec-plans/active/pallas-semantic-layer.md) |
| 布局/源形态 | 非默认物理布局、动态 grid、未审核块维度/包装/effect 等需独立准入；源图能保留不等于能规范化 | [原覆盖](history/stage-14.md)、[后续矩阵](history/stage-23.md) |
| 分析规模 | 非单调 owner/碎片 mask/有限动态地址可能耗尽预算；不承诺任意合法实例默认预算内完成 | [阶段 25](history/stage-25.md)、[阶段 26](history/stage-26.md) |
| 数学查询 | 完整投影不等于 CAS 可判定；复杂条件、大型查询可能超过默认 60 秒 | [代数边界](../algebraic-equivalence.md#当前边界) |
| 第一版之外 | 一般多轴 mesh 全局值组装/协议证明、全部 JAX 消费者统一迁移继续单列 | [当前计划](../../exec-plans/active/pallas-semantic-layer.md) |
| 用户排除 | `core_map`、`mpmd_map`、输入输出共享 HBM 暂不支持 | 不作为近期完成阻塞项 |

本表是语义规范化/证明覆盖边界；不得据此断言相应源图都无法结构比较，也不能把拒绝等同程序不等价。
历史包含受限 alias 试验，不能据此恢复一般输入输出共享 HBM 的支持承诺。

## 证据必须分别报告

| 证据 | 能说明什么 | 不能替代什么 |
| --- | --- | --- |
| 精确源图/规范化比较 | 支持范围内图及条件符合规则 | 数学不等价所需反例/完整 CAS 结论 |
| 来源证据重放 | 条件、原图与分析结果保持绑定，篡改被拒绝 | 独立算法验证 |
| 独立 oracle | 独立 NumPy 完整输出、有限逐点/逐单位状态机发现实现错误 | 任意输入的精确证明 |
| 官方 CPU interpreter | 具体场景的完整数值及启用的 race/bounds 检查 | TPU 硬件行为、一般正确性 |
| trace/lower/IR verify/AOT | 原始源码在该后端可编译；Executable 生成 | TPU 实际执行、精度或性能 |

每条新规则覆盖正例、真实反例、拒绝路径与条件/证据篡改；测试不能只复述实现。
针对 grid/memory/通信改动优先相关 core 测试，再跑受影响 CPU/AOT；只有新增变化/失败才扩大或重复验证。
独立参考不调用被测坐标/状态 helper。保留失败批次及修复后的源码/范围，去重 pytest ID 后报告覆盖；
不同源码阶段、重叠批次不能相加称为单次全量通过。

测试脚本按 DRY 保持运行设施唯一来源：CPU 隔离 worker 复用
[pallas_oracle.py 的 run_worker](../../../tests/pallas_oracle.py)，离线编译复用
[pallas_aot.py](../../../tests/pallas_aot.py)；新增 case 只填源码、环境和预期差异参数。
独立 NumPy oracle、关键反例与业务断言继续独立，不能用生产 helper 算 expected，
也不能为减少重复而合并掉不同准入/拒绝路径；[整理验收](../../exec-plans/completed/test-doc-context.md)按需查阅。

CPU 的 `run_cpu` 入口与所属 case 放在一起；只有复用或独立职责明确时才新建测试模块。
在已安装 libtpu 的环境中，用 `python -m tests.pallas_aot --suite remote_grid --output /tmp/remote-grid.json`
编译选定套件，`--suite all` 运行全部。旧文件到新入口的[映射与验收](../../exec-plans/completed/test-file-consolidation.md)按需查阅。

## 已确认的源后端限制

锁定环境为 Python 3.12、JAX/jaxlib 0.11.2；近期 AOT 使用 libtpu 0.0.48.1、离线 `v5:2x2x1` megacore。
没有 TPU 实机精度/性能结论；用户确认图与编译工作无需实机。

- TPU lowering 与 TPU 专用 CPU interpreter 仍拒绝 `load_p(mask=...)`；通用 CPU Pallas 可做相应数值对照。
  不改写原源码绕过后，再声称该原源码编译成功。
- TPU CPU swap race checker 使用整个 view 的写足迹，未按 mask 裁剪；阶段 26 一个不相交 store/DMA
  样例仅关闭这项检测做数值对照，该次运行不能称为解释器无竞争证明。
- 交错串并轴的 CPU interpreter 有 prefix 限制；阶段 27 同源码 native AOT 成功，两个限制回归不计数值成功。
- 默认/IMMEDIATE 非连续重访、双缓冲 ANY、部分 Ref transpose 等源限制与本项目拒绝分别记录。

## 最近验证与详细日志入口

| 阶段 | 准确摘要 | 打开原文的理由 |
| --- | --- | --- |
| [27：远端多维 grid](history/stage-27.md) | 分批去重 1,183 项；6 份 AOT；CPU 20 项中 18 数值/race + 2 源限制 | 两份协议、单 core 反例、绑定篡改、原始日志 |
| [26：尾块/mask](history/stage-26.md) | 分批去重 2,167 项；最终受影响 423 项 + 共享 core 368 项 | 初次 2 失败及修复、四缓冲预算、mask oracle、原生拒绝 |
| [25：并行 ANY 尾块](history/stage-25.md) | 1,975 不同 pytest ID；9 份 AOT | 72 次完整参考、有限域 13,966 点、owner 证明与篡改 |
| [24：ANY/远端原存储](history/stage-24.md) | 原 Ref、多维足迹与恢复的独立对照 | padding 生命周期、线性协议地址与真实 shape |

这些是当时的验证事实，不是本次文档拆分重跑的测试；更早记录按[历史索引](history/index.md)定位。
原始工件多位于当时 `/tmp` 目录，路径保留供本机复查，但长期可复现范围以仓库源码/固定测试与记录命令为准。

## 维护门槛

改算法时在专题记录准入、反例和证明义务，在独立阶段记录命令、结果、失败与源码状态；入口只更新短链接。
新增/移动文档检查本地目标与 fragment，运行 `uv run python tools/check_docs.py --changed`、`git diff --check`。
不为压缩上下文删除历史；不以 HTML 折叠代替文件分层。需要准确失败/限制时定向加载原文。
