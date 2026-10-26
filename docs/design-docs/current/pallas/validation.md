# Pallas 验证与当前边界

从[任务入口](../pallas-jax-equivalence.md)到此；当前范围于 2026-10-26 对照源码与固定测试核对，
未声明任意合法 Pallas 语法全覆盖。2026-10-22 及更早验收、锁定后端观察已移至
[历史快照](../../../exec-plans/completed/pallas-history/validation-snapshot-2026-10-26.md)，本次文档整理未重跑这些测试。

## 已确认的正确性缺陷

2026-10-26 审查复现整数归约/点积溢出被数学层遗漏，普通 JAX 与 Pallas 完整函数、设备局部数学入口均受影响。
例如 `int8` 常量归约的回绕结果会与无回绕常量被错误判等价；浮点实数解释警告不覆盖这项整数语义错误。
修复和回归要求登记为 [ALG-006](../../../exec-plans/active/pallas-semantic-layer.md#2026-10-26-审查确认的正确性缺陷)。现有验收正例不能说明该边界正确；
涉及整数归约/点积且未排除溢出的数学结果需按此缺陷核对。

## 当前缺口与排除项

| 范围 | 当前限制 | 处理位置/证据 |
| --- | --- | --- |
| 数据条件证明 | 远端 masked store 的条件未定义状态、互补写入和受保护读取已有受限支持；实际消费仍需证明，任意不同数据谓词之间的等价关系不作猜测 | core；[2026-10-22 验收](../../../exec-plans/completed/pallas-local-equivalence-closure.md) |
| 状态驱动映射 | 串行输入/输出 index_map 按上一 kernel 完成后的状态求值，结果显式保留该执行前提；活跃未知地址、有效并行共享状态仍拒绝，纯死计算只在数学 lowering 删减 | core；[顺序映射与删减](../../../exec-plans/completed/pallas-output-map-and-pure-lowering.md) |
| 动态页 DMA | 只读输入的有界页号支持多笔同时在途；动态可写源/目标仍需完整状态分支证明，未知范围及任意数据控制流未普遍准入 | core 与数学地址 lowering；[范围与验收](../../../exec-plans/completed/pallas-bounded-page-dma.md) |
| 并行自动 staging | 独立 owner 间实际共享缓冲发生冲突则拒绝；不能把共享 Ref 私有化 | core；[阶段 27](../../../exec-plans/completed/pallas-history/stage-27.md) |
| 执行条件 | 普通调用不自动提供 core 绑定；远端并行要求完整单 core 点流与各设备同序 | [执行绑定](communication.md) |
| 索引/控制 | 远端非恒等向量索引、未证明动态地址、数据决定路由或终止仍非普遍支持 | core；[当前计划](../../../exec-plans/active/pallas-semantic-layer.md) |
| 布局/源形态 | 部分布局编译选项已审核；一般非默认物理布局、动态 grid、未审核块维度/包装/effect 等仍需独立准入；源图能保留不等于能规范化 | [选项回归](../../../../tests/test_pallas_dynamic_views.py)、[原覆盖](../../../exec-plans/completed/pallas-history/stage-14.md)、[后续矩阵](../../../exec-plans/completed/pallas-history/stage-23.md) |
| 分析规模 | 非单调 owner/碎片 mask/有限动态地址可能耗尽预算；不承诺任意合法实例默认预算内完成 | [阶段 25](../../../exec-plans/completed/pallas-history/stage-25.md)、[阶段 26](../../../exec-plans/completed/pallas-history/stage-26.md) |
| 数学查询 | 完整投影不等于 CAS 可判定；复杂条件、大型查询可能超过默认 60 秒 | [代数边界](../algebraic-equivalence.md#当前边界) |
| 第一版之外 | 一般多轴 mesh 全局值组装/协议证明、全部 JAX 消费者统一迁移继续单列 | [当前计划](../../../exec-plans/active/pallas-semantic-layer.md) |
| 用户排除 | `core_map`、`mpmd_map`、输入输出共享 HBM 暂不支持 | 不作为近期完成阻塞项 |

本表是语义规范化/证明覆盖边界；不得据此断言相应源图都无法结构比较，也不能把拒绝等同程序不等价。
历史包含受限 alias 试验，不能据此恢复一般输入输出共享 HBM 的支持承诺。
2026-10-22 的设备局部入口、常见源码、条件初始化及开源数学完成率修复，见
[修复与循环审查验收](../../../exec-plans/completed/pallas-local-equivalence-closure.md)。

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
[pallas_oracle.py 的 run_worker](../../../../tests/pallas_oracle.py)，离线编译复用
[pallas_aot.py](../../../../tests/pallas_aot.py)；新增 case 只填源码、环境和预期差异参数。
独立 NumPy oracle、关键反例与业务断言继续独立，不能用生产 helper 算 expected，
也不能为减少重复而合并掉不同准入/拒绝路径；[整理验收](../../../exec-plans/completed/test-doc-context.md)按需查阅。

CPU 的 `run_cpu` 入口与所属 case 放在一起；只有复用或独立职责明确时才新建测试模块。
在已安装 libtpu 的环境中，用 `python -m tests.pallas_aot --suite remote_grid --output /tmp/remote-grid.json`
编译选定套件，`--suite all` 运行全部。旧文件到新入口的[映射与验收](../../../exec-plans/completed/test-file-consolidation.md)按需查阅。

## 维护门槛

改算法时在专题记录准入、反例和证明义务，在独立阶段记录命令、结果、失败与源码状态；入口只更新短链接。
新增/移动文档检查本地目标与 fragment，运行 `uv run python tools/check_docs.py --changed`、`git diff --check`。
不为压缩上下文删除历史；不以 HTML 折叠代替文件分层。需要准确失败/限制时定向加载原文。
