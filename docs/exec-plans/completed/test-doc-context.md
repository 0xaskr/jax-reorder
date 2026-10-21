# 测试与文档的按需加载整理

状态：2026-10-21 已完成。目标：减少重复维护与默认 AI 上下文，同时保留语义覆盖、
独立验证和历史证据。生产实现、比较契约与受保护文档不修改。

## 设计与完成条件

- 文档入口仅提供当前状态和按任务导航，规则、验证、历史分别按需读取。
  历史原文保留；不用 Markdown 折叠块冒充上下文隔离。移动后修复相对链接。
- 测试按 DRY 原则让运行设施与配置只有一个维护来源；共享重复的进程运行、编译设施。
  新增场景仅声明差异参数；场景参数与独立 NumPy 参照
  留在各自用例中，不建立掩盖源差异的通用 kernel 工厂。
- 比较 pytest collection 的原始 node ID，保留原测试/参数与拒绝路径；
  对共享设施检查超时、失败传播及环境隔离，运行全部受影响回归。
- 核对生产代码和受保护文档哈希不变，完成格式、文档链接和差异检查。

## 工作项

- [x] 文档拆分、短入口、当前规则与历史导航。
- [x] 合并 CPU worker 的重复进程驱动。
- [x] 合并离线 AOT 的重复编译驱动。
- [x] 独立测试覆盖审计、回归与按需阅读规模核对。

## 进展与验证

初始快照：`/tmp/jax-reorder-test-doc-compression-vgsg_un5/`。
整理前 tests 按 `splitlines()` 统计为 30,475 行；Pallas 主文档 3,372 行，
语义层 active 计划 638 行。规模仅作诊断，不以删测试或删证据达成指标。

## 最终结构

- [Pallas 入口](../../development-docs/pallas-jax-equivalence.md)从 3,372 行、282,598 字节缩至
  64 行、4,999 字节；四个专题各 58–71 行。默认入口字节减少约 98%，历史不自动加载。
  当前语义层计划从 638 行缩至 45 行，详细旧计划和旧索引另存快照。
- 原主文档全部 146,774 字符按章节迁至 36 个历史文件，另存 3 个计划/索引/技术债快照。
  原文逐字核对，仅本地链接目标随文件位置调整；历史失败与后端拒绝仍保留。
- [CPU 驱动](../../../tests/pallas_oracle.py)统一 20 份测试的 23 处调用；CLI 与 JSON 函数入口
  共用环境、目录和阶段超时。原进程协议/清理逻辑不变，独立 oracle 与场景断言不变。
  统一仓库根目录使原先继承调用目录的 10 处调用也可从其他目录运行。
- [AOT 驱动](../../../tests/pallas_aot.py)统一 6 个脚本的编译阶段和 CLI；
  case 保留源码 factory、shape/dtype、sharding、输入输出数和预期拒绝。
  两部分含公共设施及新增回归，合计净减 247 行；不建立通用 kernel 工厂或插件框架。

## 验证结果与复查

| 检查 | 结果 |
| --- | --- |
| 全仓库测试收集 | 3,940 个原 node ID 全保留；仅新增 1 个 JSON 参数/环境/目录隔离测试 |
| CPU worker 回归 | 受影响 26 文件中 `pallas_communication` 标记的 236 项通过，797 项未选中 |
| 驱动错误与清理 | `uv run pytest tests/test_pallas_oracle.py`：5 项通过；与上述 236 项不重叠 |
| 调用静态审计 | 20 文件其余测试逻辑 AST、原进程协议/清理 AST、23 处 expected/设备数/阶段超时保持一致 |
| 离线 AOT 新旧对照 | `v5:2x2x1` 的 45 条记录与脚本退出码完全一致：40 编译成功、1 预期拒绝、4 source_rejected |
| AOT 故障注入对照 | 60 组阶段对照、450 条记录；阶段字段、错误、stdout、签名、sharding、topology 和退出码一致 |
| 范围核对 | 82 个生产文件、11 个受保护文档/配置/依赖文件哈希不变 |
| 仓库检查 | Ruff、402 文件格式检查、本地 Markdown 目标及 fragment、`git diff --check` 通过 |

本次未重跑全仓库 pytest，未执行 TPU 硬件；保留测试条目与实际通过数量分别报告。
原版与新版 AOT 的拒绝也逐项比较，不将源码拒绝计为编译成功。

工件目录同上：`runner-test-files.json` 给出 CPU 批次的完整文件列表，
`runner-cpu.log` 保存结果；`runner-audit.json`、`runner-invocation-audit.json` 保存调用审计。
`aot-before/`、`aot-after/`、`check_aot_contracts.py` 和 `aot-summary.json` 保存编译对照；
`audit_docs.py`、`doc-preservation.json`、`doc-audit.json` 保存原文与链接校验，
`root-invariants.json` 保存全量收集与范围核对。临时工件用于本机复查，长期维护入口为仓库中的测试和专题。
