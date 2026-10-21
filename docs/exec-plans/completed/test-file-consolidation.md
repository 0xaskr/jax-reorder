# 测试文件按职责合并

状态：2026-10-21 已完成。用户明确要求删除冗余测试文件、合并零散脚本；在上一轮 DRY 设施整理基础上减少文件碎片。

## 范围与取舍

- 六组离线 AOT 场景使用一个可选择 suite 的命令入口，删除被取代的包装文件。
- 小型 CPU worker 与所属 case 放在一起，删除只负责转发或单场景执行的独立脚本。
- 按验证职责合并阶段性 review/guards/补丁测试；避免仅按行数拼成大文件。
- 独立正反例、参考实现、参数化和标记保持；真实重复项只有核对覆盖后才删除。
- 保留生产代码、受保护文档及已有用户改动；修正允许修改文档的路径引用。

## 完成条件

- [x] 删除被合并文件，维护调用和复现入口。
- [x] 核对旧→新测试 ID、参数与断言；检查 helper 冲突、模块导入和标记。
- [x] 运行受影响测试、CPU worker 和离线 AOT；分别报告实际通过与预期拒绝。
- [x] 完成 Ruff、文档目标/fragment、差异及生产哈希检查，归档本计划。

## 证据

开始时的工作区快照：`/tmp/jax-reorder-test-consolidation-7d10w__h/`。
基于工作区内容比较，不覆盖或重新暂存用户已有的 Git index。

## 文件归属

测试目录 Python 文件从 158 个减至 136 个：删除 23 个被替代文件，新增 1 个统一入口。
总行数 30,228→30,035，净减 193 行；没有删除独立测试场景。

| 原入口 | 现入口与职责 |
| --- | --- |
| `pallas_aot_support.py` 和六个 `pallas_*_aot.py` | [pallas_aot.py](../../../tests/pallas_aot.py)，统一编译入口 |
| 七个场景专用 CPU worker | 对应 `*_cases.py` 的 `run_cpu`，场景与独立数值执行就近维护 |
| `pallas_communication_extended_worker.py`、`pallas_staging_worker.py` | 对应 `test_pallas_communication_extended.py`、`test_pallas_remote_staging.py` 的 `run_cpu` |
| `test_pallas_local_reshard_guards.py`、`test_pallas_local_reshard_review.py` | [local_reshard](../../../tests/test_pallas_local_reshard.py)，局部 reshard 准入与证书反例 |
| `test_pallas_remote_grid_staging.py` | [remote_grid](../../../tests/test_pallas_remote_grid.py)，展开、完整输出与存储恢复 |
| `test_pallas_remote_grid_review.py` | [remote_grid_binding](../../../tests/test_pallas_remote_grid_binding.py)，owner、绑定和进展 |
| `test_pallas_remote_nd_padding.py` | [remote_nd](../../../tests/test_pallas_remote_nd.py)，多维远端内存与 padding |
| `test_pallas_padding_large_cas.py` | [padding_algebra](../../../tests/test_pallas_padding_algebra.py)，尾块完整输出代数证明 |
| `test_pallas_storage.py` | [dma_regions](../../../tests/test_pallas_dma_regions.py)，区间存储独立 oracle 与 DMA 版本 |

七个场景 worker 为 buffered_revisit、composed_dma、local_reshard、remote_grid、remote_mask、
remote_nd、remote_revisit；调用改为 `run_worker("tests.pallas_<场景>_cases", ..., entry="run_cpu", expected=...)`。
较大的共享通信 worker 和有独立语义的参考实现继续保留。

原 `python -m tests.pallas_<suite>_aot` 改为 `python -m tests.pallas_aot --suite <suite>`。
suite 可选 buffered_revisit、composed_dma、dynamic_range、remote_grid、remote_revisit、scheduling 或 all。
单套件 JSON 字段和接受状态保持；all 输出按套件分组的 JSON 对象。
scheduling 保留原默认输出 `/tmp/pallas-scheduling-aot.json`，其他入口显式传 `--output`。
历史文档保留当时的命令拼写，当前复现使用以上映射。

## 已完成审计

- 3,941 个 pytest 实例全部保留，其中 65 个按文件移动，旧→新 ID 一一匹配，无新增或丢失。
- 合并模块的 106 个函数/类 AST 与全局名字绑定一致；两个含义不同的 helper 分别保留为
  `native_trace` 和 `remote_at_point`。两项存储 oracle 函数 AST 原样保留。
- 9 个 CPU 执行体只删 argv/CLI 适配；原参考、READY/RESULT、参数、expected、超时和设备数保持。
  环境仍由公共 runner 在子进程 import 前设置。
- 新入口 45 个离线 AOT 记录与上一轮已核对哈希的基线完全一致：40 编译成功、1 预期拒绝、
  4 source_rejected；60 组阶段故障对照、450 条记录一致。未执行 TPU 硬件。
- 生产和受保护文件共 93 项哈希不变；实现/工具/CI 无已删除入口的遗留引用。

工件中的 `complete-node-map.json`、`cpu-worker-migration.json`、`test-module-map.json`
记录精确映射；`aot-merge-summary.json` 与 `aot-contracts.json` 记录编译和失败对照。
`affected-command.json`/`affected-test-files.json` 保存实际回归命令与文件列表。

## 最终验证

受影响 12 个测试模块一次运行：**583 passed in 384.17s**，含本轮迁移的所有 CPU 执行入口、
合并后的纯结构/代数/拒绝路径及区间存储 oracle；日志为 `affected-pytest.log`。
全仓库仅做 3,941 项收集核对，不将其称为全量执行通过。
Ruff、381 文件格式检查、本地 Markdown 目标、52 个 fragment 和 `git diff --check` 全部通过。
源码、依赖与受保护文档保持初始哈希；未提交或重新暂存现有改动。
