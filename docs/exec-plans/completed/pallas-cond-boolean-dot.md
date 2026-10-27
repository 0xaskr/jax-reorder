# Pallas 条件义务与布尔点积修复

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

## 目标与边界

修复上一轮已复现的两项问题：相同分支的动态 cond 丢失条件义务，以及布尔
dot_general 被送入数值乘加模型。保留原有改动，不变更受保护文档、依赖或支持边界。

## 规则与反例

- cond 的 selector 仍是控制依赖；即使所有分支返回/写入相同值，也必须保留其
  原始定义条件和内存有效性。尾块 padding 参与求和后控制 cond 应拒绝；
  `log(x) > 0` 应保留 `x > 0`。这不是未使用纯计算的删减。
  local/tensor 两条路径共用同一契约，合法的全定义条件仍可得到代数等价结论。
  实现用独立、带内部标记的 select DAG 保存控制义务，跨写入覆盖与串行迭代保留，
  嵌套分支按可达条件合并；不向每个 Ref 复制 guard，原值合并优化保留。
  switch 选择器的静态有序整数 clamp 在内部 guard 中精确解释，避免新增检查导致
  合法 switch 退化；仅在已证分支穷尽时消去共同项，selector 自身的定义与内存条件
  仍独立保留。普通整数算术政策不变。
- bool dot 采用 AND/OR 收缩，空收缩为 False；保留自由轴、batch 轴、完整输入域
  与原定义/内存条件。显式数值 accumulator 使用 0/1 乘加计数；普通数值 dot 不变，
  有限归约共用先扣预算的枚举规则；未建立模型的 dtype/参数
  继续明确拒绝，不以内部 TypeError 代替准入判断。
- 空域验证另发现已存在的 empty reshape 坐标除零：外层 `(8,0)` reshape 后求和、
  广播并送入 Pallas 的合法源码触发内部异常。空 view 没有实际元素坐标，应在
  形状/置换校验后保留空域含义，不能执行除零地址计算；用非空归约输出验证。

## 工作项

- [x] 修复 local/tensor 条件合并，补安全正例及原始定义性反例。
- [x] 修复布尔点积，补普通 JAX/Pallas、空域/批量轴、错误候选及拒绝路径。
- [x] 修复空域检查发现的 reshape 坐标除零，补公开入口正反例。
- [x] 独立复审、相关回归、离线 TPU AOT、静态检查；记录证据后归档。

## 验证

原反例在 `/tmp/review_branch_condition_validity.py`、
`/tmp/review-local-selector-domain.py` 和 `/tmp/review-bool-dot.py`。
两类源码此前均已通过离线 TPU 编译；修复后重新检查最终测试源码。
本轮日志、快照与最终校验清单写入 `/tmp/jax-reorder-cond-bool-dot-fix/`。
使用现有 Python 3.12 venv；CPU/NumPy 数值对照、公共比较结果、离线 AOT 分别报告，
没有 TPU 实机执行。

目标测试：cond 33 项、switch 4 项、bool dot 24 项、empty reshape 11 项通过。
独立 switch 复审的 6 组公开比较、21 组 CPU 边界对照通过，定义与内存条件未被
共同项化简丢弃。最终源码离线编译已通过 cond 29 份、switch 4 份、bool dot 50 份、
empty reshape 24 份，共 107 份；
日志分别在 `/tmp/review-cond-guards-aot.log`、本轮目录的 `dot-aot.json` 与
`dot-count-constant-aot.json`、`/tmp/review_empty_reshape_matrix_aot.json`，
switch 记录在 `/tmp/review-cond-switch-final-aot.log`。
回归按最终测试 ID 去重，共 **1708 项通过**，其中本轮新增 72 项；这是分批验证，
不是一次全仓测试运行。命令、各批 JUnit 和文件 SHA256 见本轮目录的
`final-verification.json`，增量审查差异见 `incremental-review.diff`。

| 批次 | 结果 |
| --- | --- |
| 普通代数与显式符号 | 738 passed |
| Pallas 代数、padding、远端值、公共接口 | 362 passed |
| core/协议/内存初次批次 | 600 passed，4 个旧工厂失败，见下文 |
| 最终受影响的 grid/state、local scope、tensor memory | 209 passed |
| switch 最后化简后，完整 grid/state 再验 | 82 passed |

初次 core 批次启动时加载了旧空布尔归约工厂，4 个用例的数值求和表达式发生 CAS
未决。最终 bool 工厂使用空布尔归约，float 工厂保留空数值归约；11 项定向测试
及上述完整受影响模块均通过。去重结果使用复验记录覆盖旧结果，未把中间失败
计入通过。原空 dot/reshape 探针也已从内部异常恢复为正确的等价结论。

`.venv/bin/ruff check .`、`.venv/bin/ruff format --check .`、
`.venv/bin/basedpyright`、`.venv/bin/python tests/inspect_jaxpr.py`、
`.venv/bin/python tools/check_docs.py --changed` 与 `git diff --check` 通过。
本轮以进入时快照为边界复核，其他 27 份已有文件保持原 SHA256；未改受保护文档、
依赖与环境配置，未提交或推送。

已知 CAS 边界：OR 与浮点计数 `>0` 的原始大表达式对照可能达到 60 秒超时；
动态数值计数 `+1` 的负例可能未决。修复不扩大 CAS，明确布尔归约正反例、数值计数
正例及常量计数与 OR 的反例均通过；不能把这些未决报告成已证明等价或不等价。
