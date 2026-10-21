# Pallas 比较审查修复

完成于 2026-10-21。

## 目标与完成条件

修复支持范围内的两项问题：一维 reduce/dot 不得丢失收缩范围；已支持的远端并行
执行绑定能传入公开代数比较。保留结构/代数结论、协议有效性和外部执行条件的区分。

## 设计与反例

- `sum/max/dot` 的一维块也使用已有 tensor 坐标与输入范围模型，不依赖 rank 决定
  运算是否逐元素。纯值快路径不能把未经审核的跨 lane 运算当作逐元素表达式。
- 长度 256、每块求和后广播：block=128 与 256 输出不同，内部比较不能返回等价；
  公开代数入口必须能处理已支持的 reduction 参数，正例继续可判等。
- 公开代数与函数入口传递单绑定或左右两侧绑定，投影证据和假设保留它们；显式请求
  的绑定不能被结构快路径忽略，缺失、非法或不适用的绑定明确拒绝。
- 使用现有二维/展平远端 grid 正例接通完整链路；错误绑定和数据差异仍不得判等。

## 工作项

- [x] 修复一维跨 lane 路由与逐元素边界，补直接/control 路径的正反例。
- [x] 接通公开代数、函数入口的执行绑定及证据，补正例与拒绝路径。
- [x] 审查差异，运行受影响回归、Ruff、格式、tracing 与文档检查。
- [x] 更新简短专题的当前约束，记录验证后归档计划。

## 验证

复现基线见 `/tmp/jax-reorder-review-confirm.py`。回归复用既有测试文件，CPU 数值
参考保留独立 NumPy 结果。无实机需求；本任务不提交或推送。

## 已确认结果

- 一维 `sum/max/dot` 的直接/cond 路径、外层实参均保留收缩范围；6 组不同分块反例
  在内部明确需要数学后端，公开代数返回 `inequivalent`，相同范围的重排仍可判等。
- 远端二维/展平 grid 的左右绑定可经 jaxpr、function、trace 三入口进入公开代数。
  显式绑定同源时检查协议、保存证据后仍返回 structural，无需 CAS；函数限定 auto + forward。
- 核心结构入口 `compare_program_views` 同源且显式绑定时也校验协议并保留规范化证据。
- 分批覆盖 **2,085 个不同 pytest ID**，全部通过；不是单次全量测试。没有实机精度/性能结论。
- 全仓 Ruff、格式、basedpyright、tracing 示例、文档链接及 `git diff --check` 通过。

| 批次 | 通过数 | 范围与去重 |
| --- | ---: | --- |
| 归约与投影 | 244 | tensor_memory、algebra、local_entry、semantic_slice、remote_values 五模块，含 22 项通信标记用例 |
| 公开绑定与函数 | 106 | remote_grid_binding、functions 两文件分批覆盖最终收集清单；同一增强用例重跑只计一次 |
| 扩大 Pallas 与错误通道 | 1,452 | 其余全部 test_pallas 文件及 comparison_errors；排除上述六个 Pallas 模块，246 项通信标记用例未执行 |
| 普通代数 | 279 | test_algebra_structural、test_algebra |
| 包与架构 | 4 | package_layout；同时运行的 41 项 core_execution 已计入扩大批次 |

归约批次命令：`pytest -q tests/test_pallas_tensor_memory.py tests/test_pallas_algebra.py
tests/test_pallas_local_entry.py tests/test_pallas_semantic_slice.py tests/test_pallas_remote_values.py`。
扩大批次按上述文件范围运行 `pytest -q ... -m 'not pallas_communication'`，完整选中清单及结果
保存在 `pallas-regression.xml`；耗时 773.44 秒。最终去重清单为 `coverage-manifest.json`。

验证摘要与完整日志在 `/tmp/jax-reorder-review-fixes-20261021/`；P1 原始摘要另在
`/tmp/jax-reorder-p1-regressions.txt`，P2 分批摘要在 `/tmp/jax-reorder-p2-validation/`。
只有重定向运行的批次保留完整日志，其他记录是工具返回的准确摘要。

两项审查复现及附加拒绝检查已独立重跑，结果见 `review-confirm.log` 和
`review-boundaries.log`。当前支持范围与拒绝边界见
[契约](../../development-docs/pallas/contracts.md)和[内存](../../development-docs/pallas/grid-memory.md)。
