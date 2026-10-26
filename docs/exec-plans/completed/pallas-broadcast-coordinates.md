# Pallas 一维广播坐标修复

## 目标与完成条件

修复长度为 1 的向量参与隐式广播时丢失输入 shape、使不同 Pallas 程序被误判等价的问题。覆盖 kernel 内部控制流及外层 JAX 输入绑定；保留已有未提交改动。

## 设计与取舍

按输入和输出 shape 识别需要广播坐标的运算，复用语义层的 tensor 投影，将 singleton 输入坐标固定为 0。不能仅特判 `iota(1)`；相同 shape 的逐元素计算继续使用已有路径。不涉及 SymPy 广播规则或新增公共 API。

## 工作项

- [x] `tensor_domain.requires_tensor_coordinates` 统一 kernel 与外层绑定的 shape 判定；嵌套 region 分别建立类型表。
- [x] 两份既有测试文件增加 20 项回归：iota/输入/常量，outer/direct/cond，隐式与显式广播及独立 NumPy/CPU/投影参照。
- [x] 相关回归、静态检查通过，更新[精简开发说明](../../design-docs/current/pallas/grid-memory.md)。

## 验证结果

修复前：`x + arange(1)` 与 `x + arange(128)` 在外层绑定和 kernel cond 两种位置均误报等价；CPU 数值不同，四个原始程序均已通过离线 TPU 编译。复现脚本：`/tmp/jax-reorder-review-broadcast-confirm.py`。

修复后：输出坐标不同，核心比较返回 `cas_undecided`；公开代数入口返回 `condition_undecided`。
整数坐标转 float 的原始条件尚不足，不能宣称这些反例已得到 `INEQUIVALENT`；此限制属于既有数学后端边界。
9 个隐式/显式广播正例均返回等价；负例严格断言具体未决分类，CPU 结果确认不等。

后续[有界整数 cast 修复](pallas-bounded-integer-casts.md)已使其中 3 个公共代数广播反例
明确返回 `INEQUIVALENT`；上述未决结论保留为本阶段的历史结果。

2026-10-21 验证：

- 新增广播回归 20 passed；13 个受影响模块的已有回归 607 passed；包分层检查 4 passed，共 631 项。
- 新增回归命令：`.venv/bin/python -m pytest -q tests/test_pallas_tensor_memory.py tests/test_pallas_algebra.py -k rank_one_broadcast`。
- 既有回归模块：tensor_memory、algebra、local_entry、semantic_slice、grid_expansion、grid_extended、masked_access、local_scope、dynamic_views、padding_domains、ref_composition、remote_values、functions。
- Ruff、format、basedpyright、tracing 示例、文档链接及 `git diff --check` 全部通过。
- 6 个 outer/direct/cond 正反例源码均通过离线 TPU v5:2x2x1 megacore 的 trace/lower/IR verify/backend compile；未执行 TPU kernel。
- 独立探针覆盖嵌套 fori_loop、外层常量/输入和非零切片偏移。256 与 2²⁴ 元素的并行循环均消耗 38 单位分析预算；非零切片保留 program 基址与偏移。

验证日志集中在 `/tmp/jax-reorder-broadcast-fix/`；新增回归日志为 `/tmp/jax-reorder-broadcast-tests.log`。
此前未提交的两项修复完整保留，本轮没有提交或推送。

## 后续边界

审查另确认外层快速路径的既有范围缺口：shape `(128,)` 的输入在 identity Pallas 调用前计算
`x + (jnp.arange(128, dtype=jnp.int32) + 3).astype(jnp.float32)`，CPU 执行合法，
`normalize_view` 却在 `binding_value → pure_value → check_integer_range` 读取缺失的
`ranges['global']` 时抛出 `KeyError`。新旧路由均复现，已登记为
[PALLAS-003](pallas-broadcast-coordinates.md#已解决问题编号)，本轮不扩展到该快速路径范围修复。

## 外层坐标范围跟进

2026-10-21 按用户后续要求修复 PALLAS-003：`binding_value` 的一维快速路径按每个
当前运算的 shape 绑定 `global ∈ [0, size)`，保留调用方设备轴范围，不修改其范围字典。
标量不添加向量坐标；输入数据的值域仍未知。没有修改代数 cast 规则。

修复前，`i+3` 与 `(i+1)+2` 在同一个 identity Pallas 调用前转 float32，两个公共入口
都报含 `KeyError: 'global'` 的 `ComparisonInternalError`；修复后
`compare_algebraic_jaxprs`、`compare_functions` 均返回等价，CPU 结果与独立 NumPy 参照一致。

- 在既有 `test_pallas_algebra.py` 新增 8 项回归：两个公共入口、128/256 元素、完整域与
  singleton 正例、block 范围及数据范围反例；反例均明确判不等。
- 新增 8 项与受影响既有 301 项全部通过，共 309 项。既有回归覆盖 pallas_algebra、
  pallas_local_entry、pallas_semantic_slice、pallas_tensor_memory、functions、pallas_local_scope、package_layout。
- Ruff、format、basedpyright、tracing、文档链接与 diff 检查通过；独立探针另验证设备轴
  `device:global`、标量广播及除余拒绝边界。
- 原始两份源码的离线 TPU 编译证据沿用前次只读审查；此次没有更改 kernel 源码或执行 TPU。
- 日志：`/tmp/jax-reorder-outer-range-fix/`；筛选新增回归使用
  `-k 'outer_vector_coordinate_arithmetic or outer_coordinate_bounds'`。

## 已解决问题编号

以下编号于 2026-10-26 从问题总表归入本计划，保留原已解决状态、优先级和依据；本次文档合并未重跑历史验收。

| 编号 | 问题 | 原优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- |
| PALLAS-003 | 外层一维快速投影已按当前运算 shape 绑定 `global` 范围，两个公共入口恢复正常；设备轴范围与未知输入数据值域保留 | P1 | 已解决 | [坐标范围修复验收](pallas-broadcast-coordinates.md) |
