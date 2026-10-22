# 串行 kernel 后计算下一输入 index_map

## 目标与用户决定

用户明确规定：当前 kernel 完成状态更新后，再计算下一 program 的 index_map。
本任务把这一逻辑执行顺序接到既有 grid 展开与版本化存储；不把默认自动流水线的
实际预取时序视为已证明。公开语义/数学结果保留该执行前提。

## 设计与范围

- 首个 program 使用实际 scalar-prefetch 初值；每步先用当前状态计算输入地址，
  加载输入，执行 kernel，再以更新后状态计算下一步。地址不因随后状态写入而改变。
- 复用 tensor executor 求值 index_map，只允许对 prefetch 的只读 effects。
  先支持串行有限 grid、每步可精确求定的整数输入块地址；未知初值、越界、有效并行轴、
  独立 program/core 回放继续拒绝。不得假设初值为零或改为输入 HBM 写回。
- 输出 index_map 本轮保持纯静态规则；不新增状态驱动 writer 的覆盖/重访证明。
  每个当前输出窗口仍在 kernel 前绑定，不能用 kernel 更新后的状态移动写回目的地。
- grid 规划阶段可延迟输入域，但每步都验证真实块范围；最终证据保存实际全部 program
  的有限映射及范围证明，不能导出临时占位域。只处理语义层，不依赖 SymPy。
- 结果的 assumptions 保留串行映射前提，证据重放也检查它，避免把条件性结论当作
  自动流水线的无条件等价。默认未使用该路径的结果保持原有前提。

## 工作项

- [x] 接入串行输入 map 求值、状态版本和完整输入域证据。
- [x] 传播与重放执行前提。
- [x] 固化非零初值、步长、错误状态更新以及并行/越界/未知/输出状态映射拒绝。
- [x] 独立 Python/NumPy 全坐标 oracle、受影响回归、静态检查与文档检查；归档。

## 验证

复用 `tests/pallas_source_cases.py` 与 `tests/test_pallas_source_coverage.py`。
独立 oracle 按用户规定的顺序计算，不使用会丢弃 scalar-prefetch 更新的普通
`interpret=True` 作为该状态语义的参考。不把该顺序模型的验证称为 TPU 实机验证。

## 进展

- 既有 tensor executor 在每个 program 的 kernel 前计算输入 maps，读真实 prefetch
  存储版本，再验证范围并记录输入 staging 版本。后一步使用前一步更新后的状态。
- grid 规划中的临时输入域只提供存储形状，任何 kernel 消费前均替换为本步实际域；
  最终证明由完整有限映射表重建。独立复审确认没有占位域成功返回路径。
- 新增 `MemoryProjection.assumptions`，语义、局部数学及完整函数数学入口传播，
  证明重放拒绝删改前提；投影聚合也保留该字段。
- 首批 prefetch 17 项通过，含三组各 384 坐标的独立 oracle、两个错误改写、两种前提
  篡改及精确拒绝原因。独立负步长、零步长、双 Ref、单例并行轴、只读映射五案共
  1,920 个坐标通过。常量数组初值的额外安全拒绝已复现，继续补齐同值不同写法。
- public 三模块阶段回归 112 项通过、7 项通信标记取消选择，无失败/跳过；这批在常量
  数组修复前完成，后续单独复验受影响分支。日志保存于 `/tmp/jax-reorder-ordered-index-map/`。

## 最终验收

- 补齐 `jnp.array` 整数数组初值：仅新路径的 prefetch 标量读取可按原 shape、每轴
  坐标、整数 dtype 和完整字节长度取精确值；浮点、未知坐标与一般 cast 保持原处理。
  初值 0 的正向与初值 2 的反向两个源码，通过三个公开入口及全部输出坐标对照。
- 相关状态六模块 332 项通过、32 项通信用例取消选择（179.35 秒）；public 三模块
  112 项通过、7 项通信用例取消选择（110.69 秒）。两批没有失败、错误或跳过。
  常量修复只影响新增状态映射路径，修复后完整 prefetch 专项 fresh 25 项通过
  （13.81 秒），含参数改名和重复覆盖，不能把三批直接相加当作不同用例数。
- 独立 `(2,3)` 全串行 grid 的 768 个输出坐标通过，实际映射为
  `3 * program_id(0) + program_id(1)`；此前五案 1,920 坐标也通过。
- 全库 Ruff/格式检查通过（400 个 Python 文件），basedpyright 零错误/警告，tracing
  示例报告 equivalent，文档链接与 `git diff --check` 通过。
- 分批 JUnit、最终文件哈希与测试 ID 保存于
  `/tmp/jax-reorder-ordered-index-map/manifest.json`，常量专项原始证据为
  `/tmp/indexmap-prefetch-final.xml`。旧拒绝复现保留，不冒充此前已经支持。

重新执行本轮测试：

```bash
JAX_PLATFORMS=cpu PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_pallas_source_coverage.py tests/test_pallas_local_entry.py \
  tests/test_pallas_local_values.py tests/test_pallas_grid_state.py \
  tests/test_pallas_tensor_memory.py tests/test_pallas_dynamic_ranges.py \
  tests/test_pallas_dynamic_views.py tests/test_tensor_owner_partition.py \
  tests/test_pallas_core_execution.py -m 'not pallas_communication'
```

## 保留边界

这是用户指定顺序下的条件性等价；没有将自动预取的实际执行时序证明为该顺序。
输入 map 仅允许只读 prefetch，且每一步必须可确定为合法整数块地址。输出状态映射、
未知初值、有效并行共享状态及逐 program 独立 core 回放仍拒绝。不会补隐式 barrier
或 wait；原显式 DMA 的初始化、在途冲突与完成义务继续验证。
