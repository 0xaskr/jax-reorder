# 带通信的 jaxpr 验证

## 目标与完成条件

在当前 CPU 范围内实际尝试带通信的 jaxpr，提供可运行、可查看原图与重排图的示例，并把结果固化为回归测试。验证独立通信换序、通信与计算组合、通信参数和依赖变化；分别报告图结论与数值结果。

## 设计与取舍

- 使用完整 `shard_map` 外层图，保留 mesh、设备顺序、分片规格和子图轴环境。
- 默认两个逻辑 CPU 设备，另在四设备运行同一示例，避免单设备通信退化。
- 不引入代数规则。即使 all-reduce 与 reduce-scatter + all-gather 数值相同，图仍不同。
- 不把 GPU 点对点通信或多主机执行算作 CPU 验证通过。
- 设计与实验见 [通信记录](../../development-docs/collective-jaxpr.md)。

## 工作项

- [x] 核对现有 collective 原语、轴环境适配和 CPU 限制。
- [x] 添加中文注释的独立通信自检示例及正反例。
- [x] 回归验证图判断、嵌套重排、幂等性与 CPU 执行。
- [x] 同步中英文 README、开发索引与验证记录，审查差异并归档。

## 决策与进展

现有逐原语用例已覆盖 psum、pmin、pmax、all_gather、all_to_all、ppermute、reduce_scatter；不足在于通信之间及通信与计算的真实组合。优先补足这些证据；若发现规范化缺陷，再修实现。

2026-10-11 完成。组合暴露出默认 check_vma 路径下缺少 core.pvary_p 的身份登记，已按锁定 JAX 实现显式加入；没有更改 BFS 规则或放宽未知语义检查。嵌套用例保留变化类型约束和接口顺序，cond 闭包捕获换序另存为图不同但样本相同的案例。

后续修正：该 cond 闭包案例在 [绑定规范化](../../development-docs/cond-bindings.md) 中修复，现应判为 equivalent；上述内容和下方数量保留首次验证的历史记录。

## 验证结果

- `uv sync --locked`：通过。
- `uv run python examples/collective_jaxpr.py --show-jaxpr`：2 个 CPU 设备，22 个组合 × 4 组输入，通过。
- `uv run python examples/collective_jaxpr.py --devices 4`：4 个 CPU 设备，同样 22 个组合 × 4 组输入，通过。
- `uv run pytest tests/test_collectives.py -q`：49 passed。
- `uv run pytest -q`：565 passed。
- `uv run ruff check .`、`uv run ruff format --check .`：通过。
- `uv run python examples/inspect_jaxpr.py`、`uv run python tests/lax_inventory.py`：通过。
- 四设备脚本已加入 CI；只报告本地验证，不宣称线上 CI 已运行。

## 未验证条件

多主机、GPU/TPU 和异步点对点通信不在本轮单进程 CPU 实验范围内；没有据此宣称这些场景支持。CPU 既有拒绝用例继续通过。未使用通信保留为 unknown，沿用已有 BFS-001 边界。
