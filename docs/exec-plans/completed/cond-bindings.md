# cond 内部参数绑定规范化

## 目标与完成条件

修复 gated RMSNorm 在 cond 闭包中仅重排独立计算却被判 different 的问题。同步置换 cond 实参和所有分支形参后应判 equivalent；外部参数绑定错误、分支交换及操作数变化仍须被识别。返回图须可执行、幂等，原图不变。

## 设计与取舍

见 [设计记录](../../development-docs/cond-bindings.md)。仅处理 cond 原语（包括 lax.switch）；不推广到 scan carry、while、jit 或 shard_map 的接口重排，不引入代数规则。

## 工作项

- [x] 复现错误，核对锁定 JAX 0.11.2 的 cond 类型检查与 lowering。
- [x] 保存失败回归与 RMSNorm hardcode 示例。
- [x] 在父图 BFS 前统一 cond 实参/形参绑定，保留完整语义。
- [x] 运行正反例、嵌套/通信/业务测试与全量检查。
- [x] 同步文档与旧通信案例标签，审查并归档。

## 决策与进展

2026-10-11：原版 cond 捕获 `(x, scale, gate)`，候选捕获 `(gate, x, scale)`；外部函数参数相同。现有逻辑逐位置比较调用输入，遗漏内部同步置换。已在三个签名及真假分支复现。

## 验证结果

- 修复前，RMSNorm 闭包候选的 FP32、BF16、小形状三个回归均失败，实际状态为 different。
- 修复后，`uv run python examples/cond_bindings.py --show-jaxpr`：三个候选 × 三个签名 × 十组输入，90 次样本检查通过，含真假分支、独立 NumPy 参照与两侧重排执行。
- `uv run pytest tests/test_cond_bindings.py -q`：20 passed；业务测试新增 10 项，本次共新增 30 项回归。
- 首次全量中发现测试构造污染 JAX 缓存图：破坏性测试直接修改追踪结果，后续参数化用例受到影响。改成只修改副本并断言原图不变。
- 最终 `uv run pytest -q`：662 passed（46.45 秒）。
- 双设备通信由 pytest 验证；`uv run python examples/collective_jaxpr.py --devices 4`：22 个组合通过，含修正后的闭包等价案例。
- Ruff、格式检查、tracing 示例、原语清单一致性与 `git diff --check` 通过。在线 CI 未运行。

## 未解决问题

多个全分支未使用的 cond 实参保留并返回 unknown；其他调用/控制流接口尚不做同步置换，已登记 BFS-005。
