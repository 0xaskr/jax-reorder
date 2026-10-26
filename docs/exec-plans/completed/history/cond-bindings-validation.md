# cond-bindings 历史实施与验收

> 状态：历史阶段记录，保留原失败、数字和当时接口口径。对应早期设计见 [设计归档](../../../design-docs/archive/cond-bindings.md)。

## 实施与验证结果

2026-10-11 修复完成。修复前，RMSNorm 三个签名的门控提前案例均被判 different；修复后全部判 equivalent，错误绑定仍判 different。可重放 [cond_bindings.py](../../../../examples/cond_bindings.py)（支持 `--show-jaxpr`）。九个候选/签名组合共 90 次样本检查，包含 NumPy 参照与真假分支执行。

新增 20 项 [绑定测试](../../../../tests/test_cond_bindings.py) 和 10 项 [业务回归](../../../../tests/test_business_layers.py)，全量 **662 passed**。覆盖父图生产者编号、多层 cond、三分支 switch、多输出、不同类型和形状、重复实参、常量、调试参数名称、错误绑定、未使用部分和非法图。通信闭包例同时通过双/四设备验证，四设备脚本共 22 例通过。

Ruff、格式检查、tracing 示例、原语清单一致性与 `git diff --check` 通过。修复和独立重放脚本已接入本地 CI 配置；不宣称线上工作流已经执行。更多过程与边界见 [已完成计划](../cond-bindings.md)。

## 四层复杂 cond 的额外验证

[四层重放脚本](../../../../examples/nested_cond_four_levels.py)完全由 AI 手写并固定保存，原版及正负候选均不通过脚本、AST 或 IR 变换生成。每层的真假分支都进入下一层，实际追踪结果是 15 个 `cond` 节点、最大深度 4、16 条可执行路径；不是连续调用四次浅层 cond。

案例包含共享中间结果、闭包数组常量、广播、sin/cos/tanh、平方、归约、残差，以及矩阵、行统计、整数路径编号三个输出。正例在四层内部都重排独立运算，提前不依赖嵌套结果的归约，并在第四层同步置换显式实参/形参。负例分别交换最深真分支的 sin/cos 输入、取反第三层条件。

```bash
uv run python examples/nested_cond_four_levels.py --report /tmp/four-level-cond.json
uv run python examples/nested_cond_four_levels.py --show-jaxpr
uv run pytest tests/test_nested_cond_four_levels.py -q
```

本次 [11 项回归](../../../../tests/test_nested_cond_four_levels.py)全部通过。FP32、BF16 输入与小形状三个签名，每个签名使用 5 组数值输入并遍历全部 16 条路径：3 个候选 × 3 个签名 × 80 个样本 = **720 组检查**。内部数值运算统一使用 FP32；输入为 BF16 不表示每一步都用 BF16 计算。

- 四层重排候选全部为 `equivalent`；两个负例全部为 `different` 并有实际输出差异。
- 所有样本核对独立 NumPy 参照、两侧原始和规范化图的执行；图通过 JAX 校验和幂等检查。
- 测试额外检查每一层确实发生方程换序，并用 0–15 的输出编号确认全部路径执行。最深层错误绑定只影响第四层为真的八条路径，另外八条路径保持相同。
- 当前实现无需修改即可支持这个案例。该结果不扩展未使用槽位、effects 等已有支持边界，也不能外推到任意深度或任意控制流。
