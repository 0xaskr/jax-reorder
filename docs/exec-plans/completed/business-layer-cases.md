# 真实层级场景的 AI 固定改写

## 目标与完成条件

从 pallas-kernel 与 SGLang-JAX 选取不超过单层的真实业务计算作为 base，由 AI 直接编写候选源码并 hardcode 到脚本。每例在运行比较器前保存图标签、理由及数值预期；执行独立数值参照、两侧重排执行、幂等性和回归检查。

## 设计与取舍

- 选 gated RMSNorm、RoPE、causal attention、KV cache 更新、MoE biased top-k 路由五个场景。
- 每个 base 和候选都是独立 Python 函数。不使用 AST 变异、脚本生成改写、随机选择改写或 reverse/wrong 开关。
- 固定候选代码和标签；随机种子只用于数值输入。保留多组边界输入与数值不同的见证。
- 提取指定配置的纯 JAX 数值路径，记录来源版本及去除的运行时包装；不声称验证完整上游模型。
- 开发说明见 [业务层级案例](validation/business-layer-cases.md)。

## 工作项

- [x] 核对用户要求：AI 直接写候选，保存到脚本并 hardcode。
- [x] 保存五个 base、明确来源及独立候选函数。
- [x] 添加预判、NumPy 参照、随机与边界输入、结果报告和 pytest。
- [x] 运行并审查结果，同步中英文 README、CI 与文档，归档计划。

## 决策与进展

2026-10-11 完成。五个 base 各配四个 AI 直接手写候选：两个独立定义换序正例、两个有数值反例的业务负例。所有源码、预判及理由写入 `examples/business_layer_cases.py`，无脚本生成改写。来源版本、文件摘要、固定配置与许可证已记录；KV 样本选取展平更新核心，MoE 样本止于 router，不冒充完整 attention runtime 或专家层。

## 验证结果

- `uv sync --locked`：通过。
- 默认脚本：20 个候选 × 3 种签名 ×（2 个边界 + 3 个随机输入），300 次样本检查通过；JSON 报告保存 30 个负例组合的完整数值反例。
- `uv run python examples/business_layer_cases.py --trials 10 --seed 42 --report /tmp/jax-reorder-business-seed42.json`：720 次样本检查通过。
- `uv run pytest tests/test_business_layers.py -q`：67 passed，含原图确实不同、故意错误预判和两边同时错误的检测。
- `uv run pytest -q`：632 passed。
- `uv run ruff check .`、`uv run ruff format --check .`、`git diff --check`：通过。
- `--scenario causal_attention --trials 1 --show-jaxpr`、tracing 示例、原语清单一致性与改动文档本地链接：通过。
- 本轮只新增案例与回归，不修改核心规范化逻辑；在线 CI 尚未执行。
