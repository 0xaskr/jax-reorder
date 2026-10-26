# `-inf` mask 的受限代数支持

## 目标与完成条件

保留原始 DSA/MLA 的 `-inf` mask，通过抽象 jaxpr 比较检验有限分支的代数改写。
不得执行 layer、替换 mask、把未知结论当作通过，或把非有限值带入实数多项式规则。

## 设计与取舍

- `-inf` 常量按 dtype、shape、完整内容保存为不透明特殊值；`+inf` / NaN 常量仍拒绝展开。
- Tensor 增加内部的“可能含非有限值”标记。该标记阻止相关乘加、消项、幂和求和展开；索引视图与原语调用保留。
- 有限输入的子表达式仍可规范化，mask、softmax、top-k 等保持算子、参数、操作数和分支的完整对应。
- JIT/cond/scan/while 传播标记；含非有限常量的循环保守标记浮点 carry，避免第一轮有限、后续轮非有限的错误证明。共同 JIT 也必须检查隐藏的常量。
- 同样检查 gather `fill_value` 等参数；非有限值不限于字面量/捕获数组。
- 不新增关于全 mask 行、softmax 输出有限性或一般扩展实数运算的推断。这些没有证明时保留 unknown。
- 独立证明校验器重新运行相同语义转换，局部代数规则不扩充。

## 工作项

- [x] 实现常量编码、非有限值传播和规则限制。
- [x] 覆盖 mask 下有限改写、改变 mask、Inf/NaN 消项、嵌套调用及循环反例。
- [x] 重跑 GLM 10 对图并记录下一处真实边界。
- [x] 完成全量测试、Ruff、示例和文档检查，归档计划。

## 验证结果

2026-10-15，Python 3.12 / 锁定 JAX 0.11.2，命令前缀为
`UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline`。

- 新增专用回归：41 passed（4.35 秒）。
- small 图比较：4 equivalent / 6 unknown，按缺口契约退出 2，报告 `/tmp/glm52-neginf-small.json`。
- GLM 回归：41 passed（315.16 秒），small/FP32、glm/FP32、glm/BF16 结果一致，全程禁止编译/执行 layer。
- 参数传播补齐前的代数回归：771 passed。
- 最终完整 `pytest -q`：**1952 passed，1 warning，963.69 秒**。唯一 warning 来自已有上游 Pallas 弃用接口，与本次纯 JAX DSA/MLA 无关。
- `ruff check .`、`ruff format --check .`、`python examples/inspect_jaxpr.py`、`python tools/check_docs.py --changed`、`git diff --check` 均通过。
- 剩余 6 个 unknown 均在 `algebra.outputs`；PV 有限性未证明，5 个反例未误接受。

核心文档受保护，设计与验证记录仅更新 development-docs / exec-plans。

## 后续边界

参见 [ALG-003](../index.md#待处理) 与 [GLM 核心图比较](validation/glm52-decode-dsa-mla.md)。
PV 改写需要对 guarded softmax 做独立的有限性证明；本次没有假设该性质。
完整层始终只作抽象追踪与比较，没有增加编译、数值采样或 Pallas 实现。修改未提交、未推送。
