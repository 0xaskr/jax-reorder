# GLM-5.2 核心层的 jaxpr 健壮性测试纠偏

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

## 目标与完成条件

用户澄清目标是利用 GLM-5.2 MLA/DSA layer 的 jaxpr 检验当前比较器，不需要在 CPU 编译或执行层。修正 `tests/glm52_decode_dsa_mla.py`：只构造抽象签名，追踪固定上游 base 与 5 个等价 / 5 个不等价的手写核心改写，再比较图并报告能力缺口。

## 已确认问题

- 旧正例只改外围投影，共同的 DSA/MLA 函数被作为完整相同调用复用，没有覆盖核心内部改写。
- 旧脚本默认 CPU 编译、数值执行与 NumPy 对照，与用户目标不符。
- 5 个反例的任意代数 `unknown` 都可通过旧验收；真实 GLM 内部维度只在手动 runner 中做图比较。
- 审查复现：将原 sparse MLA 的 scale 移到 QK 点积之前，结构不同、代数返回 `algebra.constant: non-finite constant`；原样手写复制则结构等价。这是应揭示的能力缺口。

## 设计与取舍

- 保留固定纯 JAX `streamindex_topk_ref` / `sparse_mla_ref` 基线及完整来源快照。
- 目标边界为两个 reference 组成的 decode 核心，输入已完成投影/RoPE/缓存更新；移除外围投影改写。
- 用普通 Python helper 复用分页和循环骨架，公式与变更点直接手写。骨架在追踪时完整展开；不能把改变的核心留为相同的共同 JIT 调用。
- `ShapeDtypeStruct` → `trace_function` → `compare_function_traces`；不产生随机样本、不执行 JIT、不调用 lower/compile。
- 模型配置使用 64 attention heads、512 latent、64 rope、32×128 indexer、2048 top-k；抽象追踪不分配这些张量。另提供 small 签名和 FP32/BF16。
- 用例预期关系与比较器实际结论分别报告。正例 unknown 计能力缺口，反例 equivalent 计误接受；结构 different 不冒充数学不等价证明。保留具体拒绝阶段/原因，不把任意 unknown 当全通过。
- 本轮只修测试设计和入口，明确呈现现有比较器边界，不放宽核心算法规则以制造通过结果。

## 工作项

- [x] 只读审查最近提交并复现核心改写缺口。
- [x] 改写独立脚本，新增核心内部的 5+5 版本与抽象图报告。
- [x] 回归保证执行路径只追踪，并验证实际核心改写、拒绝原因和完整模型签名。
- [x] 更新开发记录、完成检查并归档；不主动提交或推送。

## 验证结果

2026-10-15，锁定 JAX 0.11.2。命令使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline`。

| 检查 | 结果 |
| --- | --- |
| `python -m tests.glm52_decode_dsa_mla --profile small --json /tmp/glm52-jaxpr-small.json --jaxpr-dir /tmp/glm52-review-jaxprs` | 完成 10 组图对比，2 个结构正例、8 个 unknown；按能力缺口契约退出 2 |
| 默认 `glm` 签名的同一入口，报告 `/tmp/glm52-jaxpr-glm.json`、图目录 `/tmp/glm52-jaxpr-glm` | 同样 2 equivalent / 8 unknown，导出 base 与 10 个候选的 jaxpr，未编译/执行 layer |
| `pytest -q tests/test_glm52_decode_dsa_mla.py --tb=short` | 41 passed，283.53 秒；small/FP32、glm/FP32、glm/BF16 全部在禁止后端编译及 eager 运算的条件下对比 |
| `pytest -q tests/test_functions.py tests/test_algebra_common_calls.py --tb=short` | 35 passed，7.99 秒 |
| `ruff check .` / `ruff format --check .` | 通过 |
| `python tests/inspect_jaxpr.py` | 通过 |
| `tools/check_docs.py --changed` / `git diff --check` | 通过 |

回归通过只证明测试工具、两种结构换序和当前拒绝原因符合已记录事实，不表示 8 个 unknown 的公式关系已得到证明。没有运行 GLM 层数值计算，也没有运行全仓 pytest。

## 后续能力缺口

8 个改写均在 `algebra.constant: non-finite constant` 处停止，记录为 [ALG-003](../index.md#待处理)。保留原始 `-inf` mask 的正确语义需要独立设计；本轮不通过修改原层或放宽比较规则掩盖该边界。当前测试方法与入口见[开发记录](validation/glm52-decode-dsa-mla.md)。
