# GLM-5.2 DSA/MLA 核心的 jaxpr 健壮性测试

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

> 状态：历史验证与固定测试资产说明；原数字、接口口径和“当前缺口”属于记录时点，归档不表示测试资产废弃。现行支持状态见[贡献者指南](../../../../CONTRIBUTING.md)，现行契约见[设计索引](../../../design-docs/index.md)。

## 目标与来源

用真实纯 JAX layer 的原始图及手写改写图检验当前比较器。流程是 **抽象输入签名 → 原始/候选 jaxpr → 结构与代数对比**。不分配模型输入、不编译或运行 layer、不做数值采样。初版外围投影改写和 CPU 验收偏离此目标，其记录保留在[历史计划](../glm52-decode-dsa-mla.md)。

来源为 [SGLang-JAX DSA reference](https://github.com/sgl-project/sglang-jax/blob/c98ced4a7a64e83261f6672937ce41f0095828ba/python/sgl_jax/srt/kernels/dsa/ref.py)，提交 `c98ced4a7a64e83261f6672937ce41f0095828ba`。完整[源码快照](../../../../tests/fixtures/sglang_jax/python/sgl_jax/srt/kernels/dsa/ref.py.txt)保持未修改，由 SHA-256 与独立 Git blob 摘要校验，Apache-2.0 许可证保留在快照目录中。

base 组合原始 `streamindex_topk_ref(one_token_per_seq=True)` 与 `sparse_mla_ref`。输入为已投影并完成 RoPE 的 indexer query/weights、已吸收的 MLA query、已更新的 paged cache、有效长度及页表；输出为压缩 latent attention 和 top-k 索引。没有将外围线性层、完整模型或 Pallas 后端算作该核心边界。

## 签名与手写改写

[脚本](../../../../tests/glm52_decode_dsa_mla.py)默认 `glm` 签名保留 64 个 attention heads、512 latent、64 rope、32×128 indexer 和 2048 top-k，batch=2，每条序列最多 4096 token；不再为 CPU 数值执行缩小 head 数或 top-k。另提供 `small` 签名。FP32 和 BF16 均只用于 abstract tracing，动态长度、索引和页表都是 `ShapeDtypeStruct`，没有固定为随机样本。

所有变更点直接手写。普通 Python helper 复用原始循环、分页骨架，并在追踪时展开到候选核心的 jaxpr；没有从 base 的 AST/jaxpr 自动生成候选。共同未改动的另一核心可以复用，但所选核心的内部公式/依赖实际发生变化。

| 版本 | 预期关系 | 核心内部变化 |
| --- | --- | --- |
| `mla_schedule` | 等价 | MLA 循环内 mask 与索引计算移到 paged gather 之前 |
| `dsa_schedule` | 等价 | DSA 循环内 query/gate 读取移到页表读取之前 |
| `mla_scale_first` | 理想实数等价 | `(Q Kᵀ)s` 改为 `(Qs)Kᵀ`，FP32 cast 位置不动 |
| `mla_value_reduce` | 理想实数等价 | `P V` 改为广播乘积与沿 token 求和 |
| `dsa_head_reduce` | 理想实数等价 | head 加权 einsum 改乘积归约，保留 ReLU 位置 |
| `mla_missing_scale` | 不等价 | 去掉 QK scale，改变 softmax 相对概率 |
| `mla_unmasked_padding` | 不等价 | -1 padding 被当成第 0 个 token 参与注意力 |
| `mla_wrong_softmax_axis` | 不等价 | 沿 head 而非选中 token 归一化 |
| `dsa_missing_relu` | 不等价 | 负 QK 项进入 indexer 加权打分，改变 top-k 排名 |
| `dsa_inclusive_length` | 不等价 | 将有效长度上界从 `< length` 改为 `<= length` |

预期关系由公式推导标注，脚本内为反例保留局部解析反例说明；它们不是比较器输出，也不是数值运行得出的标签。代数正例只在理想实数解释下成立；浮点改写可能改变舍入、溢出和排序结果，不保证逐位等价或误差很小。

## 图比较与验收

使用 `trace_function` 单独记录原层和候选的完整函数契约及 jaxpr，再调用 `compare_function_traces`。比较器检查 pytree、抽象值、所有输入输出绑定、常量与嵌套语义；不使用 `check_cpu`，不调用 `lower`、`compile` 或数值 evaluator。

JSON 分开记录 `expected_relation`、`structural`、实际 `status`、`outcome`、`failure_kind`、`difference`、使用的 `rules` 和浮点警告。`different` 只说明图不同；`unknown` 对正例和反例都计为 `coverage_gap`，不再汇总成“5+5 全通过”。反例必须有完整输出的 `inequivalent` 证据才计为 `proven_inequivalent`；被报告为 equivalent 则是 `false_acceptance`。

命令退出码：0 表示所有预期关系已证明；1 表示误接受/预期冲突；2 表示存在 unknown 或只有结构不匹配的能力缺口。pytest 验证测试工具自身、已证明的换序和精确的当前缺口，不把通过的回归数量当作语义证明数量。

## 当前发现

2026-10-15，锁定 JAX 0.11.2：

- 两个核心内部换序都得到 **structural equivalent**，回归还检查对应 scan 循环体的原始方程顺序确实不同。
- 初始纠偏后，其余 8 个核心改写都在 `algebra.constant: non-finite constant` 停止。核心变化后不能整体复用相同调用，因而暴露原层 `-inf` mask 的支持缺口。
- 加入[受限非有限值支持](../../../design-docs/current/algebraic-equivalence.md#当前边界)后，`mla_scale_first` 与 `dsa_head_reduce` 得到 **algebraic equivalent**，保留浮点警告。合计 4 个正例已证明（2 个结构、2 个代数）。
- 加入 guarded-softmax 有限性定理后，`mla_value_reduce` 得到 **algebraic equivalent**，完整输出及两轮 scan 一起复核。合计 **5 个正例已证明（2 个结构、3 个代数）**。定理核对 mask、归约域、广播和最终零 guard，保留全 mask 行内部的部分求值，不删 mask 或替换有限常量。
- 5 个反例仍为 **structural different / algebra unknown**，原因为 `algebra.outputs: normal forms differ; no algebraic proof found`；没有误接受，也没有把 unknown 当作不等价证明。新接入的实数多项式负证据不适用于含 softmax、top-k、动态 gather 和 scan 的完整图。
- 两个 DSA 负例可以通过最终 `indices` 输出构造反例，三个 MLA 负例需连同 indexer 输入证明最终 attention 输出不同；不能把 indexer 输出设为任意自由变量。锁定 JAX 在非 TPU 上的近似 top-k fallback 可以继续审计，但本轮未适配其精确选择、并列与候选容量语义，因此不声称完整反例已取得。剩余缺口见 [ALG-003](../../index.md#待处理)。

## 运行

```bash
uv run python -m tests.glm52_decode_dsa_mla \
  --json /tmp/glm52-jaxpr.json --jaxpr-dir /tmp/glm52-jaxprs
uv run python -m tests.glm52_decode_dsa_mla --dtype bfloat16
uv run pytest -q tests/test_glm52_decode_dsa_mla.py
```

前两条命令当前预期退出 2，报告 5 个已证明正例与 5 个负例能力缺口。`--jaxpr-dir` 可保存 base 和 10 个候选的 jaxpr 文本；`--profile small` 使用小签名，仍只追踪。

回归在禁止后端编译和 eager 数组运算的条件下执行整个比较流程，并覆盖 small/FP32、glm/FP32、glm/BF16、CLI 报告、全部 jaxpr 导出、来源和无 Pallas 检查。
初始纠偏结果保留在[完成记录](../glm52-jaxpr-robustness.md)，受限 `-inf` 支持及全量 1952 项回归的更新验证见[完成计划](../negative-infinity-masks.md)。

SymPy/负证据/guarded-softmax 接入后，全量 **2095 passed，1 warning，944.74 秒**；三组签名及 CLI 均通过上述回归。[当前设计](../../../design-docs/archive/algebraic-equivalence.md#sympy-接入与数学不等价证据2026-10-15)记录新增规则与保留边界。
