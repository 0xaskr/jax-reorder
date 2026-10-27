# 真实层级场景的 AI 固定改写

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

> 状态：历史验证与固定测试资产说明；原数字、接口口径和“当前缺口”属于记录时点，归档不表示测试资产废弃。现行支持状态见[贡献者指南](../../../../CONTRIBUTING.md)，现行契约见[设计索引](../../../design-docs/index.md)。

## 契约

用户要求由 AI 直接编写改写并 hardcode 到脚本。本次候选为 AI 手写的独立函数，无脚本变异器、源码生成器或随机改写开关。测试输入可以使用固定种子随机生成；源码、标签与判断理由保持固定，不能用比较器的输出反推标签。

选取 gated RMSNorm、RoPE、causal attention、KV cache 更新及 MoE biased top-k 路由，最大边界为单层。只提取固定配置的纯 JAX 计算，保留操作数、计算分组、dtype 转换和输出含义；不运行上游模型加载、Flax 模块或 TPU 内核。不执行代数规范化。

## 来源与配置

- pallas-kernel 本地提交 `da2954714912aabc1367260d252ddeafa65b4a7a`：`tops/cpu/ops/kda/short_conv.py` 的 `kda_gated_rmsnorm_reference`；`tops/cpu/ops/mla/mla.py` 的 `causal_softmax_attention`，固定为普通 causal mask、无窗口/packed 参数。
- SGLang-JAX 提交 `c98ced4a7a64e83261f6672937ce41f0095828ba`：`layers/embeddings.py` 的 `apply_rotary_emb`，固定 Neox 半维旋转；`mem_cache/memory_pool.py` 的 `merge_kv` / `set_kv_buffer_legacy`，提取三维展平缓存与 K/V 交错写入核心；`layers/gate.py` 的 `GateLogit.__call__`（FP32、HIGHEST precision、sigmoid）及 `TopK._biased_topk_jax`，固定 k=2、无后续归一化或 scaling。
- 完整路径均位于各项目的上述目录，SGLang-JAX 路径前缀为 `python/sgl_jax/srt/`。两个项目均为 Apache-2.0，许可证副本见 [Pallas](../../../../tests/LICENSE.pallas-kernel)、[SGLang-JAX](../../../../tests/LICENSE.sglang-jax)。

| 场景 | base 与固定范围 | 上游源码 SHA-256 |
| --- | --- | --- |
| gated_rmsnorm | `norm_base`；最后维归约、epsilon=1e-6，保留中间转换回输入 dtype 的舍入边界 | `ac13b959aa60f5b8701693e43ea0a2cf14daff80f7ee197510622c47e89625be` |
| rope | `rope_base`；Neox 半维旋转，不含频率表生成、MRoPE 或模型包装 | `bdf329b84ff61d37ee8da0fdb704860d16226c90a3f9280310a53be6828e0c07` |
| causal_attention | `attention_base`；完整 QK→缩放→causal mask→softmax→V，FP32 累加，覆盖 prefill/prefix/decode | `43ddab73ac2abca111d431d4575f50bd79f57105f07aec7440179f579b43cec3` |
| kv_cache | `cache_base`；输入缓存已为三维，保留 K/V 交错与无效写入丢弃；不含 5D 包装、TPU 对齐和分片配置 | `e4bf3a814264c0987b45308b1afc2727673a384b290a2746eb3c12a41370a9a2` |
| moe_router | `router_base`；路由投影、sigmoid 与 biased top-k；专家 MLP、dispatch/combine 不在本例范围 | `896069b7a8d608195e6f2e5b1a507f81bcbe366e2319eb0ee81abb9c86312d52` |

对应源码：[SGLang embeddings](https://github.com/sgl-project/sglang-jax/blob/c98ced4a7a64e83261f6672937ce41f0095828ba/python/sgl_jax/srt/layers/embeddings.py)、[KV cache](https://github.com/sgl-project/sglang-jax/blob/c98ced4a7a64e83261f6672937ce41f0095828ba/python/sgl_jax/srt/mem_cache/memory_pool.py)、[router](https://github.com/sgl-project/sglang-jax/blob/c98ced4a7a64e83261f6672937ce41f0095828ba/python/sgl_jax/srt/layers/gate.py)。Pallas 来源为本地 `/home/askr/work/pallas-kernel` 的上述固定提交。

## AI 手写候选

完整源码及 `SCENARIOS` 中的预判、数值要求与理由保存在 [business_layer_cases.py](../../../../tests/business_layer_cases.py)。每个候选是独立函数，不调用带切换开关的 base 来构造变体。两个正例保留原操作数顺序，两个负例改变业务计算；本次没有交换律/结合律改写。

| base | equivalent 候选 | different 候选 |
| --- | --- | --- |
| gated RMSNorm | `norm_gate_first`：门控提前；`norm_prepare_scale`：转换与门控交错 | `norm_changed_epsilon`：1e-6→0.1；`norm_wrong_gate`：sigmoid→SiLU |
| RoPE | `rope_second_half_first`：先算后半；`rope_products_first`：四个独立乘积交错 | `rope_wrong_sign`：减号→加号；`rope_swapped_outputs`：输出两半互换 |
| causal attention | `attention_mask_first`：mask 提前；`attention_value_late`：K/Q 换序且 V 推迟 | `attention_wrong_scale`：缩放指数 -0.5→0.5；`attention_future_mask`：反转 mask 方向 |
| KV cache | `cache_pack_first`：先打包；`cache_interleaved_preparation`：交错准备索引与更新值 | `cache_invalid_to_zero`：负位置覆盖零槽；`cache_swapped_kv`：K/V 槽位写反 |
| MoE router | `router_bias_first`：偏置提前；`router_prepare_weights`：转换与广播交错 | `router_biased_weights`：权重误含偏置；`router_wrong_correction_sign`：偏置符号错误 |

## 验证

预期正例仅重排独立定义，负例针对业务参数或数据连接。base 与独立 NumPy 参照核对；分别检查每个候选的图预判、样本结果及两侧规范化后的实际执行。负例必须找到数值反例，记录种子与样本编号，不能以图不同代替数值证据。

每场景有三组签名：`float32`、`bfloat16`、`small`（FP32 小形状）。固定两个边界样本，再用 `--seed` 生成 `--trials` 组数值输入。边界包括 RMSNorm 零/小范数、RoPE 的单位/90° 旋转、attention 均匀分数与非等长缓存、KV 无效/有效位置、MoE 偏置主导及并列专家分数。有效 scatter 索引互异，错误落到零槽的候选也不会产生重复目标写入的不确定性。

独立 NumPy 参照使用按 query 读取可见前缀、按 token/head 写缓存、完整排序选专家等不同实现方式。所有输出检查 shape、dtype；整数索引必须精确一致，浮点输出必须有限，FP32 的 rtol/atol 为 2e-5，BF16 为 2e-2。这是采样容差验证，不是逐位等价承诺。

[回归测试](../../../../tests/test_business_layers.py)还检查：正例在 BF16 路径确实形成不同的原始方程顺序；故意错写 AI 图标签会失败；即使 base 与 candidate 同时算错，独立 NumPy 参照仍会阻止假通过。比较器不参与候选或预期标签的生成。

后续增加了 [cond 闭包复现脚本](../../../../tests/cond_bindings.py)：将 gated RMSNorm 放入条件分支，固定保存门控提前、显式传参对照及错误绑定三个候选。三个签名都执行真假分支，原先的捕获换序误判已修复；设计与本次验证见 [cond 绑定记录](../../../design-docs/archive/cond-bindings.md)。原有五个 base 和 20 个候选保持独立。

```bash
uv run python tests/business_layer_cases.py --report /tmp/business-layer-cases.json
uv run python tests/business_layer_cases.py --scenario causal_attention --show-jaxpr
uv run python tests/business_layer_cases.py --trials 10 --seed 42
uv run pytest tests/test_business_layers.py -q
```

2026-10-11 首轮默认种子 `20261011`：20 个候选 × 3 种签名 × 5 组输入，共 **300 次样本检查**；30 个图等价、30 个图不同的候选/签名组合均符合预判，30 个负例组合均有完整数值反例。每次样本检查包括 NumPy 参照、base/candidate 数值要求和两侧重排执行；结构判定每个签名只做一次。JSON 报告保存每例函数名、预判与理由、差异位置及反例输入/输出。

第二轮 `--seed 42 --trials 10`：60 个候选/签名组合、**720 次样本检查**全部通过。新增 **67 项回归通过**，全量 **632 passed**；Ruff、格式检查、tracing 示例、原语清单一致性、本地文档链接与 `git diff --check` 通过。`--scenario causal_attention --show-jaxpr` 的筛选与展示入口也已运行。CI 增加了脚本入口及 JSON 报告路径的重放命令；在线 CI 未在本次验证中运行。本轮没有扩大核心规范化算法的支持范围。
