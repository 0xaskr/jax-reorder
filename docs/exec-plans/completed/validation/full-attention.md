# 完整 Attention、block 与实际后端边界

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

> 状态：历史验证与固定测试资产说明；原数字、接口口径和“当前缺口”属于记录时点，归档不表示测试资产废弃。现行支持状态见[贡献者指南](../../../../CONTRIBUTING.md)，现行契约见[设计索引](../../../design-docs/index.md)。

在 SGLang-JAX 提交 `c98ced4a7a64e83261f6672937ce41f0095828ba` 上验证原始类，不用简化的 attention 核心代替原调用链。[来源快照](../../../../tests/fixtures/sglang_jax/README.md)保留完整原文件和独立摘要；[加载器](../../../../tests/full_layers/upstream.py)提取未改写的原始定义并注入实际依赖。硬件内核依赖直接加载原文件。

候选由 AI 直接编写，源码、预判和理由全部 [hardcode](../../../../tests/full_layers/candidates.py)。脚本只准备参数、数值输入和报告。原构造函数中固定的 head/LoRA 维度保持不变，仅缩小合法 token、head、hidden 配置。权重来自独立非对称随机值，以动态数组输入；cache、position、length、mask 也保留真实接口。

## 已验证的 CPU 前向路径

| 原始类与调用链 | 声明配置 | 固定候选/签名组合 |
| --- | --- | --- |
| `Glm5Attention` → `NativeAttention` → 原 MHA KV pool 更新 | 非 absorbed MHA；2 heads，hidden 8，原 Q LoRA 2048、KV LoRA 512、QK/V 256、RoPE 64；prefill/decode × FP32/BF16 | 16：8 equivalent、8 different |
| `LlamaAttention` → `NativeAttention` → 原 KV pool 更新 | GQA，4 Q heads / 2 KV heads，head dim 128，bias；prefill/decode × FP32/BF16 | 12：4 equivalent、8 different |
| `CLIPEncoderLayer` | 原 norm、完整 MHA、残差、MLP；2 heads，hidden 8；显式 mask/causal × FP32/BF16 | 8：4 equivalent、4 different |
| `FluxAttention` | 原 joint 文本/图像 attention，`sdpa`，QK norm、两种投影输出；2/3 图像 token × FP32/BF16 | 8：4 equivalent、4 different |

总计 4 个原始类、16 个配置、11 个按类登记的候选条目（9 个不同候选函数）、44 个候选/签名组合。每个组合使用 3 组数值输入，共 132 组。20 个等价正例均确认原始方程次序或操作数绑定序列实际改变，24 个实际错误候选都保存数值反例。不能将它们称为所有模型、所有层配置或所有参数的穷举。

GLM5 的 `use_qk_norm` 构造开关在配置中覆盖开/关，但锁定原源码的这个前向路径没有使用构造出的 q_norm/k_norm，不能据此称为已验证两种 QK norm 算法。RoPE 环境选择明确固定为 `_ROTARY_INTERLEAVED=False`。Flux joint attention 使用 `image_rotary_emb=None`、非 causal SDPA，完整 Flux block 的状态另列。

每例执行原始 IR 检查、图结论、原图对应证据、对称性、规范化幂等、原图不变、两侧规范化前后 CPU 执行和独立 NumPy 参照。NumPy 参照自行实现投影、归一化、RoPE、mask、softmax、残差和 cache 更新，不调用原类或候选的数值核心。dtype/shape/pytree 精确检查；浮点容差只用于数值证据。

GLM5 的两个正例分别提前完整 KV 路径和交错 Q/KV 投影；错误候选改变 position 或返回旧 cache。后者即使当前 hidden 输出一致，状态输出也不一致，报告会定位到 cache 端口。

## 求导与未使用索引器

四类各取第一个 FP32 配置及第一个正例，分别追踪前向、JVP、VJP。对 `hidden states` 求导（`argnums=(1,)`），权重和其他状态仍为动态普通输入。全部切向量、每个输出的余切向量保留完整动态接口，执行时使用非对称随机种子。12 份模式证据中前向/JVP 共 8 份等价，VJP 共 4 份图不同：独立投影调度影响 JAX 的 `add_any` 累加顺序。本轮不应用浮点交换律；数值接近仍报告图不同。双方原图及规范化图均执行检查。

GLM5 另验证 prefill FP32、decode BF16 的原 `GlmDsaIndexer`，固定 `_INDEXER_ROPE_CONCAT=False`、`use_dsa_sparse=False`。保留原 32 个 index heads 和 128 维 Hadamard、RoPE、top-k；结果虽未参与最终输出，完整图必须保留。两组正例换序得到 equivalent，两组删除索引器的反例得到 different，数值返回值一致。这 4 份证据单列，不算作实际输出不等价的反例。

## 已确认的拒绝与未支持

| 原始路径 | 图状态 | CPU 结果 |
| --- | --- | --- |
| GLM5 absorbed MQA，prefill/decode | 两次追踪的原图结构比较 equivalent；接口搜索预算为 1 | 原正常 lowering 拒绝：CPU 只支持 interpret |
| `Glm5DecoderLayer`，absorbed attention + norm/残差/dense MLP，prefill/decode | 同上；完整原 Decoder 图进入结构比较，不认证改写候选 | 同上，没有执行完整 Decoder |
| `WanT2VCrossAttention` → 原 USP/Flash Attention | 原图结构自比较 equivalent；未认证候选 | 同上，没有替换核心 |
| `EPMoE`，2 experts、top-1、原 GMM | 原图结构自比较 equivalent；未认证候选 | 原后端在 CPU 自动使用 interpret，执行并通过逐专家 NumPy 参照 |
| `FluxSingleTransformerBlock` | 未得到可比较原图 | 锁定 JAX 的 Explicit mesh 不接受原 `with_sharding_constraint`；原追踪拒绝 |
| GLM5 MHA 单 head 配置 | 未得到可比较原图 | 原 reshape 后的显式 sharding 不一致，原追踪拒绝 |

2026-10-19 公共 Pallas 结构入口接入后，初始结果为 4 份预算异常、2 份原图自比较、
2 份 trace_rejected。2026-10-20 使用[接口不变量细化](../../../design-docs/current/region-bindings.md)消除 GLM5
8 槽位的阶乘搜索，四个原配置均通过 `search_budget=1` 的完整比较与对应证据复核，
单例约 71–93 秒，组合 4 passed / 322.26 秒。没有提高预算或删除未使用计算；
小图另验证合法槽位置换、错误反馈、幂等与剩余对称组的预算拒绝。
运行器现在报告 6 份原图结构比较、2 份 trace_rejected；原图比较与候选等价分开统计。
GLM5 Pallas 配置只显式给定原后端允许的 block/VMEM 参数以避免依赖 TPU 自动调参，
没有启用替代 interpret。EPMoE 的 interpret 是原源码自身的 CPU 分支。

稀疏 DSA 后端、完整 MoE Decoder、其他配置、权重求导、高阶模式、多主机及 GPU/TPU 尚未验证。Flux 完整 block 已保存的候选也未通过原图前置检查，不能计入等价覆盖。

## 重放与报告

```bash
uv run pytest tests/test_full_attention.py
uv run python -m tests.full_layers.runner --supplemental --report /tmp/full-layers.json
uv run python -m tests.full_layers.runner --family glm5 --samples 3
```

[运行器](../../../../tests/full_layers/runner.py)报告 `cases` 与前向 `counts`；`supplemental` 分别保存 derivatives、indexer、boundaries、unverified，`supplemental_counts` 单独统计。unknown、CPU 拒绝和未验证不会并入 equivalent。每条包含来源、配置、模式、实际/预期结论、差异和数值证据。大权重不嵌入 JSON，使用固定输入生成 seed、签名和逐叶 SHA-256 精确重放；错误输出保存端口、索引和实际值。

报告是本地 CPU 证据；GitHub CI 是否通过须查实际运行结果。总体回归见[验证矩阵](../../../design-docs/current/verification-matrix.md)。
