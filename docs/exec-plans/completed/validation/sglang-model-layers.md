# SGLang-JAX 全模型层级对抗验证

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

> 状态：历史验证与固定测试资产说明；原数字、接口口径和“当前缺口”属于记录时点，归档不表示测试资产废弃。现行支持状态见[贡献者指南](../../../../CONTRIBUTING.md)，现行契约见[设计索引](../../../design-docs/index.md)。

本文记录 36 个选定层的原有覆盖。后续增加的完整 GLM5/Llama Attention、CLIP encoder、Flux joint attention、求导、索引器和后端拒绝路径，单独见[完整层验证](full-attention.md)，不混入下列历史统计。

## 范围与验收

目标是固定上游版本中每个支持模型至少一个真实层的比较验证，不等同于完整模型推理或全部模型配置通过。固定提交为 `c98ced4a7a64e83261f6672937ce41f0095828ba`，通过 GitHub API 于 2026-10-11 核实为当时 main。

范围同时核对四个入口，不能仅按 README 的简略列表或仅按语言模型目录计数：

- `srt/models/*` 的 `EntryClass`：44 个架构名，含继承别名、草稿及编解码器模型。
- `srt/multimodal/models/**` 的 `EntryClass`：11 个条目；同名 UMT5 与 Kimi 按各自源码路径核对。
- `StageConfigRegistry` 与多模态各 `ConfigRegistry`：合并后 26 个不同配置模型名，保留全名/短名对应关系；关键词匹配指向同一组固定配置，不将任意匹配字符串计为新模型。
- 流水线 `get_model_class` 的 15 个组件名，其中 14 个用于当前 YAML；逐项核对编码器、扩散模型、VAE 与音频组件。

来源的完整文件快照及 SHA-256 保存在 [fixtures](../../../../tests/fixtures/sglang_jax/README.md)。注册清单须由快照独立提取，与手工指定的模型→层映射精确核对，缺失或多余条目均失败。

## 基准和 AI 候选

基准加载上游选定类/函数的原源码；不重写其数值方法，不通过正则或 AST 生成候选。仅提供锁定版本的 JAX/Flax、选定小尺寸配置、真实线性层和模型参数，绕过服务器初始化及权重下载。抽取范围、配置和继承/调用路径须明确记录。

AI 候选是独立固定函数，预判和理由先于比较执行保存。独立计算换序、参数准备交错为正例；错误参数绑定、遗漏归约、改变激活/精度/常量为反例。代数或执行元数据变化仍须按图契约判断，数值相同不能代替图匹配。

每层测试多个 dtype、shape、边界与固定种子输入；基准另与独立数值参照核对。比较前后的图均校验、执行并检查幂等。实际不等价候选必须记录完整输入与不同输出。无法判定不能算作支持通过，须登记和修复，或明确保留为未完成项。

## 运行与结果

```bash
uv sync --locked
uv run pytest tests/test_sglang_model_layers.py
uv run python -m tests.sglang_layers.runner --report /tmp/sglang-layers.json
uv run python -m tests.sglang_layers.runner --layer llama --profile bf16 --show-jaxpr
```

2026-10-11 验收：36 个不同层 × 2 个固定候选 × 3 个签名 = **216 个比较组合**，每个组合使用 5 组独立权重/输入，共 **1080 组数值验证**。新增 182 项 pytest 检查，全仓库 **844 passed**；无跳过项、无 `unknown`、无预期正例误判。完整逐模型映射及来源链见 [覆盖清单](../../../references/sglang-model-coverage.md)，任务过程见 [完成计划](../sglang-model-layers.md)。

三个签名为 FP32 `(tokens=3, hidden=4, intermediate=8)`、同形状 BF16，以及 FP32 `(2,6,10)`。视觉 Gemma 使用三维输入；残差块、视频卷积分别使用 NHWC、NDHWC；音频 decode 使用两级整数码本索引。权重始终独立采样，避免上游多个投影使用相同初始化种子使错误绑定偶然相等。五组样本包括零激活、正负边界及三个固定随机种子。

每个正例保留算子输入位置与分组，负例改变激活绑定、稳定项、归一化轴、残差符号、缓存拼接或码本。预判依据保存在 `candidates.py` 的 `REASONS` 中。DSpark 的方法展开保持原始方程顺序；MiniMax2 在 FP32、路由器在默认 FP32 权重配置下，类型准备换序会在 tracing 中消去。这些正例如实标记 `raw_order_changed=false`，没有计作实际调度换序；其余 BF16 正例另要求原始方程列表确实变化。

精确图比较之外，基准与独立 NumPy 参照分别核对 shape、dtype 和数值；FP32 容差为 `rtol=atol=3e-5`，BF16 为 `0.035`，后者允许激活函数内部低精度舍入差异。正例两侧及各自重排前后使用 `2e-5` 数值容差；负例必须至少有一组输入在 `rtol=atol=0.002` 下输出不同，报告保存该组参数/输入及两侧输出。所有规范化结果通过 JAX IR 校验与幂等检查。测试还故意注入错误 AI 标签、两侧同时算错的基准，确认验收过程能够拒绝假通过。

## 实际边界

- 环境为 Python 3.12、JAX/jaxlib 0.11.2、Flax 0.12.10、单 CPU 设备、显式 `data/tensor` mesh；完整流水线、模型权重、服务、性能及多设备层执行不在本轮结论内。
- 基准通过 AST 提取**未经改动的整个类定义**及真实依赖，执行原始方法。模型服务器的模块导入和构造被隔离；推理 dropout、非量化、小形状及 GLM5 非融合路径均显式固定。
- 音频量化器验证 `decode`，不是 `encode`；视频卷积验证带历史缓存路径；Grok 验证残差 MLP。所选层是模型确实使用的子层，不宣称每个 decoder block 的全部分支或每个 checkpoint 配置已验证。
- 上游 `.value` 使用触发 Flax 弃用提示；测试只过滤这一条已知提示，保留原始源码和其他告警。
- 本轮新增层未发现新的规范化误判。前置 `cond` 捕获绑定 bug 已修复，具体规则、负例和未支持边界见 [cond 记录](../../../design-docs/archive/cond-bindings.md)。
