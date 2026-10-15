# 固定的 SGLang-JAX 源码快照

来源：<https://github.com/sgl-project/sglang-jax>，提交 `c98ced4a7a64e83261f6672937ce41f0095828ba`。

`manifest.json` 保存路径、完整文件 SHA-256 与字节数。以 `.txt` 结尾的文件是原文件的逐字节副本，保留上游版权、注释和上下文，不参与本仓库的代码格式化。许可证见 [LICENSE](LICENSE)。快照供离线核对注册模型、调用关系及运行选定层的方法；不是整个上游仓库，也不用于导入服务器或下载权重。

模型案例只加载明确指定的类/函数，注入实际 JAX/Flax 依赖与小尺寸权重。候选改写由 AI 直接编写，加载上游源码不生成候选，也不生成预期标签。

`upstream-model-tree.json` 保留 GitHub tree API 返回的两个模型目录的文件清单与 Git blob 摘要，独立检查模型目录是否漏收文件。108 份原文件覆盖全部模型定义、静态配置以及所选层使用的辅助函数，不包含模型权重。

完整 Attention 验证另加入 22 份原始后端、KV pool、ForwardBatch、norm、Pallas 内核和 MoE 依赖，共 130 份原文件。新增来源由 `upstream-attention-tree.json` 中的独立 Git blob 摘要核对，统一加入 `manifest.json`。加载完整类时只注入实际原定义；环境开关、后端选择及拒绝路径见[完整层记录](../../../docs/development-docs/full-attention.md)。没有把替代 attention 核心的结果计作原类通过。

GLM-5.2 decode 测试继续加入同一提交的 `kernels/dsa/ref.py`，现有 131 份原文件。该文件是仅依赖 JAX/JNP 的原始 reference；原样加载 `streamindex_topk_ref` 和 `sparse_mla_ref`，仅以抽象签名追踪并比较 jaxpr，不编译或执行 layer，不加载 Pallas。来源也通过上述两个摘要表校验。测试入口及明确边界见[开发记录](../../../docs/development-docs/glm52-decode-dsa-mla.md)。
