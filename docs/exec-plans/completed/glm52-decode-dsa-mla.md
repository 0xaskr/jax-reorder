# GLM-5.2 纯 JAX decode DSA/MLA 对抗测试

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

本页为 2026-10-14 初版的历史记录。用户随后明确要求仅用真实层的 jaxpr 测试比较器；外围投影改写与 CPU 数值验收不满足该目标。当前入口已按[开发记录](validation/glm52-decode-dsa-mla.md)改为核心内部的 5+5 改写及纯抽象图比较；本页的运行方式和“全通过”结论不作为当前能力声明。

## 目标与完成条件

固定开源纯 JAX reference，增加独立测试脚本，包含原始基线、5 个手写等价版本和 5 个手写不等价版本。实际执行索引选择与稀疏 MLA；区分结构证明、理想实数代数证明和数值反例。不调用 Pallas。

## 设计与取舍

- 来源为 SGLang-JAX `c98ced4a7a64e83261f6672937ce41f0095828ba` 的 `kernels/dsa/ref.py`，完整文件只导入 JAX/JNP。`streamindex_topk_ref(one_token_per_seq=True)` 与 `sparse_mla_ref` 原样加载。
- 测试层组合上游索引器、稀疏 MLA，以及 `Glm5Attention._forward_mqa` 中的 query absorption / value expansion 和输出投影。输入为已投影 query、索引器 query/weights、已更新 paged cache、投影权重和 decode 元数据；不包含前端投影、RoPE 构造、缓存写入、MoE 或完整模型。
- 保留上游 token top-k 行为（含 approx_max_k），不替换为 dense attention。非线性共同调用保持完整原图；仅在调用前后的线性算式上手写代数改写。
- CPU 使用小批次/短上下文；覆盖 GLM 的 latent=512、rope=64、nope=192、value=256、index heads=32 / dim=128。独立 NumPy 参照与非恒等页表验证 gather 和有效长度。
- 受保护文档保持只读；开发阶段不主动提交或推送，完成后用户明确授权提交到远端。

## 工作项

- [x] 查找公开实现，确认纯 JAX 路径和具体提交。
- [x] 保存完整 reference、许可证及来源哈希。
- [x] 编写基线、5+5 手写版本、独立数值参照和运行入口。
- [x] 执行结构/代数比较、数值验证、反例和静态检查。
- [x] 审查差异，记录结果并归档。

## 验证结果

2026-10-14：Python 3.12、JAX 0.11.2、CPU。命令使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline`。

| 检查 | 结果 |
| --- | --- |
| `python -m tests.glm52_decode_dsa_mla --json /tmp/jax-reorder-glm52.json` | 默认 GLM 内部维度、两个种子，5+5 全部通过；1 个结构正例、4 个代数正例；每个反例都记录实际输出差异 |
| `pytest -q tests/test_glm52_decode_dsa_mla.py tests/test_sglang_model_layers.py::test_snapshot_is_complete_and_unmodified tests/test_full_attention.py::test_attention_dependencies_match_independent_git_blob_digests --tb=short` | 31 passed，80.01 秒；其中新增 29 项，另 2 项覆盖完整来源清单 |
| `ruff check .` / `ruff format --check .` | 全部通过，146 个 Python 文件格式一致 |
| `python tests/inspect_jaxpr.py` | tracing 示例通过 |
| 新脚本 `--help`、`tools/check_docs.py --changed`、`git diff --check` | 通过 |

初次运行的新测试遍历器遇到 JAX 0.11 的 `jaxpr` 自引用属性，现已先访问 `eqns`，最终整组回归通过。没有修改比较器算法或原始 reference，也未删除来源中的动态索引、近似选择或 mask。

源码、候选、数值反例及完整边界见[开发记录](validation/glm52-decode-dsa-mla.md)。没有执行全仓 pytest；本次没有核心算法变更。报告写到 `/tmp`，未加入版本控制。

## 未验证范围

TPU/GPU、完整模型权重、长上下文性能、IndexShare 跨层传播不属于本次单层 CPU 测试。
