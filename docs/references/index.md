# 参考文档索引

本目录收录查阅资料、生成清单与来源记录。算法和契约设计见[设计索引](../design-docs/index.md)，
实际实施与验收见[执行记录](../exec-plans/index.md)。

| 主题 | 入口 |
| --- | --- |
| 自动生成的 CPU lax 原语与测试绑定清单；通过 `tests/lax_inventory.py` 更新和校验 | [lax-coverage.md](lax-coverage.md) |
| jaxpr 两阶段比较、三种结论与异常契约 | [equivalence-results.md](equivalence-results.md) |
| Oryx / MLIR / StableHLO 参考路线 | [ir-comparison.md](ir-comparison.md) |
| SGLang-JAX 注册模型、所选层与实际调用链 | [sglang-model-coverage.md](sglang-model-coverage.md) |
| SGLang-JAX 层测试的完整源码快照 | [快照说明](../../tests/fixtures/sglang_jax/README.md) |
| 快照逐文件来源与摘要 | [manifest.json](../../tests/fixtures/sglang_jax/manifest.json) |
| 快照源代码许可证 | [LICENSE](../../tests/fixtures/sglang_jax/LICENSE) |

上游快照保留原项目的背景与历史数据。本仓库规则以根目录架构和贡献指引为准，原始实验不能直接视为本仓库的验证结果。
