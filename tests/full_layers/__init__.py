"""完整上游 Attention/block：原类、AI 固定候选及独立数值参照。

upstream.py 从已校验快照加载原始类和计算依赖；builders.py 构造小尺寸
配置并准备显式参数/输入；candidates.py 保存 AI 手写的正例与反例；
oracles.py 用 NumPy 独立重算；runner.py 执行图比较、数值见证和可重放
报告；evidence.py 与 blocked.py 另列求导、索引器及实际硬件拒绝路径。

入口：`uv run python -m tests.full_layers.runner --supplemental`。结论
只覆盖声明配置和 CPU 前向路径，见 docs/exec-plans/completed/validation/full-attention.md。
"""
