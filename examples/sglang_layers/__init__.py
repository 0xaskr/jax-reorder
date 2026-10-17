"""SGLang-JAX 固定源码、AI 手写候选和逐模型覆盖清单。

upstream.py 从已校验快照抽取原始定义并解析注册关系；catalog.py 手工登记
选定层与模型映射；builders.py 构造小尺寸实例和独立权重；candidates.py
保存 AI 候选与人工预判；oracles.py 用 NumPy 独立重算；runner.py 逐层比较、
核对数值见证并生成报告。

入口：`uv run python -m examples.sglang_layers.runner --report /tmp/sglang-layers.json`。
结论覆盖固定提交的 36 个选定层、声明签名与 CPU 环境，不是完整模型推理。
"""
