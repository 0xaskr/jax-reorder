"""固定 DeepSeek-V4.1-Flash 原始 mHC 层与 AI 代数改写。

upstream.py 从已校验快照加载原始函数；candidates.py 保存 AI 手写的正例和
反例；cases.py 提供声明签名、数值输入及独立 NumPy 前向/JVP/VJP 参照；
runner.py 负责结构比较、三模式证明、数值检查和 CPU 性能测量。

复现入口：`uv run python -m examples.deepseek_v41.runner --json /tmp/deepseek-v41.json`。
结论只覆盖该固定提交、声明签名、锁定 CPU 环境及请求的模式，不证明一般
Python 等价，也不给出浮点误差界或 TPU/GPU 性能结论。
"""
