# jax-reorder

[English](README.md) | [简体中文](README.zh.md)

通过 jaxpr 结构规范化，比较 AI 生成代码与原始 JAX 代码的计算一致性。

`jax-reorder` 旨在检查 AI 生成或改写的 JAX 代码是否与原始代码等价。预期流程是在相同输入签名和静态参数下追踪两个版本，将它们的 jaxpr 转换成统一的结构表示，再进行比较。

## 开发快速开始

开发使用 Python 3.12，通过 [uv](https://docs.astral.sh/uv/) 管理环境与依赖。

```bash
git clone https://github.com/0xaskr/jax-reorder.git
cd jax-reorder

# 安装锁定版本的环境，包含开发工具
uv sync --locked

# 查看算子排列顺序不同的两个计算
uv run python examples/inspect_jaxpr.py
```

`uv sync` 会创建本地 `.venv`，准备 CPU 版 JAX 和开发依赖。如需直接激活环境，执行 `source .venv/bin/activate`。

常用检查：

```bash
uv run ruff check .
uv run ruff format --check .
```

`pytest` 已列为开发依赖。实现规范化算法时，应在 `tests/` 下加入有意义的测试，再在 CI 中启用。

## 示例与项目导航

[`examples/inspect_jaxpr.py`](examples/inspect_jaxpr.py) 追踪两个计算 `sin(x) + cos(y)` 的函数，它们的独立运算排列顺序不同：

```python
import jax.numpy as jnp


def first(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def second(x, y):
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b
```

在这个示例中，`first` 代表原始实现，`second` 代表一种 AI 可能产出的候选改写。

