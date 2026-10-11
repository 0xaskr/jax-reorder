# jax-reorder

[English](README.md) | [简体中文](README.zh.md)

通过 jaxpr 结构规范化，比较 AI 生成或改写的 JAX 代码是否与原始代码等价。

当前实现第一轮计算图比较：从输出反向 BFS 编号，再按依赖深度重排独立方程。保留普通运算的操作数位置、参数、常量、dtype、shape、共享关系及嵌套子图，不执行交换律、结合律等代数变换。

## 开发快速开始

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。锁定环境为 JAX/jaxlib 0.11.2、CPU。

```bash
git clone https://github.com/0xaskr/jax-reorder.git
cd jax-reorder
uv sync --locked

# 查看原始 jaxpr、BFS 编号与重排结果
uv run python examples/inspect_jaxpr.py

# 手动自检：图比较、重排前后执行及数值反例
uv run python examples/check_correctness.py

# 多设备通信：比较通信换序、参数反例及分片矩阵乘法
uv run python examples/collective_jaxpr.py

# 重放 AI 的经典组合和 SGLang-JAX LlamaMLP 正反例
uv run python examples/bfs_reordering.py

# 重放 AI 手写并 hardcode 的真实业务改写，保存完整数值反例
uv run python examples/business_layer_cases.py --report /tmp/business-layer-cases.json

# 检查 cond 闭包捕获换序与实际参数绑定错误
uv run python examples/cond_bindings.py --show-jaxpr

# 重放 SGLang-JAX 全部注册模型所选层的 AI 改写
uv run python -m examples.sglang_layers.runner --report /tmp/sglang-layers.json
```

`uv sync` 创建本地 `.venv` 并安装开发依赖。默认缓存不可写时，设置 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`。

[自检脚本](examples/check_correctness.py)支持 `--show-jaxpr` 查看原图和重排结果，
`--trials 100 --seed 42` 调整数值样本。可修改其中的 `original`、候选函数和 `make_inputs`
测试自己的计算；`CASES` 保存预期标签。检查失败时脚本返回非零退出码。

[通信示例](examples/collective_jaxpr.py)默认使用两个逻辑 CPU 设备，包含 22 个组合，
支持 `--devices 4`、`--trials 10 --seed 42` 和 `--show-jaxpr`。它分别验证图判断和两侧
重排前后的实际执行；通信参数改变、遗漏归约等反例会打印输入与输出。当前验证范围是
单进程 CPU 集合通信，详细边界见[通信记录](docs/development-docs/collective-jaxpr.md)。

[业务案例脚本](examples/business_layer_cases.py)包含 gated RMSNorm、RoPE、causal attention、
KV cache 更新、MoE router 五个基准及 20 个 AI 手写的独立候选函数。
源码、预判与理由均 hardcode；`--seed` 和 `--trials` 只改变测试输入。
可用 `--scenario causal_attention --show-jaxpr` 查看指定场景的原图与重排图。
每例检查独立 NumPy 参照、图结论和两侧重排执行，详见[来源与案例记录](docs/development-docs/business-layer-cases.md)。

[SGLang-JAX 模型层测试](docs/development-docs/sglang-model-layers.md)按固定上游版本的全部注册入口选取 36 个真实层，执行原始源码与 AI 手写正负候选。覆盖 FP32、BF16 和不同 shape；模型别名共享层时逐项核对实际调用链。使用 `--layer llama --profile bf16 --show-jaxpr` 可检查单例；完整模型、配置和层对应关系见[覆盖清单](docs/references/sglang-model-coverage.md)。

## 使用

```python
import jax
import jax.numpy as jnp
from jax_reorder import compare_jaxprs


def original(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def candidate(x, y):
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b


signature = jax.ShapeDtypeStruct((4,), jnp.float32)
left = jax.make_jaxpr(original)(signature, signature)
right = jax.make_jaxpr(candidate)(signature, signature)
result = compare_jaxprs(left, right)
print(result.status.value)  # equivalent
```

结果为 `equivalent`（计算图等价）、`different`（计算图不等价）或 `unknown`（无法判定）。`different` 提供差异位置，但不证明数学公式不等价。存在未使用分支或未支持的语义时返回 `unknown`，并说明原因。

`canonicalize(jaxpr)` 可单独获取重排后的 jaxpr、规范结构、编号、深度与变量引用映射。调用方须保持追踪上下文一致，并核对函数输入输出 pytree；本接口只接收展平后的 jaxpr。

`cond` / `lax.switch` 支持内部实参与所有分支形参的同步换序，外部函数参数顺序固定。这样可以识别独立计算换序引起的闭包捕获位置变化，同时检出实际绑定错误；多个在所有分支中均未使用的实参暂返回 `unknown`。详见 [绑定规则与回归](docs/development-docs/cond-bindings.md)。

## 验证与文档

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python tests/lax_inventory.py
```

测试覆盖公开 lax 原语、独立分支组合、嵌套图、19 种 dtype 与多种 shape，以及 18 个保存了 AI 判断和数值证据的组合案例。集合通信测试默认使用两个逻辑 CPU 设备。有限测试样本不限制原语的合法参数域，也不代表穷举所有组合。

- [架构与比较契约](ARCHITECTURE.md)
- [BFS 设计与 review](docs/development-docs/bfs-reordering.md)
- [逐原语覆盖与 CPU 限制](docs/development-docs/lax-coverage.md)
- [AI 案例及模型来源](docs/development-docs/bfs-cases.md)
- [通信组合与多设备验证](docs/development-docs/collective-jaxpr.md)
- [真实业务基准与 AI 固定改写](docs/development-docs/business-layer-cases.md)
- [文档索引](docs/index.md)
