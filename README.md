# jax-reorder

通过 jaxpr 结构规范化，比较 AI 生成或改写的 JAX 代码是否与原始代码等价。

## 如何使用

使用 Python 3.12 和 uv，在 CPU 环境即可运行：

```bash
git clone https://github.com/0xaskr/jax-reorder.git
cd jax-reorder
uv sync --locked
```

将下面的示例保存为 `compare_demo.py`，在仓库根目录执行 `uv run python compare_demo.py`。
`ShapeDtypeStruct` 声明输入的 shape 和 dtype，无需准备数值样本。

```python
import jax
import jax.numpy as jnp

from jax_reorder import compare_functions


def original(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def reordered(x, y):
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b


signature = jax.ShapeDtypeStruct((4,), jnp.float32)
cases = (
    ("独立计算换序", original, reordered),
    ("分配律改写", lambda x, y: x * (y + 1), lambda x, y: x * y + x),
    ("错误改写", lambda x, y: x + y, lambda x, y: x - y),
)
for name, left, right in cases:
    report = compare_functions(left, right, signature, signature)
    print(name, report.status.value, report.equivalence_kind)
```

## 示例效果

上述示例输出：

```text
独立计算换序 equivalent structural
分配律改写 equivalent algebraic
错误改写 inequivalent None
```

- `structural`：独立计算调整顺序后，完整计算图仍匹配。
- `algebraic`：计算图不同，但数学后端证明输出在声明的数学解释下等价。
- `inequivalent`：得到数学不等价依据；此时没有等价类型，因此显示 `None`。

**浮点说明：**代数比较使用理想实数解释。分配律等改写可能改变舍入、溢出等行为，
不保证误差很小或逐位一致；相关前提和警告保存在 `report.modes[0].algebra`。

## 检查选项

默认只比较前向，先检查结构，再尝试代数比较。只检查完整图可传入
`equivalence="structural"`；该模式的 `different` 仅表示图不同，不证明数学不等价。

在上面的脚本中加入以下代码，可分别检查前向、JVP 和 VJP：

```python
report = compare_functions(
    original, reordered, signature, signature, modes=("forward", "jvp", "vjp")
)
for mode in report.modes:
    print(mode.mode, mode.status.value, mode.equivalence_kind)
```

这个例子的三个模式均输出 `equivalent structural`。比较只覆盖声明的输入签名、静态参数、
追踪环境和请求模式；未支持语义、未决或预算/超时通过 `ComparisonError` 子类报告。
当前已知缺陷见[贡献者指南](CONTRIBUTING.md#当前支持状态)。

更多示例见[矩阵乘法与乘加归约互认](examples/algebraic_equivalence.py)、
[指数恒等式](examples/sympy_equivalence.py)及 [Pallas 比较入口](docs/design-docs/current/pallas/contracts.md)。

开发入门、当前支持状态与后续路线图见[贡献者指南](CONTRIBUTING.md)。
