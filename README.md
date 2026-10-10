# jax-reorder

对 JAX 的 jaxpr 做规范化，比较不同写法产生的计算。

**当前状态：项目初始化完成，reorder 与比较 API 尚未实现。**

## 目标

- 统一变量命名和无依赖算子的顺序，保留计算的数据依赖。
- 增加交换律、结合律规范化；浮点表达式发生代数改写时发出 warning。
- 区分结构匹配和代数匹配，让比较结果明确说明能够确认到哪一层。

在相同输入签名、静态参数和执行环境下，如果两个纯计算的结构规范化结果
完全一致，可以作为所追踪计算等价的充分条件。结果不同不代表程序不等价。
JAX 的 jaxpr 只描述特定追踪得到的计算，不包含任意 Python 副作用，参见
[JAX tracing 文档](https://docs.jax.dev/en/latest/jit-compilation.html)。

经过交换律或结合律改写后匹配，需要单独标记为代数匹配。尤其是浮点结合律
并不普遍成立，例如 float32 中 `(1e20 + -1e20) + 3.0` 与
`1e20 + (-1e20 + 3.0)` 可以得到不同结果；warning 不应承诺误差一定轻微。

比较需要包含输入类型、常量值、primitive 参数、嵌套 jaxpr、输出顺序及
pytree 结构，不能只比较打印文本。jaxpr 的结构参见
[JAX IR 文档](https://docs.jax.dev/en/latest/601/jaxpr.html)。

## 开发环境

使用 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/0xaskr/jax-reorder.git
cd jax-reorder
uv sync --locked
uv run python examples/inspect_jaxpr.py
```

`uv sync` 会创建本地 `.venv`，默认安装 CPU 版 JAX 及开发依赖。
如需直接进入环境，执行 `source .venv/bin/activate`。

```bash
uv run ruff check .
uv run ruff format --check .
```

`pytest` 已列为开发依赖；算法实现后再加入相应测试并启用 CI 测试步骤。

## 目录

```text
.agents/                 项目上下文与语义约定
.github/workflows/       CI 格式、静态检查及 tracing 示例
src/jax_reorder/         Python 包骨架
examples/               可运行的 JAX tracing 示例
tests/                  后续算法测试
AGENTS.md                Agent 开发指引
pyproject.toml           项目与工具配置
uv.lock                  锁定依赖
```

## 下一步

1. 实现变量规范化及保持依赖关系的确定性拓扑排序。
2. 实现包含常量、类型和输出结构的完整表示与比较。
3. 加入交换律、结合律规则、改写记录和浮点精度 warning。
4. 逐步支持嵌套 jaxpr，并明确处理 effects 和不支持的 primitive。
