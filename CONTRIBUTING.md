# 贡献者指南

本页集中维护开发入门、当前支持状态和后续路线图。代码模块与调用链见[代码导读](docs/design-docs/current/code-map.md)，设计的生效与过期状态见[设计索引](docs/design-docs/index.md)。

## Quick Start

使用 Python 3.12 和 uv；当前锁定 JAX/jaxlib 0.11.2、SymPy 1.14.0，可在 CPU 环境开始开发。

```bash
git clone https://github.com/0xaskr/jax-reorder.git
cd jax-reorder
uv sync --locked

# 查看原始 jaxpr、BFS 编号与重排结果
uv run python tests/inspect_jaxpr.py

# 检查图比较、重排前后执行与数值反例
uv run python tests/check_correctness.py
```

`uv sync` 创建 `.venv`，默认同时安装 `dev` 和 `test` 两组依赖：
`dev` 包含 Ruff、basedpyright，`test` 包含 pytest 和真实模型层验证使用的 Flax。
只安装运行时和测试依赖可用 `uv sync --locked --no-default-groups --group test`。
缓存不可写时，设置 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`。

使用函数入口比较两个函数，并分别检查前向与一阶求导：

```python
import jax
import jax.numpy as jnp

from jax_reorder import compare_functions


def original(x, y):
    a = jnp.sin(x)
    b = jnp.cos(y)
    return a + b


def candidate(x, y):
    b = jnp.cos(y)
    a = jnp.sin(x)
    return a + b


signature = jax.ShapeDtypeStruct((4,), jnp.float32)
report = compare_functions(
    original, candidate, signature, signature, modes=("forward", "jvp", "vjp")
)
for mode in report.modes:
    print(mode.mode, mode.status.value, mode.equivalence_kind)
# forward / jvp / vjp 均为 equivalent structural
```

默认 `equivalence="auto"` 先比较结构，结构不同后尝试数学比较；只检查完整图可设置
`equivalence="structural"`。更多示例见[集合通信](tests/collective_jaxpr.py)、
[业务改写](tests/business_layer_cases.py)和[真实模型层](docs/exec-plans/completed/validation/full-attention.md)。

## 当前选择暂不支持

以下三项是已明确的阶段范围取舍，不列入当前实现待办，也不作为近期完成条件：

- `core_map`。
- `mpmd_map`。
- 输入输出共享 HBM，包括将历史受限 alias 试验扩大为一般支持承诺。

该决定记录在 [Pallas 当前计划](docs/exec-plans/active/pallas-semantic-layer.md#当前待办)。
上一节的动态地址、共享 staging、布局及多轴全局证明属于尚未完成的能力，不归入这份主动排除清单。

## 当前支持状态

以下按 2026-10-26 的源码审查整理，不代表任意合法 JAX/Pallas 程序均已覆盖。

| 范围 | 已支持 | 主要边界与详情 |
| --- | --- | --- |
| JAX 结构比较 | 完整 jaxpr 规范化、独立方程换序、嵌套区域绑定、未使用子图及原图对应复核 | 保留操作数、常量、类型、参数与依赖；结构不同不推出数学不等价。[完整图](docs/design-docs/current/complete-graph.md) |
| 函数与求导 | 输入输出 pytree、静态参数、追踪环境；分别检查 forward、JVP、VJP | 仅针对声明签名和请求模式，不外推一般 Python 行为或高阶求导。[函数契约](docs/design-docs/current/function-and-derivative-contract.md) |
| JAX 代数比较 | SymPy 完整输出与定义条件比较，受限索引、归约、非线性及区域组合 | 浮点使用理想实数解释，不保证逐位一致；普通 JAX 的纯死计算删减尚未完成。[代数边界](docs/design-docs/current/algebraic-equivalence.md#当前边界) |
| Pallas 结构与资源 | 完整图及设备局部比较；受限 grid、内存版本、初始化、DMA 与通信协议证明 | 已有远端条件初始化、只读源有界动态页多笔在途 DMA；执行绑定仍由调用方保证。[Pallas 导读](docs/design-docs/current/pallas-jax-equivalence.md) |
| Pallas 数学比较 | 将已证明合法的输出值投影接入 SymPy；支持完整函数与设备局部范围 | 局部入口每侧一个静态 kernel，`device_local` 的函数接口仅接受 forward；局部结论不代表整个分布式程序等价。[支持边界](docs/design-docs/current/pallas/validation.md) |

结果区分 `equivalent`、结构 `different` 和数学 `inequivalent`；未支持语义、未决、预算耗尽或超时
通过 `ComparisonError` 子类报告，不返回 `unknown`。CPU 数值检查、解释器验证、离线 TPU AOT
与实机精度/性能证据分别记录，不能互相替代。

## 尚未完成的能力

以下是实现或覆盖尚未完成的缺口，后续按路线图推进；已有受限实现的项目明确列出剩余范围。
它们与下一节主动排除的功能分开维护，也不表示对应源图一定无法做结构比较。

| 范围 | 还没做到什么 | 已有能力与跟踪 |
| --- | --- | --- |
| 普通 JAX 纯死计算删减 | 数学转换仍处理未使用的纯计算，并保留其定义条件 | Pallas 数学投影已做受限删减；普通 JAX 待实现，见 ALG-009 |
| JAX 索引与数学转换 | 动态索引域、对角读取、任意步长覆盖、部分排序/近似 top-k 残差，以及特殊精度/分片尚无完整规则 | 已有静态 reshape/切片、受限索引与归约；见 ALG-001/003/004 |
| JAX 循环与跨区域改写 | 一般递推、变化 carry 的历史输出、数据依赖终止和非默认调用尚无通用证明 | 已有受限 cond、独立 scan 累加和默认纯 JIT 组合；见 ALG-002/004 |
| 求导适配与覆盖 | 仅存在于求导规则中的动态捕获、规则新增的有序操作/未解释 effects 仍需适配；高阶求导和任意变换组合尚无整体覆盖 | 普通函数 forward/JVP/VJP 已分别支持；见 AD-001～003 |
| Pallas 数据条件与动态地址 | 更一般的数据谓词关系、未知范围地址、动态可写源/目标的普遍处理，以及数据决定的路由/终止仍缺证明规则 | 已有受限条件初始化、只读源有界动态页多笔在途 DMA；动态可写访问仍须完整状态证明 |
| Pallas 并行与布局 | 并行共享自动 staging 的一般安全复用、非默认物理布局、动态 grid 及未审核的包装/effect 尚未普遍覆盖 | 已有静态多维 grid、受限布局选项和显式执行绑定；真实存储冲突仍须拒绝 |
| Pallas 全局比较 | 一般多轴 mesh 的跨设备值组装与全局协议证明尚未完成 | 有限单轴协议、多维 kernel grid 和设备局部比较已实现，不能外推一般全局结论 |
| 统一图表示与资源控制 | 全部 JAX 消费者迁入统一视图、持久化图格式及数学 worker 的进程内存硬上限尚未完成 | 当前有展开预算与 worker 超时；不保证所有合法实例都在默认预算内完成 |

编号对应[待处理清单](docs/exec-plans/index.md#待处理)；Pallas 的具体准入条件与剩余覆盖见
[当前边界](docs/design-docs/current/pallas/validation.md)和[实施计划](docs/exec-plans/active/pallas-semantic-layer.md)。

## 已知正确性缺陷

以下已有实现，但会产生错误结论，属于待修复缺陷：

- **ALG-006：**整数归约/点积遗漏溢出语义，可能错误判等价；普通 JAX 与 Pallas 数学入口均受影响。
- **ALG-007：**`iota` 遗漏 dtype 值域，可能错误判等价。
- **ALG-008：**空整数极值归约的单位元错误，可能把相等结果判为不等价。

复现与修复要求见[缺陷记录](docs/exec-plans/active/pallas-semantic-layer.md#2026-10-26-审查确认的正确性缺陷)。

## 比较前提与验证范围

这些是使用条件或证据范围，不计入上述“选择暂不支持”清单：

- **比较契约：**结论针对声明的输入签名、静态参数、追踪环境和模式，不证明一般 Python 等价。
  浮点代数采用理想实数解释，不提供一般 IEEE 浮点逐位等价或误差上界证明。
- **局部范围与执行前提：**`device_local` 每侧一个静态 kernel，函数入口仅接受 forward；
  执行绑定由调用方提供并保证，不配置实际调度。远端并行仍要求完整单 core 点流与设备同序。
- **未验证与上游限制：**现有 CPU、解释器和部分离线 TPU AOT 证据不覆盖多主机及任意 GPU/TPU 执行，
  没有 TPU 实机精度/性能结论。JAX 追踪或后端自身拒绝的组合单独记录，不能归为本项目主动排除；
  详见[Pallas 源后端限制](docs/exec-plans/completed/pallas-history/validation-snapshot-2026-10-26.md#已确认的源后端限制)。

## 后续路线图

1. **先修复正确性缺陷。**补齐上述整数语义，增加普通 JAX 与 Pallas 公共入口的正例、反例和拒绝路径。
2. **补齐 JAX 数学比较。**实现普通 JAX 的纯死计算删减；按具体用例扩展动态索引、循环及跨区域改写，保留定义条件和完整输出检查。
3. **扩展 Pallas 局部语义。**逐类覆盖更一般的数据谓词、动态 DMA 地址与状态、共享 staging 安全复用、数据路由/终止和非默认物理布局。
4. **独立推进全局与架构工作。**扩展多轴 mesh 全局值组装与协议证明，迁移 JAX 消费者到统一视图；持续处理已确认的大图和 CAS 预算瓶颈。

具体待办与完成条件维护在[待处理清单](docs/exec-plans/index.md#待处理)和
[Pallas 当前计划](docs/exec-plans/active/pallas-semantic-layer.md)；推进范围以本页“尚未完成的能力”为准，
主动排除项单列于“当前选择暂不支持”。

## 开发与验证

实现按 `core`（结构与资源语义）、`algebra`（数学后端）、`functions`（函数契约编排）分层，底层不得反向依赖上层。
新增规则先明确支持范围、反例和拒绝条件，在 `docs/design-docs/current/` 维护设计与验证方法，
在 `docs/exec-plans/` 记录实施过程和实际验收结果。
不能通过丢弃依赖、效果或未定义状态获得匹配；抽样数值结果不能替代图或数学比较。

提交前运行 Ruff、格式检查和 tracing 示例；算法变更还需运行 pytest。文档变更检查链接和差异：

```bash
uv run ruff check .
uv run ruff format --check .
uv run python tests/inspect_jaxpr.py
uv run pytest
uv run python tools/check_docs.py --changed
git diff --check
```

运行时、开发工具、测试依赖分别通过 `uv add`、`uv add --dev`、`uv add --group test` 管理，
并同步 `uv.lock`。原语与适配清单可运行
`uv run python tests/lax_inventory.py`、`uv run python tests/adapter_inventory.py`；
测试分组与环境边界见[系统验证矩阵](docs/design-docs/current/verification-matrix.md)。

## 测试用例来源与组织

允许使用已有开源软件栈中的模型层、子模块、Pallas kernel 或单个函数作为基准，
提取所需代码及依赖上下文，配合 AI 生成的正例和反例，验证 jax-reorder 的 JAX 与 Pallas
结构、资源语义和代数比较。用例可以针对某一层级或函数，不要求运行完整模型。

- **来源可追溯：**记录上游仓库、固定提交、文件与符号、许可证，以及提取时必要的适配。
  原始快照与清单放在 `tests/fixtures/`，保留许可证和来源校验信息。
- **候选固定保存：**将 AI 生成的正反例源码、预期结论及理由提交到仓库；测试运行时不调用 AI，
  不根据比较器的实际输出反向生成标签。预期由原始语义、公式推导或独立参照核对。
- **覆盖实际语义：**正例注明保持图结构还是仅保持数学关系；反例区分结构不同、数学不等价和
  应当拒绝的输入。JAX 保留签名、静态参数与求导模式；Pallas 还需保留 grid、内存、mask、
  DMA/通信及执行绑定等必要上下文，不能把 kernel 换成纯 JAX 参照后声称覆盖了 Pallas 路径。
- **分开记录证据：**需要数值验收的用例提供独立参照或反例见证；只做追踪的用例注明限制。
  抽样相同不证明等价，结构不同也不证明数学不等价；CPU、解释器、AOT 和实机证据分别记录。

可复用候选、模型层与可运行入口统一放在 `tests/`，pytest 断言放在 `test_*.py`。
现有入口包括 [SGLang 层](tests/sglang_layers/)、[完整 Attention/block](tests/full_layers/)、
[DeepSeek 层](tests/deepseek_v41/)和 [Pallas kernel](tests/pallas_kernels.py)。

更多约定见 [AGENTS.md](AGENTS.md)，比较契约见 [ARCHITECTURE.md](ARCHITECTURE.md)。
架构文档中的旧接口说明尚待同步，当前接口与异常行为以本页链接的现行设计文档及源码为准。
