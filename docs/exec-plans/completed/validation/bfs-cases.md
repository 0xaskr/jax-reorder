# AI 组合案例与 SGLang-JAX 模型层

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

> 状态：历史验证与固定测试资产说明；原数字、接口口径和“当前缺口”属于记录时点，归档不表示测试资产废弃。现行支持状态见[贡献者指南](../../../../CONTRIBUTING.md)，现行契约见[设计索引](../../../design-docs/index.md)。

本轮只判断计算图等价。案例由 AI 在开发阶段编写，先保存判断、节点对应关系和预期数值，再调用比较器复核；CI 不调用在线 AI，也不把比较器的输出反向用作真值。

## 固定代码与判断

[tests/bfs_reordering.py](../../../../tests/bfs_reordering.py) 保存完整基准和候选代码、输入、静态参数、判断与理由。`reverse=True` 为独立定义换序，`wrong=True` 为表中的负例。`verify_case` 先复现预先写下的数值证据，再检查图比较结果。[测试](../../../../tests/test_ai_cases.py) 还执行两个重排后的 jaxpr，并检查幂等性。

每类有一个图等价正例和一个图不等价负例，共 18 例。所有正例按中间值角色一一对应：a 与 a、b 与 b、共享前驱与共享前驱、gate/up/down 与同名投影；定义的文本位置改变，使用关系不变。静态标志仅决定追踪时选择哪段代码，不属于运行时输入。

| 基准 | 正例调整 | 负例差异 | 固定 CPU 反例：基准 → 候选 |
| --- | --- | --- | --- |
| 逐元素多支汇合 | 独立 square、neg 换序 | 汇合 mul → add | `-12 → 1` |
| 点积与广播偏置 | dot 与 bias 广播换序 | add → sub | `[[2,4],[4,6]] → [[0,0],[2,2]]` |
| transpose、reshape、slice、concat | 两条布局分支换序 | transpose 维度 `(1,0) → (0,1)` | `[1,3,2,4,1,2] → [1,2,3,4,1,2]` |
| 归约 | reduce 与 scale 广播换序 | 归约轴 `0 → 1` | `[2,4] → [1,5]` |
| 比较与选择 | 条件与 neg 分支换序 | gt → lt | `[-2,-7] → [-5,3]` |
| sort 的多结果端口 | 两条消费分支换序 | neg 读取 values → keys | `([2,3],[-10,-20]) → ([2,3],[-1,-2])` |
| 菱形与共享前驱 | 独立 neg 与 mul 换序 | 共享 mul 读取 y → x | `(15,-54) → (13,-36)` |
| cond 嵌套区域 | 真分支内部换序 | 真分支 mul → add | `-12 → 1` |
| LlamaMLP | gate/up 投影换序 | up * silu(gate) → up + silu(gate) | `[[0]] → [[2]]` |

反例只证明这些固定候选在给定有效输入上数值不同。比较器返回 `different` 的一般情况只说明图不同；例如交换操作数或复制共享计算可能仍得到相同数值，本轮不调用代数规则判定。

## 模型来源与提取范围

- 来源：本地 `/home/askr/work/sglang-jax-private`，提交 `538abe3fdba84b586a6a2fd151cac67cc974d35f`。
- `python/sgl_jax/srt/models/llama.py`：`LlamaMLP` 的三次投影、`silu` 及中间乘法；源文件 SHA-256：`1ec85cab1d3a6c6aa9faeac5474d75af95f049c3566110d7332007a7e07c840e`。
- `python/sgl_jax/srt/layers/linear.py`：`LinearBase.__call__` 的 `dot_general`、`preferred_element_type` 和默认 `out_sharding`；SHA-256：`da2fbba0348c665a8d1020415587ac05ac7489f19b95ee74e111953c49643006`。
- 原项目使用 Apache-2.0，许可证副本位于 [tests/LICENSE.sglang-jax](../../../../tests/LICENSE.sglang-jax)。示例重写为纯函数，移除模块构造、权重加载及日志，保留下面选定配置的计算。

选定无偏置、无量化、全部投影 `params_dtype=float32` 的 LlamaMLP。CPU mesh 为 `(data=1, tensor=1)`，两轴均 Explicit。输入采用 `P("data", None)`；gate/up 权重为 `P(None, "tensor")`；down 权重为 `P("tensor", None)`。投影按输入最后一维与权重第零维收缩，输出分片与原 `LinearBase` 的 `kernel_axes[-1]` 规则一致。`jax.nn.silu` 产生的 jit 子图保留并递归比较。

最小反例固定输入 `x=[[1]]`、gate 权重 `[[0]]`、up 权重 `[[2]]`、down 权重 `[[1]]`。另用 NumPy 随机种子 17 验证 `(batch=2, hidden=3, intermediate=4)` 的非平凡形状。测试不依赖本地源仓库、Flax 或模型下载；源版本和提取对应关系已固化。此基准不表示完整模型、全部 dtype 配置或 SGLang 运行时均已验证。

## 原语补充与复现

[原语清单](../../../references/lax-coverage.md)覆盖模型未用到的原语；每个 CPU 可运行固定原语用例还加入一个独立 neg 分支，在两种语句顺序下比较。另有合法方程排列、变量重命名、共享/重复节点、长链、常量字节、多输出端口及嵌套图性质测试。这些检查与 AI 判断互相补充，不称为全部计算图的穷举。

```bash
uv run python tests/bfs_reordering.py
uv run pytest
uv run python tests/lax_inventory.py
```

基线为 Python 3.12、JAX/jaxlib 0.11.2、CPU。pytest 默认使用两个逻辑 CPU 设备以验证集合通信；原语固定用例 X64 关闭，dtype/shape 扩展用例在局部启用 X64。

## 保存的实际追踪原语

按方程顺序记录；在调用节点后递归列出其子图，保留重复出现。输入和静态配置见上文及固定代码。

| 用例 | 基准 | 候选 |
| --- | --- | --- |
| `elementwise/reorder` | `mul, neg, mul` | `neg, mul, mul` |
| `elementwise/changed` | `mul, neg, mul` | `mul, neg, add` |
| `dense_bias/reorder` | `dot_general, broadcast_in_dim, add` | `broadcast_in_dim, dot_general, add` |
| `dense_bias/changed` | `dot_general, broadcast_in_dim, add` | `dot_general, broadcast_in_dim, sub` |
| `layout/reorder` | `transpose, reshape, slice, reshape, concatenate` | `slice, reshape, transpose, reshape, concatenate` |
| `layout/changed` | `transpose, reshape, slice, reshape, concatenate` | `reshape, slice, reshape, concatenate` |
| `reduction/reorder` | `reduce_sum, broadcast_in_dim, sub` | `broadcast_in_dim, reduce_sum, sub` |
| `reduction/changed` | `reduce_sum, broadcast_in_dim, sub` | `reduce_sum, broadcast_in_dim, sub` |
| `selection/reorder` | `broadcast_in_dim, gt, neg, select_n` | `neg, broadcast_in_dim, gt, select_n` |
| `selection/changed` | `broadcast_in_dim, gt, neg, select_n` | `broadcast_in_dim, lt, neg, select_n` |
| `result_ports/reorder` | `sort, broadcast_in_dim, add, neg` | `sort, neg, broadcast_in_dim, add` |
| `result_ports/changed` | `sort, broadcast_in_dim, add, neg` | `sort, broadcast_in_dim, add, neg` |
| `diamond/reorder` | `mul, neg, mul, add, mul` | `mul, mul, neg, add, mul` |
| `diamond/changed` | `mul, neg, mul, add, mul` | `mul, neg, mul, add, mul` |
| `conditional/reorder` | `gt, convert_element_type, cond, sub, mul, neg, mul` | `gt, convert_element_type, cond, sub, neg, mul, mul` |
| `conditional/changed` | `gt, convert_element_type, cond, sub, mul, neg, mul` | `gt, convert_element_type, cond, sub, mul, neg, add` |
| `llama_mlp/reorder` | `dot_general, dot_general, jit, logistic, mul, mul, dot_general` | `dot_general, dot_general, jit, logistic, mul, mul, dot_general` |
| `llama_mlp/changed` | `dot_general, dot_general, jit, logistic, mul, mul, dot_general` | `dot_general, dot_general, jit, logistic, mul, add, dot_general` |
