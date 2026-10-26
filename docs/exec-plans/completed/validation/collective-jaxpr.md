# 带通信的 jaxpr

> 状态：历史验证与固定测试资产说明；原数字、接口口径和“当前缺口”属于记录时点，归档不表示测试资产废弃。现行支持状态见[贡献者指南](../../../../CONTRIBUTING.md)，现行契约见[设计索引](../../../design-docs/index.md)。

## 输入与比较边界

本次实验沿用锁定的 JAX/jaxlib 0.11.2、Python 3.12 和 CPU。通过 `shard_map` 把显式集合通信保留在 jaxpr 中，比较完整外层图，不单独取出失去 mesh 轴环境的子图。

通信节点同样保留操作数位置、结果端口与所有 primitive 参数，包括通信轴、设备分组、置换方向、gather/scatter 维度；外层 mesh 设备顺序、输入输出分片规格及 `check_vma` 配置也属于比较信息。合法的独立通信可以随计算节点一起重排，数据依赖仍须保留。均匀控制流内的通信在各嵌套区域中独立处理。

这里验证单进程内多个逻辑 CPU 设备。多主机调用顺序、GPU/TPU lowering、异步点对点通信不由该实验验证；既有 `psend` / `precv` CPU 拒绝检查继续保留。不同通信实现数值相同也不能在本轮合并为同一图。

## 实验与验收

用例保存在 [collective_jaxpr.py](../../../../examples/collective_jaxpr.py)，回归在 [test_collectives.py](../../../../tests/test_collectives.py)。AI 预先给出图标签及样本相同/不同的预期，脚本分别执行原函数和规范化后的完整 jaxpr，验证完整结构幂等性；负例打印实际数值反例或明确记录样本相同。

首次运行发现：`check_vma=True` 的通信结果与 varying 本地值组合、或拼接输出时，JAX 插入 `core.pvary_p`；当前发现机制只扫描 lax、pjit 及 shard_map 节点，遗漏定义于 core 的该辅助原语，导致返回 `unknown`。锁定版本的 `parallel.py` 将其 lowering 为原值，但抽象求值将指定轴从 invariant 改成 varying，轴参数与输出抽象值仍必须保留，不能删除此节点。修复只显式登记该 Primitive 对象，沿用完整参数编码、轴环境与 JAX 类型检查；不扩大到任意 core 原语。

构造嵌套用例时核实了两点：`scan` carry 和 `cond` 分支除了 shape/dtype，还要求手动轴变化类型一致，因此在归约后接入本地输入，保持 varying 类型；`cond` 使用闭包时，独立语句换序可能使 JAX 改变分支捕获的参数顺序。初版将后一种情况判为 `different`；后续 [绑定修复](../../../design-docs/archive/cond-bindings.md) 对实参与全部分支形参同步规范化，闭包版本现应判为 `equivalent`。显式 `(x, y)` 传参版本继续作为对照。

## 固定组合与结果

2026-10-11，本地两个和四个逻辑 CPU 设备分别运行 22 个组合，每个组合 4 组输入，均通过预期标签和两侧重排执行检查。`cond` 的谓词由全局求和产生，正负样本分别进入通信分支与本地分支，所有设备执行相同分支。

| 组合 | 数量 | 图结论 | 数值证据 |
| --- | --- | --- | --- |
| psum 与 pmax 换序；两个 psum 换序 | 2 | equivalent | 样本相同 |
| psum、pmin、ppermute、all_to_all、scatter→gather 各自与本地计算换序 | 5 | equivalent | 样本相同 |
| jit、scan、显式传参 cond 内独立通信换序 | 3 | equivalent | 样本相同 |
| cond 闭包捕获顺序改变 | 1 | equivalent | 样本相同，保持调用与分支的实际绑定 |
| 分片矩阵乘法内独立偏置广播换序 | 1 | equivalent | 样本相同；另与 NumPy 完整矩阵乘法核对 |
| psum→pmax、通信输入绑定改变、置换路径改变、归约设备分组改变、all_to_all 维度改变 | 5 | different | 各有数值反例 |
| 激活跨越归约；分片矩阵乘法遗漏 psum | 2 | different | 各有数值反例 |
| 加法操作数交换；all-reduce→scatter+gather | 2 | different | 样本相同，不消除图差异 |
| 添加未使用的通信 | 1 | unknown | 未使用方程保留，返回原因；样本相同 |

绑定修复后合计 12 个 `equivalent`、9 个 `different`、1 个 `unknown`。这不是通信组合的穷举，也不是不同通信算法的数学等价证明。

主例中两条原始通信的顺序相反，规范化后均为 `psum_invariant(x) → pmax(y) → sub → pvary`；测试断言原始方程序列确实不同，规范化确实移动了通信节点，并检查 CPU StableHLO 含 `all_reduce`。`pvary` 保留轴参数与输出变化类型；同名的自定义 Primitive 仍被拒绝。

新增 49 项 pytest：20 个基础组合分别在 `check_vma=False/True` 的配置下验证（设备子组用例固定关闭检查，符合 JAX 要求）；独立通信另覆盖 int32、float16、float32；分片矩阵乘法正反例与 NumPy 参照；通信轴名字、mesh 设备顺序、输出分片接口变化；`pvary` 保留与身份拒绝。全部输出均检查 dtype、shape 和容差内数值，不宣称跨平台逐位一致。

```bash
uv run python examples/collective_jaxpr.py --show-jaxpr
uv run python examples/collective_jaxpr.py --devices 4
uv run pytest tests/test_collectives.py -q
uv run pytest -q
```

通信功能首次验证：通信回归 **49 passed**；全量回归 **565 passed**；Ruff、格式检查、tracing 示例、原语清单一致性与 `git diff --check` 均通过。后续绑定修复的重新验证见 [修复记录](../../../design-docs/archive/cond-bindings.md)。四设备示例已加入 CPU CI，在线 CI 结果尚未验证。原语清单仍描述原有 182 个固定原语用例；本次新增的通信组合及 `pvary` 执行证据在本页单独记录。

## 上游依据

- [JAX shard_map 文档](https://docs.jax.dev/en/latest/201/shard-map.html)：手动分片、显式 collective 与轴变化类型。
- [jax.lax 并行算子](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators)：集合通信接口。
- [JAX 多进程说明](https://docs.jax.dev/en/latest/multi_process.html)：参与通信的进程需要一致的执行顺序；本实验不外推为多主机验证。
- 本地 `.venv/lib/python3.12/site-packages/jax/_src/lax/parallel.py`：核对锁定版本的 named-axis effects 和参数；`shard_map.py` 用于核对嵌套轴环境。
