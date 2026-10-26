# 历史：现状与证据

> 状态：历史源码调查与探针记录，旧版“不支持”结论不代表当前状态；现行范围见[Pallas 设计](../../../design-docs/current/pallas-jax-equivalence.md)。

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

## 2. 现状与证据

### 2.1 当前代码已具备什么

| 层次 | 当前能力与缺口 | 入口 |
| --- | --- | --- |
| JAX 结构比较 | 比较完整图、操作数位置、参数、常量、抽象值和已适配的嵌套区域；不是只比较输出可达子图。 | [core/canonical.py](../../../../src/jax_reorder/core/canonical.py) |
| 原语识别 | 按 Primitive 对象身份识别公开 lax、已加载 lax 内部原语及特定包装原语；参数和 effects 仍需通过语义检查。尚不等于所有 JAX 原语。 | [core/support.py](../../../../src/jax_reorder/core/support.py) |
| JAX 数学比较 | 结构快路径后，将支持的完整输出公式及定义条件交给可信 SymPy；不保证任意图都能转换或判定。 | [algebra/compare.py](../../../../src/jax_reorder/algebra/compare.py)、[algebra/trusted.py](../../../../src/jax_reorder/algebra/trusted.py) |
| 坐标与公式 | 已有精确整数坐标、Indexed/Sum 和区域转换；没有完整的 Pallas 全局写入域和 TPU 同步语义适配。 | [algebra/indices.py](../../../../src/jax_reorder/algebra/indices.py)、[algebra/symbolic.py](../../../../src/jax_reorder/algebra/symbolic.py)、[algebra/regions.py](../../../../src/jax_reorder/algebra/regions.py) |
| 函数契约 | 记录 pytree、静态参数、抽象输入输出、追踪环境；前向、JVP、VJP 分别检查。 | [functions.py](../../../../src/jax_reorder/functions.py) |
| Pallas | 当前 `pallas_call` 明确抛出不支持异常，包括图与自身比较；CPU interpret 执行不等于比较器已支持。 | [test_full_attention.py](../../../../tests/test_full_attention.py) |
| 结果与错误 | 完整比较区分结构等价、数学等价、数学不等价；结构阶段的 `different` 不能推出数学不等价，未完成判定抛异常。 | [status.py](../../../../src/jax_reorder/status.py)、[errors.py](../../../../src/jax_reorder/errors.py) |

[CPU lax 清单](../../../references/lax-coverage.md)由 [lax_inventory.py](../../../../tests/lax_inventory.py) 维护，记录的是固定版本
公开 lax 原语及测试绑定，不能当作全部 JAX / Pallas / TPU 原语清单。

### 2.2 CPU tracing 实测事实（JAX 0.11.2）

以下为本轮在 CPU 上实际追踪到的事实，用于判断视图字段与适配边界是否可落地；它们不是 TPU 执行证据。

- `pallas_call` traced 参数共 11 个：`out_avals, jaxpr, debug, interpret, grid_mapping, mesh,
  input_output_aliases, compiler_params, cost_estimate, metadata, name`；外层没有 BlockSpec callable。
- `GridMapping` 保存 `grid`、`grid_names`、`block_mappings`、`index_map_tree`、`index_map_avals`、
  `vmapped_dims`、`scratch_avals`、`num_index_operands`、`num_inputs`、`num_outputs` 等字段；
  `BlockMapping` 保存 `block_shape`、`transformed_block_aval`（`AbstractRef`）、
  `index_map_jaxpr`（已 trace 的 `ClosedJaxpr`）、`index_map_out_tree`、`array_aval`、`origin`、
  `transforms`、`pipeline_mode`。
- 最小探针 A（128 分块、正序）的 index map jaxpr 是恒等；B（256 分块、反向）是 `lambda p: 3 - p`，
  两者都可用 `jax.core.eval_jaxpr` 求值。index map 已 trace **不等于**它一定纯、静态或仿射，
  仍需逐项审查。
- kernel 区域中 `get`/`swap` 的参数只有 `tree=PyTreeDef(NDIndexer(Slice(0,128,1), (128,), ()))`；
  Ref 的 aval 是 `AbstractRef(inner_aval=ShapedArray, memory_space=...)`；effects 精确到 Ref 端口；
  `program_id[axis=0]` 是普通原语，可以作为数据参与计算。
- DMA 在 CPU tracing 下即可出现：`dma_start` 参数为 `tree`（src/dst/dst_sem/src_sem/device）、
  `device_id_type`、`priority`、`add`；`dma_wait` 参数为 `tree`、`device_id_type`、
  `insert_dummy_device`、`is_wait_send`；effects 覆盖 Ref 与 semaphore 端口。
- `interpret=True/False` 的 kernel jaxpr 文本相同，只差外层 `interpret` 参数；
  `compiler_params.dimension_semantics` 承载顺序/并行语义，其余 `CompilerParams` 字段混合了
  同步、数值、检查、编译和资源语义，不能整体忽略。

本地最小探针（2026-10-18，CPU）：两份 `y = x + 1` 的 Pallas 程序。

| 项目 | A | B |
| --- | --- | --- |
| 全局输入输出 | `float32[1024]` | `float32[1024]` |
| block 大小 | 128 | 256 |
| grid | `(8,)` | `(4,)` |
| 输入与输出 index_map | `p -> (p,)` | `q -> (3 - q,)` |
| kernel 内部原语 | `get`, `add`, `swap` | `get`, `add`, `swap` |
| 内部 effects | `ReadEffect`, `WriteEffect` | `ReadEffect`, `WriteEffect` |
| 引用抽象值 | `Ref<default>{float32[128]}` | `Ref<default>{float32[256]}` |

两份图的外层原语均为 `pallas_call`。A↔A、B↔B、A↔B 的当前结构比较均抛出：

```text
UnsupportedJaxprError:
structural:left.jaxpr.eqns[0] [unsupported_semantics]: unsupported primitive pallas_call
```

两份函数以 `interpret=True` 在 `cpu:0` 执行；对唯一输入 `arange(1024, dtype=float32)`，输出均与
NumPy 的 `x + 1` 精确相同。这只是一次抽样执行证据，不是比较器已经证明等价，更不是 TPU 编译、执行
或并发验证。可复现的构造如下；使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python` 运行：

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def build(block, reverse):
    groups = 1024 // block

    def index_map(p):
        return (groups - 1 - p if reverse else p,)

    spec = pl.BlockSpec((block,), index_map)

    def kernel(x_ref, y_ref):
        y_ref[:] = x_ref[:] + 1.0

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
        grid=(groups,),
        in_specs=(spec,),
        out_specs=spec,
        interpret=True,
    )


a, b = build(128, False), build(256, True)
```

### 2.3 本轮 review 确认的缺口

| 事项 | 结论 | 依据 |
| --- | --- | --- |
| 源语义解释次数 | core 结构路径与 algebra 适配各解释一次；`freeze` 仍解析 scan FlatTree、reduce computation、cast sharding、default_jit、共同调用和非有限子图。Pallas 若照此实现会成为第三次。 | [core/params.py](../../../../src/jax_reorder/core/params.py)、[core/semantics.py](../../../../src/jax_reorder/core/semantics.py)、[algebra/adapter/transport.py](../../../../src/jax_reorder/algebra/adapter/transport.py) |
| `ProgramView` 与 `Graph` | 职责重叠：两者都保存操作数位置、参数、常量、区域和输入输出。`Graph` 是类型化比较键，但字段仍绑定 JAX 对象（`Frozen` 实际为 `Any`），不能直接承担可分析语义模型。 | [core/canonical.py](../../../../src/jax_reorder/core/canonical.py) |
| effects 与 Ref | canonical 只接受 `NamedAxisEffect` / `NoDCEEffect`；algebra 拒绝任何 effectful 图和非 `ShapedArray` 值。kernel 的 `ReadEffect` / `WriteEffect` 与 `AbstractRef` 被两层拒绝。 | `check_effects`、`freeze` |
| 原语与参数审计 | 身份扫描只覆盖 lax/pjit/shard_map/custom primitives，不覆盖 Mosaic Pallas 原语；adapter inventory 也没有 Pallas 参数 schema。 | [core/support.py](../../../../src/jax_reorder/core/support.py)、[adapter_inventory.py](../../../../tests/adapter_inventory.py) |
| 文档与 CI | Ruff 0.17 会格式化本文件内嵌 Python 代码块；原样提交会使 `ruff format --check .` 失败。README/ARCHITECTURE 仍是 UNKNOWN/无代数模式的旧描述（受保护，需授权更新）。 | CI 配置、技术债 DOC-002 |
