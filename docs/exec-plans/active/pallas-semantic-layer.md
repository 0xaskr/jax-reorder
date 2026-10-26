# JAX / Pallas 语义层：当前计划

创建：2026-10-18。状态：**实施中**；当前范围于 2026-10-26 对照源码与固定测试核对。
2026-10-22 已完成设备局部数学入口、受限远端条件初始化、顺序输出映射及有界动态页 DMA；本次文档整理未重跑测试。
默认只读本页和[开发入口](../../design-docs/current/pallas-jax-equivalence.md)，按任务选择一个专题。
已完成工作与原始命令移至[历史快照](../completed/context-history/pallas-semantic-layer-2026-10-21.md)，
快照中的旧缺口不代表当前状态。

## 目标与完成条件

比较分布式程序中单个 device 的一个 Pallas kernel，mesh 可以存在；保留必要设备/轴/实参与通信依赖。
完整结构比较、可选资源/协议证明与纯值数学判定分开。每次扩展须有正例、反例、拒绝路径和源码证据。
grid/地址/初始化/同步留在 core；SymPy 消费已证明合法的值投影，默认 60 秒。实机不阻塞图/AOT 验证。

## 当前待办

| 优先级 | 工作项 | 完成条件 / 约束 |
| --- | --- | --- |
| 优先修复 | 共享数学层的整数语义 | ALG-006 的归约/点积错误影响 Pallas 两个数学入口；同时跟踪普通 JAX 的 ALG-007/008，具体范围与验收见[缺陷复现](#2026-10-26-审查确认的正确性缺陷) |
| 后续覆盖 | 更一般的数据条件初始化 | 保留已支持的条件未定义状态、互补写入和受保护读取；按源事实扩展谓词关系，漏初始化/错误 predicate/在途 DMA 仍拒绝 |
| 后续覆盖 | 动态 DMA 地址与状态 | 保留只读源有界页号的多笔在途支持；动态可写源/目标须有完整状态证明，未知范围或非法等待仍拒绝 |
| 后续覆盖 | 并行共享自动 staging | 按真实存储和必要同步证明复用；不私有复制 Ref 或插入 barrier 绕过冲突 |
| 后续覆盖 | 一般索引、数据路由/终止、非默认布局 | 按源事实逐类增加确定规则与拒绝反例；不以采样或 CAS 替代地址/通信证明 |
| 持续 | 大图与 CAS 预算 | 只优化已确认瓶颈，保留预算收费及未决结论；一般复杂查询不承诺 60 秒内完成 |
| 独立后续 | JAX 全消费者统一视图、多轴全局协议/组装 | 不作为设备局部第一版的额外准入条件；保留各自契约 |

具体已知边界与源后端限制见[验证专题](../../design-docs/current/pallas/validation.md)。
`core_map`、`mpmd_map` 与输入输出共享 HBM 按用户决定暂不支持；MLIR/ISL/Z3 不作为当前依赖。

## 2026-10-26 审查确认的正确性缺陷

本节集中保留共享数学层的待修复缺陷。ALG-006 已在普通 JAX 与 Pallas 入口复现；ALG-007/008 的证据为普通 JAX，不能外推为已验证的 Pallas 缺陷。
以下在 `93e93ca`、Python 3.12、JAX/jaxlib 0.11.2、SymPy 1.14.0 的 CPU 环境复现，均待修复。
实际输出与公共数学入口结论分别记录；这些整数差异不能由理想实数浮点警告解释。
源码定位均在 [SymbolicBuilder._operation](../../../src/jax_reorder/algebra/symbolic.py)，具体分支列于表中。

| 编号 | 原因与影响 | 优先级 | 状态 | 修复验收要求 |
| --- | --- | --- | --- | --- |
| ALG-006 | `reduce_sum`/`dot_general` 将整数收缩当作无限精度运算，丢失溢出；`int8` 的 `[100,100]` 求和或与 `[1,1]` 点积后转 `int32`，实际为 `-56`，与 `200` 比较却返回 `equivalent`。Pallas 完整函数与局部数学入口也受影响 | P1 | 待处理 | 保留 dtype 算术或明确拒绝未支持转换；覆盖整数归约/点积、扩大类型前后顺序、普通 JAX 与 Pallas 入口 |
| ALG-007 | `iota` 直接生成无限精度坐标，未核对目标 dtype；`lax.iota(uint8,258)[257]` 转 `int32` 实际为 `1`，与 `257` 比较却返回 `equivalent` | P1 | 待处理 | 检查坐标是否可表示，不能证明时保留准确转换或拒绝；覆盖边界内外及后续 cast |
| ALG-008 | 空 `reduce_max/min` 对整数仍使用 `-∞/+∞`；空 `int32` 归约与正确 dtype 极值比较，却返回 `inequivalent` | P2 | 待处理 | 按 dtype 选择空归约单位元；覆盖有符号/无符号整数，并保留浮点与布尔行为 |

从仓库根目录使用 `.venv/bin/python` 可执行以下最小复现。打印的是当前错误行为，不是期望通过的断言：

```python
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_reorder import compare_functions


cases = (
    (
        "ALG-006 sum",
        lambda: lax.reduce_sum(jnp.full((2,), 100, jnp.int8), (0,)).astype(jnp.int32),
        lambda: jnp.array(200, jnp.int32),
        (),
    ),
    (
        "ALG-006 dot",
        lambda: lax.dot(jnp.array([100, 100], jnp.int8), jnp.ones(2, jnp.int8)).astype(jnp.int32),
        lambda: jnp.array(200, jnp.int32),
        (),
    ),
    (
        "ALG-007 iota",
        lambda: lax.iota(jnp.uint8, 258)[257].astype(jnp.int32),
        lambda: jnp.array(257, jnp.int32),
        (),
    ),
    (
        "ALG-008 max",
        lambda x: lax.reduce_max(x, (0,)),
        lambda x: jnp.array(np.iinfo(np.int32).min, jnp.int32),
        (jnp.empty((0,), jnp.int32),),
    ),
    (
        "ALG-008 min",
        lambda x: lax.reduce_min(x, (0,)),
        lambda x: jnp.array(np.iinfo(np.int32).max, jnp.int32),
        (jnp.empty((0,), jnp.int32),),
    ),
)
for name, left, right, args in cases:
    result = compare_functions(left, right, *args)
    print(name, np.asarray(left(*args)), np.asarray(right(*args)), result.status.value)
```

Pallas 复现同样使用 `int8` 收缩：kernel 将 128 个 `100` 求和、转 `int32` 并广播写入 128 个输出。
对照 kernel 写常量 `12800`。通用 CPU interpreter 的首元素分别为 `0`、`12800`；对 `interpret=False`
源函数使用 `compare_functions(..., scope="function")` 和 `scope="device_local"` 都得到错误的代数等价。
这是本地解释器与比较器的证据，没有 TPU 实机执行结论。

同次审查的定向回归命令如下，结果为 **300 passed，81.11 秒**；Ruff 与格式检查通过。
广泛快速测试批次被主动中断，未完成全仓验证；上述最小复现尚未作为回归测试加入测试集。

```bash
.venv/bin/pytest -q tests/test_canonical.py tests/test_semantics.py \
  tests/test_functions.py tests/test_correspondence.py tests/test_unrooted.py \
  tests/test_pallas_local_entry.py tests/test_pallas_local_values.py tests/test_pallas_structural.py
```

## 已确定的执行条件

- 不从 program_id 推测 core；`CoreExecutionBinding` 是调用者提供并保证的条件，不配置实际运行。
- 远端并行要求完整单 core 点流及各设备同序，独立流冲突与实际序列进展分别证明。
- 保留 scratch/staging 的真实共享身份、跨 program 在途 DMA 和各输出生命周期。
- 结构匹配不能替代通信正确性；未证明条件明确拒绝，不能据此报告程序不等价。

算法入口按[契约/API](../../design-docs/current/pallas/contracts.md)、[grid/内存](../../design-docs/current/pallas/grid-memory.md)、
[通信/绑定](../../design-docs/current/pallas/communication.md)路由，避免读全阶段历史。

## 最近完成与证据

下表为各阶段历史验收，2026-10-26 仅核对范围与代码路径；已复现缺陷按上方待办独立跟踪。

| 子阶段 | 结果 | 原始记录 |
| --- | --- | --- |
| 有界动态页 DMA（2026-10-22） | 最终六批去重 721 项通过；真实动态页 attention 的正确/错页/倍 V 数学用例完成 | [验收](../completed/pallas-bounded-page-dma.md) |
| 顺序输出映射与纯死计算（2026-10-22） | 最终两片 438 项通过；输入/输出 index_map 保留逐 kernel 状态更新前提 | [验收](../completed/pallas-output-map-and-pure-lowering.md) |
| 设备局部等价收口（2026-10-22） | 冻结源码后四片去重 1,288 项通过；包含局部公开入口和受限远端条件初始化 | [验收](../completed/pallas-local-equivalence-closure.md) |
| 远端多维 grid 与并行流 | 分批去重 1,183 项通过、6 份离线 AOT；CPU 的 2 项解释器限制不计数值成功 | [阶段 27](../completed/pallas-history/stage-27.md) |
| 普通尾块与 masked 访存 | 分批去重 2,167 项；最终 423 受影响 + 368 core；后端 masked-load 拒绝独立保留 | [阶段 26](../completed/pallas-history/stage-26.md) |
| 并行 owner ANY 尾块 | 1,975 不同测试 ID、9 AOT，独立有限域/完整输出 oracle | [阶段 25](../completed/pallas-history/stage-25.md) |

不将跨阶段结果相加。新的功能阶段单独保存准确命令、源码状态、失败及修复证据，完成后只在本页保留一行入口。
