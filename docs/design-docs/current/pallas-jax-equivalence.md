# JAX / Pallas 等价比较：任务入口

当前范围于 2026-10-26 对照源码与固定测试核对；2026-10-22 及更早验收按原阶段保留，本次文档整理未重跑测试。
先读本页，再按任务打开一个专题；历史不默认加载。
本页是路由与当前约束，比较契约仍以 [ARCHITECTURE](../../../ARCHITECTURE.md) 为入口。

## 阅读预算与路由

默认上下文仅本页；通常再读一个专题即可开始定位。需要实现细节时才打开对应源码或单个历史阶段。
不要递归读取所有链接，也不要把 `history/`、已完成计划或原始日志一次性放入上下文。

| 当前问题 | 下一份文档 | 何时再看原始证据 |
| --- | --- | --- |
| 选入口、结果含义、外层 JAX/mesh 绑定 | [契约与接口](pallas/contracts.md) | 接口来源或历史取舍不明确时 |
| grid/index_map、输出窗口、mask/ANY、Ref 变换 | [Grid 与内存](pallas/grid-memory.md) | 修改某条规则、查反例时 |
| DMA/信号量、执行绑定、远端并行与进展 | [通信与执行绑定](pallas/communication.md) | 修改协议或检查绑定义务时 |
| 还缺什么、如何验收、CPU 与 AOT 有何区别 | [验证与边界](pallas/validation.md) | 需要准确批次、失败及复现命令时 |
| 接下来做什么 | [当前计划](../../exec-plans/active/pallas-semantic-layer.md) | 不加载全部已完成工作项 |
| 追溯某次结果或讨论 | [历史索引](../../exec-plans/completed/pallas-history/index.md) | 只打开相关阶段；原文完整保留 |

## 每次任务必须知道

- 设备局部语义/数学入口比较分布式程序中单个 device 的 kernel；每侧一个静态 `pallas_call`。
  mesh 可以存在，device_id 导致的值、分支、地址和通信目标依赖必须保留。
- 局部结论标明 `scope="device_local"`；必要环境和通信接口条件不能省略，
  也不能把局部结论扩展为整个分布式程序等价。
- 结构层保留完整运算、参数、区域绑定和副作用；未使用纯计算只在代数 lowering 删减。
  结构不同不推出程序不等价。请求协议检查时，图相同也不能替代通信有效性证明。
- 适配器解释源对象；`ProgramView` 保存源事实。core 处理 grid、地址、初始化、
  内存版本和通信；SymPy 只消费已证明合法的纯值投影，默认查询限时 60 秒。
- grid 展开不插入 barrier，不复制真实共享存储，不替源码补 wait；
  串行状态依赖和并行资源独立性分别证明。
- `CoreExecutionBinding` 是调用者必须保证的外部执行条件，不是编译器配置。
  普通 `pallas_call` 不自动提供绑定；远端并行还需完整单 core 点流及各设备同序条件。
- 不支持的语义和预算耗尽明确失败；数值抽样、证据重放、IR 校验、AOT 编译
  与精确比较分别报告。编译成功不代表 TPU 已执行。

## 当前覆盖速览

| 层面 | 已接入的主要范围 | 继续阅读 |
| --- | --- | --- |
| 源图与局部接口 | 公共完整图比较、外层实参/常量/逻辑轴绑定、源证据重放 | [接口](pallas/contracts.md) |
| Grid 与内存 | 静态多维展开、串行状态驱动 index_map、独立输出窗口、跨迭代状态、ANY 恢复、尾块/mask/条件初始化 | [内存](pallas/grid-memory.md) |
| 通信 | 本地/远端 DMA、只读源有界动态页多笔在途、有限协议与计数、跨 program 版本、多维串行和附显式绑定的并行 grid | [通信](pallas/communication.md) |
| 数学 | 已支持投影接公共 SymPy 与 Pallas/JAX；坐标与同步证明仍在 core | [验证](pallas/validation.md) |

远端条件初始化与有界动态页 DMA 已有受限实现；剩余覆盖是更一般的数据谓词关系、
共享自动 staging 的并行复用、动态地址/数据路由/终止与物理布局。已复现的整数数学转换缺陷需先修复，
不能用这些范围内的历史正例替代边界正确性。
详细准入条件、预算与源后端限制集中在[验证与边界](pallas/validation.md)，不以此表宣称任意合法语法全覆盖。
用户明确暂不支持 `core_map`、`mpmd_map` 和输入输出共享 HBM；实机精度/性能不阻塞图与离线编译工作。

## 代码阅读路线

| 阅读目的 | 入口与职责 |
| --- | --- |
| 完整源图结构比较 | [canonical.py](../../../src/jax_reorder/core/canonical.py) 调用 [structural adapter](../../../src/jax_reorder/core/adapters/pallas/structural.py)，保留 Pallas 参数、捕获、资源和顺序；[correspondence.py](../../../src/jax_reorder/core/correspondence.py) 复核原图映射 |
| 语义源事实 | [adapters/jax.py](../../../src/jax_reorder/core/adapters/jax.py) 及其 Pallas 子适配器审核 schema，构造 [model.py](../../../src/jax_reorder/core/model.py) 中的 `ProgramView`；这一消费者的准入条件更严格 |
| 局部接口与实参 | [local.py](../../../src/jax_reorder/core/local.py) 编排 `compare_pallas_kernels` / 局部投影；[call_bindings.py](../../../src/jax_reorder/core/call_bindings.py) 保留输入、常量、外层计算和轴来源 |
| 完整函数值投影 | [normalization.py](../../../src/jax_reorder/core/normalization.py) 按调用边界选择本地或远端状态证明；四个 memory 模块分工见[内存专题](pallas/grid-memory.md#投影路径与模块分工) |
| DMA 与协议 | [communication.py](../../../src/jax_reorder/core/communication.py) 管理本地在途传输；[remote.py](../../../src/jax_reorder/core/remote.py) 构造跨设备事件，交给 [protocol.py](../../../src/jax_reorder/core/protocol.py) 检查因果、计数与冲突 |
| 数学比较 | [algebra/compare.py](../../../src/jax_reorder/algebra/compare.py) / [algebra/local.py](../../../src/jax_reorder/algebra/local.py) 编排范围；[adapter/pallas.py](../../../src/jax_reorder/algebra/adapter/pallas.py) 冻结已证明投影，[algebra/pallas.py](../../../src/jax_reorder/algebra/pallas.py) 构造完整输出公式，交给 [trusted.py](../../../src/jax_reorder/algebra/trusted.py) 的 SymPy worker |

完整结构消费者直接使用 structural adapter；资源与数学消费者使用 `ProgramView`。
全部 JAX 消费者统一视图仍是独立后续工作，结构准入不保证值投影或协议检查能够完成。

历史阶段的数字、AOT 与解释器结果见[近期证据摘要](../../exec-plans/completed/pallas-history/recent-evidence.md)
和[实施与调查索引](../../exec-plans/completed/pallas-history/index.md)，不在现行设计中重复维护。

## 维护方式

新规则先更新相关专题的当前范围/义务，再为本次验证写独立记录；本页只保留路由与少量最新证据。
历史文件不继续追加新状态，旧失败和后端拒绝不得改写成成功；通过链接保留证据，不在多个入口复制整段日志。
当专题超过约 150 行时按问题拆分；默认入口维持约 60–100 行，历史归档不计入默认阅读预算。
