# 历史：9.16 真实双缓冲 DMA 的跨迭代证明（2026-10-20，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../../design-docs/current/pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.16 真实双缓冲 DMA 的跨迭代证明（2026-10-20，完成）

本阶段固定 JAX 0.11.2 revision `32544801e26115ac1794926d027148abf3baf009`，保存
[完整上游测试文件](../../../../tests/fixtures/pallas_kernels/jax_tpu_pipeline_test.py.txt)、
[来源与 SHA-256/git blob 清单](../../../../tests/fixtures/pallas_kernels/jax_pipeline_manifest.json)
及 [Apache 许可证](../../../../tests/fixtures/pallas_kernels/LICENSE.jax)。同时核验实际执行的
`jax/_src/pallas/mosaic/pipeline.py` 与该 revision 完全相同。
来源为 [JAX 官方 pipeline 测试](https://github.com/jax-ml/jax/blob/32544801e26115ac1794926d027148abf3baf009/tests/pallas/tpu_pallas_pipeline_test.py)
中的 `PallasCallMultipleBufferedPipelineTest.test_copy` 和 `test_matmul`。

两个基线均提取完整原始嵌套 kernel，保留 512×512 输入、128×128 tile 和两个输入/输出
buffer。copy 有 4×4 个块，matmul 有 4×4×4 次块计算。使用原版 `emit_pipeline` 生成
prologue/scan/epilogue 和真实 DMA start/wait；没有替换为手工展开的同步实现。
追踪目标明确为 TPU v5 的空轴 abstract mesh，适配与比较在同一环境进行。
AI 变体只在独立 AST/临时替换上下文中修改调度方法，每次构造新的 kernel 函数，避免
JAX trace 缓存掩盖修改；不修改已安装 JAX。实现见
[源案例与执行器](../../../../tests/pallas_pipeline_cases.py)、[回归测试](../../../../tests/test_pallas_pipeline.py)。

初次追踪成功，局部适配首先拒绝 ANY 存储。逐层补齐的源事实包括 `run_scoped` 捕获与
分配、DMA semaphore 数组槽位、切片起点及降秩索引、同步复制附加的完整恒等视图。
官方条件分支声明的是分支 effect 并集：仅在 cond 绑定处核对所有分支实际 effects 的
并集与声明一致，不删除空分支上的声明。scan carry 按锁定 JAX 的 `typematch` 校验，
仍保留两端完整类型事实；静态布尔转换和模运算辅助原语也按实际类型解释。

已实现的 core 规则：

- `run_scoped` 只接空 collective_axes、无分配变换的 VMEM/semaphore；捕获与新分配
  分开绑定。VMEM 初始未知、semaphore 初值为零。禁止 Ref 逃逸，退出前检查所有涉及
  局部分配的 DMA 已完成、普通信号量计数归零。普通信号量的并行 core 目标限制同样
  适用于新作用域，不能通过把分配移进 scope 绕过。
- 本地 DMA semaphore 每个合法数组元素有独立资源身份和代次。等待由**槽位和消费
  字节数**确定；同一存储/dtype 下，wait 的地址描述符并非生产者身份，改变虚拟地址
  但保持计数及槽位的变体是正例。错误槽位、漏等待、未完成目的读取/源覆盖均拒绝。
- DMA 两端整体 Ref 可不同形状，选中的逻辑矩形必须同形状、同 dtype；降秩保留
  原存储坐标，并建立完整元素对应。冲突使用矩形交集，计数使用实际元素数。
  新增的非连续矩形限 32 位 dtype、默认布局，父存储末维及切片末维起点/长度按 128
  对齐、倒数第二维起点/长度按 8 对齐；原有未证明的 64 宽非连续路径仍拒绝。
- 有限循环跨迭代共享 pending DMA、源版本和 semaphore 代次，不在每轮清空。
  串行 grid 在同一输出窗口内保留状态；跨 program 尚未完成的 DMA 若读取自动 staging
  输入，则在重装边界拒绝。ANY 整块 Ref 不发生该自动重装，允许已证明的跨轮次传输。
  ANY 直接 get/swap 和输入输出 alias 仍需另行证明，当前明确拒绝。
- 循环重放、槽位分配、矩形切分和证据重放均受分析预算约束；不枚举全局数据元素，
  不把同步或内存有效性交给 SymPy。通信检查通过后才投影完整值表达式。

具体正反例及结果：

| 源修改 | 结论与证据 |
| --- | --- |
| copy/matmul：先 wait 当前块，再预取下一块 | 与原版 normalized 等价，完整证据重放通过；copy 公开 SymPy 入口也等价。 |
| copy/matmul：输入 buffer 从 2 改 4 | 与原版完整输出等价；保留各槽位和多代 DMA 证据。 |
| copy/matmul：使用官方 `no_pipelining=True` | 与双缓冲原版完整输出等价。 |
| copy/matmul：仅移动 wait 的虚拟目标地址，保持 semaphore 槽位/计数 | 等价；copy 在官方 CPU interpreter 中也得到相同完整输出。 |
| copy：删除输入 wait、wait 使用相邻槽位、预取固定写槽 0 | 均拒绝：目的区域在完成等待前被访问。 |
| copy：删除输出 wait / 删除最终排空 wait | 分别拒绝源区域提前覆盖 / 作用域退出仍有 pending DMA。 |
| copy：来源列块改为下一块，等待保持正确 | 通信合法；完整值投影变为 `x[row, (col + 128) % 512]`。独立 NumPy/CPU 对照确认输出改变；公开代数入口返回 inequivalent，给出 `(0,0)` 的整数 0/1 见证。 |
| 单独构造的串行 grid 边界探针 | program i 发起、i+1 等待，ANY 输入正例通过；自动 staging 输入和强制复用槽 0 两个反例拒绝。此探针单列，不冒充上游 kernel。 |
| 越界槽位、数据依赖索引、ANY alias/直接读、证据代次篡改、预算不足 | 分别走显式拒绝/证据校验失败/预算异常，不默认为等价。 |

独立投影解释器按两个随机种子、所有 16 个输出 tile 的内部点核验 copy/matmul。
这里发现并修正了测试 oracle 的 eager select：标量谓词只求值选中分支，避免对未选中
tile 的无效坐标求值。这是抽样交叉检查，完整等价仍由符号投影和重放证据给出。

CPU 执行使用 JAX 官方 TPU interpreter、`dma_execution_mode="on_wait"`、竞态检测、
未初始化内存填充和逻辑越界检查。10 项覆盖原始 copy/matmul、合法调度变体、搬错块、
漏输入 wait 及跨 program 探针，每个上游执行项使用两个种子并检查完整输出。
漏输入 wait 的负例**数值错误，但官方竞态标志为 false**：on_wait 模式下未执行的 DMA
未必触发竞态报告。因此没有把“无竞态”当作有效性证明；core 依赖检查仍明确拒绝。
本阶段没有真实 TPU 执行证据。

最后公开入口回归还发现一个桥接缺口：原版 copy 的整型输出可合并为直接输入读取，
搬错块的输出需要保留分块选择；后者被通用整数 `select_n` 策略转成不透明调用，导致
`CONDITION_UNDECIDED`。修正限 Pallas 已证明值投影中由 `Predicate`/布尔值控制的整数
分块选择，直接保留精确的分段值及各分支定义条件，并记录
`pallas.integer-memory-selection:v1`。这里不执行整数算术，不放宽通用 JAX 整数
运算、数据依赖不透明谓词或分支原有定义条件。坐标反例进一步限定为两侧都只是合法
整数输入读取时，对最多两个不同元素赋值 0/1：两值在源整数 dtype 中可表示，见证
记录在 SymPy 查询方法中。整数算术、符号下标和越界读取仍不能借此产生反例。

例：令 `x[0,0]=0`、`x[0,128]=1`，原版在 `(0,0)` 返回 0，错块版本返回 1。
该坐标查询的两侧仍来自完整输出，方法记录为
`bounded integer read witness: {x0[0,0]: 0, x0[0,128]: 1}`，不是把中间差异升级成结论。

本阶段新增 **42 项**：41 项 pipeline/边界/公开入口回归，加 1 项整数读取见证边界
回归。全部 Pallas 最终 collection 为 **665 项**（575 项静态/数学、90 项 CPU marker）。
分批命令统一使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run`：

| 验证范围 | 命令与结果 |
| --- | --- |
| Pallas 静态/数学全批 | `pytest tests/test_pallas_*.py -m 'not pallas_communication' -q --tb=short`：564 passed、87 deselected；执行时本阶段新文件尚为 27 项。 |
| 官方 CPU 解释器全批 | `pytest tests/test_pallas_*.py -m pallas_communication -q --tb=short`：87 passed、564 deselected；随后补齐的 3 个 CPU 项在 pipeline 全文件运行中通过。 |
| 最终 pipeline 文件 | `pytest tests/test_pallas_pipeline.py -q --tb=short`：初次 40 passed、1 failed；失败正是上述整数选择/反例桥接缺口。修正后 `-k public_algebra_entry` 的 2 项均通过（253.57 秒），其余 39 项不受该桥接修改影响；41 项分批全部通过。 |
| 最后资源边界复测 | `pytest tests/test_pallas_pipeline.py tests/test_pallas_dma.py tests/test_pallas_local_scope.py tests/test_pallas_remote_values.py -m 'not pallas_communication' -q --tb=short`：116 passed；另补 3 项 scoped 普通信号量的归零/并行限制回归，通过。 |
| 共享适配/区域与包分层回归 | `test_algebra_array_conversion.py`、`test_package_layout.py`、`test_comparison_errors.py`、`test_adapter_guards.py`、`test_semantics.py`、`test_region_bindings.py`、`test_algebra_regions.py`：231 passed。 |
| 整数见证边界与数组转换 | `pytest tests/test_algebra_array_conversion.py -q --tb=short`：新增测试后 13 passed；验证只允许合法直接读取的 0/1 见证，算术/符号下标/越界仍拒绝。 |
| 最终数学桥接回归 | `pytest tests/test_pallas_algebra.py tests/test_pallas_upstream_state.py tests/test_algebra_inequivalence.py tests/test_algebra_abs_where.py tests/test_algebra_finiteness.py -m 'not pallas_communication' -q --tb=short`：201 passed、10 deselected（512.69 秒）。 |

上述批次有重叠，665 是按最终 collection 去重的 Pallas 覆盖数，不是把日志计数相加，
也不是本轮重跑了所有非 Pallas 测试。Ruff、265 个文件的格式检查、basedpyright
（0 error/0 warning）、tracing 示例、JAX/jaxlib 0.11.2 adapter inventory、文档链接
与 `git diff --check` 均通过。本阶段已无未解决的测试失败。

范围调整（2026-10-20）：用户确认**暂不支持输入输出共用同一块 HBM**。该项移出
近期补齐清单，不作为本阶段完成的阻塞项；历史受限 alias 实现记录不构成一般共享
HBM 的支持承诺。下一优先项是在输入输出存储独立的前提下，验证跨输出块的状态
复用，即切换到其他输出块后，再回到原块继续更新。

其余待补齐边界是一般数据依赖地址/不定长循环、任意布局与变换、远端 semaphore
数组及跨设备流水协议、非后缀窗口复用，不宣称任意合法 Pallas 流水线均已支持。
