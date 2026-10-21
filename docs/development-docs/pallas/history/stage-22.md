# 历史：9.22 时序重访、尾块有效域与 Ref 组合（2026-10-20，完成）

> 按需查证，非默认上下文。这里的“当前/未完成”指记录当时；现状见[任务入口](../../pallas-jax-equivalence.md)。
> 原章节编号、失败与验证记录保留；跨节编号在[历史索引](index.md)查找。

### 9.22 时序重访、尾块有效域与 Ref 组合（2026-10-20，完成）

用户确认以下分工并要求继续实现。三项都是既有语义/代数接口的覆盖补齐：

- 远端输出窗口重访不单独引入新的执行语义。串行 grid 展开后，依照原程序的
  发起顺序和已证明的 DMA/信号量完成依赖重放存储版本。`A→B→A` 的完整覆盖
  由最终版本决定结果；输出暂存区离开窗口后失效，不能假设返回 A 会自动回读旧值。
  放开窗口唯一性和输出 HBM 单次写限制时，仍须验证覆盖、初始化、在途版本及竞态。
- 对已分配缓冲的合法 padding，语义层携带原 shape、原访存和逐位置有效域，
  不因其随后可能被屏蔽而改变源图，也不在读入时无条件拒绝整块。
  core 不调用 SymPy；代数 lowering 把选择分支与有效域一起转换，证明最终输出
  已定义后才比较。`sum(where(mask, x, 0))` 可按有效域缩求和范围；
  mean 的原分母、max 的补零、非零 other 等贡献必须保留。真实越界、
  未完成 DMA、缺失同步和未被屏蔽的未定义结果继续拒绝或报告条件未决。
- Ref 的 reshape/transpose/index 在 core 组合成原存储地址映射，保留同一资源
  身份及真实足迹。有界动态索引复用有限范围与分支绑定；DMA 的源/目标变换须
  保留搬运次序、物理区域和在途冲突，不默认把非连续区域当作连续矩形。

验证要求：窗口 A/B/A 的最终值与错目标、漏 wait、缺同步及无回读累加反例；
padding 的先屏蔽后 reduce、错 mask、mean/max/other 差异和未定义值逃逸；
Ref 链的 NumPy 坐标对照、动态越界/预算、变换 DMA 的值映射和别名冲突。
可执行的源例分别报告 CPU 数值、原生 AOT 和语义证明；不要求 TPU 实机。
`core_map`/`mpmd_map`、输入输出共享 HBM 和多轴 mesh 全局证明仍不纳入。

**源语义核对。** 显式 `ANY` HBM Ref 的重复写有源码 DMA 与 wait，可直接
按已证明协议重放最终版本。默认自动输出缓冲则额外涉及后端生成的写回：
锁定 JAX `RevisitMode.IMMEDIATE` 将非连续重访的缓冲状态视为未定义，
官方 TPU CPU interpreter 会提前拒绝这种访问序列。普通 `pallas_call` 走原生
XLA pipeline emitter，不能借用 Python `pipeline.Scheduler.wait_out` 的实现作为
本路径的写回顺序证明。[官方流水文档](https://docs.jax.dev/en/latest/pallas/pipelining.html#buffer-revisiting)
明确区分输出暂存和 HBM，并不保证重返窗口时自动恢复旧值。
本轮自动窗口 A/B/A 的原生 AOT 成功只说明编译接受；Mosaic dump 仍保留窗口和
迭代参数，未提供完整外层 DMA 完成顺序。因此先落地显式 HBM 重访，自动窗口重访
继续准确报告缺少源写回时序契约，不把这个源边界改造成假同步。

**已接通的接口。** 语义视图适配保留原始 padded read；完整值投影的默认入口
仍要求有效值，代数桥显式使用 `defer_padding=True` 取得携带逐位置有效域的投影。
`RewriteEvidence.deferred_padding` 参与重放，未完成代数初始化证明的投影不能
冒充 normalized 等价证据。代数值另有 `memory_validity`，与 log 等函数的数学
定义域分开；对完整输出逐项证明内存有效性后才判等。已支持选择分支、逐点运算、
reduce 和 dot 的有效域传递，未接入的外层消费者明确拒绝待解义务。
本轮只放开合法输入 padding，不放开未初始化 scratch、真实越界、DMA 源尾部
或 padded Ref 变换。

局部求和范围收缩由 `algebra/pallas_domains.py` 完成：只分片已知输出坐标的
`floor(axis / 正整数)` 常值区间，再按一元整数仿射 mask 的临界点切分原 Sum。
非零 other 按实际区间长度保留，mean 仍除以原块宽，输出 shape 和 core grid
不变。未知 guard 保留原式，分片和区间先计分析预算；最终等价判定仍交 trusted
SymPy。真实 `(8,384)` 输入、256 分块的 sum/mean × 补 0/补 7 四种源码，
均可与独立 JAX 分块求和/广播/拼接取得完整公共代数等价结论。

**范围改写的交叉审查。** 整体输出有定义，不能推出被 mask 隐藏的分支在扩大的
前缀区间也有定义。例如 `sum(where(k>=3, log(k-2), 0))` 在原 `k=0..7`
范围内合法；若改为两个从 0 起始的 log 前缀和，会引入 `log(0)`。
实现因此仅对已证明完整定义、初始化且有限的输出启用此分域步骤；选中的分支
还须通过有限叶子和多项式运算检查，才可沿用已有的前缀差分规范化。log、倒数、
sqrt 等保留实际选中区间，不能靠消项抹去新增的未定义值。
独立审查完成 936 个仿射 mask 输出点、231 个多轴/多商输出点、144 个非零起点
与边界例，以及上述三种部分函数反例；均与直接枚举一致。

**代数结论的实际边界。** 小尺寸错误 mask、mean 分母、max 补零及非零 other
已有不等价或初始化未决回归。真实尺寸的错误 mean 分母仍可能返回 CAS 未决；
两个错误 mask（383 漏一个有效 lane、385 多读一个未初始化 lane）在公共比较
中均触发原有默认 60 秒超时，未取得数学结论。它们另有独立数值或初始化反例，
不将这些样本写成全域 CAS 证明，也不靠放大通用求和展开预算消除该边界。
大规模 Max 构造/证明仍可能昂贵，其 CPU/AOT 和小尺寸 CAS 结果分别记录。

Ref 链在原 storage 上处理有界动态标量/切片；DMA 端点当前要求精确足迹合并为
单个连续或已支持的对齐矩形。变换 DMA 按逻辑坐标重组多版本源片段后映射到目标，
不填平地址洞。远端路径暂不接此变换；交叉审查发现混合 kernel 中局部变换 DMA
会进入远端解释器、以及变换后的局部 wait 可能错误消费较大远端计数，已为所有
start/wait 增加统一拒绝并补真实源码反例。

**离线 TPU 编译与数值证据。** 使用隔离的 JAX/jaxlib 0.11.2、
libtpu 0.0.48.1、`JAX_PLATFORMS=cpu` 和 compile-only `v5:2x2x1` /
megacore 拓扑，没有执行 TPU kernel。

| 具体源码组 | 原生后端结果 | 独立验证及限制 |
| --- | --- | --- |
| 显式 HBM 完整/部分 A/B/A、错块、多输出 | 4 份生成 Executable | 官方 CPU 在 eager/on_wait 两种 DMA 模式下分别检查数值和竞态；错块由值反例识别 |
| 默认自动输出缓冲 A/B/A、错块 | 2 份生成 Executable，仅作审计 | 官方 CPU interpreter 拒绝非连续重访；原生编译不能补出写回时序契约 |
| 动态 Ref 读/写/切片、本地变换 DMA、上游 reshape DMA 两种 slice-first 顺序 | 7 份生成 Executable | 7 项官方通用 CPU 对照；上游来源为已保存的 `jax_tpu_state_test.py::test_dma_reshaped_ref` |
| `(8,384)` 输入、256 分块的 sum/补 7/mean/max | 4 份生成 Executable | 完整 CPU 输出与独立 NumPy 对照；max 的原生编译不表示任意大规模 Max 已有 CAS 证明 |
| transpose Ref 读、transpose DMA | 2 份 lowering 明确拒绝 | 语义层可证明坐标映射，锁定 TPU 后端仍不接受该源变换 |

合计 17 份源码生成 Executable，其中 2 份自动窗口例子仅审计；2 份 transpose
拒绝单列。原始报告位于 `/tmp/jax-reorder-followthrough-b2r7obh_/`：
`remote-aot-final.json`、`refs-aot.jsonl`、`padding-aot.json`。
远端重访编译用例保存在 `tests/pallas_remote_revisit_aot.py`；源码、结构证明、
CPU 数值与编译结论分别报告，不以其中一种替代其他证据。

**主工作区最终验收。**

| 检查 | 实际结果 |
| --- | --- |
| Pallas 最终 collection | 1308 项，较本子阶段基线净增 167 项，最终用例均有分批通过证据 |
| 初次合入三分支后的全部非通信 marker | 1069 passed、193 deselected，723.34 秒；含尾块直接 CPU 对照 |
| 全部官方 CPU 通信 marker | 193 passed、1069 deselected，380.46 秒 |
| 尾块范围补丁、既有 padding/代数与包分层组合 | 130 passed、1 failed，119.56 秒；唯一失败为下述导入环 |
| 消除导入环后的完整新增尾块与包分层组合 | 86 passed，55.67 秒；82 项尾块测试及 4 项包守卫 |
| 通用代数、显式符号、Nerva 和可信后端 | 756 项功能测试通过；同批另有 3 项包守卫通过、1 项导入环失败，修正后 4 项包守卫全部通过 |
| 共享结构、对应证据、错误分类与无根图 | 70 passed，31.03 秒 |
| Ruff、format、basedpyright | 通过；303 个 Python 文件格式一致，类型检查 0 errors / 0 warnings |
| tracing、adapter inventory、文档链接、diff 空白检查 | 通过 |

包分层检查发现 Pallas 初始化规则为访问 `SCALAR` 而反向导入 `symbolic`，
构成 `symbolic → pallas → symbolic`。主线程将原语名称集合以只读
`SymbolicBuilder.pointwise_primitives` 提供，消除反向导入，未改动原语集合
或数学规则；独立只读复核和上述最终 86 项验证均通过。
通用回归原始汇总为 `759 passed, 1 failed`，1315.94 秒；没有把这个中间失败
隐藏为一次性全绿，也没有重复全部昂贵 CAS 查询。

分批结果存在交叉，不能将表内各行简单相加。原始日志与本轮相对工作区快照的
差异保存在 `/tmp/jax-reorder-followthrough-b2r7obh_/`，包括
`integrated-pallas-static.log`、`integrated-pallas-cpu.log`、
`integrated-padding-final.log`、`integrated-final-cycle-fix.log`、
`integrated-algebra.log`、`integrated-core.log`、`integrated-collection-final.log`
和 `integrated-current.patch`。保留此前用户改动，未修改受保护文档或依赖文件，
未提交/推送。本子阶段完成；自动缓冲时序契约、远端变换和大型 CAS 等剩余范围
继续在整体 active 计划及技术债中跟踪。
