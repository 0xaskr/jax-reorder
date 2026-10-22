# 有界动态页地址与在途 DMA

## 目标与完成条件

补齐真实 JAX paged attention 中固定长度 128、运行时页表 `pages & 7` 的语义投影。
连续 `start` 多页之后再 `wait` 时，保留每笔搬运与页号、源版本、目标窗口及信号量的关联。
三个完整有序输出必须保留；覆盖源码等价改写、错误页/值反例、漏 wait 与冲突拒绝。

## 已确认根因

旧路径在动态 Ref 地址处枚举当前区域剩余程序。第一个 K 页的分支启动 DMA 后，
第二个页又需要地址分支，被 `fork` 的在途 DMA 入口限制拒绝。页号界已证明为 0～7；
本例尚未执行任何 wait。直接展开八个独立页号会造成 8^8 组合，不能作为正常支持路径。

## 设计与范围

- 在单笔本地 DMA 内处理有界动态源地址，保留原始页号表达式，分别验证有限源窗口并
  捕获对应 payload；不枚举整个后续 kernel。目标窗口保持可求定，现有一般控制分支规则保留。
- 真实逻辑字节/元素数与所有可能源窗口的保护足迹分开：后者供冲突和版本检查，不能
  被误当成一次真实搬运量。完成 wait 后按同一页号表达式选择实际值。
- 首先覆盖普通只读输入 HBM 源；其他存储如扩展，必须证明完整候选源的初始化、版本与
  在途冲突。未知范围、越界、未初始化、漏/错/不足 wait 继续拒绝，不插入隐式 wait/barrier。
- 状态、地址、同步在 core 解决。数学后端消费合法完整投影；动态 peer、动态 grid、
  只读 prefetch 的并行准入及 squeezed 输出的其他缺口不属于本轮。
- 候选 payload 的共同纯表达式在 core 提取，仅在不同的地址操作数保留选择；连续页号
  恢复为原始 selector，避免对整个 attention 展开逐页 Piecewise。数学 lowering 仅对
  活跃地址使用精确低位掩码及经过 dtype 范围检查的整数运算，保留地址自身定义域。

## 工作项

- [x] core 接入每笔 DMA 的动态源选择、足迹和源版本证据。
- [x] 小型真实源码正反例、完整 NumPy oracle、wait/冲突/预算拒绝。
- [x] 完整上游 paged attention 动态合法页表的语义入口与 CPU 验证。
- [x] 默认预算下完整数学正反例。
- [x] 复审、受影响回归、静态检查和开发文档；归档。

## 分工与验证

core、专用 DMA 测试、真实 attention fixture 三项并行；主代理整合、审查证据与数学入口、
维护记录。测试复用既有文件和独立 oracle，避免重复执行设施。先跑最小专项，再扩大到
受影响的本地 DMA、动态视图、窗口/资源回归。CPU 数值、图证明与编译证据分别报告。
本轮不提交/推送、不改受保护文档，保留工作区已有改动。

## 进展与结果

- 已保存旧拒绝的精确跟踪：`/tmp/paged_dma_branch_probe.py` 与同名 `.log`。
- 实现采用逐笔有限源选择，避免将纯地址的可能值组合扩展成整个程序的指数级路径。
- 最终 core 四模块回归 244 passed、2 deselected（两项错页数学反例另批验收），
  `/tmp/final-freeze-core.xml`。真实 attention 三输出全部 1,040 个坐标，
  两张页表 × direct/inline/倍 V/错 K 页共八组均通过独立 NumPy 对照；官方 CPU
  interpreter 五项通过，启用 race 检查。两类证据不替代精确数学结论。
- 整合发现数学反例访问 `data[pages[i] & mask]` 时旧 witness 没有先代入内部页表读；
  已补递归地址求值及每层 bounds 检查。低位掩码、页号补数/置换、地址定义域保留、
  非连续低位掩码及可能溢出/窄化保守处理，十项专项通过。
- 小型错 K/V 页公共数学两项通过。真实 attention 默认预算反例曾耗尽；20,000 的
  临时诊断可给出倍 V 不等价，但不作为默认准入成功。有限页置换的 Piecewise 补显式
  default，消除 SymPy ITE 构造错误。随后消除静态坐标重复求值并改善精确反例搜索顺序，
  不提高预算或用抽样一致性代替判等。一次全局改搜索顺序触发旧 RoPE wrong_sign 预算
  回退，恢复其原顺序、仅对动态地址优先搜索后该回归已通过。
- 当前反例采用同一输入赋值递归代入页表，先验证所有输出定义域与每层实际读取范围；
  纯坐标不做无用输入求值。先逐轴再组合选择输出坐标，防大 shape 的前 16 个笛卡尔积
  点全部落在第一块。非均匀输入只用于精确负例搜索，未找到反例不会推出等价。
- 真实源码来自锁定 JAX 包的 `experimental/pallas/ops/tpu/paged_attention/`
  `paged_attention_kernel.py`。长度固定 128，物理 8 页，运行时 `pages & 7`；两张页表
  包括乱序、重复、负数和超出掩码范围的原始值。错页反例仅轮转 K 页，V 保持不变。
  父函数只返回 Y 不影响设备局部比较仍保留 Y、max、sum 三个完整有序输出。
- core 缓存仅复用固定几何、动态位置关联及不可变表达式深度。每次实际读取、初始化、
  版本、冲突与 selector 范围仍重新验证。真实 direct/错 K 页分别关闭两类缓存，四次
  完整投影及通信证据均一致；源码哈希与结果见 `/tmp/final-freeze-core-evidence.json`。
- 错 K 页精确赋值可令最大分数输出为 128 与 0，且全部输出定义条件成立。单热整片
  输入只用于负例搜索；已知 cast 仍逐值验证可精确表示，不能被当作恒等函数忽略。
- 两份新增本地 DMA 源码通过 TPU lowering 和 StableHLO `operation.verify()`；这是
  IR 校验，不是 executable AOT，更不是 TPU 实机精度或性能验证。
- 数学反例内，只有词法坐标范围已证明恒定的 floor/Mod 才提前化简；不从输入 shape
  推断数据范围。范围证明缓存只规范化缓存键中的轴名，保留独立轴、重复轴相关性、
  每个轴范围及未绑定符号，不重写实际捕获关系。cast 缓存仍先求值并验读原操作数，
  再复用相同精确值及源/目标 dtype 的表示性结论。
- 复审发现预化简可经 `0 * read` 或 `Eq(read, read)` 擦掉尚未验证的非法读取，也可能
  抵消未验证的闭合 cast。已采用惰性算术重建，数据/转换关系先验证操作数；新增越界、
  非精确 cast、同范围独立轴、同名嵌套轴拒绝测试。独立 14 个探针也全部保守拒绝。
- 最终错 K 页单项已在公开入口默认 10,000 预算和 60 秒限制内通过，37.91 秒，
  `/tmp/pallas-bounded-pages-math-page.xml`。完整开源数学九项随后全部通过，真实动态
  页 attention 的 direct↔inline 为等价，V×2 与仅 K 错页均为数学不等价。

## 最终验证

| 批次 | 结果 | 证据/范围 |
| --- | --- | --- |
| DMA 四模块，冻结 core 后 | 244 passed，2 deselected | `/tmp/final-freeze-core.xml`；排除的错 K/V 页数学两项另跑 |
| 真实 attention 源码组 | 13 passed | `/tmp/pallas-bounded-pages-source-final.xml`；语义/三输出 oracle、未知范围拒绝、5 项 CPU interpreter |
| 精确反例/收缩域 | 147 passed | `test_algebra_sum_domains.py`；包含本轮地址递归、范围缓存、非法读/cast 拒绝 |
| 通用不等价与小型错页 | 42 passed | `test_algebra_inequivalence.py` 与两项错 K/V 页公开语义/数学比较，66.87 秒 |
| 完整开源数学 | 9 passed | `/tmp/pallas-large-math-final.xml`；attention/RoPE/paged attention，全部默认预算与超时，整批 155.15 秒 |
| 通用 Pallas 代数 | 266 passed | `/tmp/pallas-bounded-final-algebra.xml`；`test_pallas_algebra.py` 与 `test_pallas_padding_algebra.py`，390.01 秒 |
| Ruff / 格式 / 类型 / tracing | 通过 | 402 文件格式检查；basedpyright 0 errors、0 warnings；示例输出 equivalent |

以上六个最终测试批次没有重复 pytest ID，共 721 项通过，不包含中间诊断或重复专项，
不声称全仓测试。最终代数源码前后哈希与独立安全探针记录于
`/tmp/pallas-bounded-final-algebra-safety.json`；所有计划项完成。

## 可复现入口

均使用锁定环境，`JAX_PLATFORMS=cpu PYTHONPATH=.:src`。单纯追踪、语义比较及本轮
CPU/IR 验证不要求 TPU 实机。

```bash
.venv/bin/pytest tests/test_pallas_dma.py tests/test_pallas_dma_regions.py \
  tests/test_pallas_dma_composition_complete.py tests/test_pallas_dynamic_views.py
.venv/bin/pytest tests/test_pallas_source_coverage.py
.venv/bin/pytest tests/test_pallas_large_math.py tests/test_algebra_sum_domains.py
.venv/bin/pytest tests/test_pallas_algebra.py tests/test_pallas_padding_algebra.py
```

小型 fixture 和拒绝测试在 `test_pallas_dynamic_views.py`；真实源码包装与独立三输出
oracle 共用 `pallas_paged_attention_cases.py`，语义/数值与数学分别由 source coverage
和 large math 测试驱动。使用默认预算的数学断言必须为 `EQUIVALENT/INEQUIVALENT`，
不接受 `UNKNOWN` 作为通过，不扩大预算掩盖重复分析。
