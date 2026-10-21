# Pallas 常量与远端布尔转换修复

## 目标与完成条件

修复两项已复现问题：大整数常量 cast 经 Python float 中转造成二次舍入并误判；
远端 kernel 的静态 bool→float 被错误地送入整数范围检查。保留此前未提交改动。

## 设计与取舍

- 常量转换保留源整数精度，再执行目标 dtype 转换；不以浮点中间值承载整数。
  覆盖有符号/无符号及符号两侧的舍入边界，保留不透明表达式与定义条件。
  只修复 cast 建模，不放宽整数溢出或未知数据值域的证明。
- 协议静态求值区分 bool、整数目标与普通浮点值；浮点 cast 由已有值投影处理。
  不将浮点值用作静态路由或计数，不放宽数据依赖通信与同步检查。
- 正反例放入既有测试模块，独立 NumPy/CPU 输出验证不依赖生产转换 helper。
  离线 TPU 编译与数学比较分开记录，不执行 TPU kernel。

## 工作项

- [x] 修复常量 cast，覆盖公共代数/函数接口的等价与不等价结论。
- [x] 修复远端 bool cast，覆盖协议、值投影及非法路由拒绝路径。
- [x] 独立审查、相关回归与静态检查，更新短入口并归档。

## 验证基线

启用 x64 后，`N=2**54+2**30+1` 在外层 JAX 从 int64 转 float32，随后进入只复制
float32 的 Pallas kernel：两个公共数学入口都误判它等于 `2**54`，不等于正确值
`2**54+2**31`。CPU 和独立 NumPy 128 元素输出一致；没有整数溢出。

远端 DMA exchange 后加 `(device_id == 0).astype(float32)`，协议和数学入口均报
`index expression must have integer dtype`；等价 `where` 和经 int32 的源码可比较。
上述两组共 6 份原始源码均已通过离线 `v5:2x2x1` megacore 完整后端编译。
基线源码与结果：`/tmp/jax-reorder-review-after-cast/review.json`。

本轮日志目录：`/tmp/jax-reorder-cast-review-fixes/`。

## 实现与审查

整数常量使用能容纳值的 int64/uint64 存储后转目标 dtype；Pallas 的合成 int32 标签
不能代表已证明坐标常量的实际宽度。仅源整数类型进入该规则；非精确整数或超 64 位范围
明确拒绝，普通浮点与闭合不透明表达式继续原有规则。不声称所有目标格式都理想单次舍入，
例如 bf16 的实际转换仍与锁定 NumPy/JAX 行为一致。

远端静态求值将 bool 目标保留为 bool，整数目标仍做范围检查，浮点目标交给值投影。
解除错误拒绝后发现 Pallas 的标量 bool/浮点字面量被按向量坐标读取，触发 zip 长度异常；
已在代数桥的 cast 入口恢复显式广播，保留原始定义条件、内存有效性及 Source 链。
不改变普通 JAX 的 shape 契约或张量收缩规则。

新增测试 31 项通过：常量转换 20 项（两个公共入口、正负/无符号、舍入中点、三种浮点目标格式及闭合整数归约），
远端 11 项（值正反例、标量浮点、路由及信号量计数拒绝）。独立探针另验证 int64/uint64
极值到 float32/float64、普通 JAX 入口，以及标量/一维/二维广播的条件保留。

首次测试中的 uint64 隐式 Python int 被 JAX 当作有符号整数解析，已改为显式源 dtype；
非法路由同图数学比较会走结构短路，测试改为与合法 exchange 对比以覆盖数学投影，
协议入口仍显式检查同图。修正后分别完整重跑 16/11 项；首次完整失败日志未保留，
测试修正摘要在日志目录的 `remote-retry-observation.txt` 及本记录中。

收尾反回归检查发现直接要求 `sp.Integer` 会拒绝旧路径支持的闭合整数 Sum；
现先调用已有预算化简器，再检查精确整数。增加两种归约 × 两个公共入口的 4 项
正反例与 CPU 对照；连同常量转换及旧闭合 opaque 测试共 22 项重跑通过。
超出现有有限展开上限且不能精确化简时仍保守拒绝，不再使用旧路径的隐式浮点求值。

## 离线编译

从最终测试工厂及参数读取的 36 个编译用例全部 trace/lower/IR verify/backend compile 通过：
24 个常量转换正反例、6 个闭合整数归约正反例、6 个远端 bool cast/where/经整数转换正反例。
常量转换位于外层 JAX，Pallas 复制 float32 输出；这不取消既有 float16 输出的源后端限制。
离线拓扑仍为 `v5:2x2x1` megacore，没有 TPU 实际执行或精度/性能结论。

复现脚本 `aot.py` 读取仓库测试工厂，复用 `tests.pallas_aot.run_cases`；原始阶段结果在
`aot.json` / `aot.log`，闭合归约为 `aot-closed-sum.json` / `.log`。
首次脚本选择了错误的参数装饰器，在创建 case 时停止，尚未调用编译；
修正选择后完整重跑，原记录为 `aot-harness-error.log`，不计作源码编译失败。

## 最终验收（2026-10-21）

- 常量转换原始复现：两个公共入口均对正确值判等、错误值判不等；远端 bool cast
  协议检查正常完成，数学正例判等、改错设备谓词的反例判不等。
- 首轮相关回归：Pallas/函数/分层 324 项、普通代数 390 项、远端 341 项全部通过。
  闭合整数 Sum 补修后，完整转换相关模块 177 项重跑通过；分批去重共 **1,059 项**，
  含 **31 项新增**。重叠测试不重复计数，不称为单次全量或所有批次均在最后一次补修后运行。
- 独立最终审查确认预算、opaque、定义条件与内存有效性保持；Ruff、format、默认
  basedpyright、tracing 示例、文档链接和 diff 检查通过。此前未提交改动保留。

最终重跑命令：

```bash
.venv/bin/python -m pytest -q tests/test_pallas_algebra.py tests/test_algebra_cast_sharding.py tests/test_symbolic_explicit.py tests/test_algebra_scalar_ir.py
```

其余批次：`pallas.xml` 覆盖 tensor_memory、padding_algebra、padding_domains、functions、package_layout
及最终重跑已覆盖的 pallas_algebra；`algebra.xml` 覆盖 algebra、algebra_structural、algebra_finiteness
及最终重跑已覆盖的三个普通代数模块；`remote-values.xml` 与 `remote.xml` 覆盖 remote_values、remote、
protocol、remote_mask、remote_grid_binding、remote_nd、remote_grid。最终去重 ID、文件 hash 与
36 个编译记录存于日志目录的 `final-verification.json`。
