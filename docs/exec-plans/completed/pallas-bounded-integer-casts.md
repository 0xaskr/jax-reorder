# Pallas 有界整数坐标转浮点

## 目标与完成条件

让 `i∈[0,127]` 时 `(i+1).astype(float32)` 与 `i.astype(float32)+1` 正常判等，
并让真实不同的完整输出可判不等。保留未经证明的转换、舍入和原始定义条件。

## 设计与取舍

在 algebra 消费 Pallas 投影时使用 core 已证明的坐标域和收缩绑定域，
仅对范围完整落在目标浮点连续精确整数区间的整数坐标 cast 建立恒等值映射。
覆盖外层输入绑定、kernel 和嵌套收缩；不从 tensor shape 推断任意输入数据范围，
不放宽普通 JAX 的不透明整数运算。超出证明范围继续走已有保守规则。

## 工作项

- [x] 保留投影坐标及收缩绑定范围，接入预算内的精确 cast 规则。
- [x] 补正例、真实反例、精度边界、未知输入及嵌套范围回归。
- [x] 审查原始条件/初始化和预算，运行相关检查并归档。

## 验证结果

修复前正反例均 `condition_undecided`，同余正例可判等；6 份源码已通过离线 TPU 编译。
复现：`/tmp/jax-reorder-review-cast-boundary.py`。

实现限定源为 core 的 `IndexExpr`，仅接受已审核的四种目标格式；闭区间阈值为
`±2**(nmant+1)`。按 emitter 词法域缓存范围，节点与边计入预算；lookup 区间无法证明时
回到已有不透明路径，预算异常仍传播。

审查发现精确转换使一个旧边界变得可达：闭合 `Sum` 中的不透明 cast 没有自由符号，
但不能转成 Python float。已修正通用 cast 准入：保留 narrowing 的原始定义条件，
其他改变 dtype 的闭合不透明表达式继续不透明。增加两组完整 kernel 回归。

初次离线 TPU 编译中 10 份源码通过，8 份 16-bit 输出源码因 block128/array256 不满足
源后端 tiling 规则被拒绝；测试已改为这些 dtype 使用完整 block256，保留 float32 双 program。
原失败日志与修改后的验证分别保存，不把原失败源码计为编译成功。

2026-10-21 验收：

- 原始 5 对复现分别得到 3 个等价、2 个不等价，符合独立 CPU 结果。既有 3 个广播反例
  从 `condition_undecided` 升级为 `INEQUIVALENT`；未将采样结果当作公共数学证明。
- 新增 29 个测试 ID：direct/outer/cond 和函数入口、正负精度边界、未知整数数据、
  sum/dot/嵌套收缩输入范围、闭合不透明 cast。与既有 3 个广播反例共 32 项受影响用例。
- Pallas/函数/分层回归 438 passed，普通代数回归 279 passed，分批去重共 717 项。
  Pallas 批次启动后仅调整了测试 helper 的 16-bit tiling；受影响精度测试 12 项重跑通过，
  以新结果覆盖同名 ID，不重复计数，也不称为单次全量运行。
- Ruff、format、默认 basedpyright、tracing 示例、文档链接及 diff 检查通过。
- 独立审查覆盖 lookup 保守回退、预算异常传播、词法域缓存、收缩输入域及原始定义条件；
  未发现误报等价或剩余必修问题。
- 最终源码共 17 份离线 TPU AOT 成功：原 10 份 float32、修正 tiling 的 4 份 bfloat16、
  3 份闭合 cast。另 4 份 float16 输出源码 trace/lower/IR verify 通过，但当前后端在
  `tpu.pack_subelements` legalization 失败；保留为源后端限制，不能算 AOT 成功。
  使用 `v5:2x2x1` megacore 离线拓扑，未执行 TPU kernel，不提供实机精度/性能结论。

复现命令（Python 3.12、仓库锁定依赖）：

```bash
.venv/bin/python -m pytest -q tests/test_pallas_algebra.py tests/test_pallas_tensor_memory.py tests/test_pallas_local_entry.py tests/test_pallas_semantic_slice.py tests/test_pallas_padding_algebra.py tests/test_pallas_padding_domains.py tests/test_pallas_remote_values.py tests/test_functions.py tests/test_package_layout.py
.venv/bin/python -m pytest -q tests/test_algebra.py tests/test_algebra_structural.py
.venv/bin/python -m pytest -q tests/test_pallas_algebra.py -k coordinate_cast_precision_bounds
```

日志、JUnit、AOT 逐阶段结果及最终校验清单在 `/tmp/jax-reorder-bounded-cast-fix/`。
离线 AOT 脚本从上述测试工厂读取源码，复用 `tests.pallas_aot` 的编译设施；
原始 tiling 失败保存在 `aot-before-tiling.log`，最终 16-bit 结果在 `aot-valid-tiling.log`，
闭合 cast 在 `aot-closed.log`。此前未提交改动保留，本轮没有提交或推送。

## 后续边界

本次只证明已知整数坐标的精确转换，不实现任意整数数据或有舍入 cast 的完整数学模型。
