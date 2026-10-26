# 外层浮点 iota 准入修复

## 目标与设计

修复一维外层 `iota` 丢失 dtype 后被当作精确整数坐标的问题。反例：外层分别生成
`arange(512, bfloat16)` 与 `arange(512, int32)`，进入相同 Pallas kernel 转 float32；
索引 257 分别为 256、257，原比较却返回等价。

复用 core 的整数类型/范围校验，与现有 tensor 和 kernel 路径一致。浮点 iota 的
语义投影明确拒绝，不增加浮点序列的舍入模型；原始结构比较仍保留 dtype。
整数 iota 及有界整数转浮点继续支持。具体边界入口见
[接口契约](../../design-docs/current/pallas/contracts.md)。

## 工作项

- [x] 补一维准入，新增公开数学/函数/设备局部入口的拒绝路径和整数正反例。
- [x] CPU/JIT 独立数值对照、离线 TPU AOT、相关回归与静态检查。
- [x] 独立审查、更新验证结果并归档；保留已有未提交改动。

## 验证

32 项新增回归通过：三个公开入口、四种浮点 dtype、cast 在调用前/后、CPU eager/JIT
独立数组对照、整数坐标正反例及嵌套小窗口。范围校验按 iota 自身 shape，不能借用
外围 block 的长度：例如 block512 中合法的 int8 iota128 必须继续支持。

独立复核确认 core 三个 iota 入口均检查整数准入；原反例与 cond 路径明确拒绝，
小窗口正例判等。相关回归 535 passed，包含上述 32 项新增测试，不重复计数。
Ruff、format、basedpyright、tracing、文档链接与 diff 检查均通过。

离线 TPU `v5:2x2x1` megacore：8 份源码通过 trace/lower/IR verify/backend compile，
包括原 bf16/int32 两侧、f32、外层转 f32 的 f16、两个整数改写及 int8/int32 小窗口。
另 1 份 kernel 内 f16 转 f32 的源码通过前三阶段，但后端因 f16 VMEM load 拒绝；
保留为源后端限制。未执行 TPU kernel，编译成功不代表比较器支持浮点 iota 投影。

回归日志位于 `/tmp/jax-reorder-float-iota-fix/`；AOT 逐阶段记录位于
`/tmp/jax-reorder-float-iota-audit/` 和 `/tmp/review-iota-short-aot.log`。

```bash
.venv/bin/python -m pytest -q tests/test_pallas_algebra.py tests/test_pallas_tensor_memory.py tests/test_pallas_local_entry.py tests/test_pallas_semantic_slice.py tests/test_pallas_grid_state.py tests/test_pallas_dynamic_views.py tests/test_pallas_remote_values.py tests/test_functions.py tests/test_package_layout.py
```

本轮仅修改一维 iota 准入、现有代数测试及开发文档；未修改受保护文档、提交或推送。
