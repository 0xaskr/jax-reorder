# Pallas scratch 范围与布尔运算审查修复

## 目标与完成条件

按用户要求循环复现、修复、验证与复审，直至本轮审查没有剩余已确认问题。
不将一般 CAS 未决或既有支持边界当作缺陷，不宣称任意程序均正确。

## 已确认问题与规则

- scratch512/输出128：`where(arange(512)<128,1.,2.)` 写入 scratch 后读 `[128:256]`，
  真实全2却与全1判等。局部纯值计算误借输出 block 范围；须保留每个中间操作自身坐标域，
  内存读写时再按窗口变换。以正确全2、错误全1及多种窗口/控制形态验证。
  复审确认同一合法 scratch 窗口在无 jit 时误入旧 full-block 路径；路径选择应读取
  实际 scratch/访存需求，不能依赖无关 jit 包装。
- 布尔 `lax.max/min`：谓词被送入 SymPy 数值 extrema，公开接口发生内部异常。
  应按 bool dtype 分派为 OR/AND，保留输入定义条件；普通 JAX 与 Pallas 入口均验证。
  相邻 `lt/le/gt/ge` 同样已复现；遵循 `False < True` 转为布尔逻辑，其他 dtype 不变。
  bool `reduce_max/min` 复用已有 OR/AND 归约及空域单位元，保持原预算、定义域与内存条件；
  Pallas 只覆盖已有 reduce_max，普通 JAX 同时覆盖 reduce_min，不扩 core 原语支持。
  组合回归又确认：常量 bool 归约需要恢复被标量化的 Source 广播；bool where 结果不能
  作为数值 Piecewise 传给逻辑运算。两者均在代数适配边界保留类型、来源和原条件。
  布尔 Piecewise 统一在符号张量边界转换，覆盖常量、where、pad、concat 与区域输出。
  逐分支串联 ITE 在 1024 分支实测触发递归上限，改用保持 first-match 的平衡 ITE DAG；
  不做 DNF、不展开坐标，原生成预算及定义/内存条件继续有效。

源码合法性用离线 TPU AOT，数值用独立 CPU/NumPy，二者不代替公共比较结论。
原有未提交改动保留；不修改受保护文档，不提交或推送。

## 工作项

- [x] 修复 scratch 范围，补正例、反例及拒绝边界。
- [x] 修复布尔 extrema，检查相邻布尔运算并补回归。
- [x] 合并后独立复审、相关测试及静态检查，记录实际证据并归档。

## 验证

日志与逐批 JUnit XML 在 `/tmp/jax-reorder-scratch-bool-fix/`。本轮使用已有 `.venv`，
不改依赖；回归命令如下（各批分别运行，最后按测试 ID 去重）：

```bash
.venv/bin/python -m pytest -q tests/test_pallas_{grid_state,local_scope,local_entry,semantic_slice,remote_grid_binding,remote_nd,remote_revisit,local_any,dma,protocol,tensor_memory}.py
.venv/bin/python -m pytest -q tests/test_pallas_{algebra,tensor_memory,padding_algebra,padding_domains,remote_values}.py tests/test_functions.py tests/test_package_layout.py
.venv/bin/python -m pytest -q tests/test_algebra*.py tests/test_symbolic_explicit.py
```

core 批次初次 492 passed、1 failed：部分输出只写一半仍被正确拒绝，但扩展路径将
拒绝原因从“partial Ref 不支持”改成真实的“输出未初始化”。测试更新为检查未初始化，
该项单独重跑通过；不把旧错误文字当作行为契约，也未放宽内存安全断言。

新增目标回归共54项（scratch 7、布尔47），全部通过；最终测试工厂的84个离线 TPU
编译用例（scratch30、布尔54）全部通过 trace/lower/IR verify/backend compile，
使用 `v5:2x2x1` megacore，没有实机执行。早期两个直接 bool max/min 的 kernel 被 TPU
lower 拒绝，最终测试使用原生可编译的外层 JAX 形式；未把原失败源码计为成功。

独立复审：12组scratch窗口正反例、21组定义域/初始化义务、8组布尔Piecewise消费者、
常量数组原反例、30点重叠guard真值检查均符合预期；Source和通信/alias边界未被放宽。
128/1024分支常量的完整公开比较均通过；1024分支为4100个唯一节点、深度13。
序列化会展开共享节点（约587KB），没有宣称序列化体积严格线性。

目标测试/AOT日志分别在 `/tmp/review-local-range-route-target.xml`、
`/tmp/review-local-range-fixed-aot.log`、`/tmp/review-local-route-aot.log` 和
`/tmp/jax-reorder-bool-scalar-fix/`；独立义务验证在 `/tmp/review_boolean_domains-final.log`。
Ruff、format、basedpyright、tracing、文档链接和差异检查通过。最终 Pallas 批次439项、
数学批次722项全部通过；合并 core 的通过项及修正后的单项重跑，按测试 ID 去重共
**1569项通过**，包含全部54项新增回归，不称为单次全量运行。最终清单与文件校验和在
`/tmp/jax-reorder-scratch-bool-fix/final-verification.json`。

本轮已确认问题全部修复，独立复审未发现剩余必须修复项。既有不支持边界及 CAS 未决
规则保持不变；上述证据不表示对任意程序或整个仓库作无缺陷保证。
