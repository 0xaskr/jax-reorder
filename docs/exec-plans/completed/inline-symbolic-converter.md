# 将 jax2sympy 转换代码并入项目

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

## 目标与完成条件

用户要求把 jax2sympy 代码直接纳入项目，最多在文件开头保留许可证；随后明确未用到的代码也迁入，便于以后复用。
删除独立 vendor 包和 manifest；生产用到的 SSA 遍历、精确常量和标量映射并入现有构造器，
其余完整数组映射与逐元素入口集中到普通项目模块，来源和完整 MIT 声明放入文件头。

## 设计与取舍

生产入口只调用带 builder 的遍历路径以及 36 个标量映射；上游通用数组适配并未参与生产数学判断。
将遍历直接并入 SymbolicBuilder.graph，保留常量、输入绑定、输出顺序、shape 检查、预算和未使用计算的定义条件。
逐点映射直接操作 SymPy 标量，删除 NumPy 零维数组包装；不修改定义域或后端判等策略。
精确常量转换沿用现有实现。当前设计入口为[代数开发记录](../../design-docs/current/algebraic-equivalence.md)。

未用于生产的完整数组映射和逐元素遍历迁入 algebra_explicit.py，保留可调用的
jaxpr_to_sympy_expressions 入口与原有行为；该公式接口不跟踪定义条件，不能替代生产比较。
旧 vendor 测试中的 23 个公式场景改用这个项目模块，并继续阻断 JAX 执行、使用独立公式核对。
规模脚本保留显式/惰性对比，只改导入位置。历史上游评估工具支持传入转换函数和映射表，
原来的外部 checkout 评估方式仍可运行，测试无需伪造上游包名。
受保护文档、依赖、公开比较 API 与 Nerva 原始 fixture 不变。本轮未授权提交或推送。

## 工作项

- [x] 核对生产调用、旧基准和测试依赖。
- [x] 并入生产实现，保留完整数组转换代码和文件头许可，删除 vendor 包。
- [x] 迁移公式测试和规模脚本，更新当前开发文档与引用。
- [x] 运行标量映射前后对照、转换/数学回归、Ruff、示例、打包和文档检查。
- [x] 审查差异、记录验证证据并归档。

## 验证结果

- 生产 36 个标量映射的 114 组独立符号输入在改动前保存结果；改动后逐项精确相同，包含布尔映射、整数负幂和常量。
- 原数组映射中的 57 个函数/赋值定义经 AST 对照全部原样保留；两个承载文件的完整 MIT 声明与原 LICENSE 逐字核对一致。
- 显式转换的输入/输出读取函数及遍历函数也经 AST 对照保留，仅移出生产 builder 分派；其余 60 个 primitive 映射完整保留并可调用。
- 首轮转换/可信后端/资源回归 **46 passed，35.93 秒**，其中逐元素探针包含 23 个独立公式场景，并阻断 JAX 编译和执行。
- 规模脚本正常运行：长度 8/32/128/512 时，逐元素表达式唯一节点为 18/66/258/1026，惰性表达式均为 12，与迁移前一致。
- 全部代数测试、Nerva 和函数入口回归 **741 passed，570.91 秒**。本次是转换代码迁移，没有重跑不相关的全仓测试。
- 空输入、空输出及无输出时未使用的部分函数条件分别验证通过：前两种保持空结果，后一种仍明确拒绝。
- Ruff、175 个文件的格式检查、全仓文档链接与 `git diff --check` 通过；回归期间源码、测试和实验脚本哈希保持不变。
- tracing 与 SymPy 示例通过。离线 sdist/wheel 构建通过；wheel 无 vendor 包，两个源码文件的完整许可随包保留。
  隔离导入验证生产的 equivalent/inequivalent/unknown 三条数学路径，且不加载逐元素参考模块；
  随后导入并调用逐元素入口成功，完整 60 个 primitive 映射保留。

复现命令（只读缓存环境可设置 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`）：

```bash
uv run pytest -q tests/test_symbolic_explicit.py tests/test_trusted_sympy.py tests/test_algebra_resources.py
uv run pytest -q tests/test_algebra*.py tests/test_nerva_symbolic.py tests/test_functions.py
uv run python tools/experiments/equivalence_backends/trusted_sympy_scaling.py
uv run python tests/inspect_jaxpr.py
uv run python tests/sympy_equivalence.py
uv run ruff check .
uv run ruff format --check .
uv run python tools/check_docs.py --changed
uv build --offline --out-dir /tmp/jax-reorder-inline-converter
git diff --check
```

上述验证覆盖非有限常量、闭包常量、输出换序、嵌套区域、空输入/输出及未使用部分函数条件；
不以数值抽样作为公式转换正确性的依据。未提交或推送。

验收末尾出现并发新增的 `symbolic-code-equivalence-research-2026-10-16.md` 及开发索引修改，
本轮保留，未纳入转换代码迁移范围；已验证的源码、测试和实验脚本没有并发变动。

## 未解决问题

无新增边界；保持当前支持范围和 unknown 分类。
