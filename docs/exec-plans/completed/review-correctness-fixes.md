# 修复仓库审查确认的求导与内部绑定问题

## 目标与完成条件

修复 2026-10-14 审查复现的三项问题：普通 custom VJP 混合输出的整数余切类型错误、全整数输出漏比 VJP 前向规则、仅由无根计算使用的私有输入同步置换漏认。前两项不得再把实际一阶程序不同的图判为 equivalent；第三项必须通过结构证据，不能依赖代数后备路径。

仅修改实现、相关回归及开发记录。受保护文档保持只读，不执行提交或推送。

## 设计与取舍

- [求导契约](../../design-docs/current/function-and-derivative-contract.md)：按锁定 JAX 的 symbolic_zeros 配置恢复整数余切；全整数输出保留前向和实际反向模式，空输出保留前向图和残差。
- [区域接口](../../design-docs/current/region-bindings.md)：只以有序输出和有序根可达节点的使用位置锚定私有输入；其余输入进入已有有界槽位搜索，各候选重新规范化完整图。
- 保留独立对应校验、未知语义拒绝和预算限制。cond 的全部分支共同确定锚点；scan/while 的角色边界和位置元数据仍同步移动。

## 工作项

- [x] 复核现有实现、锁定 JAX 适配及审查反例，记录修复设计。
- [x] 添加能在修复前失败的回归，覆盖正例、反例、零余切和预算拒绝。
- [x] 修复 VJP 物化及私有槽位规范化，通过针对性测试。
- [x] 运行全量 pytest、Ruff、格式、示例及原语/适配清单。
- [x] 审查最终差异，完成文档检查、验证记录并归档计划。

## 决策与进展

2026-10-14：以干净的 HEAD 4921068 开始修复。审查基线为 1775 passed、1 个上游弃用 warning。已确认 JAX 0.11.2 在普通 VJP 中实例化零余切，在 symbolic 模式中使用 SymbolicZero；当前实现无条件使用后者。无根输入漏认由首次使用顺序依赖旧形参编号导致，复用现有完整候选搜索处理，不忽略未使用计算。

运行公开 `jax.vjp`/pullback 后补充确认：全整数输出的 pullback 在收到显式 float0 时仍调用 bwd，symbolic 模式也保留这个显式数组。输出模式因此需要枚举整数叶子的显式/符号情况，不能只增加前向记录；空 pytree 才只保留前向。全整数输入不触发注册 fwd，不扩展输入激活范围。

首批 25 个回归在未修复实现上全部失败，涵盖两种余切配置、整数/布尔、标量/向量、空输出、有序前向，以及 jit/cond/scan/while/shard_map 的私有绑定。修复后连同原有相关测试共 61 项通过（14.87 秒）；追加 symbolic float0 组合预算与无效整数输出 bwd 的拒绝回归，随后进入完整验证。

全量 1802 项测试通过，比基线新增 27 项；保留原有 1 个上游 Pallas 弃用 warning。私有输入换序通过独立对应校验、反向比较、原图不变、规范化幂等和执行核对；错误共享及 donation 绑定仍判为不同，预算不足返回 unknown。完成实现差异审查，计划移入 completed，归档链接检查通过。

## 验证结果

以下命令均使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline` 前缀，在 Python 3.12、JAX/jaxlib 0.11.2、CPU 环境执行。

- `pytest -q tests/test_custom_derivatives.py tests/test_unrooted.py`：首批修复后 61 passed，14.87 秒；随后追加 2 项拒绝回归，一并纳入全量测试。
- `pytest -q`：1802 passed、1 warning，414.62 秒；warning 来自上游 `jax.experimental.pallas.reciprocal` 弃用入口，与审查基线一致。
- `ruff check .`、`ruff format --check .`：通过。
- `python examples/inspect_jaxpr.py`、`python examples/bfs_reordering.py`：通过。
- `python examples/collective_jaxpr.py --devices 2` 和 `--devices 4`：各 22 个通信组合通过，每个组合 4 组数值样本；图比较与执行抽样分别验证。
- `python tests/adapter_inventory.py`、`python tests/lax_inventory.py`：与锁定环境一致。
- `python tools/check_docs.py --changed`：4 个 Markdown 文件、25 个本地链接通过；`git diff --check` 通过。

## 未解决问题

本轮不扩大到其他未适配求导、一般代数恒等式或新的执行后端。
