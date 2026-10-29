# KDA 完整比较与既有回归修复

2026-10-28 阶段验收：结构搜索、后续前端覆盖与回归修复完成；最终清单共
5,039 项通过、2 项默认跳过，计数合并全量主运行与修正后的定向复测。
完整 KDA 资源/数学证明仍受 PALLAS-005 阻塞，详见下方证据。

## 目标与完成条件

解决 [KDA 历史验证](kda-backward-history.md) 暴露的问题：

1. 真实 mega kernel 在默认预算下可完成结构规范化；保留所有运算、资源、
   接口与独立原图对应校验，不能删除副作用或丢弃未使用分配以取得匹配。
2. 推进该真实 kernel 的局部资源语义与数学比较，修复随后出现的实际阻塞，
   用既定历史版本和 AI 正反例复验，不以通过拒绝路径代替支持。
3. 核对并解决全量回归中的 7 个原有失败；测试预期只能根据比较契约和独立
   证据调整，不能为了绿色结果放宽语义。
4. 更新可复现报告及待办状态，通过相关回归与最终全量检查。

## 实施原则

- 以当前源码与已保存原图为依据。结构与数学分层，纯死计算仅按既有契约
  在数学 lowering 中处理；不凭结构差异推断数学不等价。
- 接口消歧先利用置换不变的类型、使用和资源属性；只在可证明完全对称时
  合并搜索分支，其余仍保留有界精确搜索和独立对应校验。
- 本计划记录实现设计与实验；不修改未获得具体授权的受保护架构/设计文档。
- 现有 KDA 测试和报告为前阶段工作，保留源码快照与改写预期。

## 工作项

- [x] 定位 9 个未消歧槽位的角色，构造最小正反例并修复结构搜索。
- [x] 补齐随后暴露的原语、Ref 视图和编译参数覆盖，定位完整 KDA 的下一处阻塞。
- [x] 修复既有 7 项回归，验证共同定义域和反例构造的正确性。
- [x] 更新历史/AI 报告，完成回归、类型、文档及差异检查。

## 进展与验证

- 保留上一阶段 KDA 测试资产；已实现私有分配的精确对称剪枝。
- 结构瓶颈位于 `scan → scan → run_scoped → run_scoped` 的私有资源排序。
  已逐槽位确认 9 个位置 `(10,11,18,19,20,21,22,23,25)` 是完全未引用的
  新分配资源，并非只出现在 effects 的捕获值；包含重复 VMEM 占位缓冲和
  一组未使用 DMA semaphore，所有 `ref_transforms` 均为空。
- 新鲜私有分配按完整 aval 和逐槽位元数据分组。仅当资源在输出、全部节点、
  effects、参数与上下文字段中完全没有引用时，才合并同组的相同排列；保留
  分配数量、类型、内存空间和 transforms。普通捕获值与仍有引用的槽位继续
  有界精确搜索；含绑定的类型/元数据不使用此简化。
- 完整 KDA base 在 `search_budget=1` 下自比较已得到 `equivalent`，耗时约
  10.5 秒；独立原图对应校验和 JAX 类型检查照常执行。
- 结构专项回归 `test_pallas_structural.py`、`test_interface_refinement.py`、
  `test_unrooted.py` 共 80 项通过；涵盖未使用资源置换、元数据绑定与
  effects/参数中的隐藏引用。涉及文件 Ruff、格式、类型及差异检查通过。

## 后续前端与资源覆盖

- 新增 `pad` 的精确坐标投影，保留边界填充、负边界裁剪、interior padding、
  填充值和空维度；`exp2` 的理想实数语义为 `2**x`。
- 显式 VMEM block 使用既有 buffered 生命周期。动态 slice 或 scalar 选择的
  多层 NDIndexer 使用 Ref view chain；每个中间视图仍分别检查越界。
- `vmem_limit_bytes`、`disable_bounds_checks` 经具体审核后可进入完整 tensor
  资源证明；它们仍保留在源契约中，不能关闭检查器的访问证明。
  未审核的 semaphore/barrier 选项继续拒绝。
- 新增 28 项回归，另有动态范围、张量内存、局部值及代数 347 项、分片与
  Ref 视图等 155 项通过。独立审查枚举 493 种一维 padding，与 `lax.pad`
  逐点一致；实际越界、两侧编译参数不同及中间视图越界均有永久回归。

## 保留的完整 KDA 阻塞

推进后的公共局部语义和数学入口都到达原始源码的首次 scratch 读取：

```python
gate_dA_acc[:] = jnp.zeros_like(gate_dA_acc[:])
gate_dbias_acc[:] = jnp.zeros_like(gate_dbias_acc[:])
```

位置见[最新 mega 原文](../../../tests/fixtures/upstream_tpu/pallas_kda/objects/61dbbc60aabb62b72b1472e26f0f93b5d11247e9.txt)。
对应 jaxpr 保留 `get → broadcast zero → swap`；`get` 的值未使用，
但仍是未初始化 Ref 读取。现有
[`test_unused_uninitialized_read_is_still_checked`](../../../tests/test_pallas_semantic_slice.py)
明确要求资源检查保留此读取义务，不能将它当成无副作用纯函数删掉。

本阶段保留 base 和既有契约，记录 `UnsupportedJaxprError`，不以解释器数值
通过替代完整资源/数学证明。改为 `jnp.zeros(ref.shape, dtype=ref.dtype)` 可作为
后续独立候选研究，不能替换原始 base，也不能提前承诺后续所有计算都已支持。

## 代数回归调查

- 2026-10-28：`log(x)` 对 `x`、`x/x` 对 `x+1` 及未使用 `log` 的负例，
  可在共同合法输入 `x=1` 上获得完整输出反例。测试应核对精确见证，而非要求
  条件未决；普通 JAX 死计算的定义条件仍保留，本次不扩大 DCE 范围。
- DeepSeek BF16 错缩放已可取得精确反例；错转置和提前舍入的多个输出坐标重复
  检查同一完整定义域，默认预算耗尽。现仅缓存固定输入赋值下的完整定义条件
  检查结果，键保留输入 shape/dtype、cast 映射、输入模式和原始条件。未证实的条件
  只可复用为该模式不可接受，不能推出等价；预算异常继续传播。
- `test_algebra_inequivalence.py`、`test_algebra_sum_domains.py` 和
  `test_deepseek_v41_layer.py`：**225 passed**。包含缓存跨输入类型、转换格式、
  输入模式和原始条件的隔离，以及非法读取、非精确 cast、其他输出定义域和
  预算异常不得记作失败候选的拒绝用例。
- 这三个 DeepSeek 舍入相关分支恢复为默认预算内的 `condition_undecided`，不声称
  已有符号不等价证明；BF16 错缩放则在 forward/JVP/VJP 全部判不等。
- 本子任务修改文件通过 Ruff、格式、basedpyright 和差异空白检查；未修改
  受保护设计文档，未提交。

## 最终验证

- 最终源码复跑六配置的全部 53 个有效历史/AI 组合：完整结构 17 等价、
  36 不同；两个局部入口各 53 项均因上述原始 scratch 读取明确拒绝。
  没有结构搜索预算失败或 worker 失败。
- 官方 TPU CPU interpreter 共 106 次执行，无竞态报告或非有限公开输出；
  五个 AI 候选的两种子完整输出数值预期 60/60 符合。重算、BF16 gate 输出和
  DMA batch 尾窗均已复验，详细结果见[报告](kda-backward-history.md)。
- 报告的全部生产源码和三个 KDA 脚本哈希与最终测试版本逐一核对一致。
  两个真实 `compute_dav` 小 tile 正例另外通过结构/代数证明，不充当完整
  kernel 的证明。
- 全量运行发现 `test_pallas_dynamic_views.py` 仍将已审核的
  `disable_bounds_checks` 列为未知执行选项。改为检查仍未审核的
  `disable_semaphore_checks`，并补齐布局开/关与资源选项的组合正例及
  越界拒绝。两个相关文件复测 **136 passed**；没有修改生产代码或来源快照。
- 全量主运行 `uv run pytest -q`：**5,035 passed、1 failed、2 skipped**，
  唯一失败是上述旧参数预期。随后两文件 **136 passed**。将两个测试清单按
  node id 核对、以复测结果替换对应文件后，最终 5,041 个用例覆盖为
  **5,039 passed、2 skipped**；该计数合并主运行与定向复测。
- 两项跳过是默认关闭的 Diffusers SNR 60 秒超时测量，本轮 KDA 用例没有跳过。
  Ruff、格式、basedpyright、tracing 示例、文档链接及差异空白检查均通过。

## 已解决问题编号

| 编号 | 原优先级 | 结论与证据 |
| --- | --- | --- |
| PALLAS-004 | P1 | 只合并完全无引用且同型同元数据的私有分配排列，保留所有资源；完整 base 在搜索预算 1 下自比较通过，最终 53 组结构比较完成 |
| TEST-001 | P2 | 核实共同定义域反例并缓存重复条件检查，225 项相关回归通过；未决的舍入证明仍保持未决 |

后续读取准入问题单独登记为 PALLAS-005，本阶段不宣称完整 KDA 资源/数学已支持。
