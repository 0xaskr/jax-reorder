# Pallas 完整结构比较

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

创建及完成：2026-10-19。状态：本轮公共结构入口与定向验证完成。
设计入口：[Pallas / JAX 等价比较](../../design-docs/current/pallas-jax-equivalence.md)。

## 目标与完成条件

按本轮用户收缩后的目标，Pallas 层只比较完整计算图。沿用锁定的 JAX/jaxlib
0.11.2、TPU、同 mesh、每侧一个 `pallas_call` 和前向计算边界。原语、参数、
嵌套区域、Ref 绑定、常量、输入输出位置与有副作用操作的相对顺序均须保留。
独立纯计算可重排；不同 block/grid、通信协议或归约分组允许返回结构 `different`。
跨分块和数学证明不作为本次结构覆盖的前置条件。
“完整”指不删减收到的图，不表示任意官方语法组合或任意规模都已验证。

完成条件：公共 `compare_jaxprs`、`canonicalize` 和函数结构模式可以消费 Pallas；
自动模式的结构快路径可直接通过，结构不同仍进入独立代数流程，未支持转换明确失败。
信号量按资源绑定和操作拓扑比较，不模拟计数、不猜测 start/wait 配对、不要求死锁证明。

## 设计与取舍

- 复用已有 `Graph`、BFS/无根图搜索及原图 correspondence，不另写一套图匹配器。
  Pallas 源格式解释集中在 `core/adapters/pallas`，由现有语义编码器调用。
- 现有 `ProgramView`/normalized 分析保留供后续数学 lowering 使用；其受限准入条件
  不用于阻塞通用结构比较，也不将 normalized 结果伪装为原图结构证据。
- 副作用操作及包含它们的区域按原顺序加边。纯计算可穿过这些节点，但不能改变
  数据依赖；不跨越任何副作用节点强行移动 DMA start 到 wait 前。
- 信号量身份由分配/形参绑定、作用域、设备及索引表达，图间使用一致的 Ref 映射；
  signal/wait/read、计数参数、发送/接收等待角色与别名均参与精确比较。
- 未知原语、effect、描述符或 schema 仍明确失败。官方覆盖清单独立记录；拒绝测试
  不能计入支持完成度。整数溢出、运行时延迟和硬件调度不作为本次证明目标。

## 工作项

- [x] 同步合并后的主分支，恢复现有两处文档改动。
- [x] 审计锁定 TPU lowering 原语及 Pallas/Ref 描述符，建立结构适配和清单。
- [x] 补齐 effects 的编码、顺序约束及嵌套区域处理，复用公共图规范化与对应复核。
- [x] 公共 API 回归：结构通过/不同、函数契约、自动模式及代数拒绝边界。
- [x] 覆盖语义组合：多维/动态索引、循环、局部资源、DMA、信号量与随机状态。
- [x] 按用户追加要求，选择 GitHub 代表性 Pallas 项目，锁定源码及许可证，编写本地 AI 正反例。
- [x] 开源 kernel 测试分别报告原图结构、变体判定、CPU interpret 数值与 TPU 验证范围。
- [x] Ruff、格式、类型、结构/既有 JAX 回归、inventory、示例及文档检查。
- [x] 审查最终差异，记录实际支持与验证边界并归档。

## 决策与进展

- 原有 Pallas adapter 只接受受限分析片段；直接扩展公共结构消费者能复用既有
  JAX 内部端口规范化和独立对应验证，避免用手写新 matcher 重复这些能力。
- 受保护文档及权限配置保持不变。本次只更新开发设计和执行记录。
- 公共结构消费者直接使用 source codec；原 ProgramView 的有限存储/协议消费者保持独立。
  GridMapping 内的 index map 在原 trace_env 下校验；scoped 捕获绑定不编码 tracer 载荷。
  `ref_transforms` 跟随局部分配形参，不能误当作捕获实参的逐槽元数据。
- 官方 TPU frontend 延迟注册 pipeline/einshape 和 PRNG；源适配在固定身份集合前先导入它，
  两种库导入顺序的独立进程测试均通过。当前身份集合 150 个，Pallas/state 显式 schema 44 项；
  清单准入不等于逐原语全部组合已验证。
- 大型 pipeline 暴露既有无根搜索退化；仅对没有隐藏参数入边、完整参数及实际邻接相同的
  孪生顶点剪枝。加入隐藏绑定共享/分离、24 种重命名、8 个无关重复节点的回归；不删节点。
- 初次结构回归 1026 passed、2 failed：旧测试将现在已支持的 debug print 当作未知 effect。
  改为真实 Python callback 拒绝路径；未知 effect 自身仍有独立拒绝回归。两个文件复测 42 passed。
- 新 API 测试的 stochastic_round 调用曾遗漏 random_bits/keyword 参数，修正为锁定版本签名；
  剪枝预算测试曾混入实际参数引用目标，调整为八个真正无关的重复节点后测试通过。
- 选取 Tokamax 的完整 RoPE 函数（含量化分支）与 vLLM TPU Flash Attention 完整单步函数。
  原源码、两个许可证、commit/blob/SHA-256 固定；17 组 AI 候选由本地源码记录预期。
  六组结构等价、八组错误且有数值见证、三组结构不同但数值一致。每组两套输入，共 34 对
  CPU interpret；原结果另核对独立 NumPy 参照。Pallas tracing 自身会删除未使用 sin，已单独注明。
- 原有 full-layers 边界脚本同步为四条 GLM5 预算异常、Wan/EPMoE 两条自比较及两条追踪拒绝。
  自比较不能计为候选等价，CPU 正常编译拒绝与 EPMoE 原 interpret 分支分别报告。

## 验证结果

- 新结构/开源/无根搜索三文件：`pytest -q tests/test_pallas_structural.py
  tests/test_pallas_upstream.py tests/test_unrooted.py --tb=short`：**79 passed，32.54s**。
  其中开源固定内核 **19 项**（2 个来源摘要、17 个候选），Pallas 结构 **47 项**。
- `tests.full_layers.evidence.verify_boundaries()`：8 条边界记录全部符合预期，
  GLM5 四条预算异常、Wan/EPMoE 自比较、两个原始追踪拒绝；EPMoE 数值参照通过。
- `pytest tests/test_full_attention.py -k 'original_glm5 or original_cross or original_moe'`：
  **10 passed，21 deselected，121.73s**，一个上游 reciprocal 迁移弃用警告。
- Ruff、格式检查（229 文件）、basedpyright（0 errors、0 warnings）、adapter/lax inventory 通过。
  `python -m tests.pallas_kernels` 的 17 行结构矩阵通过；tracing 与 BFS 示例通过。
- 结构相关 23 文件最终回归：**1029 passed，244.87s**。文件为 adapter_guards、ai_cases、
  call_bindings、canonical、collectives、combination_matrix、comparison_errors、cond_bindings、
  correspondence、custom_derivatives、feedback_bindings、functions、lax_primitives、mixed_regions、
  nested、nested_cond_four_levels、package_layout、region_bindings、semantics、shard_bindings、
  symbolic_explicit、unrooted、algebra_structural（均为 `tests/test_<name>.py`）。
- `pytest -q tests/test_pallas_*.py --tb=short`：**262 passed，104.51s**，包含全部既有 Pallas
  片段、存储、DMA、远端协议、隔离通信模拟与本轮结构/开源测试。
- `pytest -q tests/test_depth.py tests/test_stress.py --tb=short`：**6 passed，32.26s**。
  包含共享嵌套区域、2000 层参数、6000 节点依赖链、512 路扇出。
- Markdown 链接与 `git diff --check` 通过；确认受保护文档/权限配置无改动，用户已有纯死代码
  设计记录与索引改动保留。未提交、未推送。
- 曾启动全量 pytest，在旧版本进程已加载且仍修改代码时中止，不报告全量通过；
  本轮使用上述最终源码分组回归，不重复未修改的全量 SymPy worker 长时测试。

## 未解决问题

- 大型 emit_pipeline 搜索性能、GLM5 八个歧义 carry 槽位的排列预算仍是实际限制；
  不增大默认预算掩盖边界，不以自比较短路绕过未知语义。见技术债 LAYER-001/PALLAS-001。
- 没有真实 TPU 编译执行证据；多主机非本地常量、全部官方语法组合、跨 block/grid 数学
  lowering、一般协议证明与 Pallas↔JAX 仍未完成。结构接入不宣称这些目标已完成。

后续边界已记入[待处理清单](../index.md#待处理)，原语义中间层的更广目标继续由
[后续计划](../active/pallas-semantic-layer.md)维护。

## 已解决问题编号

以下编号于 2026-10-26 从问题总表归入本计划，保留原已解决状态、优先级和依据；本次文档合并未重跑历史验收。

| 编号 | 问题 | 原优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- |
| LAYER-001 | 公共 Pallas 结构入口与大图接口细化已接入；GLM5 absorbed Attention/Decoder 的 prefill/decode 四个真实配置在 search_budget=1 下通过完整结构比较，Wan/EPMoE 原图自比较通过。CPU 正常 lowering 仍拒绝 TPU 内核；自比较不计为任意改写候选认证 | P2 | 已解决 | [实际后端边界](validation/full-attention.md#已确认的拒绝与未支持)、[接口细化](../../design-docs/current/region-bindings.md) |
