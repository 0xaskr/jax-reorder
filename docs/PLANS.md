# 执行计划管理

跨文件功能、算法实现、大范围迁移或复杂重构在 [active](exec-plans/active/) 中维护执行计划。简单修改直接执行，无需建立形式化计划。

## 当前计划

当前没有进行中的执行计划。

## 已完成计划

- [计算图等价比较的完善与系统验证](exec-plans/completed/graph-equivalence-validation.md)：完整图、函数契约、注册一阶求导、区域反馈及独立原图复核完成；原 GLM5/Llama/CLIP/Flux 完整层与实际后端边界有报告，全量 1078 项通过。
- [SGLang-JAX 全模型层级对抗验证](exec-plans/completed/sglang-model-layers.md)：全部注册入口映射到 36 个真实层，216 个候选/签名组合、1080 组数值检查通过；新增 182 项回归，全量 844 passed。
- [cond 内部参数绑定规范化](exec-plans/completed/cond-bindings.md)：修复捕获槽位同步换序误判，新增 30 项回归，全量 662 passed。
- [真实层级场景的 AI 固定改写](exec-plans/completed/business-layer-cases.md)：五个 base、20 个 AI hardcode 候选；独立参照与两组种子验证通过，新增 67 项回归，全量 632 passed。
- [带通信的 jaxpr 验证](exec-plans/completed/collective-jaxpr.md)：补齐 pvary；22 个组合通过双/四 CPU 设备验证，新增 49 项回归，全量 565 passed。
- [第一轮：基于 BFS 的 jaxpr 计算图规范化与比较](exec-plans/completed/bfs-reordering.md)：实现与自审完成，516 个测试通过；CPU 原语逐项覆盖、AI 经典案例及 LlamaMLP 正反例已固化。

## 计划应包含

- 目标、范围和可观察的完成条件。
- 子任务、依赖、设计链接与尚未确定的选择。
- 已作决定及理由，进展、失败和下一步。
- 实际验证命令、结果和未验证条件。

可使用下面的结构：

```markdown
# 计划标题

## 目标与完成条件

## 设计与取舍

## 工作项
- [ ] 工作项

## 决策与进展

## 验证结果

## 未解决问题
```

任务进行时更新原计划，保持可恢复。完成后移入 [completed](exec-plans/completed/)；尚未解决的缺口记录到 [技术债](exec-plans/tech-debt-tracker.md)，并保留相关证据或工单链接。
