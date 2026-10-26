# 历史研究与选型索引

本目录保存 2026-10-15 至 2026-10-16 的后端评估、探针和研究综述。状态为研究参考，
不是已生效的产品契约；2026-10-26 的文档迁移没有重新安装候选、运行探针或复验外部资料。
固定探针与测试资产继续保留，不因归档而废弃。
现行能力见[贡献者指南](../../../../CONTRIBUTING.md)，现行方案见[设计索引](../../../design-docs/index.md)。

| 记录 | 状态与已被替代的部分 |
| --- | --- |
| [开源后端首轮评估](open-source-equivalence-backends.md) | 历史选型；候选优先级、独立证明要求和正常结果 `unknown` 的口径已被后续生产方案替代 |
| [组件改造与架构复用](equivalence-backends/architecture-reuse.md) | 历史改造实验；jax2sympy 转换逻辑的生产接入后来已完成，文中的“下一步”不是当前路线图 |
| [CAS 后端](equivalence-backends/cas.md) | 研究参考；保留固定版本符号子问题与真实 helper 探针，不能当作完整模型证明或当前能力上限 |
| [等式饱和与张量规则](equivalence-backends/rewriting.md) | 研究参考；区分上游规则、本地补充规则和未构建候选，不将当时未证明解释为引擎不可能支持 |
| [张量验证与 SMT](equivalence-backends/verification.md) | 研究参考；记录精确模型、源码审计与运行边界，局部 SMT 结果不能直接替代源程序结论 |
| [编译器与图转换](equivalence-backends/compiler.md) | 研究参考；保留 MLIR/TVM 探针和其他候选的文档核对，不表示项目已采用这些后端 |
| [符号代码等价研究综述](equivalence-backends/symbolic-code-equivalence-research-2026-10-16.md) | 研究参考；论文与工具路线不整体标为失效，选型建议须结合当前实现重新评估 |

已生效的数学路线以[代数比较设计](../../../design-docs/current/algebraic-equivalence.md)为准；
生产迁移过程见[可信 SymPy 后端](../trusted-sympy-backend.md)和[转换器并入项目](../inline-symbolic-converter.md)。
研究记录中的 `unknown` 是当时的探针或比较器口径，不能据此恢复已删除的公共结果状态。
