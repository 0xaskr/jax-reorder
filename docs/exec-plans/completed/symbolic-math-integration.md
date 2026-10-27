# 数学等价后端接入、完整验证与 main 发布

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

## 目标与完成条件

2026-10-15，用户授权在本地修改和完善当前工作；全量验证后最新明确要求直接提交到 main，覆盖先前独立分支/PR 的发布安排。此任务延续两份 JAX 程序的计算图等价、数学等价、数学不等价和 unknown 四类语义目标。模型 layer 用于验证比较器；不运行模型作数值判等。

本轮把[开源评估](research/open-source-equivalence-backends.md)中的可行路径接入正式比较流程，而不止保留实验脚本：

- 正式依赖与公开比较入口能够使用 SymPy 的数学变换，复用现有索引 IR、图契约和独立证据复核。
- 增加带明确证据的数学不等价结果；结构不同、证明失败、局部节点不同不能替代完整输出不等价。
- 推进真实 GLM 核心 5+5 的完整图验证，重点是 guarded-softmax/PV、mask、索引和区域的组合。逐项记录实际证据，不降低原有验收口径；若存在理论或上游语义限制，必须完成实证分析并显式保留 unknown。
- 覆盖正例、负例、拒绝、预算与篡改证据；保持空域、非有限值、索引绑定和全部有序输出的语义。
- 完成全量 pytest、Ruff、格式、tracing 示例、文档链接与差异检查，并经独立审查。
- 保留并纳入本任务相关的已有 GLM 纠偏、`-inf` 支持及评估产物；不改受保护文档。
- 验证完成后按最新授权提交并推送 main，核验远端提交与本地一致。

## 初始状态与约束

- 本地 main 与远端 main 初始为 `01eeb8275dc102ec18a52012a4f686682bb94ac1`。
- 工作区有此前未提交的 GLM 抽象追踪纠偏、`-inf` 支持、开源评估和实验脚本。本轮保留这些内容，不执行 reset/revert 或覆盖用户工作。
- 当前正式代数流程为 Lowerer → PolynomialNormalizer → 独立 verify_algebraic_proof；SymPy 仅存在于实验脚本。
- 现有 GLM 本地基线为 4 个已证明正例、6 个 unknown；五个负例尚未取得完整输出的不等价证据。
- `README.md`、`ARCHITECTURE.md`、`AGENTS.md`、`CLAUDE.md`、`docs/index.md`、`docs/PLANS.md`、`docs/design-docs/` 保持只读；设计与进展写在 development-docs/exec-plans。
- 浮点数值误差不作为本轮判等标准；数学定义域、类型转换、显式 `-inf`、空域和排序/索引语义不能省略。

## 设计与工作分解

1. 调查既有 IR/proof/函数结果契约，明确外部变换的证据接口；先写开发设计再改算法。
2. 保留结构快路径与已证明的索引乘加规则，将 SymPy 放入受控的数学变换层；避免 `.equals`、随机采样和无条件 `simplify` 判等。
3. 对超出纯乘加的函数/有理式规则保存条件和可重放步骤；预算在表达式增长之前检查。独立验证器不能调用同一个搜索过程冒充复核。
4. 数学不等价只在可解释的完整有序输出上构造证据；求解器反例须满足源输入约束，未解释函数或中间义务反例不直接上升为源程序结论。
5. 分析并实现 guarded-softmax 的数学域推理，使合法有限输出能进入后续 PV 代数规则；保留全 mask 和未知边界。
6. 用真实 GLM 原图/改写图和通用算子组合验证，不写按用例名/源代码模板直接接受的路径。

三个 subagent 先分别调查 SymPy/证明接口、完整 GLM 组合、数学不等价结果契约；主代理负责设计整合、依赖、函数入口和最终验证/PR。确定模块所有权后再并行编辑，避免同文件冲突。

## 工作项

- [x] 核对工作区、目标授权、保护规则及评估结果。
- [x] 创建独立本地分支；记录开发设计和模块接口。
- [x] 接入正式 SymPy 数学规则与独立证据复核。
- [x] 接入完整输出的数学不等价证据及公开结果分类。
- [x] 改进 guarded-softmax/PV 与真实 GLM 5+5 的完整图比较。
- [x] 增加边界、反例、拒绝和证明篡改回归，更新相关实验/开发记录。
- [x] 独立代码审查并修复确认的问题。
- [x] 全量验证并审查最终 diff；记录剩余具体限制，发布后归档计划。
- [x] 提交、推送并核验远端 main；归档发布记录。

## 决策与进展

- 2026-10-15：已重新读取当前仓库，而非仅沿用上一轮结论；启动三个调查分工。用户已经授权最终分支推送与 PR，不另加确认关卡。
- 分支为 `feat/symbolic-math-backends`；正式锁定 `sympy==1.14.0`（依赖范围 `>=1.14,<1.15`）。设计写入 `algebraic-equivalence.md`，三个代理按独占模块实现 CAS/证据、输出负证据、guarded-softmax 有限性；主代理负责公共状态、入口、报告和整体验证。
- 新增 `inequivalent` 独立于 `different`；汇总表示至少一个所请求模式已有完整输出反例，不从求导反例外推前向结论。结构不匹配在 GLM 报告中仍算未完成验证，退出码为 2。
- 全量验证完成后，用户明确要求“提交到main上”；已核对本地和远端 main 仍为初始基线，发布方式改为直接 main，不再创建临时 PR。
- 完整 GLM 负例的反例需要合法 indexer 输入以及近似 top-k/动态读取/循环语义，首批实数多项式负证据不覆盖这些算子；不能沿用局部 QK 反例宣布整层成功。
- 完整 glm FP32 已实跑抽象比较：5 个正例 equivalent（2 结构、3 代数），5 个负例 unknown；CLI 按缺口退出 2。全过程未执行模型。small 与 BF16 签名由最终整层回归复核。
- CAS 与负证据模块交叉审查无已确认误证。对嵌套安全除法未提升外层 exp 总性的完整性缺口，新增独立 `exp_defined` 证据规则并补定义域拒绝与篡改回归。
- 有限性审查补查默认 NaN gather fill、窄化与 float8 无穷转换、复数 tanh 极点和内部 token。另发现既有 `_constant` 未拒绝非有限复数，导致经 abs 转实数后错误消项；已明确拒绝非有限复数常量，并修复复数 tanh 在 opaque、共同 JIT 和循环反馈中的部分性传播。独立复核实际反例均为 unknown。

## 验证记录

- 核心接口、原多项式和索引 IR：192 passed；另补求导不等价与前向结论分离后，functions 33 passed。
- CAS + 独立 proof + `-inf`：96 passed。负证据 + proof：75 passed；交叉审查另有禁止编译/eager 的真实 JAX 探针。
- 最终有限性 + 原 `-inf`：82 passed；独立复核共同 JIT/scan 隐藏部分性路径 15 passed。
- 代码冻结后全量 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline pytest`：**2095 passed，1 warning，944.74 秒**。唯一警告来自既有 SGLang-JAX Pallas fixture 的弃用接口，GLM-5.2 新核心及其比较回归无 Pallas。
- `uv sync --locked --offline`、`ruff check .`、`ruff format --check .`（169 个 Python 文件）、`python tests/inspect_jaxpr.py`、`python tools/check_docs.py --changed` 与 `git diff --check` 全部通过。核心保护文档及 `.codex`/`.agents` 无改动。
- GLM 回归覆盖 small/FP32、glm/FP32、glm/BF16，禁止后端编译/eager；每组 5 个正例全部通过，5 个负例 unknown。CLI JSON/jaxpr 导出及退出码 2 均已验证。

## 保留的能力边界

- CAS 接口已落实为每个 AlgebraStep 的 ScalarRewrite 见证，规则生成与独立复核分离。当前只实现白名单 exp/有条件除法；通用分式、更多超越恒等式和条件域推理仍是后续能力，不因化简失败而宣布不等价。
- 多项式负证据要求全部输出属于受支持的有限实数片段且没有区域 barriers，张量捕获常量、未解释函数、整数/复数等仍拒绝。搜索或复核超预算返回 unknown。
- GLM 五个完整负例仍未取得反例证据。近似 top-k、动态索引、循环和最终输出的组合必须从源语义适配；局部 QK/PV 差异不能替代。精确剩余工作已记录为 ALG-003/004。
- SymPy 是候选生成器，现有索引 IR 和语义适配仍属于证明基础；这不是任意 Python/JAX 程序数学等价的完备判定，也不承诺逐位浮点等价。

## 发布记录

按用户最新授权直接发布 main，实现提交为 [`2bcbc5e`](https://github.com/0xaskr/jax-reorder/commit/2bcbc5ed256b5b3d1871036ba5179bdc4d6f2760)。推送后已核对远端 main 与本地完整 SHA 一致；未创建 PR。本计划及索引的归档作为后续纯文档提交，不改变已验证实现。
