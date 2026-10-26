# Pallas 覆盖补齐与代数接入

创建及完成：2026-10-19。状态：本轮覆盖补齐、公共代数接入及 60 秒默认超时调整完成。
用户指定先覆盖、再代数，后续明确要求将默认 SymPy 时限从 10 秒改为 60 秒。

## 目标与完成条件

先盘点锁定 JAX/jaxlib 0.11.2 的 TPU Pallas API、原语与参数族，建立可执行覆盖清单，
补齐遗漏的官方构造、组合和明确拒绝路径。继续沿用每侧一个 pallas_call、同 mesh、
前向结构比较范围；CPU tracing/interpret 与实际 TPU、多主机证据分别记录。

覆盖检查完成后，将已有中间语义层的全局值投影接入公共代数流程，复用可信 SymPy，
验证跨 block/grid、计算表达式改写及 Pallas/JAX 输出关系；没有完整投影的 effect/区域
不能被忽略，也不能以局部公式差异推断整个程序不等价。

## 工作项

- [x] 读取当前契约、权限与上一轮验证，保留全部现有改动。
- [x] 建立官方 API/原语覆盖盘点，区分实际追踪通过、源端拒绝和范围外 API。
- [x] 修复覆盖盘点发现的结构适配缺口，补充参数、资源和嵌套组合正反例。
- [x] CPU 数值复核；检查硬件可用性并记录实际验证边界（仅 CPU，通信专项纳入最终回归）。
- [x] 覆盖专项通过后，完成中间层到 SymPy 的公共代数桥接及完整输出证据。
- [x] 跨分块、代数正反例、Pallas/JAX、effects 与不支持转换的回归。
- [x] 按用户追加要求，将默认 SymPy worker 超时从 10 秒改为 60 秒，复测既有五个超时用例及超时回收。
- [x] Ruff、格式、类型、相关回归、示例、文档和最终差异审查；归档执行记录。

## 设计与取舍

- 清单由实际源码和真实 jaxpr 产生；原语身份准入不计作执行覆盖，不用原图自比较冒充改写验证。
- 先修真实缺口，不改默认搜索预算掩盖失败；必要的算法修复与反例一起记录。
- 保持 core → algebra → functions 单向依赖，源格式解释仍属于 adapter。
- 不修改受保护文档及权限配置，不提交或推送。

## 进展与验证

覆盖阶段：`test_pallas_coverage.py` 66 passed；相关核心/结构回归 101 passed；
basedpyright 0 errors。119 个公开 callable 导出逐一分类，其中 73 个有真实追踪用例，
其余是声明、工具、host helper、范围外 MPMD 与源端拒绝（不是 119 个执行通过）。
CPU interpreter 对可执行候选进行正反数值核验；Ref transpose discharge 和 Indirect
数组索引的上游拒绝分别固定为测试。设备检查仅一个 CPU，没有 TPU 硬件证据。
开发设计入口：[Pallas/JAX](../../design-docs/current/pallas-jax-equivalence.md)。

后续补入官方 pipeline lowering 的 `pipeline_body` 与未知 HiPrim 拒绝，覆盖变为 40 类。
最终覆盖/核心/接口回归 153 passed；公共代数桥接专项 26 passed；官方 TPU interpreter
通信模拟 15 passed；包分层与真实 GLM5/Wan/EPMoE 后端边界 10 passed（1 条上游弃用警告）。
Ruff、格式、basedpyright、lax/adapter inventory、tracing/BFS 示例及文档检查通过。
扩展常规回归按原 10 秒默认值完成：1916 passed、5 failed、391 deselected，2131.63 秒。
五个失败均为既有超时，已在修改前版本复现，单列 ALG-005；这是旧时限下的历史结果。
用户随后明确要求默认时限调整至 60 秒，保留调用方显式覆盖与超时 kill/回收机制。
三个公共入口与独立 worker 默认值均已核对为 60.0；原五个失败用例按新默认值全部通过。
新默认值专项（五个历史失败 + `test_trusted_sympy.py`）**49 passed，144.92 秒**，
包括显式短超时、预算耗尽、worker 回收及下一次查询恢复。没有重跑整个 1916 项已通过
批次；最终修改由本节数学/核心及超时专项补充验证，不将分阶段计数简单相加。

修复取舍：小接口先按返回类别缩小搜索；大接口若仍有歧义，保留原阶乘预算预检。
因此默认 pipeline 能比较，GLM5 absorbed 四配置仍快速报告预算不足，没有用增加预算
或原图自比较捷径宣称支持。Pallas 值投影的全局读拥有独立 Source 身份，不能因来源
数组相同而合并不同坐标的有限性事实。

代数收尾补充：拆分 DMA 的分段投影与 JAX 拼接先暴露定义域坐标范围不一致，
改为在合法输出坐标内比较条件；输入值的部分性条件继续保留。扩展回归随后发现
SymPy 把条件化简成布尔 Piecewise，不能直接作为 And 操作数，现复用既有 ITE
转换后再施加边界。相关部分函数正反例 11 passed；新增普通拼接、输入定义域
拒绝和分片全局组装拒绝均已验证。最终数学组合 **82 passed，197.97 秒**：
`test_pallas_algebra.py`（28 项）、`test_algebra_array_conversion.py`、
`test_algebra_padding.py`、`test_algebra_sympy.py`。计数与前述专项存在重叠。
最终深度/规模 6 passed（31.24 秒），覆盖深层区域、长依赖和高扇出。

## 未解决条件

真实 TPU 或多主机执行需要实际可用设备，不能由 CPU 逻辑设备模拟替代。
代数投影暂限一维静态 Blocked 和已验证的本地 DMA；多维、一般控制流、全局分片
组装、远端 DMA/普通信号量及非默认执行策略继续明确拒绝。更广目标仍由
[语义层计划](../active/pallas-semantic-layer.md)维护；本轮不宣称任意合法 Pallas 全覆盖。
