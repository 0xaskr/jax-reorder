# 远端 DMA：协议依赖到全局存储版本

创建/完成：2026-10-19。状态：已完成（下述有限协议范围）。用户要求先完成远端通信语义层整合，并授权子代理调研 GitHub 上 MLIR/TVM 的策略。

## 目标与完成条件

- 将已有有限远端协议分析接入 core 规范化：证明初始化、源版本稳定、只读输入、调用内生命周期、输出覆盖与跨设备值对应。
- 支持单 concrete mesh 轴、每设备一个完整向量块、静态设备分支与路由、零初始化 DMA 信号量单代全量等待；按真实 shard_map 端口映射组装全局输出。
- 请求—回复及三设备直接/中继能够比较；错误路由保留值差异；竞争、等待环、未初始化、漏写及错误复制声明明确拒绝。
- 公共数学入口只接收完成全局组装的纯值图。SymPy 不推断协议或修复缺失依赖。

## 设计与取舍

设计见 [Pallas 设计 §9.10](../../development-docs/pallas/history/stage-10.md)。复用现有协议图与区间 Storage；不引入 MLIR/TVM 运行时依赖。
先证明所有冲突区间在所有合法调度下有序，再沿拓扑顺序传播版本。start 保存逻辑源快照，send_complete 结束源保护，receive_complete 发布目标版本。wait 消费计数，不配对单个传输。
设备实例可以有限展开，元素只按区间边界处理。源 Ref 别名由区域实参绑定解析，外部 alias 和动态分配仍拒绝。

## 工作项

- [x] 调研 MLIR / AIR / IREE / TVM，区分编译变换与验证规则。
- [x] core 远端版本传播、分片全局组装与可重放证据。
- [x] 公共投影接入及平台假设传递。
- [x] 正例、反例、证据篡改、规模预算及官方 CPU interpreter 对照。
- [x] 回归、审查差异、记录边界并归档。

## 决策与进展

- 不按 AIR 编译 pass 的方式给竞争补依赖；仅消费源程序已有的顺序和完成依赖。
- 单次拓扑遍历的版本相同不能代替全调度冲突检查。源写回旧值也不能解除在途版本保护。
- 保留旧的完整结构比较准入；更严格的初始化/进展条件只属于远端值规范化。
- 实现在 `core/remote_memory.py`，复用 protocol 的全调度依赖证明及 Storage 的边界区间；
  设备解释器在状态操作前暂停，由事件推进。source start 保存逻辑快照，receive_complete 才发布新版本。
- `RewriteEvidence.remote` 保留协议、传输、区间版本和全局端口组装。直接/中继的图不同但终端纯值相同；
  公共数学入口同样保存证据与远端进展假设，不引入 MLIR/TVM 依赖。
- 初次测试发现复制分片既可为 `P()` 也可为 `P(None)`，两者都按一维复制解释。
  普通 JAX 参考必须在相同 mesh 下追踪，保持完整输出抽象契约；没有放宽契约检查。
- 额外坐标测试发现浮点 iota 不在现有投影规则内；改用受支持的整数 iota 转浮点后，core 投影与
  CPU 参考一致，CAS 仍报条件未证明。保留拒绝回归，记录为 ALG-004 的具体边界。

## 验证结果

环境：Python 3.12、锁定 JAX/jaxlib 0.11.2。下列命令使用 `.venv/bin/`，与 `uv run` 同一环境。

- `pytest tests/test_pallas_remote_values.py -m 'not pallas_communication' -q --tb=short`：
  **31 passed，6 deselected，47.37s**。正反例、证据篡改、多输入输出、静态条件绑定、复制契约、
  大形状预算、公共完整输出数学比较及明确的条件未判定路径。
- `pytest tests/test_pallas_remote.py tests/test_pallas_remote_values.py -m pallas_communication -q --tb=short`：
  **15 passed，42 deselected，79.31s**。双设备交换、请求—回复，三设备直接/中继，累计切片交换、
  scratch 源复用及普通切片坐标，各跑 eager/on_wait；另有静态等待环的隔离超时观察。
  正常用例同时用独立的小数组值 DAG 求值器核对全局投影，再与官方 interpreter/race detector 对照。
- `pytest tests/test_pallas_dma.py tests/test_pallas_semantic_slice.py tests/test_pallas_sharding.py tests/test_package_layout.py -m 'not pallas_communication' -q --tb=short`：
  最终 **120 passed，6 deselected，19.98s**，包括局部坐标域没有被误当作全局域的既有比较路径。
- 较早的 DMA/区间/sharding/semantic/remote/protocol/storage/package 组合：**171 passed，25 deselected，19.07s**。
- `pytest tests/test_pallas_algebra.py -q --tb=short`：**28 passed，167.77s**。本地投影、跨分块、JAX 桥接与拒绝路径回归。
- Ruff、格式检查、basedpyright、adapter inventory、tracing 示例、文档链接及 `git diff --check` 通过。
  未修改受保护文档、权限配置或依赖；未提交/推送。未宣称全仓库测试或 TPU 硬件验证通过。

CPU interpreter 是独立数值及竞争对照，不等于真实 TPU 验证或一般并发证明。

## 未解决问题

多轴/多程序、动态控制流、信号量跨代复用与部分等待、混合本地/远端协议、一般 alias 不在本规则范围。
普通信号量、一般复制性代数证明、浮点 iota 及上述分段转换条件也未扩展；详见
[PALLAS-001 / ALG-004](../tech-debt-tracker.md)。
