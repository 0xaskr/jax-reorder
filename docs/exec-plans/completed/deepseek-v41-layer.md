# DeepSeek-V4.1-Flash 真实 JAX 层的代数改写验证

## 目标与完成条件

从网上取得 DeepSeek-V4.1-Flash 的真实 JAX 层作为 base；保留固定版本的原始源码、许可证和来源校验。由 AI 手写代数变换与优化候选，用 jax-reorder 分别验证结构、代数及实际求导图，修复本次发现的问题。数值采样和性能测量独立报告，不以近似相同冒充证明。

必须使用指定模型的实际层，保留其完整计算与接口；不可用更早版本、玩具公式或自行重写的 base 替代。原始层大小可用受控输入配置运行，但同时核查模型真实参数配置及原始调用关系。优化需说明工作量变化并测量当前可用后端，不能凭公式断言加速。

## 当前来源

- 官方模型发布：[DeepSeek-V4.1-Flash](https://www.deepseek.com/news/deepseek-v4-1-flash/)。
- JAX 实现：[Inferact/tpu-megakernels PR #3](https://github.com/Inferact/tpu-megakernels/pull/3)，2026-10-14 查询为 OPEN。
- 固定提交：`d11defb7e5e2fdd8d355b2f6225c74d2a15fa1e1`，来源 fork `jawadaminGOOG/tpu-megakernels`。上游含 `deepseek_v41/reference.py`、模型配置、XLA decode 调用与 CPU 交叉验证。
- 选定完整 `hc_post` 残差混合层；原始模型配置 `H=4,D=5120`。以 `hc_mixes` 和 20 次 Sinkhorn 作为真实上下文补充验证；设计见[开发记录](validation/deepseek-v41-layer.md)。

## 工作项

- [x] 核查当前工作区、模型版本及公开 JAX 来源。
- [x] 固定源码、许可证、哈希及真实层调用关系；写入开发设计。
- [x] 保留原始 base，编写 AI 固定的正例、错误反例、独立数值参照及复现入口。
- [x] 用结构、代数、独立证明和分模式求导验证，复现并修复所有本轮发现的问题。
- [x] 完成正确性矩阵、CPU 性能与工作量证据，全量测试、静态检查和文档检查。
- [x] 逐项核对目标完成证据，审查差异并归档。

## 决策与进展

2026-10-14：保留前一轮尚未提交的 3 项审查修复，基线为 1802 passed。本目标取得模型与 JAX PR 的明确来源证据，并在读取外部代码前核实 PR 当前提交。

受保护文档维持只读；开发记录与本计划承担设计和验证记录，未提交、未推送。当前 JAX 仅有 CPU 设备，硬件为 Intel Core Ultra 7 155H。

已下载 6 个固定上游文件（含 Apache-2.0 许可证），完整原始定义由 AST 原样加载。AI 手写 `post_matmul`、`post_blocked` 和 3 个错误反例，补充 NumPy 前向/一阶公式与复现 runner。

已复现并修复：默认纯 JIT 的 VJP 残差组合（初始 3 个失败回归）、低精度转置 cast 的默认空网格描述（2 个失败回归）、共同非线性调用的结构复用及原始端口绑定、排序键重复遍历、分块 VJP 的安全零扩展（8 个失败回归）、单位分块的梯度索引条件。共新增 68 项测试，包含实际原始模型、独立证明、数值参照、篡改证据及拒绝路径。

H=3 的额外测试发现单位分块的梯度索引需在前缀指标约束下化简，已据实际规范项补充整数条件规则，两个模型回归及 23 个 padding/共同调用测试通过。最终 runner 已证明真实 D=5120、FP32/BF16、两候选的三模式与完整 20 次 Sinkhorn 上下文，错误候选均被拒绝。初步并行计时已弃用，最终数字在全部验证进程结束后独立重测。

完整 BF16 上下文最初虽证明成功，但耗时仍高。堆栈采样定位 `map_factors` 内部排序没有沿用局部缓存；已将缓存传入索引、域见证和前缀指标转换。修复后单次 BF16 完整上下文 VJP 为 13.59 秒、工作量 3942；相关 114 项测试通过（16.54 秒），包括非默认 cast memory kind 的拒绝。最终 runner 的两种 BF16 完整上下文三模式分别耗时 29.58 / 31.13 秒；该诊断计时与全量测试并行，不作为独立性能基准。

全量回归定位一个测试构造差异：原语清单的 FFT 用例手工创建证据时仍使用单侧 Lowerer，独立校验器则按当前双侧共同调用规则重建表达式。该测试改为按生产入口建立双侧 `CommonCalls`，继续绕过结构直通，保留全部展开、规范项相等及独立证明断言；拒绝清单不变。

## 验证结果

命令均以 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run --no-sync --offline` 运行，结果如下：

| 检查 | 结果 |
| --- | --- |
| `pytest -q --tb=short` | 1869 passed、1 failed、1 warning，555.56 秒；唯一失败为上述旧 FFT 手工构造测试 |
| 修正构造后的 `pytest -q tests/test_algebra_structural.py --tb=short` | 371 passed，8.77 秒；保留全部原有断言 |
| 全量结束后 `pytest --lf --last-failed-no-failures=none -q --tb=short` | 1 passed，0.95 秒；无遗留失败，1870 个用例均有通过记录 |
| `python -m tests.deepseek_v41.runner --json /tmp/jax-reorder-deepseek-v41-final.json` | 独立层与完整 mHC 上下文 24 组正例模式通过；18 组错误候选模式拒绝；NumPy 前向/JVP/VJP 参照通过 |
| `ruff check .` / `ruff format --check .` | 通过，142 个 Python 文件格式一致 |
| tracing、BFS、cond、algebraic 示例 | 全部通过 |
| `tests/adapter_inventory.py` / `tests/lax_inventory.py` | 锁定环境清单一致 |
| `tests/collective_jaxpr.py --devices 2` / `--devices 4` | 各 22 个通信组合通过 |
| `tools/check_docs.py` / `--changed`、`git diff --check` | 通过；新增文件与归档链接纳入检查 |

唯一 warning 为保留上游的 Pallas reciprocal 弃用提示。本次算法及模型测试没有遗留失败。

独立性能阶段在所有上述进程退出后调用 `tests.deepseek_v41.runner.benchmark(profile, dtype, 200)`，形状依次为 `(1,1,4,5120)`、`(8,1,4,5120)`、`(1,128,4,5120)`，dtype 为 FP32/BF16；写回同一临时 JSON 的 `benchmarks`。与公开 `--benchmark --repeats 200` 入口使用相同函数。

matmul 的单 token 中位加速比为 FP32 1.053×、BF16 1.109×，其他形状更慢；blocked 全部更慢。base 的临时缓冲为 0，两候选均增加临时空间，不能宣称普遍加速或实际省内存。完整工作量、误差、延迟和内存表见[开发记录](validation/deepseek-v41-layer.md)。JSON 原始数据位于 `/tmp`，未纳入仓库。

## 未解决问题

本轮发现的问题均已处理。范围限定所选完整 mHC 层及明确追踪签名；GPU/TPU 性能未验证。其他既有通用边界继续由技术债记录维护，已更新 ALG-002 的实际支持范围。受保护文档没有改动，已有未提交修复保留。

## 后续交付

用户随后授权提交并推送远端，并明确验证脚本只需测试 DeepSeek-V4.1-Flash。因此增加 [verify_deepseek_v41.sh](../../../tools/verify_deepseek_v41.sh)，只调用真实层验证入口，默认输出 JSON；性能测试通过 `--benchmark` 可选开启。

交付前相关回归 518 passed（195.04 秒）。最终精简脚本从 `/tmp` 执行，命令为 `UV_OFFLINE=1 bash /home/askr/github/jax-reorder/tools/verify_deepseek_v41.sh --trials 1 --json /tmp/jax-reorder-deepseek-v41-script.json`，全部三模式、反例、独立数值和完整上下文检查通过。Bash 语法、帮助入口、Ruff、格式、tracing 和文档链接检查通过。

首次暂存检查提示完整上游快照 `xla_decode.py.txt:1626` 带文件末尾空行；为保持下载文件及 SHA-256 不变，保留原样，其余暂存文件通过 `git diff --cached --check`。提交范围不包含受保护文档、运行报告或环境文件。
