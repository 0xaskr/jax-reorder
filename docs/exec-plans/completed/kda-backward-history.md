# KDA backward mega kernel 历史与 AI 候选验证

2026-10-28 验收：最新 base + 三个历史版本 + 五个 AI 候选，六配置中 53 组
完成完整结构比较（17 等价、36 结构不同），106 次官方 CPU 解释器执行完成，
AI 数值预期 60/60 符合。完整局部语义/数学均停在原始未初始化 scratch 读取；
数值一致不能替代证明。详细机器数据按需读取[结果 JSON](validation/kda-backward-history-results.json)。

## 目标与完成条件

从相邻 `pallas-kernel` 仓库提取 KDA backward mega kernel，以核实后的远端
main 版本为基准，对比多个有实质变化的历史提交以及固定的 AI 正反例。
保留真实 kernel、完整输出、grid、refs 和 DMA；将来源、可复现命令、结果及
未完成证明的具体原因保存到本仓库。

## 范围与取舍

- 使用提交中的源码，不修改相邻仓库的工作区。
- 复用 `tests/upstream_tpu/sources.py` 的原文快照与哈希校验。
- 优先设备局部的非 CP 配置；数值验证、结构比较、语义投影和数学证明分别记录。
- 历史变更不预设等价；AI 候选先固定改写内容及预期，再执行比较。
- 不要求 TPU 实机，CPU tracing/interpreter 证据不称作 TPU 编译或执行。
- 本轮是测试与证据建设，不修改受保护的架构和设计文档。

## 固定测试资产

- [来源清单](../../../tests/fixtures/upstream_tpu/pallas_kda/manifest.json)：
  固定提交、每个依赖的版本、SHA-256、Git blob 和 Apache-2.0 许可证。
- [原始源码加载与 AI 候选](../../../tests/upstream_tpu/kda_cases.py)：
  AI 仅替换原 mega kernel 实际调用的计算依赖，避免复制整份 kernel。
- [独立参照与解释器](../../../tests/upstream_tpu/kda_oracle.py)、
  [回归](../../../tests/test_upstream_tpu_kda.py)、
  [可复跑报告入口](../../../tests/kda_history.py)。

| 来源 | 固定提交 | 完整结构比较 |
| --- | --- | --- |
| 最新 main | `a861cfb60717d74dff52ed9d508cc7eab1782c1b` | 自比较 6/6 等价；mega 文件最后修改为 `c500d1bb` |
| 2026-10-09 | `114f2aa10c9bafaca7f4975763a421c2c5d054c7` | 5/6 等价；BF16 gate 输出配置不同 |
| 2026-09-08 | `8de99b4db8a5b67e3971a635723d67d5d1a76aaf` | 6/6 结构不同 |
| 2026-07-30 | `6b69fe68754ea2892007285f7d5a158c3be4bcdf` | 5/5 结构不同；无 BF16 gate 输出参数 |

所有配置固定 `H=B=1`、`K=V=128`、chunk size 64、非零 initial state/dht；
完整检查 `dq, dk, dv, db, dg, dh0`。六组配置分别为单 chunk FP32、双 chunk
FP32、双 chunk BF16、双 chunk FP32 重算、双 chunk BF16 gate 输出，以及
三 chunk / DMA batch 2 的尾窗。每组两种种子（7、23）。

| AI 候选 | 比较前确定的数学预期 | 完整结构比较 | 独立数值观测 |
| --- | --- | --- | --- |
| `dav_reordered` | 独立 dAqk/dv0 计算换序，等价 | 6/6 等价 | 全部六输出与 base 逐元素相同 |
| `dav_scale_inside` | 将 dAqk 缩放移入矩阵乘左操作数，理想实数等价 | 6/6 不同 | 六输出在固定容差内一致，存在微小舍入差 |
| `dav_wrong_scale` | dAqk 缩放翻倍，不等价 | 6/6 不同 | `dq/dk/dg` 暴露差异 |
| `dv_wrong_transpose` | dv0 使用 A 而非 A 的转置，不等价 | 6/6 不同 | `dk/dv/db/dg/dh0` 暴露差异 |
| `cumsum_forward` | gate 梯度从前向后累加，不等价 | 6/6 不同 | `dg` 暴露差异 |

## 工作项

- [x] 核实最新远端提交，选定基准及历史源码/依赖。
- [x] 固化源码、来源与许可证，建立真实 mega kernel tracing。
- [x] 添加固定 AI 正反例，执行分层比较及独立数值验证。
- [x] 保存机器可读结果，添加回归，记录能力缺口。
- [x] 完成相关检查并归档计划。

## 决策与进展

- 2026-10-28：源仓库位于 `/home/askr/work/pallas-kernel`，工作区有无关 DSA
  修改；只读取提交。远端 main HEAD 为
  `a861cfb60717d74dff52ed9d508cc7eab1782c1b`，本地 origin/main 较旧，
  需要在隔离目录读取远端历史后再确认基准。
- 已在 `/tmp` 的隔离 Git 目录获取远端 main，确认该版本的 mega 源文件最后修改
  为 `c500d1bb`。固定 `114f2aa1`、`8de99b4d`、`6b69fe68` 三个历史版本；
  同版本依赖一起提取，33 个来源条目按 Git blob 复用 18 份原文对象。
- 采用各版共同接受的 128 宽 `Aqk/Akk`；保留原 launcher 的输入预处理、
  kernel、scratch、DMA、循环和六个公开输出。仅固定解释模式和 TPU v7
  配置；静态对齐工具的 `cdiv` 绑定到同公式的 `pl.cdiv`。
- AI 候选包括独立计算换序、缩放移入点积，以及错误缩放、错误转置、
  错误累加方向三个反例。标签在比较前由源码关系确定。
- 初次测试暴露出配置判断错误：9 月版本已经提供 `cast_dg_to_q_dtype`，
  仅 7 月版本没有。已改为读取真实签名判断，补入 9 月 BF16 gate 用例。
- 首轮结构比较被 9 个完全未引用的私有分配排列阻塞；现已修复，随后补齐
  `pad/exp2`、VMEM、Ref 索引链及受审核编译参数，见[修复记录](kda-comparison-fixes.md)。
  AI 换序版保留原 helper 的类型转换与字面量，仅改变独立计算的顺序。

## 验证结果

报告 schema 2 保存 checker HEAD、全部生产源码 SHA-256 和三个 KDA 脚本的
SHA-256。每个候选在隔离进程执行；阶段完成即保存结果，超时或异常不能覆盖
先前结果或被记为证明。本次六个配置的所有版本/源码哈希一致，无 worker 失败。

| 证据 | 实际结果 |
| --- | --- |
| 原始 Pallas tracing | 53 组成功；7 月版本不提供 BF16 gate 输出选项，该组合明确标为不可用 |
| 完整结构 | 53 组完成；17 `equivalent`、36 `different`，后者只表示图不同 |
| 设备局部语义 / 设备局部数学 | 各 53 组均为 `UnsupportedJaxprError`；停在 base 的未初始化 scratch 读取，无完整资源/数学证明 |
| 官方 TPU CPU interpreter | 106 次执行；六输出均有限，无竞态报告 |
| 固定 AI 候选 | 两正三负 × 六配置 × 两种子，60 次完整输出数值检查符合预期 |
| FP32 独立参照 | 四组 FP32 配置中 base 与两个正例通过；base 最大相对 L2 误差约 `3.14e-7` |
| BF16 独立参照 | 保存实际误差，不以 FP32 容差认证；base 最大相对 L2 误差约 `0.47%`，未证明误差界 |

独立参照使用 NumPy 逐 token 递推和手工反向导数，另与 JAX autodiff 核对六个
梯度。官方 TPU CPU interpreter 开启竞态检测、未初始化值填 NaN 和逻辑边界检查。
所有比较图保留 `interpret=False`；本轮没有 TPU 后端编译或实机执行。

历史版本在所测普通配置中与 base 的六输出逐元素相同。开启 BF16 gate 输出时，
`114f2aa1` 仅 `dg` 不同，最大绝对差 `6.103515625e-5`；`8de99b4d` 与 base
仍相同。源码差异将 `direct_fp32_dg` 改为仅在 `CP_ACTIVE` 时启用，
本轮非 CP 配置因此在 reverse cumsum 前恢复 BF16 ref-store/reload 舍入。
这是实际数值差异，不能直接推出理想实数语义不等价。

补充诊断：抽取真实 `compute_dav`，在小 tile、`scale=0.25` 下证明换序版结构
等价、缩放移入矩阵乘版理想实数等价，后者带浮点改写警告。此证据仅针对
helper，不能填补完整 kernel 的资源/数学缺口。

### 回归检查

全量主运行 5,035 项通过、1 项旧参数预期失败、2 项默认跳过；修正该预期并
补齐布局组合后，两个相关文件 136 项复测通过。按最终用例清单核对并去重，
**5,039 项通过、2 项跳过**，计数合并主运行与定向复测。原有 7 项失败已
修复，详细依据见[修复记录](kda-comparison-fixes.md#最终验证)。

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python -m tests.kda_history \
  --output /tmp/kda-history.json
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest -q
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest \
  tests/test_pallas_dynamic_views.py tests/test_pallas_kda_primitives.py -q
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run basedpyright
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python tests/inspect_jaxpr.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python tools/check_docs.py --changed
git diff --check
```

可用 `--config` 选择配置，或 `--graphs-only` / `--numerics-only` 选择证据阶段。

## 未解决问题

- 原始 mega 使用 `gate_dA_acc[:] = jnp.zeros_like(gate_dA_acc[:])` 等初始化
  写法，首次写入前保留一次无消费者的 Ref `get`。当前契约仍检查该读取，
  局部语义和数学均明确拒绝。没有修改固定 base，也没有删除读取来获得匹配；
  具体源码与现有拒绝回归见[修复记录](kda-comparison-fixes.md#保留的完整-kda-阻塞)。
- 本轮只覆盖 dense、CP inactive 的设备局部路径；packed/分段、实际 CP、
  fused gate 参数梯度、更大 head/batch 配置和 TPU 后端执行未覆盖。
