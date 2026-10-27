# GitHub 前十 TPU JAX/Pallas 仓库验证

创建与完成：2026-10-27。状态：本轮所选范围验证完成；剩余能力继续在待处理清单跟踪。

## 目标与完成条件

按用户确认的口径，以 GitHub Stars 排序，筛选确实包含 TPU JAX/Pallas 实现的开源仓库，
固定检索快照与前十名单。逐库提取真实层、子模块或函数，配合 AI 固定生成的正反候选，
验证 jax-reorder；分析失败、修复可修问题，对剩余问题记录具体触发条件和暂不支持的原因。

每个仓库至少有一组实际执行的原始源码与正反候选；同时含 JAX/Pallas 的仓库分别选取入口。
完成指这十个仓库的所选范围均有可复现结果，不表示全仓库、完整模型或 TPU 实机已验证。
仅调研、下载、基准自比较或源码追踪失败不计为成功用例；相关阻塞单独列出。

## 范围与证据

- 排名以记录的公开检索、Stars 观察时间和资格证据为准；保留排除项及理由。
- 固定源码提交、文件、函数/层、许可证、内容摘要及必要依赖；不得用简化重写替代上游基准。
- AI 候选与预期先固定，再调用比较器；记录结构、数学结论与独立数值证据。
- Pallas 源图保留 kernel、grid、Ref、存储及通信语义；CPU interpreter 与实际源图分开验证。
- 失败区分上游加载/追踪、比较器缺陷、语义尚未支持、预算/超时及执行后端限制。
- 比较器修复需要最小回归、反例与拒绝路径；不能以忽略语义获得通过。
- 本目标明确要求实际测试，取代前一阶段“直接提交、不运行测试”的限制。

## 工作项

- [x] 固定 Stars 候选、资格依据与前十名单。
- [x] 建立上游源码快照与可复用的隔离加载/结果记录入口。
- [x] 分批编写 JAX/Pallas 正反候选，实际运行并分类结果。
- [x] 分析失败，修复可修问题并重跑相关回归；记录剩余边界。
- [x] 汇总十库覆盖矩阵、固定版本、命令、结果与未验证范围。

## 进展

- 已完成示例资产并入 `tests/`、默认安装 `dev` 和 `test` 依赖组，以及真实代码/AI 候选规范。
- 已完成最终十库及两库额外覆盖的真实源码、固定 AI 候选与整合回归；五项修复完成，四类剩余能力单独跟踪。

## 验证结果

最终整合命令：

```bash
JAX_REORDER_RUN_DIFFUSERS_SNR_BATCH=1 uv run pytest tests/test_upstream_tpu_*.py tests/test_pallas_algebra.py -q
```

**360 passed / 570.95 秒，无失败、无跳过**。其中最终入选十库 141 项、额外 AlphaFold 3/Trax
36 项、Pallas 代数回归 183 项。两项默认 60 秒的 SNR 批量超时已显式重跑并符合预期。
通过数包含来源检查、独立数值验证以及准确的未决/拒绝/超时断言，不能当作数学证明数量。

| 验证文件标识 | 通过项数 |
| --- | --- |
| ai_engineering | 7 |
| keras | 36 |
| google_research | 11 |
| jax | 12 |
| diffusers | 16 |
| mujoco | 12 |
| alphafold | 29 |
| dalle_mini | 8 |
| vision_transformer | 4 |
| cleanrl | 6 |
| alphafold3（额外） | 29 |
| trax（额外） | 7 |
| test_pallas_algebra | 183 |

此前定位与修复过程：

- 迁移后全量收集 4,634 项；既有 AI/cond/lax/Pallas 上游定向批次 423 passed（34.76 秒）。
- 首批新 DALL·E Mini RMSNorm 与 ViT MLP：8 passed、4 failed；4 个失败均定位到缺失 `rsqrt` 数学转换，结构重排与独立数值检查通过。Keras BatchNorm 独立复现同一缺口。

## 排名快照与逐库覆盖

观察日期：2026-10-27。[机器可读排名](../../../tests/fixtures/upstream_tpu/ranking.json)
保存检索式、返回仓库和 Stars、资格源码链接、排除理由与审查深度。
纳入归档仓库、仍留在源码树中的旧实现，以及含真实计算代码的教学仓库；
外部项目链接、纯清单、数据展示工具、仅有 TPU 章节标题均不算本仓的计算实现。
未建立 TPU 证据的通用 JAX 代码不因可移植性自动入选。

这是公开检索和逐项核实形成的可复现快照，不宣称穷尽 GitHub 的未索引内容。
较高 Stars 的资源列表及明显无关框架保留元数据初筛标记，不能把初筛当全源码审计。
Stars 在查询期间可能变化；固定测试提交独立于排行榜观察时间。JAX 选择与本地运行时一致的
0.11.2 源码；Keras 固定下载时的提交，其默认分支在调查期间已前进。
所有 43 份源文件/许可证/资格材料另与固定提交的上游原始内容逐项核对，SHA-256 全部一致。

表内“正/负”是预先定义的语义标签。结构不同不等于数学不等价；数值反例、未决、拒绝
均单独列出。每行测试入口为 `tests/test_upstream_tpu_<标识>.py`，完整提交和许可证在对应 manifest。

| 排名 | 仓库 / Stars | 真实计算范围 | 比较与独立验证结论 |
| --- | --- | --- | --- |
| 1 | [rohitg00/ai-engineering-from-scratch](../../../tests/fixtures/upstream_tpu/ai_engineering/manifest.json) / 74,254 | 完整 MNIST `forward`、`init_params`；784→256→128→10 | 换序结构等价、bias 拆分代数等价；错 bias 结构不同且有数值反例，数学因 common JIT 定义条件未决 |
| 2 | [keras-team/keras](../../../tests/fixtures/upstream_tpu/keras/manifest.json) / 64,390 | JAX BatchNorm、attention、真实 TPU Splash wrapper | BatchNorm 正负与 attention 正例通过；attention 部分负例未决；Splash 结构/解释器通过，函数数学输出组合尚未支持 |
| 3 | [google-research/google-research](../../../tests/fixtures/upstream_tpu/google_research/manifest.json) / 39,037 | IMP FAVOR+ 特征变换；ALX Gramian | FAVOR+ 正负通过；ALX 两个 CPU 设备实际 psum/pmean 有独立数值反例，数学转换拒绝 effectful call |
| 4 | [jax-ml/jax](../../../tests/fixtures/upstream_tpu/jax/manifest.json) / 36,503 | 原 `squareplus`；原 TPU matmul launcher + scratch kernel | JAX 正负通过；Pallas 两种比较范围正负通过，漏初始化拒绝；官方 CPU TPU 解释器及 NumPy 通过 |
| 5 | [huggingface/diffusers](../../../tests/fixtures/upstream_tpu/diffusers/manifest.json) / 34,816 | 保留的旧版 TPU Flax ControlNet `compute_snr`、`l2` | l2 正负通过；SNR 换序结构等价，scalar 数学未决，batch 数学默认 60 秒超时；数值正负已独立验证 |
| 6 | [google-deepmind/mujoco](../../../tests/fixtures/upstream_tpu/mujoco/manifest.json) / 15,905 | MJX-JAX 完整 `quat_mul`，原生和外层 vmap | 换序结构等价、代数重组等价、逆序乘法不等价；独立 Hamilton 矩阵及非交换基准通过 |
| 7 | [google-deepmind/alphafold](../../../tests/fixtures/upstream_tpu/alphafold/manifest.json) / 14,911 | 原 `mask_mean` | 负轴/全轴/广播/drop-mask-channel/零 mask 覆盖；结构与代数正例、错误广播及错误 eps 负例通过 |
| 8 | [borisdayma/dalle-mini](../../../tests/fixtures/upstream_tpu/dalle_mini/manifest.json) / 14,717 | 原 Flax `RMSNorm` | 结构换序、代数重结合与漏 scale 负例通过；两种 shape、双种子 NumPy 通过 |
| 9 | [google-research/vision_transformer](../../../tests/fixtures/upstream_tpu/vision_transformer/manifest.json) / 12,762 | 原 Flax `MlpBlock` | Dense-GELU-Dense 原计算、换序/bias 拆分正例、错误 bias 负例通过；NumPy tanh-GELU 参照通过 |
| 10 | [vwxyzjn/cleanrl](../../../tests/fixtures/upstream_tpu/cleanrl/manifest.json) / 10,608 | 原 DDPG `Actor`，256 宽隐藏层 | 换序与 bias 拆分正例通过；错误 action bias 有结构和数值反例，数学 common JIT 定义条件未决 |

额外覆盖（不计入最终前十）：AlphaFold 3 的原 `mask_mean`（与 AF2 的 epsilon 规则不同），
以及 Trax 原 `LayerNorm.forward`。两者均完成真实函数正反例；分别保留自己的来源与测试。
最终名单修正发生在高星候选源码复核阶段，额外成果保留，不用它们替代入选仓库。

## 本轮剩余缺口

这些是尚未完成的能力，**不是选择排除的功能**；均保留固定候选和实际异常回归。
遇到未建立的语义继续明确失败；未决或拒绝断言通过不计为数学证明成功。

| 跟踪 | 触发条件与定位 | 后续需要补什么 |
| --- | --- | --- |
| UPSTREAM-001 | Diffusers SNR 动态 timestep：scalar `dynamic_slice` 的定义条件未决，batch `gather` 常量表展开为 Piecewise 后 60 秒超时；`algebra/symbolic.py` / `sympy_ops.py` | 索引范围与分段条件证明、避免指数级展开；不能靠扩大超时或数值抽样宣称证明完成 |
| UPSTREAM-002 | CleanRL、AI Engineering 共享 ReLU JIT 输出成为 `call_*` 与 `defined_call_*`，输出残差可非零但无法建立原调用域；Keras attention 负例也有 CAS/条件未决 | 给共同纯调用提供可复核的定义域/精确求值模型，保留全部输出条件 |
| UPSTREAM-003 | Google Research ALX `shard_map` + `psum`：`algebra/adapter/calls.py` 拒绝 ordered/effectful call；普通 JAX 数学区域还没有 shard_map 语义 | 绑定 mesh、参与者、轴和归约语义，不能直接移除 effects guard |
| UPSTREAM-004 | Keras Splash 四个调用输出经 broadcast/选择主输出后，`core/normalization.py` 报 `outer postprocessing needs output composition` | 组合外层值运算与输出来源，保留未返回资源/副作用检查 |

另外：Keras 错误 V 的局部入口改变真实输入端口绑定，当前明确报契约不一致；
这不是数学不等价结果。squeeze 与一般 reshape 的 div/mod 坐标化简仍可能未决。
本轮没有 TPU 硬件；CPU 解释器的 race/OOB 检查不是 TPU AOT、实机精度或性能验收。

## 已确认修复设计

### rsqrt 数学转换

真实归一化层使用 JAX `rsqrt`；当前逐点映射仅有 `sqrt`，数学比较因此拒绝。
按现有理想实数契约，将默认精度的 `rsqrt(a)` 映射为 `a**(-1/2)`，
同时保留原始严格定义条件 `a > 0`。不能仅沿用 `sqrt` 的 `a >= 0`，也不能因后续
乘零或相减消项而丢弃该条件。非默认 accuracy 继续按已有规则拒绝，复数不扩展。
验证覆盖正域的等价改写、非等价表达式、零/负输入域的定义条件与部分函数消项，
并重跑触发问题的真实层。

### Pallas 数学投影中的 squeeze

Keras 的真实 Splash wrapper 经 `vmap` 产生 `squeeze`；完整结构比较与官方 TPU
解释器执行均通过，但函数和设备局部数学投影尚未准入该原语。
只接受真实 `lax.squeeze_p`，完整校验 `dimensions` 参数：必须是无重复、在输入秩范围内的
整数元组，且只能移除长度为 1 的轴。输出 shape 必须恰好删除所列轴，dtype、weak type
和其余标量元数据保持一致。被移除的输入坐标固定为 0，其余坐标按保留轴顺序映射；
保留原操作数表达式、来源和部分函数条件，不通过丢弃计算获得匹配。
验证包括 Pallas kernel 内与外包装、函数与设备局部入口、不同输入来源和错误数值反例、
非法维度/输出 shape/dtype 的拒绝，以及 squeeze 后消项仍保留原定义条件。
重跑真实 Splash 后若出现下一项未支持语义，保留准确失败位置，不替换原 kernel。

### Pallas 调试 metadata

补齐 squeeze 后，原 Splash 停在 `pallas_call.metadata`。锁定 JAX 0.11.2 的
`pallas_call.py` 将该参数定义为调试/分析用途，传入时固定为 `FrozenDict`；
`mosaic/pallas_call_registration.py` 原样转交，`tpu_custom_call.py` 仅将其 JSON
写入 `mhlo.frontend_attributes.kernel_metadata`，未用来选择 kernel 数值运算。
实际 Splash 字段是 `xprof_metadata`，值为完整 block 配置 JSON 字符串。

仅准入精确类型的字符串键到字符串值 `FrozenDict`，保留原键值，不解析或删减 JSON，
未知对象不转为 repr；`None` 和空字典分别编码。字段进入 `PallasCallView` 的结构契约，
以及每侧规范化证据和复核比较。它不成为数组值的相等条件：不同 metadata 必须结构不同，
数学比较可证明输出相同，但保留“锁定后端调试属性不改变 kernel 值”的前提及两侧来源字段。
验证包括合法字段、None/空差异、错误类型、source view 与证据篡改拒绝，再推进真实 Splash。
锁定 TPU lowering 会自动添加 `mesh_axes`，与用户同名键冲突时拒绝；在完整命名 mesh
上下文准入前，显式提供该保留键也明确拒绝，不能把它当作无条件无害的描述字段。

### Pallas scalar-prefetch 的 None 占位

metadata 专项初步 20 项通过后，真实 Splash 到达 index-map 输入布局检查。
其三维 grid 和两个预取 Ref 共五个叶子，实际 `index_map_tree` 还保留一个无叶子的
`None` 位置：`((*, *, *, *, *, None), {})`。锁定 JAX `core.py` 将 scalar-prefetch
原始参数树保留在此字段，kernel 与 index-map jaxpr 则接收展平后的两个 Ref；旧适配器
要求参数恰好为纯叶子元组，因而误拒这个已展平的合法输入。

仅增加位置元组中的 None 占位：根节点必须是 `(args, kwargs)`，kwargs 为空字典，
args 是普通元组，前 grid-rank 个位置必须全是叶子，后续仅允许叶子或 None；有效叶数
必须与 `index_map_avals`、`num_index_operands` 及每个 region 输入数完全一致。
不准入嵌套容器或自定义 pytree。完整 leaf/None 布局进入 call 结构字段和每侧规范化
证据，并参与证据复核；按已追踪 region 的展平输入顺序计算，不删除原始槽位事实。
验证合法占位、占位前移/缺失、错误叶数、kwargs/嵌套树拒绝与证据篡改拒绝。

### 不同定义域中的共同合法反例

AlphaFold 2/3 的真实 `mask_mean` 暴露数学负例完成率缺口：修改广播系数或混淆
`sum(mask) + eps` 与 `maximum(sum(mask), eps)` 会改变原定义域；当前 worker 在
无法证明两侧定义域相同时直接未决，尚未尝试共同合法输入上的精确输出反例。
两库首轮 58 项中 55 通过、3 项因此失败，独立 NumPy 及结构反例已经通过。
现有 `exact_witness` 在保留两侧全部输出定义域后，已能用全一输入分别证明
`24/25 != 8/3`、`24/25 != 1`、`1 != 24/25`。

仅在定义域比较尚未证明时，复用现有 `witness_at` 搜索有界输出坐标上的反例：
小输出沿用最多 64 个坐标，大输出沿用最多 16 个端点/中点见证坐标；预算与
worker 超时不变。每个候选必须通过两侧全部输出的量化定义域、该点读取合法性、
输入 dtype 可表示性以及精确值不等检查，才返回数学不等价。没有合法反例时
继续报告原条件未决；不能从定义域不同直接推出不等价，也不能以样本相同判等。

最小回归覆盖共同合法输入上的负例、无共同定义域、相同值但未证明定义域一致、
其他输出或同输出其他坐标的定义条件阻止反例；随后重跑真实 AlphaFold 2/3、
trusted SymPy、部分函数定义域和有限性相关测试。CPU 数值证据与数学结论分开。

本项验证（2026-10-27）：

- 最小共同域回归修复前 `3 failed, 3 passed`；修复后连既有消项拒绝路径 `8 passed`。
  另补共同域非空、但其他输出限制排除全部不等值输入的反例，`1 passed`。
- `.venv/bin/python -m pytest tests/test_upstream_tpu_alphafold.py tests/test_upstream_tpu_alphafold3.py -q --tb=short`：
  `58 passed`（13.65 秒）。两库原始完整 `mask_mean`、换序及广播改写、错误广播系数、
  加 eps/maximum 混淆负例，以及零 mask、AF3 小 mask、默认 eps、轴和输出形状均有证据。
- `.venv/bin/python -m pytest tests/test_trusted_sympy.py tests/test_algebra_sympy.py tests/test_algebra_finiteness.py tests/test_algebra_sum_domains.py tests/test_upstream_tpu_diffusers.py -q --tb=short`：
  `285 passed, 2 skipped`（172.85 秒）；2 项为显式 opt-in 的 Diffusers 批量 SNR 超时回归。
  该批次与前述最小回归重叠，不相加为新的独立测试总数。
- `trusted.py` 的 basedpyright 检查为 0 errors / 0 warnings；本项相关源码和测试的 Ruff、格式与 diff 检查通过。

Diffusers 遗留 ControlNet 训练脚本的完整 `main.compute_snr` 和 `main.train_step.l2`
通过固定提交、SHA-256/Git blob 核验加载；闭包累计 alpha 数组显式提供，未执行或替换整模型。
其定向结果为 `14 passed, 2 skipped`（8.72 秒）：l2 的结构正例、代数正例和错误 mean
负例均完成，SNR 的换序结构比较和独立 NumPy 正反例完成。动态标量 SNR 正例仍为
`condition_undecided`、错倍率负例仍为 `cas_undecided`；动态批量 `(4,)` 的两个数学查询
已分别实测默认 60 秒超时（60.216/60.224 秒），不能算数学等价/不等价结论。
批量回归用 `JAX_REORDER_RUN_DIFFUSERS_SNR_BATCH=1` 显式启用，默认不重复等待。
共享源码加载器兼容回归（Keras 来源、DALL·E Mini、ViT）另 `13 passed`（14.28 秒）。

## Keras、Trax 与 Pallas 准入验证记录

- Keras 固定提交 `cfebb2af8a202fcb31b8b86766f44eabe6295a7a`，原 `batch_normalization`、
  `_apply_masks`、`_dot_product_attention_core`、`wrap_flash_attention` 完整定义保持原样。
  BatchNorm 的结构重排、代数因式改写与错误 offset 符号均获预期结论；attention 的
  direct 与 transpose/matmul 正例通过。错误 V 与错误 scale 有结构差异和独立数值证据，
  数学路径分别仍为 `SymPy did not establish output identity` 与
  `original conditions are not established`，明确记录 undecided，不计数学反例证明。
- 实际 Splash 原 wrapper 和 AI wrapper 都调用锁定 JAX 的真实 TPU kernel；结构正例
  匹配、错误 V 结构不匹配。官方 TPU interpreter 在 CPU 上运行，开启 race detection、
  浮点计算、NaN 初始化和逻辑边界检查；原实现与独立 NumPy attention 最大绝对误差
  约 `8.18e-8`，AI 正例通过、错误 V 有超过 `0.1` 的数值差异。这不是 TPU 实机证据。
  原生 CPU lowering 仍按上游要求拒绝 native TPU kernel。
- squeeze、metadata、None 布局补齐后，真实 Splash 的函数数学路径停在
  `outer postprocessing needs output composition`：pallas_call 有四个输出，外层通过
  broadcast 恢复 batch 轴并只返回主输出，当前 bound-call 规范化要求输出直接逐项返回。
  设备局部数学路径针对错误 V 候选报 `local inputs, outputs or axis bindings differ`：
  候选将 V 绑定换成 K，不能把局部输入契约不一致当作数学不等价结论。此次不扩展输出组合。
- Trax 固定提交 `31022d6cd7dd525ed11a04d84cd3936228499173`，完整提取
  `LayerNorm.forward`，显式提供其 weights、center 与 epsilon 属性；JAX NumPy 绑定与
  原 `fastmath/jax.py` 的后端字典一致。范围是方法的原始计算，不覆盖 Layer 生命周期。
  center 开/关各验证 direct、除 sqrt 改乘 rsqrt、错误 scale 三个固定候选，六组结构与
  数学结论均符合预期；每组两组种子对照独立 NumPy 结果，7 项回归通过。

运行记录：

- `uv run --locked pytest tests/test_pallas_index_map_bindings.py tests/test_pallas_metadata.py
  tests/test_pallas_squeeze.py tests/test_upstream_tpu_keras.py tests/test_pallas_tensor_memory.py
  tests/test_pallas_remote.py tests/test_pallas_local_scope.py tests/test_pallas_local_entry.py
  -q --tb=short`：307 passed / 1 failed（168.90 秒）。唯一失败是新证据测试错误地将带
  deferred-padding 的代数投影交给完整规范化复核；已改成显式 `normalize_view` 构造完整证据。
- 修正后 None 布局专项和 Trax：26 passed（14.77 秒），包含该失败点。
- 最终补充 bool/int 类型混淆的证据篡改拒绝后，
  `uv run --locked pytest tests/test_pallas_index_map_bindings.py tests/test_pallas_metadata.py
  -q --tb=short`：44 passed（15.43 秒）。
- 上述改动涉及的 14 个实现/测试文件 Ruff 与格式检查通过；`git diff --check` 通过。
  未重跑全量测试，未运行 TPU 硬件。

## 最终验收命令与代码入口

```bash
uv sync --locked
uv run pytest tests/test_upstream_tpu_*.py
# 同时执行两项已测得的 60 秒批量 SNR 超时回归，及完整 Pallas 代数回归文件：
JAX_REORDER_RUN_DIFFUSERS_SNR_BATCH=1 uv run pytest tests/test_upstream_tpu_*.py tests/test_pallas_algebra.py
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest --collect-only -q
uv run python tools/check_docs.py --changed
git diff --check
```

- [来源加载器](../../../tests/upstream_tpu/sources.py)：完整定义/原装饰器/显式闭包依赖，SHA-256 与 Git blob 双校验。
- [原始快照](../../../tests/fixtures/upstream_tpu/)与[固定 AI 候选](../../../tests/upstream_tpu/)：无运行时生成候选或在线下载。
- `src/jax_reorder/algebra/symbolic.py`、`trusted.py`：严格 `rsqrt` 条件与共同合法输入反例。
- `src/jax_reorder/core/adapters/pallas/call.py`、`model.py`、`normalization.py`、`local.py`：新参数准入、原始事实保留与独立复核。
- `src/jax_reorder/core/tensor_values.py`、`tensor_domain.py`、`adapters/jax.py`：受限 squeeze 坐标转换。

独立审查没有发现本轮补丁的正确性阻塞项；额外检查了定义域不同但无共同合法输入、
共同域非空却排除所有差异输入、65 维输出末坐标限制反例等情况，均保持明确未决。
相关 285 项代数回归通过，新增跨输出安全回归另 1 项通过；Pallas metadata/index-map 44 项、
squeeze 43 项通过。以上批次有重叠，不相加成全仓测试总数。
全仓 4,917 项收集成功；Ruff、446 个 Python 文件格式、basedpyright 与差异检查通过。
归档后文档检查覆盖 141 份 Markdown、1,189 个本地链接，全部通过。
本次未执行全仓 4,917 项测试，未提交或推送本轮改动；不覆盖 TPU 实机。
