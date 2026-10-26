# 计算图比较的验证矩阵与升级检查

状态：当前生效的验证策略。结果分类与 CI 分组于 2026-10-26 对照源码和工作流核对；所链接的验收记录保留当时的环境与结果，不代表本次文档整理重跑了测试。

基线为 Python 3.12、JAX/jaxlib 0.11.2、CPU。验收目标是保持完整图语义、识别已声明的同构，并对未解释的语义显式拒绝。测试覆盖有限的类别交叉，不声称穷举任意 JAX 源码、所有 shape 或无限组合。

## 规则与证据映射

| 覆盖维度 | 固定证据 |
| --- | --- |
| 公开 lax 身份、参数与 CPU 能力 | [逐原语清单](../../references/lax-coverage.md)、[原语测试](../../../tests/test_lax_primitives.py)：171 个身份，169 个有 CPU 执行证据；另列追踪消去与 CPU 拒绝 |
| 链、菱形、扇入/扇出、多输出、共享、常量、顺序 effects | [图回归](../../../tests/test_canonical.py)、[语义字段](../../../tests/test_semantics.py)、[AI 固定组合](../../../tests/test_ai_cases.py) |
| conv/dot/广播、QR/Cholesky、多结果与 precision | [新增组合](../../../examples/combination_matrix.py)、[验收](../../../tests/test_combination_matrix.py)：FP32/FP64，合法 conv/dot BF16，非方阵与三种固定调度，检查对称/传递性及 NumPy 参照 |
| cond、switch、scan/while 捕获和循环反馈 | [区域绑定](region-bindings.md)、[捕获测试](../../../tests/test_region_bindings.py)、[反馈测试](../../../tests/test_feedback_bindings.py)：零/一/多次，FP32/BF16，私有槽位联动和错接反例 |
| jit 结果及 donation、shard_map 接口、混合控制流 | [调用测试](../../../tests/test_call_bindings.py)、[分片测试](../../../tests/test_shard_bindings.py)、[混合测试](../../../tests/test_mixed_regions.py)：cond→scan→while→cond、scan→cond→scan、jit→cond→scan |
| 函数 pytree、静态参数、环境与模式 | [函数测试](../../../tests/test_functions.py)、[契约](function-and-derivative-contract.md) |
| custom JVP/VJP 规则、残差、透传、闭包和 symbolic zeros | [求导测试](../../../tests/test_custom_derivatives.py)：不同规则对象、错误梯度、stop_gradient、残差重排、全动态种子、激活模式、jit/vmap/控制流及显式拒绝 |
| 完整无根图、对称接口、预算 | [无根图](../../../tests/test_unrooted.py)、[cond](../../../tests/test_cond_bindings.py)：共享差异、未使用常量、有限精确搜索，无 DCE |
| 独立原图复核和故意破坏证据 | [对应关系测试](../../../tests/test_correspondence.py)：遗漏节点、共享、端口、分支、effects，以及规范化器漏参数、相同非法图、冒名 Primitive |
| 压力 | [深度](../../../tests/test_depth.py)、[规模](../../../tests/test_stress.py)：8/16/32 层受控共享区域、2000 层参数、6000 节点长链、512 路扇出 |
| 单进程双/四 CPU 通信 | [通信记录](../../exec-plans/completed/validation/collective-jaxpr.md)、[脚本](../../../examples/collective_jaxpr.py)：22 个固定组合、参数反例和规范化前后执行 |
| AI 错标签、两侧共同错误、独立参照 | [业务验收](../../../tests/test_business_layers.py)、[SGLang 层验收](../../../tests/test_sglang_model_layers.py) |
| 完整真实业务层及硬件拒绝 | [完整层](../../exec-plans/completed/validation/full-attention.md)、[测试](../../../tests/test_full_attention.py)：原 GLM5/Llama/CLIP/Flux、索引器、JVP/VJP、真实 Pallas 调用 |

19 种 dtype、标量/零长度/单元素/广播等已有原语矩阵仍保留。每项测试使用原语合法域；JAX 本身拒绝的组合单独报告。NaN/Inf/有符号零、参数与常量精确编码不能被数值容差掩盖。

## 结论与失败分类

结构 `equivalent` 需要完整图匹配和原图双射复核；`different` 是确定图差异，未必数值不同。自动模式在结构不同时继续数学比较，数学 `equivalent` / `inequivalent` 由支持语义下的完整输出及定义条件比较给出。数学不等价表示两函数并非恒等，不表示所有输入都得到不同值。实际数值反例与符号判定分别记录，多次样本相等不能证明等价。

正常结果不返回 `unknown`。未支持语义、未决、预算耗尽与超时通过 `ComparisonError` 子类报告；函数追踪、编译和契约失败分别保留 `trace_rejected`、`cpu_compile_rejected`、`context_mismatch` 或 `contract_mismatch` 等异常分类。当前定义见[结果状态](../../../src/jax_reorder/status.py)、[异常分类](../../../src/jax_reorder/errors.py)及[函数契约](function-and-derivative-contract.md)。

AI 标签是待验证预期。运行器核对源图实际换序、IR、接口、对应关系、幂等、原图不变、规范化执行及独立参照；脚本不生成候选。业务报告分开统计类、不同实现、候选条目、签名组合、数值样本及确实改变追踪表示的正例。应完成的正反例须命中预期；预期拒绝须核对对应异常，不能声称它们匹配成功。历史报告中的 `unknown` 保留其当时含义。

## CI 与本地命令

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run pytest
uv run pytest -m 'not business and not stress and not pallas_communication'
uv run pytest -m business
uv run pytest -m stress
uv run pytest -m pallas_communication
uv run python tests/adapter_inventory.py --report /tmp/adapter-inventory.json
uv run python tests/lax_inventory.py
uv run python examples/inspect_jaxpr.py
uv run python examples/collective_jaxpr.py --devices 4
uv run python tools/check_docs.py --changed
git diff --check
```

[CPU CI](../../../.github/workflows/ci.yml)将快速、业务、压力测试分组；快速组排除 Pallas 通信，并运行 Ruff、basedpyright、文档检查、清单和示例。Pallas 通信在独立任务运行并保存 JUnit 报告；集合通信另以 2/4 设备运行，适配与业务任务保存 JSON 报告。业务和压力分组由 [conftest](../../../tests/conftest.py)登记，Pallas 通信测试使用 `pallas_communication` 标记。上方全量命令与分组命令是不同运行方式，`uv run pytest` 仍运行全部。缓存只读时使用 `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache`。这里描述工作流配置，不宣称本轮执行或远端运行通过。

## JAX 升级审核

[适配清单脚本](../../../tests/adapter_inventory.py)与[锁定快照](../../../tests/fixtures/jax-adapter-0.11.2.json)核对版本、私有 IR slots、各区域 Primitive 参数/类型和公开 lax 身份。规范化入口遇到未经审核的 JAX/jaxlib 版本抛出 `UnsupportedJaxprError`；未知描述符子类也拒绝，避免继承字段未被编码。

[手动升级工作流](../../../.github/workflows/jax-upgrade.yml)使用独立 uv 环境检查指定候选，不改 `uv.lock`、快照或版本允许表。若私有 API 已变到无法导入/追踪，任务失败即表示需要适配；不能把未生成候选清单当作升级通过。常规基线仍须保持全绿。

扩展支持须逐项审核新增字段、参数、抽象值、effects 和求导 thunk，补字段反例，重新运行原语、完整层、深度及多设备验证，最后用 `uv add` 更新依赖/锁文件和快照。仅改版本字符串或复制新快照不能完成审核。

仍未覆盖的持久格式、高阶求导、规则专有动态捕获及硬件路径见[待处理清单](../../exec-plans/index.md#待处理)。该阶段的全量数字与命令结果保存在[执行计划](../../exec-plans/completed/graph-equivalence-validation.md)；历史文档中的旧数字只代表当时的回归。
