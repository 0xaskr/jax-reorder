# 等式饱和与张量规则评估

评估日期：2026-10-15。对应[总计划](../../exec-plans/completed/open-source-equivalence-backends.md)。本工作线只构造符号图和求解公式，没有执行 GLM layer，没有随机采样，也没有修改生产比较器或项目依赖。

## 结论

egg/egglog 可复用搜索、共享表达式和同余推理基础设施，但不能替代张量语义与规则库。实际运行的 egglog 上游实验数组规则没有证明共同集中的 E2/E3/E4；这表示本次规则和预算未完成证明，不表示引擎不能实现这些推理。补上带索引条件的基础规则后，E3 可以合并，但收益来自我们提供的语义规则，不应算成社区已有张量证明能力。

TASO 的现成张量公理有真实复用价值：未添加目标公式，用原有公理在 Z3 中导出了 E3。它仍不是完整 JAX 数学等价后端。Tensat 整体构建较重，当前源码的旧 verifier 规则和算子元数存在不一致，优先借鉴规则，暂不建议直接接入其整个程序。SPORES 的索引关系代数最贴近仓库现有表示，但本轮只完成论文核对，不能声称已获得可用软件包。

## 版本和证据等级

| 候选 | 固定版本或快照 | 证据 |
| --- | --- | --- |
| egglog Python | PyPI `egglog==14.0.0`；源码 `22bfc5a7f22733537d32940472a9c46af59f4459` | 隔离安装、实际运行、源码核对 |
| egglog Rust | 上述 Python 源码 Cargo.toml 固定 `90635860397ce710f8c0a4eeb04154a8ebc3ac05` | 通过 Python wheel 实际调用；没有另行编译 Rust CLI |
| egg | 官方仓库和 explanations 文档，未固定源码版本 | 文档核对；没有单独运行，不把 egglog 结果冒充 egg 结果 |
| Tensat | `4fd1ad4a5082155e6bfb1685037f7d7fa07e5e0a` | 已下载源码、核对依赖；未构建 |
| TASO | `f11782c4db180ad7d1e7ff7f4bb82e574ac3b6b8` | 已下载源码；原有符号公理实际运行；未构建 C++ runtime |
| SPORES | 作者论文 arXiv `2002.07951v2` | 论文核对；本轮未找到并验证可运行的官方源码发布物 |
| DialEgg | `96b0b040d7868181554956d7184694085dfecf06` | 官方仓库已下载、源码核对；未构建或运行 DialEgg |

egglog 是 Rust 推理引擎的 Python 接口，支持规则和条件关系；候选比较应把两图放入同一个 e-graph 检查等价类，不能分别提取“最便宜表达式”后把文本不同当作不等价。其高层 Python API 尚未提供完整证明输出流程。[官方接口说明](https://github.com/egraphs-good/egglog-python/blob/22bfc5a7f22733537d32940472a9c46af59f4459/docs/reference/egglog-translation.md)

egg 提供从规则应用生成 equality explanation 的接口，值得独立评估可重放证明；解释仍相对于所注册规则，不能自动保证规则数学正确。[官方 explanations 文档](https://egraphs-good.github.io/egg/egg/tutorials/_03_explanations/index.html)

## 共同集实际结果

`unknown` 包括未合并、缺少算子语义或求解超时，均没有被计为“数学不等价”。

| 用例 | egglog 上游 array API | egglog 手写索引适配对照 | TASO 原有公理 + Z3 |
| --- | --- | --- | --- |
| E2：分配律 | 8 轮后 unknown | 未添加待证分配律规则，未测 | 源码已有该公理；没有把直接公理命中计为独立证明实验 |
| E3：收缩前后移动标量 | 8 轮后 unknown | 相同等价类 | `unsat`：否定目标与原有 44 条公理矛盾 |
| E4：dot 与广播乘积求和 | 8 轮后 unknown | 按定义展开即相同表达式；不是新增证明能力 | 未测，现有公理语言无对应通用 reduce 算子 |
| E5：指数加法恒等式 | unknown：没有 callable 数组 exp | unknown：exp 保持未解释 | 未测 |
| E6：mask softmax、全遮蔽 | 未测 | 未实现 | 未测 |
| N1：遗漏缩放 | unknown | unknown | 5 秒超时，unknown |
| N2：错误归约轴 | 未测 | unknown，轴名及 extent 保留 | 未测 |
| N3/N4/N5 | 未测 | 未测 | 未测 |
| U1：top-k/scan | 未测试转换 | 显式 unknown：适配器不支持 | 未测 |

上游数组探针提供兼容的 shape 元数据、`float64` dtype 和有限性假设；只建立符号变量，没有分配张量值。`array_api.exp` 这一属性实际来自通配导入的 `egglog.exp` 模块，并非数组指数函数。存在属性不等于支持该算子；Rust 的数值 `f64.exp()` 也不等于符号张量指数恒等式库。上游已有 `NDArray.__matmul__` 和 `sum` 声明，不代表已有 E3/E4 所需的数学连接规则。[本轮核对的数组源码](https://github.com/egraphs-good/egglog-python/blob/22bfc5a7f22733537d32940472a9c46af59f4459/python/egglog/exp/array_api.py)

手写适配对照仅有三个基础规则：实数乘法交换律、结合律，以及在 `c` 不依赖归约索引 `k` 时允许 `sum_k(a) * c` 改为 `sum_k(a*c)`。E3 通过组合这些规则得出，没有直接注册 `(QKᵀ)s=(Qs)Kᵀ`。有限实数和有限归约是明确前提；索引和 extent 以符号保留。将 `c` 换成依赖 `k` 的项后，守卫不成立，探针保持 unknown。E4 的两侧分别按收缩定义和广播求和定义得到同一索引表达式，报告明确把它归为转换验证。

TASO 探针复用上游声明与 44 条公理，只追加否定目标，没有添加新数学公理。上游 CLI 有 Python 2 的 `print` 语法，因此加载 `if __name__ == '__main__':` 之前的未修改定义部分。E3 结论相对于这些公理成立，尚未重新审计所有公理与 JAX 的形状/数学语义；未解释张量排序的 `sat` 模型也不能直接当作真实张量反例。[TASO 验证源码](https://github.com/jiazhihao/TASO/blob/f11782c4db180ad7d1e7ff7f4bb82e574ac3b6b8/verify/verify.py)

## Tensat 与 SPORES 的复用边界

Tensat verifier 使用同一 e-graph 比较表达式根；optimizer 另外调用 TASO 的算子运行成本。我们只需要前者的推理思路。其构建要求 Rust、定制 egg/TASO fork、C++ bindgen，`build.rs` 链接 TASO runtime 和 protobuf。本环境没有 cargo；本轮未进行完整构建，属于依赖和接入成本评估，不是候选能力测试失败。[依赖与使用说明](https://github.com/uwplse/tensat/blob/4fd1ad4a5082155e6bfb1685037f7d7fa07e5e0a/README.md)

此外，`src/rewrites.rs` 的 verifier 规则仍包含二元 `matmul` 和一元 `transpose`，而 `src/model.rs` 对应枚举分别有三个子节点。这是源码确认的接口不一致；尚未构建复现具体报错。README 关于“未验证规则”的描述不能成为数学不等价结论，我们以未找到证明处理。[旧规则定义](https://github.com/uwplse/tensat/blob/4fd1ad4a5082155e6bfb1685037f7d7fa07e5e0a/src/rewrites.rs)、[当前语言定义](https://github.com/uwplse/tensat/blob/4fd1ad4a5082155e6bfb1685037f7d7fa07e5e0a/src/model.rs)

SPORES 将线性代数降为带命名索引的实数关系代数，复用乘法、加法、聚合规则组合更多张量改写，避免穷举矩阵恒等式。其规范形式完备性仅针对论文给出的 sum-product 片段，而且唯一性要求对任意维度都等价。论文明确给出固定维度下不同形式仍等价的反例，因此不能据此判定当前固定 tracing shape 的两图不等价；`exp`、全遮蔽 softmax、动态索引和控制流不在该完备性结果里。本轮没有证明其源码可直接接入。[作者论文第 2.3 节及附录](https://arxiv.org/html/2002.07951v2)

## 距离真实 GLM 图的缺口

主代理在禁止编译/eager 执行下得到的 inventory：原始 GLM 图递归有 196 条方程，原始加十个改写共 33 种 primitive。以下为本线的覆盖判断，不把小型符号探针外推成完整 layer 通过。

| 真实算子或区域 | 等式饱和路线的适配要求 |
| --- | --- |
| `dot_general`, `reduce_sum`, `mul`, `add` | 有标准张量定义和可复用线性规则；必须携带 batch/收缩轴、索引作用域、输出形状；本轮只验证 E3/E4 小型片段 |
| `broadcast_in_dim`, `reshape`, `squeeze`, `slice`, `iota` | 可用索引映射表达；不能只替换算子名；上游有限数组声明不等于覆盖全部 JAX 参数 |
| `gather`, `dynamic_slice`, `dynamic_update_slice` | 需要索引范围、越界/填充值、更新顺序语义；本轮无现成规则覆盖证据 |
| `exp`, `div`, `max`, `reduce_max`, `select_n` 与 `-inf` | 需要定义域、分段条件、全遮蔽分支和特殊值语义；本轮无完整证明路径 |
| `top_k`, `approx_top_k` | 必须保留选择、顺序、并列以及近似算子参数；不能当成普通求和；本轮未支持 |
| `scan` | 需要递推不变量或受控展开、carry 和索引范围；同余引擎本身不会提供这些语义 |
| `jit`, `custom_jvp_call`, `stop_gradient` | 必须按仓库比较契约处理子图和求导边界，不能因外层名字相同而认定等价 |
| `convert_element_type` 与比较/布尔 primitive | 需要明确转换及整数索引/谓词语义，再生成条件事实；不能统一丢成实数原子 |

最低接入建议：保留结构比较与现有索引 IR，把可审计的有限实数 sum-product 子图送入 egglog，复用 TASO/SPORES 的规则思想；先证明额外案例有收益，再接复杂片段。需要数学不等价时，连接有语义依据的反例/受限判定后端；不能把 e-class 未合并当作拒绝证据。若优先目标是减少维护规则的负担，单独安装 egglog 不足以完成目标。

## 补查：DialEgg 的 MLIR 桥与 StableHLO 示例

DialEgg 自动转换 MLIR operation、type、attribute 等表示到 egglog，再把提取结果转换回 MLIR。它首先是优化器桥接设施；论文没有把它描述为任意两图的三态数学判定器。[作者介绍及论文入口](https://www.azizzayed.com/publications/dialegg/)

当前官方仓库确有 StableHLO 示例，不能简单说“完全没有张量规则”：`test/stablehlo/tests.egg` 包含逐元素交换/结合/分配、concat 和 dot_general 组合规则。但 `test/stablehlo_benchmarks/stablehlo.egg` 主要是操作声明，规则区为空，`reduce`、`reduce_window`、`scatter`、`while` 仍是注释。出现 `exponential`、`gather` 的声明不等于存在这些算子的数学等价规则。[StableHLO 示例](https://github.com/AzizZayed/DialEgg/blob/96b0b040d7868181554956d7184694085dfecf06/test/stablehlo/tests.egg)、[benchmark 声明](https://github.com/AzizZayed/DialEgg/blob/96b0b040d7868181554956d7184694085dfecf06/test/stablehlo_benchmarks/stablehlo.egg)

源码审查发现一个与数学拓扑直接相关的条件缺口：示例的 dot_general 结合律沿用同一任意 `?a` 属性，没有限制收缩轴。设双方收缩轴均为 0、没有 batch 轴，则二维方阵算子是 `D(X,Y)=XᵀY`，不满足普通矩阵乘法结合律。取符号非对称矩阵 `A`、`B=C=I`，可直接按定义推导：

```text
D(D(A,I),I) = A
D(A,D(I,I)) = Aᵀ
```

两图在一般非对称 `A` 上不是同一数学函数。这是对示例规则的定义推导审查，不是运行 DialEgg 后复现的错误输出，也不涉及浮点误差。它说明示例规则需要补充轴和类型条件，不能未经审计用于验证任意 StableHLO dot_general。

`EqualitySaturationPass.cpp` 写入单图节点并请求 `(extract ...)`，读取优化表达式后替换原操作；没有发现成套的双图 equality/inequality/unknown 返回协议或证明证书输出。源码对未声明操作保留 opaque value，这适合单图优化边界；接双图验证时还要建立稳定的输入、输出和 opaque 子图绑定，不能按各次分配的局部编号对应。[优化流程源码](https://github.com/AzizZayed/DialEgg/blob/96b0b040d7868181554956d7184694085dfecf06/src/EqualitySaturationPass.cpp)、[转换源码](https://github.com/AzizZayed/DialEgg/blob/96b0b040d7868181554956d7184694085dfecf06/src/Egglog.cpp)

构建需固定 LLVM/MLIR、StableHLO，以及带 cost-action 功能的旧 egglog fork；当前环境没有 standalone `mlir-opt`、`egg-opt` 或 cargo。现有 jaxlib Python MLIR 绑定不能直接替代这一 C++ pass 的构建依赖。本轮下载并审查源码，没有启动完整 LLVM 构建，也没有声称候选运行失败。结论是可借鉴转换接口和已审计规则，不作为下一步整层数学判定器首选。[官方构建要求](https://github.com/AzizZayed/DialEgg/blob/96b0b040d7868181554956d7184694085dfecf06/README.md)

## 复现与验证

脚本：[rewriting_probe.py](../../../tools/experiments/equivalence_backends/rewriting_probe.py)。实验环境、下载源码和 JSON 结果只在 `/tmp/jax-reorder-equivalence-eval/rewriting/`。

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv venv --python 3.12 /tmp/jax-reorder-equivalence-eval/rewriting/.venv
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv pip install --python /tmp/jax-reorder-equivalence-eval/rewriting/.venv/bin/python egglog==14.0.0 numpy==2.5.4 z3-solver==5.3.0.0
git clone https://github.com/jiazhihao/TASO.git /tmp/jax-reorder-equivalence-eval/rewriting/taso
git -C /tmp/jax-reorder-equivalence-eval/rewriting/taso checkout f11782c4db180ad7d1e7ff7f4bb82e574ac3b6b8
/tmp/jax-reorder-equivalence-eval/rewriting/.venv/bin/python tools/experiments/equivalence_backends/rewriting_probe.py --taso-source /tmp/jax-reorder-equivalence-eval/rewriting/taso
```

初次 PyPI 安装因沙箱 DNS 失败；获得网络提权后安装成功，不存在最终安装阻塞。Python 3.12.3、egglog 14.0.0、NumPy 2.5.4、Z3 5.3.0；egglog 探针每例最多 8 轮，TASO 求解每例 5 秒。脚本已运行；结果见上述矩阵。Ruff 和格式检查用于核对新增脚本，文档链接由主代理统一检查。
