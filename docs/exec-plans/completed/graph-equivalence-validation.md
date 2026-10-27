# 计算图等价比较的完善与系统验证计划

> 路径迁移注记（2026-10-27）：本文命令已更新为迁移后的 `tests` 路径；验收结果仍属于原记录阶段，迁移阶段未重跑这些命令。

## 状态与目标

2026-10-11：已完成 M1–M4 的实现、验证、报告与文档，按计划归档。全量 1078 项通过；最终错误分类调整另通过 31 项相关回归。尚未支持的语义、上游 CPU 拒绝和未验证配置保留明确边界，见文末与技术债。

目标是：无论源码经历了怎样的改写，只要最终得到的**完整语义计算图**相同，就在声明的输入签名、执行环境与 JAX 变换范围内识别为等价。比较不需要知道源码的改写过程。

这里的图相同指保持外部输入输出位置、普通算子的操作数端口、节点及其全部语义信息的图同构。内部区域的捕获槽位和形参允许同步换序，但实际绑定必须不变；它们不能与外部参数位置或普通算子的操作数顺序混为一谈。节点编号、变量名和无依赖计算的排列不影响结果，只比较连接形状或算子名称不够。

需要同时解决两个问题：

- 可靠性：判为 `equivalent` 的图必须确实满足比较契约，不能遗漏参数、绑定、分支、求导规则或 effects。
- 识别能力：契约允许的图同构不应因追踪顺序、捕获槽位或内部编号不同而被漏认。

有限测试不能穷举任意程序、任意 shape 和无限组合。完成条件应是明确规则的论证、独立复核、系统覆盖和反例回归，不写成“所有 JAX 代码已验证”。

## 范围与语义契约

延续 [当前架构](../../../ARCHITECTURE.md)，先覆盖锁定版本 Python 3.12、JAX/jaxlib 0.11.2 的 CPU。`jax.lax` 在 CPU 上支持的合法 shape、dtype 和参数属于目标域；测试用到的尺寸和取值不作为实现白名单。JAX 能处理而本项目不能解释的情况属于待补支持。

按用户补充，将自定义求导纳入计划：普通前向、JVP、VJP 分别追踪和比较，并记录各自适用范围。首轮实现前向与一阶求导；高阶求导、不同变换顺序必须作为独立模式登记，不能由一阶结果直接推断。

| 项目 | 比较要求 |
| --- | --- |
| 外部接口 | 输入和输出的 pytree、叶子顺序、shape、dtype、weak type，以及声明的可微参数 |
| 图节点 | Primitive 身份、全部语义参数、输出端口、抽象值、子图和 effects |
| 连接 | 普通算子的有序操作数、共享与重复计算的区别、常量绑定、区域接口的实际绑定和循环反馈关系 |
| 追踪上下文 | 静态参数、闭包常量、JAX/jaxlib 版本、CPU、精度配置、X64、PRNG 实现、mesh/sharding 等相关设置 |
| 求导 | 前向输出、切向量/余切向量接口、残差绑定、不可微参数和 symbolic zeros 对应的追踪模式 |
| 未解释语义 | 显式 `unknown`，保留原因；不能丢弃字段后继续声称等价 |

结论适用于该追踪签名所描述的有效输入域，而不只是抽样输入。它不覆盖未声明的 Python 静态分支、外部状态、副作用或其他签名。函数入口也不能自动证明任意 Python 函数纯净；调用契约须明确这一前提。JAX 对追踪行为的说明见 [jaxpr 文档](https://docs.jax.dev/en/latest/601/jaxpr.html)。

图匹配可以给出声明范围内的计算语义等价；不自动承诺原代码在任意编译方式、硬件上的逐位浮点一致。XLA 可能改变浮点表达式的执行，详见 [JAX FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)。数值检查单独报告，不能代替图证明。

保留现有三种图级结果：`equivalent`、`different`、`unknown`。`different` 仅表示存在图差异；例如增加一个数学上的恒等计算也会改变图，不能据此声称计算结果必然不同。需要声称实际输出不等价的案例，另附有效反例输入和输出。

## 实施前基线与已确认缺口

基线提交为 `034b92949117052b6c3bca0e16de0db34c687a0a`。提交前全量结果为 855 passed；这是已有验证记录，本次计划编写没有重跑全量测试。

- [原语清单](../../references/lax-coverage.md)覆盖 171 个公开 Primitive 身份，其中 169 个有 CPU 执行样例，其余分别为追踪消去、CPU 拒绝检查；这不等于所有参数和组合已经验证。
- 已修复 `cond` 捕获槽位的同步置换；四层二叉 `cond` 案例包含 15 个条件节点、16 条执行路径。
- [SGLang-JAX 案例](validation/sglang-model-layers.md)覆盖 36 个选定子层和 216 个候选/签名组合；不代表完整 Attention 或 Decoder block 已覆盖。`Glm5Attention` 和 `Glm5DecoderLayer` 尚未验证。
- [待处理清单](../index.md#待处理)中的 BFS-001、002、003、004、005 分别涉及无输出根的子图、函数契约、JAX 适配、深层遍历和区域绑定。

以下为实施前在锁定 CPU 环境用 AI 手写最小函数得到的缺口记录，修复后证据见文末：

| 探测 | 修复前结果 | 计划中的处理 |
| --- | --- | --- |
| `scan` body 只交换独立计算顺序，捕获槽位从 `(x, scale, gate)` 变成 `(gate, x, scale)` | `different`，路径 `graph.nodes[0].inputs[0][1]` | 补循环常量与 body 形参同步规范化 |
| `while_loop` body 进行相同改写 | 同样返回 `different` | 补 condition/body 捕获接口的同步规范化 |
| 普通追踪两个 `custom_jvp` 或 `custom_vjp` 函数 | `unknown`，对应调用 Primitive 尚未支持 | 物化所需求导图，增加专用适配 |
| 从 `custom_vjp_call` 参数分别取得并追踪 fwd/bwd | 简单标量规则成功得到两个普通 jaxpr；前向为 `mul x x`，反向为 `mul 2 x` 后乘余切向量 | 确认可以直接适配嵌套规则，当前缺口不是 JAX 无法提供内部图 |
| 前向都为 `x*x`，求导规则分别为 `2*x`、`3*x`；显式追踪 JVP/VJP | `different`；`x=2` 时前向均为 4，梯度分别为 4、6 | 固化前向相同、求导不同的反例 |
| 返回 tuple 与 list，但展平后的叶子计算相同 | jaxpr 接口返回 `equivalent` | 函数入口补 pytree 契约；现有底层接口没有这些信息 |

## 允许的变换与边界

| 改写 | 本计划预期 |
| --- | --- |
| 变量改名、Python 函数拆分但追踪出的调用边界和图相同 | `equivalent` |
| 任意深度的独立算子换序、独立分支交错计算 | `equivalent`，不限制为同一深度 |
| 内部捕获参数、形参、常量槽位同步置换 | 完成对应区域适配后 `equivalent`；实参到变量的绑定必须相同 |
| 多层控制流内部的同类改写 | 逐层检查接口和子图后 `equivalent` |
| 求导残差槽位同步置换 | 连同生产者、消费者一起匹配；只改一侧必须拒绝匹配 |
| 外部输入输出位置、操作数端口、分支条件、常量或精度参数改变 | 有明确图差异时 `different` |
| 交换律、结合律、分配律、CSE、DCE、常量折叠 | 会改变本轮定义的图，不作为等价规范化规则 |
| 内联已存在的调用节点、循环展开、融合算子替换、改变 remat 边界 | 需要另一套区域展开或计算改写规则，本计划保留边界，不自动接受 |
| 未支持的 effect、FFI、Pallas/自定义 Primitive、不透明 callable | `unknown`；取得完整语义适配后才能扩展 |

“Python helper 展开后图未变”和“删除 jaxpr 中已存在的调用边界”须区分；不能用前者的测试通过证明后者受支持。

## 工作项与实施顺序

### 第一阶段：固定函数契约与回归基线

- [x] 固化上述 `scan`、`while_loop` 和自定义求导探测为 AI hardcode 回归，先保存原始 jaxpr、预期和原因。
- [x] 增加函数级比较入口，调用现有图比较器；统一捕获输入输出 pytree、静态参数、抽象签名和相关追踪环境。
- [x] 权重、KV cache、随机 key 等业务状态尽量作为显式输入输出；闭包常量仍完整比较。
- [x] 区分 JAX 拒绝追踪/CPU 编译、项目缺少适配、确定图差异、数值抽样失败四种原因。
- [x] 按前向、JVP、VJP 以及显式组合模式分别返回结果。只有全部请求模式完成且等价，才能报告该范围整体等价。

拟涉及 `src/jax_reorder/` 的函数入口与结果类型，以及函数契约测试。保持现有 `compare_jaxprs` 作为底层接口。

验收：tuple/list、输出字典键、输入顺序、静态签名等差异不会被展平掩盖；未检查的求导模式明确显示“未验证”。更换追踪环境不会复用不适用的比较结果。

### 第二阶段：增加独立的原始图对应关系校验

- [x] 在规范化结果中保留原节点、变量、常量和区域接口到规范表示的映射。
- [x] 匹配成功时构造两份原图的对应表，不仅给出相同的规范表示。
- [x] 单独实现校验器，逐项检查映射是否为双射，是否保持外部位置、端口、Primitive、参数、常量、共享关系、子区域与 effects。
- [x] 校验器直接检查原图与映射，不通过再次调用 BFS 或比较同一份规范化结果来“独立验证”。语义适配器仍属于共同信任基础，需用独立预期值和字段反例检验。
- [x] 故意损坏映射：漏节点、合并共享、交换端口、错绑常量、错连一个分支、漏 effect，校验器都必须拒绝。
- [x] 补充书面论证：有序输出与操作数 BFS 对已覆盖的可达图保持同构；拓扑重排保持依赖；区域置换必须同时保持实际绑定；嵌套结论递归成立。

验收：每个 `equivalent` 都可复核全部必要对应关系。校验失败或语义不完整不得升级成 `equivalent`。书面论证和实现校验不冒称机器形式化证明。

### 第三阶段：补齐区域接口，先解决已复现的漏认

为每种区域定义接口角色及相关元数据，避免把所有子图参数当成可自由排序的普通列表。

| 顺序 | 区域 | 必须联动的内容 |
| --- | --- | --- |
| 1 | `scan` 捕获常量 | 父节点常量实参、body 常量形参、常量数量及相关参数位置 |
| 2 | `while_loop` 捕获常量 | condition/body 各自的捕获槽位、父节点实参、两侧共享绑定 |
| 3 | `jit` 与调用区域 | 父实参、子形参、分片、布局、donation 等按位置存储的元数据 |
| 4 | `scan`/`while` 内部 carry 重组 | 初始化值、body 入参和返回值、条件输入、循环反馈边、最终外部输出的联合映射 |
| 5 | `shard_map` 等已有边界 | 同步参数绑定，同时保留 mesh、轴环境、通信参数及顺序语义 |

`scan` 的常量、carry、扫描输入和输出按角色区分；保留 length、reverse、unroll 等参数。循环 carry 不能直接套用 cond 的“首次使用排序”，必须联合处理反馈关系；只有映射成立才接受。

- [x] 每条规则先添加至少两个正例、两个接错绑定反例和一个不支持/非法路径。
- [x] 混合验证 `cond → scan → while → cond`、`scan → cond → scan`、`jit → cond → scan`，以及同一子图被多个父节点引用。
- [x] 覆盖分支返回直通参数、多输出、重复实参、不同 dtype、多个同 shape 参数、闭包常量和未使用参数。
- [x] 对循环测试合法的零次、一次、多次执行；对 cond 覆盖真假和 switch 全部分支。静态分支须单独追踪，不能只改变数值输入。

验收：本次已复现的 scan/while 正例变成 `equivalent`，故意只换实参或只换形参的反例不能通过；规范化后的图仍通过 IR 校验、执行与幂等检查。

### 第四阶段：自定义求导的图与绑定

用户提出“把前向和反向分别匹配”的方向可行，但普通 `make_jaxpr(f)` 不会自动提供全部已展开的求导图。当前锁定版本实测：`custom_jvp_call` 包含 `call_jaxpr` 和 `jvp_jaxpr_fun`；`custom_vjp_call` 还包含延迟前向规则、反向 callable 和输出树信息。

`custom_jvp` 注册的是 JVP 规则，由 JAX 支持相应的反向求导；`custom_vjp` 注册 `fwd`/`bwd`，其中 `fwd` 返回残差供 `bwd` 使用。普通调用的 `f` 也必须独立比较。`custom_vjp` 不能直接接受普通前向模式 `jax.jvp`，这应登记为变换不适用，不算比较器漏支持。[JAX 自定义求导文档](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html)

`custom_vjp_call` 应作为可递归解释的嵌套计算节点正式支持。打印时 `fwd`/`bwd` 显示名称，不代表无法读取函数内容。实施前已从实际节点参数成功物化简单规则的前向和反向图；现已按以下路线加入比较器，完整规则及边界见[求导契约](../../design-docs/current/function-and-derivative-contract.md)。

实现路线：

1. 增加锁定版本的自定义调用适配，将 `call_jaxpr`、fwd、bwd 分别转为可递归比较的区域；JVP 节点对应普通前向和 JVP 规则。保留调用边界与原始 callable 的执行能力，不用名称、`repr`、地址或对象相同代替规则内容比较。
2. 根据原始参数、可微参数模式和前向残差的抽象签名物化规则。完整编码常量、输出树、symbolic zeros、残差透传映射等信息；残差生产者和消费者执行同一个内部槽位置换。
3. 通过公开 `jax.jvp`、`jax.vjp` 另行追踪连接后的计算，复核嵌套适配和规范化前后的执行；不以删除全部调用边界代替原图匹配。

一个实际需要处理的细节：在 `fwd(x) -> (x*x, x)` 的探测中，锁定版本的 fwd jaxpr 只输出 `x*x`；残差 `x` 的直接透传由 `out_trees()` 的附加映射 `[0]` 表达。只分别比较两个函数体而漏掉该绑定信息会留下漏洞。

用于交叉验证的 VJP 程序将前向和反向连接起来，避免断开残差绑定：

```python
def vjp_program(f):
    def run(x, cotangent):
        y, pullback = jax.vjp(f, x)
        return y, pullback(cotangent)

    return run


# x 与 cotangent 都作为动态参数，按其完整抽象签名追踪。
# 实际入口需推广到多参数和 pytree，并另外比较普通前向 f。
vjp_graph = jax.make_jaxpr(vjp_program(f))(x, cotangent)
```

JVP 相应追踪 `(x, tangent) -> jax.jvp(f, (x,), (tangent,))`。不能只比较 `grad(sum(f(x)))` 或固定余切向量为全 1；那只能覆盖一个投影，可能漏掉输出间抵消的错误。不能只在某个固定 x 上建立 pullback 再推广到所有输入。

- [x] 正例：前向和求导各自独立计算换序、不同 Python rule 对象但图相同、残差生产与消费同步换序、多输出梯度。
- [x] 反例：前向相同但梯度系数不同、只接错一个残差、遗漏一个输入梯度、加入 `stop_gradient`、原函数与自定义 fwd 的输出改变。
- [x] 检查 `nondiff_argnums`、静态参数、零切向量、symbolic zeros、整数/不可微叶子及复杂 pytree；symbolic zeros 的不同激活模式分别登记。
- [x] `jit`、`vmap`、`cond`、`scan` 内外的求导分别测试；不同变换顺序分别追踪，只对声明的模式给结论。
- [x] 动态 while 等 JAX 本身不支持的求导组合单独报告；不能通过改写为另一种循环而悄悄改变验证对象。

验收：前向相同、梯度不同的案例，在前向范围与包含求导的范围中得到对应结果；任一请求模式未完成，整体报告不能称为全面等价。

### 第五阶段：完整图与高深度处理

输出反向 BFS 适用于有根的可达部分；要求完整图同构还需要处理孤立纯计算、未使用常量及对称结构。

- [x] 保留全部节点和常量的数量、身份类别与共享关系，不用 DCE 绕过问题。
- [x] 对无输出根的部分先按完整语义标签和连接关系细化候选分组；有歧义时做精确对应搜索，不只比较摘要。
- [x] 共享常量、共享输入或父子区域关系必须在全图内检查，不能将有共享关系的部分当成互不关联的组件独立排序。
- [x] 设置可说明的资源预算；超出预算返回带规模和原因的 `unknown`，不能回退为任意顺序或误判 `different`。
- [x] 将深层区域、参数容器和比较诊断的递归改为显式栈，确认错误处理也不会因递归深度崩溃。
- [x] 区分纯深度压力与二叉分支数量压力；测试 8、16、32 层嵌套时使用受控节点数量，不要求构造指数规模的全部分支树。

验收：固定的未使用组件换序正例和共享差异反例全部通过；长链、高扇出和深层混合区域没有崩溃或静默退化。资源边界明确记录，不能称为无限规模均可完成。

### 第六阶段：建立组合测试矩阵

遵守用户既定要求：**AI 直接编写 base、改写候选、预期结论和理由，并 hardcode 保存；不用脚本或 AST 自动生成改写。** 脚本只执行已固定的程序、生成数值输入、遍历声明的签名并收集报告。压缩失败程序也由 AI 改写后固化。

| 维度 | 必须覆盖的类别 |
| --- | --- |
| 拓扑 | 链、菱形、扇入/扇出、跨深度交错、多输出、共享与重复子图、直通输出、孤立部分 |
| 原语 | 逐项保留已有清单；扩展一元/二元、归约、变形、索引更新、dot/conv、线性代数、随机、通信、控制流的组合 |
| 参数 | axis、dimension_numbers、precision、padding、stride、gather/scatter mode、排序与通信分组等完整字段 |
| 类型与形状 | 原语实际合法的整数、bool、浮点、复数和 key；标量、零长度、单元素、广播、非方阵、不均匀维度、weak type |
| 数值 | 零、正负边界、非对称值、极值、NaN/Inf、有符号零；按原语定义域选择有效输入 |
| 区域 | 各类捕获接口、循环反馈、嵌套、重复参数、常量、多输出、求导规则和残差 |
| 变换环境 | 声明的 jit/vmap/求导组合、单 CPU 与单进程 2/4 CPU 设备、分片接口 |

组合策略为覆盖已登记的类别交叉：原语族 × 拓扑，区域接口 × dtype/shape，求导 × 控制流，精度 × 归约等高风险三项组合。非法组合单独列为 JAX 拒绝；不能用跳过项填充支持数量。每个公开原语至少保留独立换序证据；不能把所有原语各测试一次等同于组合覆盖。

每个固定案例经过以下检查：

1. 在运行比较器之前记录 AI 预判、允许规则和图差异原因；AI 预判是待验证标签，不是独立正确性证明。
2. 双方原始 jaxpr 和规范化 jaxpr 均通过 JAX IR 校验；原图未被修改。
3. 图状态精确符合预期；正例通过原图对应校验；幂等、自反、对称，以及固定三元组上的传递性成立。
4. 声称“独立计算换序”的正例，必须确认原始追踪顺序或捕获接口确实改变。追踪后完全一样的样例另记，不能冒充新的换序覆盖。
5. 双方执行、双方规范化前后执行，与独立数值参照核对。dtype、shape、pytree 单独精确检查，浮点容差仅用于数值证据。
6. 实际计算不等价的候选必须保存可复现的输入/输出反例；只有图差异的候选可以数值相同，分别报告。
7. 加入“错误 AI 标签”“两侧同时写错”“损坏对应表”“只遗漏一个语义字段”等测试，确认验证流程能发现这些问题。

优先保证没有已知的假阳性，再逐项消除范围内假阴性；不能通过放宽比较字段让测试变绿。

### 第七阶段：真实 Attention 与完整 block

以 [固定上游源码](../../../tests/fixtures/sglang_jax/README.md)为基准，继续保留原始实现和来源哈希。新增覆盖按实际执行的类、路径和配置统计，已有模型别名映射可复用，不能用另一类简单子层抵扣。

第一批按以下顺序实施：

1. `Glm5Attention` 的 CPU 可执行路径可行性核查：真实 Q/KV 投影、归一化、RoPE、attention 后端、输出投影和返回的 `kv_fused` 全部纳入。
2. 明确 absorbed MQA 与 MHA 两条路径、prefill/decode、QK norm 开关、indexer/稀疏配置分别是否能够在 CPU 执行。选定一种只算该配置，其他路径继续标为未验证。
3. 为可执行的完整 Attention 写至少两个保持图的 AI 候选、两个真实错误候选，覆盖准备次序、捕获绑定、position/mask/cache、精度或缩放等差异。
4. 继续进入 `Glm5DecoderLayer`，包含 attention、norm、残差、实际使用的 MLP/MoE 及状态输出；根据原调用链登记不同层配置。
5. 根据上游实际结构选择代表性的普通 MHA、GQA/MLA、MoE、交叉注意力与视觉/扩散 block，逐步扩展完整层覆盖表。

完整 Attention 的前提是实际 attention 核心也被追踪和执行。若原后端仅支持 TPU/GPU 或调用不透明内核，应明确记录阻塞的类和 Primitive；可另测公开 CPU 参考实现，但不能把替换核心后的结果写为原始完整层已通过。

GLM5 当前源码内有固定的 head/LoRA 维度，应先减少 token、head 等合法配置；不能随意改写这些常数并仍声称原类未经改变。索引器即使返回值未参与输出也可能产生待保留子图，需与第五阶段衔接。

业务验收包含独立权重、非对称输入、多 dtype/shape、边界位置与 mask、cache 状态，以及适用时的前向/JVP/VJP。每份报告列出完整层与仅子层的范围，不再只给“所有模型通过”的总数。

### 第八阶段：CI、版本变更与报告

- [x] 保持锁定版本 CPU 为阻塞性 CI 基线；新 JAX 版本在独立升级验证中检查，不能静默改变结论。
- [x] 对私有 IR 字段、Primitive 参数、抽象值、effects 和自定义求导结构增加适配覆盖检查；新字段未解释时显式拒绝。
- [x] 快速回归进入每次提交检查，完整案例和多设备验证进入全量检查，较大数值输入与深度压力单独运行。
- [x] 报告包含源码案例 ID、上游版本、模式、输入签名、原语/接口覆盖、预期与实际状态、差异路径、未支持原因、数值反例和运行环境。
- [x] 分开统计函数数、不同实现数、候选数、签名组合、数值样本和实际改变图表示的正例数。
- [x] 公布支持范围内的正确识别、误判、漏认、未支持和未验证项；`unknown`、跳过和 CPU 不可执行都不计作等价通过。

## 首轮交付与后续里程碑

| 里程碑 | 内容 | 可观察完成条件 |
| --- | --- | --- |
| M1 | 固定契约、当前漏认回归、对应关系校验、scan/while 捕获绑定 | 当前两类漏认被修复；绑定错误反例拒绝；现有回归无退化 |
| M2 | 函数入口完整结果、custom JVP/VJP、混合控制流 | 前向/求导分别报告；残差同步换序通过；前向相同而梯度不同被发现 |
| M3 | 无根子图、对称绑定、深层遍历和组合矩阵 | 已登记同构正例无漏认，图差异反例无误判；预算和未支持项明确 |
| M4 | Glm5Attention 与 DecoderLayer、其他完整层及 CI 报告 | 每个声明配置的原实现、候选、独立参照和状态输出都有证据；阻塞路径保留缺口 |

上述编号表示依赖和交付顺序。GLM5 CPU 可执行性调查应在 M1 期间开始，尽早识别后端阻塞；完整层验收在所依赖的接口和子图支持就绪后进行。

每个修复都按“固定失败函数 → 确认图对应关系 → 修改算法 → 正反例 → 原语/业务回归 → 文档”完成。当前的 855 个测试作为回归基线，不以某个更大的测试总数作为项目完成条件。

## 验证命令与本次记录

实现阶段沿用以下命令，并在各里程碑补充专项测试和报告命令：

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv sync --locked
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run ruff check .
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run pytest
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python tests/lax_inventory.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python tests/inspect_jaxpr.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python tests/nested_cond_four_levels.py
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv run python tests/collective_jaxpr.py --devices 4
git diff --check
```

编写草案时只完成计划、环境核对和小规模探测。`uv sync --locked` 成功，探测使用 JAX 0.11.2 与单 CPU 设备；以下复现保留的是修复前结果，实施后的证据见末尾记录。

计划文档验证：四份变更文档的 24 个本地链接存在，两个 Python 代码块通过语法检查，下面的 scan 复现再次得到预期的当前漏认结果。`git diff --check` 通过。

`scan` 漏认的最小复现如下，`while_loop` 探测在循环体中使用相同的独立计算换序。以下函数由 AI 手写；这段记录用于后续固化回归，不属于已完成的修复。

```python
import jax
import jax.numpy as jnp

from jax_reorder import compare_jaxprs


def original(x, scale, gate):
    def body(carry, item):
        a = jnp.sin(x * scale)
        b = jnp.cos(gate)
        return (carry + a) + b, item + x

    return jax.lax.scan(body, jnp.zeros_like(x), jnp.arange(3, dtype=x.dtype))


def reordered(x, scale, gate):
    def body(carry, item):
        b = jnp.cos(gate)
        a = jnp.sin(x * scale)
        return (carry + a) + b, item + x

    return jax.lax.scan(body, jnp.zeros_like(x), jnp.arange(3, dtype=x.dtype))


args = tuple(jnp.asarray(v, dtype=jnp.float32) for v in (1.0, 2.0, 0.3))
left = jax.make_jaxpr(original)(*args)
right = jax.make_jaxpr(reordered)(*args)
result = compare_jaxprs(left, right)
print(result.status.value, result.difference)
# 当前基线：different graph.nodes[0].inputs[0][1]
# 完成捕获绑定规范化后的预期：equivalent
```

## 已确定的选择

- 完整图/接口默认搜索预算 10,000，分别按区域状态与最外层接口尝试计数。超限 unknown；受控压力为 8/16/32 层区域、2000 层参数、6000 节点链和 512 路扇出，不宣称无限规模。
- symbolic zeros 非空激活模式逐项物化，最多 64 个输入/余切组合；超限 unknown，不默默忽略模式。
- GLM5 native MHA 完整原实现通过；absorbed MQA/Decoder 与 Wan Flash 原 Pallas CPU lowering 拒绝，EPMoE 原 CPU interpret 可执行但图未知。单 head GLM5、Flux 完整 block 的原追踪拒绝与未验证稀疏 DSA 分开登记。
- 仅声明已请求的一阶模式；高阶求导、其他变换顺序、规则专有动态捕获及延迟有序 effects 不自动外推，详见技术债。


## 实施记录

- 2026-10-11，M1 开始：已重新读取现有代码与环境，确认 scan 使用锁定版本的 ft_in/ft_out 角色树。新增 AI 固定循环正反例与区域接口设计，首先固化失败，再实施捕获绑定适配。函数入口、独立对应校验、求导和后续阶段均仍待完成。
- M1 循环与对应校验：38 项循环绑定测试及 12 项对应证据测试加入后，全量 905 passed（129.22 秒）。scan/while 捕获漏认已修复；零次/一次/多次循环、类型和 shape、错绑反例及规范化前后执行均检查。
- 函数入口与调用槽位：`compare_functions` 分别追踪前向/JVP/VJP，动态余切向量覆盖完整输出；环境与 pytree 单独比较。jit 求导残差返回槽位漏认也已修复，位置元数据同步置换。函数、对应关系、循环、cond 和嵌套专项 106 passed；额外调用槽位 9 passed。加入求导适配入口后的核心专项 89 passed（11.58 秒）。这些是分阶段记录，不代替最终全量回归。
- 原始图复核的设计与论证见[对应关系文档](../../design-docs/current/original-correspondence.md)。求导规则物化和残差连接实现已开始，相关测试尚在验证。
- GLM5 早期调查：已取得锁定上游提交的 RadixAttention、NativeAttention、MLA 后端、KV cache 和 ForwardBatch 源码，核对 Git blob 摘要。确认存在 CPU native 路径；完整类和状态输出尚未执行验证，不能计入已覆盖层。

- M2–M3：函数入口、注册规则物化、残差/透传绑定、symbolic zeros、反馈槽位、shard_map、混合区域与完整无根图已实现。阶段全量 1013 passed（170.62 秒）；后续新增项仍需最终回归。
- 完整业务层：原始 Glm5Attention MHA、LlamaAttention GQA、CLIPEncoderLayer、FluxAttention joint 路径使用 16 个配置与 11 个候选条目，保留全部权重和状态。原始完整文件及 22 份新增后端依赖均核对 SHA-256 和 Git blob。
- 硬件边界：Glm5Attention absorbed MQA、含 dense MLP 的 Glm5DecoderLayer（prefill/decode）及 WanT2VCrossAttention 均在实际 pallas_call 上返回 unknown，原 CPU lowering 拒绝；EPMoE 的原 CPU interpret 路径执行且与 NumPy 参照一致，但图仍 unknown。FluxSingleTransformerBlock 的原 Explicit sharding 约束与单 head GLM5 配置在 JAX 追踪处被拒绝。未替换内核冒充通过。
- 四类完整层分别检查前向/JVP/VJP，隐状态为可微输入，全部输出的动态余切接口保留。四项专项通过：前向和 JVP 等价，VJP 因 add_any 梯度累加顺序返回 different；实际求导图及规范化图的 CPU 数值一致。
- GLM5 原始索引器：prefill FP32、decode BF16 两项通过；AI 保留索引器并重排得到 equivalent，删除未使用索引器得到 different，二者均保持实际返回值。稀疏 DSA 路径继续明确未验证。
- 组合矩阵：conv/dot 及 QR/Cholesky 的固定三元组、参数反例和 NumPy 参照，5 项通过；没有用脚本生成候选。混合区域 12 项、注册求导 31 项、长链/高扇出/函数与字段拒绝 38 项专项通过。
- 适配清单：锁定版本的私有字段、区域参数类型和公开 Primitive 身份与检查入库的 JSON 一致。CI 已拆分快速、业务、压力和双/四设备任务，新增手动候选升级检查；在线工作流尚未执行。


## 最终验收与自审

环境：Python 3.12.3、JAX/jaxlib 0.11.2、CPU。pytest 默认两个逻辑 CPU；完整层报告使用单 CPU 的 Explicit data/tensor 1×1 mesh，X64 关闭、默认 matmul precision、threefry2x32。FP64 组合在自己的 X64 上下文中验证。

| 验证 | 结果 |
| --- | --- |
| `uv sync --locked` | 通过，依赖锁文件没有变化 |
| `uv run pytest -q` | **1078 passed**，338.61 秒；1 条来自原始上游 Pallas reciprocal API 的弃用警告 |
| 函数入口与原 Flux 追踪拒绝的最终调整 | 31 passed，5.61 秒；AssertionError 正确归为 trace_rejected |
| 独立对应校验、深度与描述符拒绝专项 | 25 passed；补查相同非法图、冒名原语与未经审核版本 |
| Ruff / 格式 | 全部通过 |
| lax 清单 / 适配字段快照 | 均与锁定环境一致；候选模式在同版本完成 smoke check，不代表新版本已适配 |
| tracing / BFS / cond / 四层 cond 示例 | 均通过；四层 cond 的 9 个候选/签名、720 组检查完成 |
| 双 CPU、四 CPU 通信脚本 | 各 22 个组合通过，分别核对图与重排执行 |
| 原有业务报告 `--trials 1` | 60 个候选/签名组合通过 |
| 完整层 `--supplemental` 报告 | 44 个前向组合、132 个数值样本：20 equivalent、24 different；全部与预期一致 |
| 求导 / 索引器 / 后端补充报告 | 12 个求导模式（8 equivalent、4 different）；4 个索引器组合（2 equivalent、2 different）；8 个边界探测（6 unknown、2 trace_rejected） |
| CI 配置 | YAML 与各 run 的 Bash 语法通过；快速/业务/压力分组及双/四设备配置已写入，远端工作流未触发 |
| 文档 | 中英文 README、架构、开发记录、索引和技术债同步；本地链接与 `git diff --check` 通过 |

报告由 `uv run python -m tests.full_layers.runner --supplemental --report /tmp/jax-reorder-full-layers.json` 生成；JSON 保存来源、配置、签名/输入摘要、模式、预期/实际、差异路径、拒绝原因和重放所需 seed。全部检查的等价/图差异标签均命中预期，unknown 和 JAX 拒绝不计作等价成功。详细覆盖表见[完整层记录](validation/full-attention.md)与[验证矩阵](../../design-docs/current/verification-matrix.md)。

自审已补强：原图证据独立检查 SSA/支持域；上下文描述符子类不能隐藏继承字段；symbolic 模式重新物化保持 fwd 残差树一致；动态规则捕获须还原真实输入边；延迟有序规则不能漏掉父级顺序。前向及 JVP 一致但 VJP 累加顺序改变的业务实例被保留为正确图差异，没有放宽 add_any 或应用代数规则。

剩余限制已转入[待处理清单](../index.md#待处理)：规则专有捕获/日志/有序 effects、高阶变换、硬件内核、原上游拒绝、稀疏及其他业务配置、持久格式和后续版本审核。归档表示本计划中明确范围的实施和系统验证完成，不表示所有 JAX 代码或任意组合均已证明。未执行 Git 提交或推送。

## 已解决问题编号

以下编号于 2026-10-26 从问题总表归入本计划，保留原已解决状态、优先级和依据；本次文档合并未重跑历史验收。

| 编号 | 问题 | 原优先级 | 状态 | 依据 |
| --- | --- | --- | --- | --- |
| BFS-001 | 未参与输出的纯子图、常量及对称结构已加入完整图细化与精确搜索；预算不足抛出分类异常 | P2 | 已解决 | [完整图设计及测试](../../design-docs/current/complete-graph.md) |
| BFS-002 | 已增加函数级 pytree、静态签名、环境及分模式比较；纯函数与未声明 Python 行为仍由调用方约定 | P2 | 已解决 | [函数契约](../../design-docs/current/function-and-derivative-contract.md) |
| BFS-005 | cond 未使用槽位、scan/while 捕获与反馈、jit/shard_map 私有输入输出已联合规范化 | P1 | 已解决 | [区域绑定](../../design-docs/current/region-bindings.md)、[系统验证计划](graph-equivalence-validation.md) |
| BFS-006 | custom JVP/VJP 注册一阶规则、残差、透传、symbolic zeros 已适配；剩余边界见 AD-001/002 | P1 | 已解决 | [求导契约与测试](../../design-docs/current/function-and-derivative-contract.md) |
