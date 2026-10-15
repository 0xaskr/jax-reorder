# CAS 后端评估：SymPy、SymEngine、SageMath、Maxima

2026-10-15。目标是证明 JAX 图的数学表达和索引拓扑关系；本轮没有执行模型、随机抽样或比较浮点输出。结论仅覆盖下表中的符号子问题，不能替代完整 layer 比较。

## 结论与证据等级

| 候选 | 证据 | 可以复用 | 本轮结论 |
| --- | --- | --- | --- |
| SymPy 1.14.0 | 隔离安装、实际运行、关键源码核对 | 精确多项式规范化、函数恒等式、分段假设、符号矩阵及指标求和转换 | CAS 线首选局部后端；上游 batch 数组转换报错，但本次有限索引适配器已证明真实 QK/PV helper，尚非完整 GLM 证明 |
| SymEngine 0.14.1 | 隔离安装、实际运行、Python API 检查 | C++ 标量表达式规范化，Python/C 等接口 | 可作为将来的标量性能后端；当前没有测出超出 SymPy 的证明能力，不能据此换掉索引 IR |
| SageMath | 官方文档核对；本机无 `sage`，未安装 | 多个 CAS 的集合、符号化简、有限秩张量模块 | 本轮未发现相对直接接 SymPy 的明确收益；重量级运行环境，不列首轮集成 |
| Maxima | 官方 5.49.0 手册核对；本机无 `maxima`，未安装 | 有理式和函数化简、符号求和、`itensor` 指标运算 | 有复用价值但需专门转换；不是现成机器学习算子图等价判定器，保留备选 |

SymPy 已有张量表达式，不能把它简单归为“只能做标量”。但“能表示张量”也不意味着能判定所有 JAX 图。其他候选尚未运行的能力都不计为探针通过。

## 可复现环境与命令

探针：[cas_probe.py](../../../tools/experiments/equivalence_backends/cas_probe.py)。环境为 CPython 3.12.3、SymPy 1.14.0、SymEngine 0.14.1、mpmath 1.3.0；环境和结果仅位于 `/tmp`，未修改仓库依赖或锁文件。

```bash
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv venv --python .venv/bin/python /tmp/jax-reorder-equivalence-eval/cas/venv
UV_CACHE_DIR=/tmp/jax-reorder-uv-cache uv pip install --python /tmp/jax-reorder-equivalence-eval/cas/venv/bin/python sympy==1.14.0 symengine==0.14.1
/tmp/jax-reorder-equivalence-eval/cas/venv/bin/python tools/experiments/equivalence_backends/cas_probe.py > /tmp/jax-reorder-equivalence-eval/cas/result.json
```

实际安装命令未固定版本，解析得到上述版本；此处固定版本便于复现。首次安装遇到沙盒 DNS 限制，经网络审批后安装成功。探针最终退出码为 0。安装问题不计为算法失败。

脚本只调用明确的符号变换，禁用 `Expr.equals` 与随机探测 `_random`，没有输入模型数组或调用 JAX。SymPy 精确变换内部为排列常数而调用 `evalf` 与“以浮点采样判等”是不同的路径；本探针仅以前者得到的符号结果作为后续规范化表达式，判定证据是精确零、多项式系数、形状一致的收缩表示和符号条件。

## 实测结果

| 案例 | SymPy 结果 | 证据与范围 |
| --- | --- | --- |
| E2 分配律 | 等价 | `expand((x+y)*z-x*z-y*z)` 为精确零 |
| E3 dot 缩放移动 | 等价 | 两份符号矩阵表达式均规范为 `scale*Q*K.T`；维度是任意正整数符号 |
| E4 二维 PV 与逐项乘积求和 | 等价 | `Sum(P[i,j]*V[j,k], j=0..tokens-1)` 转为 `P*V`，数组形式均为 `ArrayContraction(ArrayTensorProduct(P,V),(1,2))` |
| E4 实际 batch 形式 | unknown：转换失败 | `P[b,h,n]`、`V[b,n,d]` 的 `Sum(P[t,i,j]*V[t,j,k], j=0..n-1)` 转换抛出 `ValueError: contracting indices of different dimensions` |
| E5 exp 加法恒等式 | 等价 | `expand_power_exp(exp(x+y))-exp(x)*exp(y)` 为零 |
| E6 有限 logits 的统一平移 | 条件等价 | 含符号长度 `Sum` 的 softmax 分式差经 `cancel` 为零；要求非空有限行、有限实数 logits/shift、分母非零 |
| E6 完整 mask/all-mask | unknown：未实现完整语义转换 | 确认 `exp(-oo)=0`、`exp(-oo-(-oo))=nan`、外层 false 分支能选择零；这三个常量事实不能替代完整 masked-softmax 证明 |
| N1 缺少缩放 | 数学不等价，局部表达式 | 差为非零多项式 `(scale-1)*x*y`，实变量独立时不恒等于零 |
| N2 错误收缩轴 | 数学不等价，局部表达式 | 二阶符号矩阵 `P*V-P*V.T` 某分量为 `p01*(v10-v01)`，是非零多项式 |
| N3 去掉 ReLU | 数学不等价，局部表达式 | 在符号条件 `x<0` 上差为 `-x>0` |
| N4 包含长度边界 | 数学不等价，局部谓词 | 取符号索引 `i=length`、容量 `length+1`，`i<length` 为 false，`i<=length` 为 true |
| N5 exp 错变成相加 | 数学不等价，局部表达式 | 差的混合二阶导数为严格正的 `exp(x+y)`；恒零光滑函数的对应导数必须为零 |
| U1 不透明 primitive | unknown：缺失算子语义 | 未知函数的分配关系保留为表达式，不因未化为零而判不等价 |

E3/E4 正例没有枚举维度、创建稠密矩阵或调用 `as_explicit()`。N2 使用有限大小的**符号反例族**，目的是否定普遍恒等式；这不构成任意维度正例证明，也没有用数值样本比较输出。

N1–N5 是共同集中的局部数学负例，不是已证明 GLM 的五个完整负例。中间节点不等价可能被下游乘零、mask、top-k 等消去，不能把局部不等价直接提升为整个 layer 不等价。E1 依赖/调度关系仍由仓库结构比较器承担。

SymEngine 实测 E2、E5 为零，`exp(-oo)=0`。安装版本的顶层 Python API 有 `DenseMatrix`、`Piecewise`、`Max`，没有 `MatrixSymbol` 或 `Sum`；E3/E4 在本探针中为 unknown，未通过固定维度展开冒充符号形状支持。`Symbol('assumed_real', real=True).is_real` 实测为 `None`，因此不能默认 SymPy 的假设构造方式可原样迁移。

## 追加：真实 JAX → SymPy 端到端探针

[cas_jaxpr_probe.py](../../../tools/experiments/equivalence_backends/cas_jaxpr_probe.py) 直接导入 [GLM 脚本](../../../examples/glm52_decode_dsa_mla.py) 的 `_qk`、`_qk_scale_first`、`_qk_without_scale`、`_pv`、`_pv_reduce`，以 `glm` profile 的真实 helper shape 创建 `ShapeDtypeStruct`。实际经过 `jax.make_jaxpr`，读取原始方程的输入输出绑定、shape、dtype、`dimension_numbers`、`axes` 和 `broadcast_dimensions`，再构造索引表达式。这里没有把手写数学式冒充转换结果。

```bash
PYTHONPATH=/tmp/jax-reorder-equivalence-eval/cas/venv/lib/python3.12/site-packages .venv/bin/python -m tools.experiments.equivalence_backends.cas_jaxpr_probe > /tmp/jax-reorder-equivalence-eval/cas/jaxpr-result.json
```

实际环境为项目 JAX 0.11.2 与临时环境 SymPy 1.14.0；结果 JSON 包含每份真实 jaxpr、primitive 计数、输出契约及转换表达式。脚本退出 0，Ruff 与格式检查通过。tracing、转换、比较期间均阻断 JAX 编译与 eager primitive 执行，同时阻断 SymPy `equals`、`_random`。

| 真实输入函数对 | 抽象输入 shape | 实际结论 |
| --- | --- | --- |
| `_qk` / `_qk_scale_first`，两边使用实际 `layer.SCALE=1/16` | Q `(2,64,576)`，K `(2,2048,576)` | 等价；两边均为 `Sum(Q[t,h,j]*K[t,k,j], j=0..575)/16` |
| `_pv` / `_pv_reduce` | P `(2,64,2048)`，V `(2,2048,512)` | 等价；两边均为 `Sum(P[t,h,j]*V[t,j,d], j=0..2047)` |
| `_qk` / `_qk_without_scale` | 同 QK shape | helper 数学不等价；将转换结果限制在均匀符号输入族 Q=x、K=y 后，差为非零多项式 `-540*x*y` |
| 原始 JAX 分配律函数对 | 三个 `(2,2)` 输入 | 等价；真实 add/mul 图转成同一个符号表达式 |
| `jax.numpy.exp` / 恒等函数 | `(2,2)` | unknown，明确拒绝不在适配器范围的 `exp` |
| 空输出的 `x` / `2*x` | `(0,)` | unknown，拒绝空索引域；不会因形式分量不同误报不等价 |
| 返回常数 1 / 返回常数 2 | 标量参数 | 数学不等价，精确常量差为 -1；无需多项式生成元 |

这个适配器是**本次手写的有限转换器**，不是 SymPy 上游自带 JAX 支持。它将 dot 的 batch 轴先绑定为输出自由索引，再依次绑定左右未收缩轴；收缩轴和 reduce 轴使用独立 `Dummy` 绑定，广播保留轴映射及 singleton 索引。SymPy 复用的是展开、提取与求和指标无关的因子、绑定变量规范化，最终比较精确索引表达式。上节失败的是 SymPy 的 `convert_indexed_to_array` 路线；本节直接保留 `Sum`，没有使用或修复该接口。

正例完全保留 576/2048 长度求和，不枚举模型元素。负例只在转换后的表达式上限制一个合法符号输入族，并将常量被加数的求和闭合为长度乘积；没有分配 Q/K 数组或抽样执行。它证明 QK helper 的输出函数不同，**不能证明完整 attention layer 不同**：该均匀族使各 key 的 logits 相同，后续 softmax 可能消除其变化，因此不能直接作为整层反例。

支持域限制为正大小静态维度（允许标量）、无 effects、无捕获数组、前向、float32 按有限实数解释的 `add`、`mul`、`broadcast_in_dim`、`dot_general`、`reduce_sum` 与同 dtype 转换；dtype 改变、其他 primitive、嵌套区域均拒绝。没有证明扩展实数 mask、全遮蔽行、真实浮点行为或求导契约。输入是扁平数组参数，本次只比较已追踪签名；不宣称任意 Python 输入结构或任意程序的等价。

交叉审查发现并复现：初版对 `(0,)` 的 `x` 与 `2*x` 仍构造形式输出分量，负例逻辑因此误报不等价，虽然实际两个输出均为空张量。根因是缺少“该输出索引存在”的前提。最终版本在转换入口拒绝非正大小/非静态维度，并加入上述拒绝回归；GLM 的正维度结果保持不变。本轮选择保守 unknown，没有实现空输出等价的专门规则。

第二个边界是负例化简成非零常数时没有多项式生成元，初版直接构造 `Poly` 会抛 `GeneratorsNeeded`。最终版本直接检查精确有理常量，并保留返回 1/2 的回归；若选定的符号输入族产生零残差，只说明该族未区分函数，返回 unknown。

## 各方案的语义与复用边界

### SymPy

上游提供 `MatrixExpr.from_index_summation`、`ArraySymbol`、`ArrayTensorProduct`、`ArrayContraction`、`ArrayDiagonal`、`PermuteDims` 以及指标/矩阵/数组互转。E4 使用上游定义与转换器，没有注册 `PV=sum(P*V)` 作为待证公理。来源：[矩阵表达式](https://docs.sympy.org/latest/modules/matrices/expressions.html)、[数组表达式](https://docs.sympy.org/latest/modules/tensor/array_expressions.html)。

上游数组转换的 batch 失败表明不能直接依赖其全自动互转。追加探针已经通过本次手写的有限适配器，将共享 batch 保留为自由索引，直接形成 `Sum`，从真实 QK/PV jaxpr 得到证明。这项转换工作属于本仓库新增内容；通用数组转换、任意 reshape、区域及选择算子仍未覆盖。

不能把 `simplify(a-b)!=0` 解释为不等价。官方明确说明通用 `simplify` 是启发式，不保证输出的特定规范形；应选择 `Poly`、`expand`、`cancel`、受约束的函数恒等变换等确定接口。[最佳实践](https://docs.sympy.org/latest/explanation/best-practices.html#avoid-simplify)

也不能直接把 `Expr.equals()` 当成“纯符号证明 API”。安装源码的 `equals` 调用 `is_constant`，后者存在随机替换和近似求值路径，官方源码也描述了 `evalf` 行为。本探针显式禁用这两个 API；这不意味着 SymPy 所有内部计算都被形式化验证。[Expr 源码](https://github.com/sympy/sympy/blob/1.14.0/sympy/core/expr.py)

定义域必须由转换器维护。探针故意保留未求值的 `x/x` 再调用 `cancel`，结果为 1，但输出没有同时携带原有 `x!=0` 条件。变量假设、分段条件、张量元素有限性、求和范围、原始分母非零条件都应进入证明义务，不能把成功约分当作条件已被证明。

### SymEngine

上游定位是 C++ 符号表达式库，提供多种语言接口，Python bindings 可通过 wheel 安装。本次没有编译 C++ 核心，也没有比较性能。现有证据支持将它作为局部标量表达式后端，而非完整的 JAX/张量图后端。[核心仓库](https://github.com/symengine/symengine)、[Python bindings](https://github.com/symengine/symengine.py)

上面的缺失 API 是安装版 Python 接口检查结果，不声称整个 C++ 项目无法扩展张量语义。若为任意 shape 自行实现 Sum、索引绑定和收缩，主要缺口仍由本仓库承担；“用 C++ 重写”本身不会增加证明能力。分段函数存在也不代表内置了 JAX select、NaN、top-k 或循环语义。

### SageMath

官方 `simplify_full` 实际依序调用数种专门化简器；名称中的 full 不代表完备判等。符号假设必须保留，官方示例里 `bool(x>3)` 在已知 `x>2` 的情况下也返回 false，因为无法确认，并非已证明相反关系。[符号表达式](https://doc.sagemath.org/html/en/reference/calculus/sage/symbolic/expression.html)、[假设接口](https://doc.sagemath.org/html/en/reference/calculus/sage/symbolic/assumptions.html)

SageMath 还有有限秩自由模上的指标张量接口，可以处理收缩等数学运算，不能据此说它“没有张量能力”；但本轮没有发现能接收当前 JAX primitive 图并保留 mask/区域的现成前端。[指标张量接口](https://doc.sagemath.org/html/en/reference/tensor_free_modules/sage/tensor/modules/tensor_with_indices.html)

Sage 部分符号功能组合 Pynac、Maxima、SymPy 等后端，部署可以使用 conda 或完整发行版；源码发行版依赖规模较大。因此本轮没有为重复标量能力安装完整系统，证据等级是文档核对，不是安装失败或能力失败。[符号后端说明](https://doc.sagemath.org/html/en/reference/manifolds/sage/manifolds/calculus_method.html)、[安装说明](https://doc.sagemath.org/html/en/installation/index.html)、[源码发行版依赖](https://doc.sagemath.org/html/en/installation/source-distro.html)

### Maxima

Maxima 通过 `ratsimp` 等接口做专门化简，`sum` 可以保留符号求和，`sumcontract`、`intosum` 等可以操作求和结构。`itensor` 直接处理自由/哑指标及收缩，不必先展开所有分量；它主要面向几何张量代数，JAX 的混合 shape、广播、batch 轴和神经网络算子仍需映射。[求和手册](https://maxima.sourceforge.io/docs/manual/Sums-Products-and-Series.html)、[itensor 手册](https://maxima.sourceforge.io/docs/manual/Package-itensor.html)

Maxima 的结构 `=` 与数学谓词 `equal` 不同；不确定的关系可以通过 `maybe` 或关闭 `prederror` 的 `is` 返回 unknown。不能用结构 false 作为数学不等价，`is(equal(a,b))` 也不是全体实函数的完备算法。[运算符](https://maxima.sourceforge.io/docs/manual/Operators.html)、[事实与假设](https://maxima.sourceforge.io/docs/manual/Maxima_0027s-Database.html)

本机不存在 Maxima CLI，此次没有安装 Lisp 运行时、构建 Maxima 或运行探针。`itensor` 的适配可列为后续专门调查，但目前没有优于 SymPy 实测结果的证据。

## 对真实 GLM 图的覆盖

主工作线通过抽象 tracing 生成的清单记录：JAX 0.11.2，原始图递归出现 196 条方程，原图与十个改写共 33 种 primitive；未编译或执行模型。CAS 探针对其中的数学子表达式有如下关系：

| GLM 部分 | 本轮可复用结果 | 仍缺少的转换/证明 |
| --- | --- | --- |
| `dot_general`、`mul`、`reduce_sum` | 实际 GLM shape 的 QK/PV helper 经有限 jaxpr 转换后证明；二维符号维度探针也通过 | 上游 batch 数组转换报错；本次适配器尚未覆盖完整 layer 与非有限值 |
| `broadcast_in_dim`、reshape/squeeze | 真实 `_pv_reduce` 的广播映射已转换成功 | 未实现通用 reshape/squeeze 索引映射 |
| `exp`、div、`reduce_max`、`select_n` | 有限实数 exp 与 softmax shift；局部 Piecewise 条件 | mask、全遮蔽、最大值条件、分母有效性不能抹掉 |
| gather、dynamic slice/update、比较、iota/rem | 符号索引与边界关系可作为局部条件 | 需显式边界/填充值/动态更新语义与域约束 |
| `top_k`、`approx_top_k` | 未验证 | 顺序、并列、近似选择的语义不能只替换为不透明函数后声称完整证明 |
| scan、jit、`custom_jvp_call`、stop_gradient | 无现成整图接入 | 区域参数绑定、循环关系以及求导契约由仓库保留并组合证明 |

建议的最小接入范围是 **SymPy 作为现有索引 IR 的可选局部后端**：先复用多项式/exp/指标收缩，返回显式条件与规范化表达式；以本轮真实 QK/PV helper 的有限转换为接入证据，继续验证 mask、选择算子和区域。不支持时保留具体 unknown。局部等价可在已建立的同上下文中组合；局部不等价要继续追到最终输出，不能提前结束整图判定。本轮不建议为获得另一套 `simplify` 同时引入 SageMath、Maxima 或 SymEngine。
