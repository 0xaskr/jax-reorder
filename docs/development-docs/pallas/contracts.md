# Pallas 契约与接口

从[任务入口](../pallas-jax-equivalence.md)到此；只在修改 API、外层绑定或结果语义时读本页。

## 比较路径

| 路径 | 结论及准入 |
| --- | --- |
| 完整结构比较 | 保留输入输出顺序、全部节点/参数/effects/嵌套绑定；独立纯运算可重排。未使用纯计算仍保留。 |
| 设备局部入口 | 每侧完整 jaxpr 包含一个静态 `pallas_call`，提取实参来源及必要轴环境；结论只针对该 device 的 kernel。 |
| 语义规范化 | 在地址、初始化、内存版本与资源证明成立后，产生规范化值及来源条件；不是放宽原始图匹配。 |
| 代数比较 | 对已支持的完整输出投影及定义条件调用可信 SymPy。只在 lowering 删除无副作用的纯死计算。 |

`DIFFERENT` 只表示结构不同。不能用局部表达式差异或采样不一致代替公共数学结论所需证据。
不支持/预算不足使用现有分类异常；`normalized` 是证据种类，不是新增结果状态。

## 公共设备局部接口

```python
from jax_reorder import compare_pallas_kernels, verify_pallas_kernel_comparison

# 完整父 jaxpr 保留调用实参和轴绑定；不要只传丢失环境的裸 kernel。
result = compare_pallas_kernels(left, right, check_protocol=True)
assert result.scope == "device_local"
assert result.protocol_checked
verify_pallas_kernel_comparison(left, right, result)
```

默认 `check_protocol=False` 只比较局部结构；显式请求的协议检查无法完成时抛异常。
`equivalence="semantic"` 自动完成资源检查后比较规范化的局部值图；成功的
`equivalence_kind` 为 `structural` 或 `normalized`，`different` 仍不表示数学不等价。
仅对等价结果重放；重放重新提取源绑定和相同规则，检查证据未被篡改，不冒充独立证明算法。
执行绑定通过现有 `execution` 参数传入；含义与限制见[通信](communication.md)。
源码入口：[local.py](../../../src/jax_reorder/core/local.py)、
[normalization.py](../../../src/jax_reorder/core/normalization.py)。

公开数学入口 `compare_algebraic_jaxprs(..., execution=...)` 接受同一绑定或
`(left_binding, right_binding)`；`compare_functions` / `compare_function_traces`
在 `equivalence="auto"` 的 forward 比较中传递该条件。显式绑定先完成协议校验，
不能因结构相同而跳过；结果的 `normalizations` 和 `assumptions` 保留执行前提。
绑定不配置调度，不适用于普通 JAX、未接入该证明的本地 kernel 或求导模式；这些情况明确拒绝。

设备局部数学比较使用 `compare_algebraic_jaxprs(left, right, scope="device_local")`；
`compare_functions` / `compare_function_traces` 在 forward 模式接受相同 `scope`。
默认 `scope="function"` 保持完整父函数契约。局部结果带显式 `scope`，始终先验证
kernel 资源义务，再报告结构、规范化或数学结果。父函数的输出包装、裁剪及后处理
不属于局部结论；kernel 自身的全部有序输出仍比较，实际使用的外层输入端口和计算
不能丢弃。函数入口仍核对输入 pytree、静态参数、变换和追踪环境。
局部显式多 core 绑定另外在 `core_executions` 保留左右协议与各 program 的投影证据，
协议的进展假设进入 `assumptions`；结构快路径和数学路径遵循相同规则。

局部轴有有限绑定时，数学适配按该范围实例化，防止反例使用不存在的 device_id。
远端局部投影保留全局输入分片来源及对端身份，输出按 `(kernel_output_port, device)`
展开并在 `normalizations[*].remote.local_outputs` 留证；不要求父函数的 replicated
out_specs 在所有设备上给出同一值，也不要求全局输出拼接成立。协议、DMA source
版本和真实提交义务仍全部验证。远端 computed shard input 缺少来源证明时明确拒绝。

## 环境与来源

- Python/NumPy 静态标量、精确数组实参和外层 JAX 纯运算保留其内容、端口和依赖。
  捕获常量不能变成任意独立输入；外层无关计算不属于局部结论。
- mesh 可多维；无关拓扑不额外进入局部键，但影响值、地址、分支或路由的轴身份、顺序、大小不能删除。
  `device_id` 可以是符号量，只有所需证明依赖其范围时才要求取得范围/相关对端证据。
- `(device_id + 1) % mesh_size` 与 `device_id + 1 % mesh_size` 是不同源表达式，不替用户修正。
- 全函数结构入口仍有独立的完整 mesh/函数契约；局部与全函数结论不得互换。
- kernel 内 replicated `reshard` 仅接受绑定的全 Manual mesh、无设备/分区重排及不变的类型/值元数据。
  源操作和参数仍保留，core 再验证后才能消去局部值映射；普通外层 reshard 不因此获准。

## 分层边界

`core/adapters` 唯一解释 JAX/Pallas 源对象与私有 schema；`ProgramView` 保存计算、区域、资源与来源事实。
Graph、存储/协议分析和数学投影读取同一视图；algebra 不反向解析 GridMapping/Ref 来补语义。
地址与初始化证明完成后才消去状态；浮点代数重排仍报告警告，不承诺逐位相同或误差很小。

scalar-prefetch 参与串行输入/输出 `index_map` 时，首步读初值并绑定窗口；当前 kernel
完成状态更新后才计算下一 program 的映射，本轮已绑定的读写地址不回溯改变。
输出逐轮检查覆盖、窗口切换及已有 ANY 恢复规则，多个输出分别维护生命周期。
该条件进入公开结果的 `assumptions`，不承诺默认自动流水线执行了同一预取时序；
[实现与验证](../../exec-plans/completed/pallas-output-map-and-pure-lowering.md)按需查阅。

纯计算是否可在数学 lowering 删减，取决于其结果是否被输出或副作用消费。
未使用的 `counter & 1` 可以删；用它选择实际输入块则必须保留地址依赖。
删减不能跳过读取初始化、地址合法性或通信义务检查。

本地 DMA 的只读输入源允许有界运行时页号：每笔 start 捕获自己的页号、候选源值及
版本，wait 提交该次捕获的值；多笔搬运可同时在途。候选窗口的保护足迹不能当作实际
搬运字节数，动态可写源/目标及未知地址范围不因此获准。数学层只对已证明范围内的
活跃整数地址作精确化简；[有界动态页 DMA](../../exec-plans/completed/pallas-bounded-page-dma.md)
记录范围、真实 paged attention 正反例与验证。

代数层可用已证明的输出坐标域及收缩输入域，证明整数坐标到 float16/bfloat16/float32/float64
的 cast 在目标格式连续精确整数区间内保持值不变；任意整数输入不能从 shape 获得数据值域。
范围不足或可能舍入时保留不透明转换与原始条件；闭合表达式含不透明函数也不能当作数值常量求值。
闭合整数常量按可容纳值的整数存储转目标 dtype，避免 Python float 中转丢失精度；
标量 cast 的广播恢复须保留定义条件和内存有效性，不改变目标格式自身的转换行为。
浮点 `iota`（含一维外层输入绑定）不能充当整数坐标，语义投影明确拒绝；
[准入修复与验证](../../exec-plans/completed/pallas-float-iota.md)按需展开。
布尔 extrema、序关系与已支持的归约遵循 `False < True` 的逻辑规则，保留所有原始
定义条件和内存义务；[布尔运算修复与复审](../../exec-plans/completed/pallas-scratch-boolean-review.md)。
相同 cond 分支仍保留 selector 义务；布尔点积按 AND/OR 收缩建模，显式数值累加类型按计数建模。
验收见[条件与点积修复](../../exec-plans/completed/pallas-cond-boolean-dot.md)。

## 按需证据

- 常量舍入及远端布尔 cast：[修复验收](../../exec-plans/completed/pallas-constant-and-boolean-casts.md)。
- 有界坐标 cast、精度边界与闭合转换：[修复验收](../../exec-plans/completed/pallas-bounded-integer-casts.md)。
- 一维收缩范围与公开执行绑定修复：[审查修复验收](../../exec-plans/completed/pallas-review-fixes.md)。
- 接口与外层来源正反例：[阶段 13](history/stage-13.md)。
- 完整结构比较、effects 和真实开源候选：[阶段 8](history/stage-08.md)。
- 纯值投影和公共 SymPy 接入：[阶段 9](history/stage-09.md)。
- kernel 内原始 `jnp.where` / reshard 条件：[阶段 26](history/stage-26.md)。
- 早期数据模型和接口草案：[原始架构](history/design-03.md)，仅用于追溯取舍。
