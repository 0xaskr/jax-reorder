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
仅对等价结果重放；重放重新提取源绑定和相同规则，检查证据未被篡改，不冒充独立证明算法。
执行绑定通过现有 `execution` 参数传入；含义与限制见[通信](communication.md)。
源码入口：[local.py](../../../src/jax_reorder/core/local.py)、
[normalization.py](../../../src/jax_reorder/core/normalization.py)。

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

## 按需证据

- 接口与外层来源正反例：[阶段 13](history/stage-13.md)。
- 完整结构比较、effects 和真实开源候选：[阶段 8](history/stage-08.md)。
- 纯值投影和公共 SymPy 接入：[阶段 9](history/stage-09.md)。
- kernel 内原始 `jnp.where` / reshard 条件：[阶段 26](history/stage-26.md)。
- 早期数据模型和接口草案：[原始架构](history/design-03.md)，仅用于追溯取舍。
