# 核心库类型与异常分类

本次工作覆盖 `src/jax_reorder` 的函数签名、类字段、容器元素和回调接口。运行基线为 Python 3.12。执行与验证见[任务记录](../exec-plans/completed/type-annotations.md)。比较算法及其适用范围不因类型标注扩大。

## 异常阶段与失败分类

`ComparisonStage` 和 `ComparisonFailureKind` 使用 `StrEnum`，从包顶层导出。异常的默认字段分别声明为 `ClassVar[ComparisonStage]` 和 `ClassVar[ComparisonFailureKind]`；实例的 `stage`、`failure_kind` 也保存枚举成员。

```python
from typing import ClassVar

from jax_reorder import ComparisonError, ComparisonFailureKind, ComparisonStage


class CustomTraceError(ComparisonError):
    default_stage: ClassVar[ComparisonStage] = ComparisonStage.TRACE
    default_kind: ClassVar[ComparisonFailureKind] = ComparisonFailureKind.TRACE_REJECTED
```

扩展异常类不要求使用任意字符串：子类可选择已有枚举成员。枚举的成员集合是封闭的，新增阶段或分类时需要在 `errors.py` 集中添加成员，不能通过继承一个已有成员的枚举来追加成员。

构造器仍接受已声明的字符串值，并立即转换为枚举；原有的字符串比较、错误展示、JSON 文本值和 worker 传输保持兼容。未知字符串现在抛出 `ValueError`，防止拼写错误静默进入分类；省略值或空字符串沿用原来的默认值行为。`at()` 返回 `Self`，因此调用方可以推导出重定位后的具体异常子类。

`path`、`reason` 是字符串；`diagnostics` 是 `dict[str, object]`，保存任意附加诊断。读取异构诊断后需要按实际载荷缩窄类型，不能假设所有错误都有同一组查询字段。

## 类型边界

- 结构层明确图、节点、值引用、区域来源、语义参数及槽位置换的容器类型。内部生成器分别标明子区域请求、送回结果和最终返回值。
- 符号层通过最小 `Protocol` 描述冻结图、方程、数组抽象值和字面量；JAX 图到 worker 的转换仍保留原有载荷及语义检查。
- 函数入口接受任意 Python pytree，位置参数、关键字参数和用户函数返回值保留明确的 `Any` 边界；抽象输入和比较结果使用具体类型。
- 异构 primitive 参数、精确编码值和 NumPy 对象数组仍有动态部分；类型标注不替代已存在的运行时校验。填充后才完整的槽位表及动态扩展模块对象在已有不变量成立的位置使用 `cast`，不改变值。
- 仅用于标注的跨模块引用放入 `TYPE_CHECKING`。包测试只对运行时导入做环检测，层次约束仍检查包含类型引用在内的全部依赖。

## 持续检查与验证范围

```bash
uv run ruff check .
uv run ruff format --check .
uv run basedpyright
uv run python examples/inspect_jaxpr.py
uv run pytest
```

Ruff 的 `ANN` 规则覆盖核心库的参数和返回值；测试、示例及实验工具不做批量注解迁移。`basedpyright` 使用 basic 模式，并额外拒绝缺失参数类型和裸泛型；它同时检查 `tests/typing/public_api.py` 的调用方类型断言，包含异常 `Self`、枚举字段和各比较入口的返回类型。包内 `py.typed` 随 wheel 发布，使安装后的调用方可以读取内联标注。

SymPy 未发布完整类型契约，检查配置不从其动态构造器源码推测类型；不承诺静态验证每个 SymPy 调用。锁定 JAX 私有接口的可用性继续由运行时版本守卫和回归测试负责。静态类型检查、运行时比较语义、数值验证仍是不同证据。

回归覆盖枚举与字符串/JSON/pickle 的兼容、非法分类拒绝、异常子类重定位、诊断载荷保留及 worker 错误重建；全库比较回归用于确认标注和必要的局部变量调整未改变现有结论。
