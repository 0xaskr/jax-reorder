"""结构层内部共用的类型别名和协议。

Parameters 里的参数值本来就是异构的（数组、描述符、回调、嵌套图），所以只在
值边界保留动态类型；这里还给编码器提供它真正读取的最小接口，避免把 JAX 的
动态导出直接当成类型定义。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

# primitive 参数随算子变化，值可能是数组、描述符、回调或嵌套图。
type Parameters = dict[str, Any]
type ValueReference = (
    tuple[Literal["input", "constant"], int]
    | tuple[Literal["result"], int, int]
    | tuple[Literal["literal"], Any, Any]
)


class EncodedRegion(Protocol):
    """Encoder 只读取区域回调结果的 graph，不依赖规范化器实现。"""

    @property
    def graph(self) -> object: ...


class TreeDef(Protocol):
    """JAX 扩展模块的 pytree 描述符接口，避免把动态导出误当作类型别名。"""

    @property
    def num_leaves(self) -> int: ...

    def node_data(self) -> tuple[type, Any] | None: ...

    def children(self) -> Sequence[TreeDef]: ...


class Device(Protocol):
    """编码器读取的设备身份字段。"""

    @property
    def platform(self) -> str: ...

    @property
    def process_index(self) -> int: ...

    @property
    def id(self) -> int: ...

    @property
    def device_kind(self) -> str: ...
