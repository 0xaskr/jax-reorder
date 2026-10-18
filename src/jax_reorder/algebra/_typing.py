"""符号层读取图所需的最小接口，同时兼容真实 JAX 图和 worker 里的冻结对象。

用 Protocol 描述字段，是为了让 builder 和隔离 worker 共用同一套读取代码；
这里只声明形状，不包含任何逻辑。
"""

from collections.abc import Sequence
from typing import Any, Protocol

from jax.typing import DTypeLike
from numpy.typing import ArrayLike, NDArray

from ..core._typing import Parameters

type Shape = tuple[int, ...]
type ObjectArray = NDArray[Any]


class ArrayAval(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def dtype(self) -> DTypeLike: ...


class ArrayAtom(Protocol):
    @property
    def aval(self) -> ArrayAval: ...


class ArrayLiteral(ArrayAtom, Protocol):
    @property
    def val(self) -> ArrayLike: ...


class NamedPrimitive(Protocol):
    @property
    def name(self) -> str: ...


class Equation(Protocol):
    @property
    def primitive(self) -> NamedPrimitive: ...

    @property
    def params(self) -> Parameters: ...

    @property
    def invars(self) -> Sequence[ArrayAtom]: ...

    @property
    def outvars(self) -> Sequence[ArrayAtom]: ...


class SymbolicGraph(Protocol):
    @property
    def invars(self) -> Sequence[ArrayAtom]: ...

    @property
    def constvars(self) -> Sequence[ArrayAtom]: ...

    @property
    def consts(self) -> Sequence[ArrayLike]: ...

    @property
    def outvars(self) -> Sequence[ArrayAtom]: ...

    @property
    def eqns(self) -> Sequence[Equation]: ...
