"""符号层读取的最小图接口，兼容 JAX 图与隔离 worker 的冻结对象。"""

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
