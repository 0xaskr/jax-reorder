"""把已校验的 jaxpr 冻结成独立符号 worker 的输入，不执行 JAX。

冻结结果只保留 SymbolicBuilder 真正会用到的字段：带稳定编号的变量描述、
方程名、已适配的参数、输入/常量/输出端口。JIT 布局、名称和编译元数据不参与
“理想实数”解释，而且 default_jit 已核对过它们是默认值，所以不用跨进程传。
无法解释或无法安全传递的语义（effects、token、非默认 scan/cast 接口、
非默认 JIT 契约）直接抛 UnsupportedJaxprError，不猜也不丢。
本模块不执行数组、不做数学判断，也不构造 SymPy 对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from jax._src import core
from jax.typing import DTypeLike

from ...core.derivatives import CUSTOM_PRIMITIVES
from ...core.params import parameter_view, semantic_parameters
from ...core.semantics import Frozen
from ...errors import UnsupportedJaxprError
from .._typing import ArrayAtom, ArrayAval, Shape, SymbolicGraph

if TYPE_CHECKING:
    from .calls import CommonCalls


@dataclass(frozen=True)
class Variable:
    """冻结图中的轻量变量描述；number 是本进程分配的稳定顺序编号。

    shape/dtype 构成 worker 侧要核对的抽象契约；aval 属性提供只含这两项的
    视图，让按 JAX aval 读取形状的接口无需依赖 core.Var。编号按首次出现
    分配，用于报告位置和跨进程对应，不暴露原始变量名。
    """

    number: int
    shape: Shape
    dtype: DTypeLike

    @property
    def aval(self) -> ArrayAval:
        """只含 shape 与 dtype 的 aval 视图，供按抽象值读取的接口使用。"""
        return cast(ArrayAval, SimpleNamespace(shape=self.shape, dtype=self.dtype))


def freeze(
    graph: core.Jaxpr,
    common_calls: CommonCalls | None = None,
    common_ids: dict[Frozen, int] | None = None,
) -> SymbolicGraph:
    """冻结一张图（ClosedJaxpr 或裸 jaxpr）为 worker 可读的普通对象。

    common_calls 是父进程构造的 CommonCalls，用于识别两侧完整相同的纯 JIT；
    common_ids 是跨两侧共享的“语义身份 → 编号”表：同一共同调用或同一
    不透明 reducer 在两张图中得到相同编号，数学层才能建立同一符号。
    变量按首次出现分配编号；字面量不占变量编号，直接携带数值与抽象值。
    嵌套子图、tuple/list/dict 参数递归冻结，不透明 callable 原样保留。
    """
    closed = hasattr(graph, "jaxpr")
    if common_ids is None:
        common_ids = {}
    inner = graph.jaxpr if closed else graph
    if inner.effects:
        raise UnsupportedJaxprError("algebra.effects", "effectful graph")
    variables = {}

    def variable(var: core.Atom) -> ArrayAtom:
        """把 Var 或字面量转换成 worker 侧的变量描述或常量值。"""
        # token 等非数组值没有实数解释，worker 也无法按元素坐标读取。
        if not isinstance(var.aval, core.ShapedArray):
            raise UnsupportedJaxprError(
                "algebra.shape", "requires shaped array values; tokens are unsupported"
            )
        if isinstance(var, core.Literal):
            return cast(ArrayAtom, SimpleNamespace(val=np.asarray(var.val), aval=var.aval))
        if var not in variables:
            variables[var] = Variable(len(variables), tuple(var.aval.shape), var.aval.dtype)
        return variables[var]

    def parameter(value: Any) -> Any:
        """递归冻结参数里的嵌套图与容器，保持其他值原样。"""
        # 只深入已知容器，其他值原样传递；不在这里猜测未知参数的语义。
        if isinstance(value, (core.Jaxpr, core.ClosedJaxpr)):
            return freeze(value, common_calls, common_ids)
        if isinstance(value, (tuple, list)):
            return tuple(parameter(x) for x in value)
        if isinstance(value, dict):
            return {key: parameter(x) for key, x in value.items()}
        return value

    inputs = tuple(variable(v) for v in inner.invars)
    constants = tuple(variable(v) for v in inner.constvars)
    equations = []
    for eq in inner.eqns:
        if eq.effects:
            raise UnsupportedJaxprError("algebra.effects", "effectful equation")
        # 取完整语义参数；下面按 primitive 只保留数学层能解释的部分。
        params: dict[str, Any] = semantic_parameters(
            eq, include_derivative_rules=False, parent=graph
        )
        if eq.primitive in CUSTOM_PRIMITIVES:
            # 自定义求导只比较实际被调用的程序；残差与 symbolic-zero 参数
            # 属于 JAX 接口元数据，不是数学函数的一部分。
            params = {"call_jaxpr": params["call_jaxpr"]}
        if eq.primitive.name == "reduce":
            from ...core.canonical import canonicalize

            # reduce 的 computation 可能是 callable 或 jaxpr。用完整参数编码
            # 加规范化后的子图组成稳定身份，并让两侧同一 reducer 共享编号；
            # 不能按函数名或对象 id 判等。
            encoded, children = parameter_view(
                eq, lambda _: None, include_derivative_rules=False, parent=graph
            )
            key = (
                "opaque_reducer",
                encoded,
                tuple(
                    (path, canonicalize(child, include_derivative_rules=False).graph)
                    for path, child in children.items()
                ),
            )
            if key not in common_ids:
                common_ids[key] = len(common_ids)
            params = {"opaque_semantic": common_ids[key]}
        if eq.primitive.name == "convert_element_type" and params.get("sharding") is not None:
            from .semantics import encode

            # 默认分片不改变布局语义，可以丢弃；非默认分片必须拒绝。
            aval = cast(core.ShapedArray, eq.outvars[0].aval)
            default = core.ShapedArray(aval.shape, aval.dtype).sharding
            if encode(params["sharding"]) != encode(default):
                raise UnsupportedJaxprError("algebra.cast", "non-default cast sharding")
            params = {**params, "sharding": None}
        if eq.primitive.name == "scan":
            # FlatTree 接口元数据不属于数学语义，只保留常量与 carry 数量。
            ft_in, ft_out = params["ft_in"], params["ft_out"]
            if (
                not ft_in.unpackable
                or not ft_out.unpackable
                or any(value is not None for value in (*ft_in.vals, *ft_out.vals))
            ):
                raise UnsupportedJaxprError("algebra.scan", "unsupported scan interface metadata")
            scan_constants, carries, _ = ft_in.unpack()
            params = {key: value for key, value in params.items() if key not in ("ft_in", "ft_out")}
            params.update(num_consts=len(scan_constants), num_carry=len(carries))
        if eq.primitive.name in ("jit", "pjit"):
            from .jit import default_jit

            if not default_jit(eq, params):
                raise UnsupportedJaxprError("algebra.jit", "non-default JIT execution contract")
            params = {"jaxpr": params["jaxpr"]}
            if common_calls is not None and eq in common_calls.invocations:
                from ..finiteness import nonfinite_graph

                key, input_ports, output_ports = common_calls.invocations[eq]
                # 子图含 -inf 等非有限值时不能写成不透明共同符号，
                # 否则会丢失特殊值语义；此类调用保留完整展开。
                if not nonfinite_graph(params["jaxpr"], common_calls.budget):
                    if key not in common_ids:
                        common_ids[key] = len(common_ids)
                    params = {**params, "common_call": (common_ids[key], input_ports, output_ports)}
        equations.append(
            SimpleNamespace(
                primitive=SimpleNamespace(name=eq.primitive.name),
                params=parameter(params),
                invars=tuple(variable(v) for v in eq.invars),
                outvars=tuple(variable(v) for v in eq.outvars),
            )
        )
    return cast(
        SymbolicGraph,
        SimpleNamespace(
            invars=inputs,
            constvars=constants,
            # 裸 jaxpr 的 constvars 只是外部绑定描述，没有可传递的常量值。
            consts=tuple(np.asarray(x) for x in graph.consts) if closed else (),
            eqns=tuple(equations),
            outvars=tuple(variable(v) for v in inner.outvars),
        ),
    )
