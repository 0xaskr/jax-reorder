"""给原语参数提供一处统一的“比较视图”。

比较编码、对应校验和代数冻结都要看同一份参数，各写一份迟早会走样，所以集中
到这里：parameter_view 输出可精确比较的编码，并把嵌套子图登记成路径；
semantic_parameters 处理自定义求导调用的参数，需要时物化一阶规则。
本模块属于 core，不依赖 algebra 或 functions，避免上层去 import 别的模块的
私有辅助函数。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import jax
from jax import lax
from jax._src import core, pjit, shard_map
from jax._src import custom_derivatives as cd

from ..errors import ComparisonError, InvalidJaxprError, UnsupportedJaxprError
from ._typing import EncodedRegion, Parameters
from .derivatives import CUSTOM_PRIMITIVES, _check_rule_order, _jvp_rules, _vjp_rules
from .semantics import Encoder, Frozen


def parameter_view(
    equation: core.JaxprEqn,
    variable: Callable[[core.Var], Frozen],
    *,
    inputs: Sequence[int] | None = None,
    outputs: Sequence[int] | None = None,
    parameters: Mapping[str, Any] | None = None,
    include_derivative_rules: bool = True,
    parent: core.Jaxpr | None = None,
) -> tuple[tuple[tuple[str, Frozen], ...], dict[str, core.Jaxpr]]:
    """保留参数内容，用子图路径作占位并另行交给图校验器。

    不读取规范图的参数编码。jit 名称和 reduce callable 是锁定版本中
    已审查的两个例外；其余值一律由完整编码检查。parameters 可传入已物化
    的语义视图；parent 提供捕获常量所在的原始区域；
    include_derivative_rules 控制要不要物化 custom_jvp / custom_vjp 的规则。

    inputs/outputs 是从左侧槽位到当前右侧槽位的映射，只用于把右侧的逐位置
    元数据投影到左侧顺序。返回参数编码及“参数路径 → 原始子图”的映射。
    """
    regions = {}

    def nested(graph: core.Jaxpr, path: str) -> EncodedRegion:
        """登记子图及其路径，让参数编码与子图内容校验分开进行。"""
        # 父参数只记录子图所在位置，内容随后逐区域核对；占位不意味着忽略子图。
        regions[path] = graph
        return cast(EncodedRegion, SimpleNamespace(graph=("region", path)))

    encoder = Encoder(nested, variable)
    values = []
    params = (
        semantic_parameters(
            equation, include_derivative_rules=include_derivative_rules, parent=parent
        )
        if parameters is None
        else parameters
    )
    value: Any
    for key, value in sorted(params.items()):
        if equation.primitive is pjit.jit_p and key == "name":
            continue
        # 逐位置元数据必须随槽位置换同步投影；inputs/outputs 为 None 时不做
        # 投影，父调用映射由上层另行校验。
        if equation.primitive is pjit.jit_p:
            if inputs is not None and key in {"in_shardings", "in_layouts", "donated_invars"}:
                value = tuple(value[i] for i in inputs)
            if outputs is not None and key in {"out_shardings", "out_layouts"}:
                value = tuple(value[i] for i in outputs)
        if equation.primitive is shard_map.shard_map_p:
            if inputs is not None and key == "in_specs":
                value = tuple(value[i] for i in inputs)
            if outputs is not None and key == "out_specs":
                value = tuple(value[i] for i in outputs)
        if equation.primitive is lax.scan_p:
            if inputs is not None and key == "ft_in":
                value = value.update(value.vals[i] for i in inputs)
            if outputs is not None and key == "ft_out":
                value = value.update(value.vals[i] for i in outputs)
        if (
            jax.__version__ == "0.11.2"
            and equation.primitive is lax.reduce_p
            and key == "computation"
            and isinstance(equation.params.get("jaxpr"), core.Jaxpr)
            and callable(value)
        ):
            # 锁定版本中 reduce 的 computation 由已追踪的 reducer 子图表达；
            # 比较该子图而不是 Python callable 身份，返回方程仍保留原 callable。
            encoded = ("represented_by_reducer_jaxpr",)
        else:
            encoded = encoder.encode(value, f"params.{key}")
        values.append((key, encoded))
    return tuple(values), regions


def semantic_parameters(
    equation: core.JaxprEqn,
    *,
    include_derivative_rules: bool = True,
    parent: core.Jaxpr | None = None,
) -> Parameters:
    """已知 callable 展开为语义图；其余原语参数原样交给通用编码器。

    返回值仅供比较使用，原方程仍保留 callable 以支持执行和 JAX 变换。
    parent 提供捕获常量的原始绑定；include_derivative_rules=False 时只比较
    普通前向调用。
    custom 调用的返回视图固定包含 call_jaxpr、num_consts、symbolic_zeros，
    为 True 时再附加物化的一阶 rules；参数模式变化时明确拒绝。
    """
    if equation.primitive not in CUSTOM_PRIMITIVES:
        return equation.params
    if jax.__version__ != "0.11.2":
        raise UnsupportedJaxprError("derivative", "custom derivative adapter requires JAX 0.11.2")
    p = equation.params
    keys = {"call_jaxpr", "num_consts", "symbolic_zeros"}
    keys |= (
        {"jvp_jaxpr_fun"}
        if equation.primitive is cd.custom_jvp_call_p
        else {"fwd_jaxpr_thunk", "bwd", "out_trees"}
    )
    # 参数增减都表示适配假设变化，不能忽略新字段后继续声称理解了完整规则。
    if p.keys() != keys:
        raise UnsupportedJaxprError("derivative", "custom derivative parameter schema changed")
    if (
        not isinstance(p["call_jaxpr"], core.Jaxpr)
        or type(p["num_consts"]) is not int
        or not 0 <= p["num_consts"] <= len(equation.invars)
        or type(p["symbolic_zeros"]) is not bool
        or len(p["call_jaxpr"].invars) != len(equation.invars)
        or len(p["call_jaxpr"].outvars) != len(equation.outvars)
    ):
        raise InvalidJaxprError("derivative", "invalid custom derivative interface")
    # 返回的是比较视图，不改 equation.params。True 时物化一阶规则；False 只
    # 比较当前调用的前向语义，供已经显式展开 JVP/VJP 的函数模式等路径使用。
    result = {name: p[name] for name in ("call_jaxpr", "num_consts", "symbolic_zeros")}
    if include_derivative_rules:
        try:
            result["rules"] = (
                _jvp_rules(p, equation, parent)
                if equation.primitive is cd.custom_jvp_call_p
                else _vjp_rules(p, equation, parent)
            )
            _check_rule_order(result["rules"])
        # 已分类的错误保留原路径；JAX 或用户规则的其他失败统一包装成不支持。
        except ComparisonError:
            raise
        except Exception as exc:
            raise UnsupportedJaxprError(
                "derivative", f"rule tracing rejected: {type(exc).__name__}: {exc}"
            ) from exc
    return result
