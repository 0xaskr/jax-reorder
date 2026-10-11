"""锁定 JAX 版本的自定义一阶求导适配。

普通调用、规则图和残差实际连接均参与比较；运行时参数和 callable 不变。
这里执行 JAX 延迟追踪接口，不按 callable 名称或 Python 对象身份判断等价。
"""

from functools import partial
from inspect import getclosurevars
from itertools import product

import jax
import numpy as np
from jax._src import ad_util, core
from jax._src import custom_derivatives as cd
from jax._src.interpreters import partial_eval as pe

from .errors import InvalidJaxprError, UnsupportedJaxprError

CUSTOM_PRIMITIVES = frozenset((cd.custom_jvp_call_p, cd.custom_vjp_call_p))
MAX_RULE_CASES = 64
_MEMOIZED_CODE = pe._memoize(lambda: None).__code__


def _abstract(aval):
    if not isinstance(aval, core.ShapedArray):
        raise UnsupportedJaxprError("derivative", f"unsupported rule abstract value {type(aval)}")
    return jax.ShapeDtypeStruct(
        aval.shape, aval.dtype, weak_type=aval.weak_type, sharding=aval.sharding
    )


def _differentiable(aval):
    return jax.dtypes.issubdtype(aval.dtype, np.inexact)


def _activations(avals, symbolic):
    indices = [i for i, aval in enumerate(avals) if _differentiable(aval)]
    if not indices:
        return ()
    count = 2 ** len(indices) - 1 if symbolic else 1
    if count > MAX_RULE_CASES:
        raise UnsupportedJaxprError(
            "derivative",
            f"symbolic-zero rule budget exceeded: {count} cases, limit {MAX_RULE_CASES}",
        )
    result = []
    patterns = product((False, True), repeat=len(indices)) if symbolic else [(True,) * len(indices)]
    for pattern in patterns:
        if any(pattern):
            active = dict(zip(indices, pattern, strict=True))
            result.append(tuple(active.get(i, False) for i in range(len(avals))))
    return tuple(result)


def _constant_bindings(constants, equation, parent):
    """把规则捕获的旧 tracer/数组还原到调用实参，避免把动态值当字面常量。"""
    parent_constants = (
        {}
        if parent is None
        else {id(value): var for var, value in zip(parent.constvars, parent.consts, strict=True)}
    )
    positions = {var: i for i, var in enumerate(equation.invars) if isinstance(var, core.Var)}
    result = []
    for value in constants:
        source = (
            value.val
            if isinstance(value, pe.DynamicJaxprTracer)
            else parent_constants.get(id(value))
        )
        if isinstance(source, core.Var):
            if source not in positions:
                raise UnsupportedJaxprError(
                    "derivative", "rule captures a dynamic value outside the custom call operands"
                )
            result.append(("input", positions[source]))
        elif isinstance(value, core.Tracer):
            raise UnsupportedJaxprError(
                "derivative", "unreviewed captured tracer in derivative rule"
            )
        else:
            result.append(("constant", value))
    return tuple(result)


def _constant_values(bindings, primals):
    return tuple(primals[value] if kind == "input" else value for kind, value in bindings)


def _check_rule_order(rules):
    """延迟规则的隐式 effects 不会出现在原调用的 effect 集中。

    原父图的顺序校验当前只覆盖已经追踪的执行图。规则新增的有序操作
    必须显式拒绝，避免交换 custom 调用后隐含改变求导 effect 顺序。
    显式 JVP/VJP 函数入口会追踪实际执行图，可沿普通 effect 路径检查。
    """
    work, seen = [rules], set()
    while work:
        value = work.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, core.Jaxpr):
            work.extend(value.eqns)
        elif isinstance(value, core.JaxprEqn):
            if value.primitive.name in {"rng_uniform", "dce_sink"}:
                raise UnsupportedJaxprError(
                    "derivative",
                    "latent ordered derivative rules require an effect binding adapter",
                )
            work.extend(value.params.values())
        elif isinstance(value, (tuple, list)):
            work.extend(value)
        elif type(value) is dict:
            work.extend(value.values())


def _jvp_rules(params, equation, parent):
    primal = params["call_jaxpr"]
    avals = primal.in_avals[params["num_consts"] :]
    result = []
    for active in _activations(avals, params["symbolic_zeros"]):
        graph, constants, out_zeros = params["jvp_jaxpr_fun"].call_wrapped(*(not a for a in active))
        bindings = _constant_bindings(constants, equation, parent)
        n = len(primal.invars)
        if len(graph.invars) != len(constants) + len(avals) + sum(active):
            raise InvalidJaxprError("derivative", "invalid custom JVP rule input count")

        def connected(*xs, graph=graph, bindings=bindings, n=n):
            return tuple(
                core.eval_jaxpr(
                    graph,
                    graph.consts,
                    *_constant_values(bindings, xs[:n]),
                    *xs[params["num_consts"] :],
                )
            )

        inputs = tuple(_abstract(aval) for aval in primal.in_avals) + tuple(
            _abstract(aval.to_tangent_aval()) for aval, nz in zip(avals, active, strict=True) if nz
        )
        result.append((active, tuple(out_zeros), jax.make_jaxpr(connected)(*inputs)))
    return tuple(result)


def _vjp_rules(params, equation, parent):
    primal = params["call_jaxpr"]
    nc = params["num_consts"]
    activations = _activations(primal.in_avals[nc:], params["symbolic_zeros"])
    cotangents = _activations(primal.out_avals, params["symbolic_zeros"])
    if len(activations) * len(cotangents) > MAX_RULE_CASES:
        raise UnsupportedJaxprError(
            "derivative", "combined input/cotangent symbolic-zero rule budget exceeded"
        )
    result = []
    for active in activations:
        thunk = params["fwd_jaxpr_thunk"]
        # JAX 的 memoized thunk 只缓存图，不恢复 out_trees 的辅助 store。
        # 不同 symbolic 模式可产生不同 residual 树；再次物化必须重新执行
        # 已审查的原 thunk，否则会把上一个模式的透传表配给当前图。
        if (
            getattr(thunk.f, "__code__", None) is not _MEMOIZED_CODE
            or thunk.transforms
            or thunk.params
        ):
            raise UnsupportedJaxprError("derivative", "unreviewed custom VJP forward thunk wrapper")
        uncached = getclosurevars(thunk.f).nonlocals["fn"]
        with core.set_current_trace(None):
            graph, constants = uncached(*active)
        bindings = _constant_bindings(constants, equation, parent)
        if len(graph.invars) != len(constants) + len(primal.invars) - nc:
            raise InvalidJaxprError("derivative", "invalid custom VJP forward input count")
        out_tree, residual_tree, forwards = params["out_trees"]()
        forwards = tuple(forwards)
        if len(forwards) != residual_tree.num_leaves or out_tree.num_leaves != len(primal.outvars):
            raise InvalidJaxprError("derivative", "invalid custom VJP result or residual interface")
        if any(
            f is not None and (type(f) is not int or not 0 <= f < len(primal.invars))
            for f in forwards
        ):
            raise InvalidJaxprError("derivative", "invalid residual input forwarding")
        nr = sum(f is None for f in forwards)
        if len(graph.outvars) != nr + len(primal.outvars):
            raise InvalidJaxprError("derivative", "invalid custom VJP forward result count")

        for ct_active in cotangents:
            # 两个边界显式保留。返回槽位与消费者一起比较，不能单独比较 bwd
            # 后遗漏 fwd 透传输入给 residual 的路径。
            @partial(jax.jit, keep_unused=True)
            def fwd(*xs, graph=graph, bindings=bindings):
                return tuple(
                    core.eval_jaxpr(graph, graph.consts, *_constant_values(bindings, xs), *xs[nc:])
                )

            nr_full = len(forwards)

            def backward(*xs, ct_active=ct_active, params=params, nr_full=nr_full):
                residuals, nonzero_cts = xs[:nr_full], iter(xs[nr_full:])
                cts = tuple(
                    next(nonzero_cts) if nz else cd.SymbolicZero(aval.to_ct_aval())
                    for nz, aval in zip(ct_active, primal.out_avals, strict=True)
                )
                gradients, logs = params["bwd"].call_wrapped(*residuals, *cts)
                if logs is not None:
                    raise UnsupportedJaxprError(
                        "derivative", "custom VJP backward logs require an effect adapter"
                    )
                # None/Zero 返回值按 JAX 的实际 cotangent 语义显式物化，保留形状类型。
                return tuple(ad_util.instantiate(gradient) for gradient in gradients)

            residual_outputs = iter(graph.out_avals[:nr])
            residual_avals = tuple(
                next(residual_outputs) if f is None else primal.in_avals[f] for f in forwards
            )
            seeds = tuple(
                _abstract(aval.to_ct_aval())
                for nz, aval in zip(ct_active, primal.out_avals, strict=True)
                if nz
            )
            backward_graph = jax.make_jaxpr(backward)(
                *(tuple(_abstract(aval) for aval in residual_avals) + seeds)
            )
            backward_bindings = _constant_bindings(backward_graph.consts, equation, parent)
            captured_inputs = tuple(
                sorted({value for kind, value in backward_bindings if kind == "input"})
            )
            local_bindings = tuple(
                (kind, captured_inputs.index(value) if kind == "input" else value)
                for kind, value in backward_bindings
            )
            backward_graph = pe.convert_constvars_jaxpr(backward_graph)

            capture_count = len(captured_inputs)

            @partial(jax.jit, keep_unused=True)
            def bwd(*xs, graph=backward_graph, bindings=local_bindings, count=capture_count):
                return tuple(
                    core.eval_jaxpr(
                        graph, graph.consts, *_constant_values(bindings, xs[:count]), *xs[count:]
                    )
                )

            def connected(
                *xs, fwd=fwd, bwd=bwd, forwards=forwards, nr=nr, captured_inputs=captured_inputs
            ):
                primals, seeds = xs[: len(primal.invars)], xs[len(primal.invars) :]
                forwarded = fwd(*primals)
                residual_values = iter(forwarded[:nr])
                residuals = tuple(
                    next(residual_values) if f is None else primals[f] for f in forwards
                )
                gradients = bwd(*(primals[i] for i in captured_inputs), *residuals, *seeds)
                return (*forwarded[nr:], *gradients)

            inputs = tuple(_abstract(aval) for aval in primal.in_avals) + seeds
            connected_graph = jax.make_jaxpr(connected)(*inputs)
            result.append((active, ct_active, out_tree, residual_tree, connected_graph))
    return tuple(result)


def semantic_parameters(equation, *, scope="all", parent=None):
    """已知 callable 展开为语义图；其余原语参数原样交给通用编码器。"""
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
    result = {name: p[name] for name in ("call_jaxpr", "num_consts", "symbolic_zeros")}
    if scope == "all":
        try:
            result["rules"] = (
                _jvp_rules(p, equation, parent)
                if equation.primitive is cd.custom_jvp_call_p
                else _vjp_rules(p, equation, parent)
            )
            _check_rule_order(result["rules"])
        except (InvalidJaxprError, UnsupportedJaxprError):
            raise
        except Exception as exc:
            raise UnsupportedJaxprError(
                "derivative", f"rule tracing rejected: {type(exc).__name__}: {exc}"
            ) from exc
    return result
