"""锁定 JAX 版本的自定义一阶求导适配。

普通调用、规则图和残差实际连接均参与比较；运行时参数和 callable 不变。
这里执行 JAX 延迟追踪接口，不按 callable 名称或 Python 对象身份判断等价。
本模块只物化注册的一阶规则；更高阶展开由调用方显式追踪，不在此递归证明。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from inspect import getclosurevars
from itertools import product
from typing import Any, Literal, cast

import jax
import numpy as np
from jax._src import ad_util, core
from jax._src import custom_derivatives as cd
from jax._src.interpreters import partial_eval as pe

from ..errors import (
    ComparisonBudgetError,
    ComparisonStage,
    InvalidJaxprError,
    UnsupportedJaxprError,
)
from ._typing import Parameters, TreeDef

type ConstantBinding = tuple[Literal["input"], int] | tuple[Literal["constant"], Any]
type JvpRule = tuple[tuple[bool, ...], tuple[bool, ...], core.Jaxpr]
type VjpRule = tuple[tuple[bool, ...], tuple[bool, ...], TreeDef, TreeDef, core.Jaxpr]


# 只接纳锁定版本中的这两个原语身份，同名用户原语不会进入规则适配。
CUSTOM_PRIMITIVES = frozenset((cd.custom_jvp_call_p, cd.custom_vjp_call_p))
# symbolic zeros 会产生组合爆炸；该上限限制规则模式数，而不是图节点数。
MAX_RULE_CASES = 64
# 仅用于识别已审查的 JAX 延迟追踪包装，不用 code 对象身份比较用户求导规则。
_MEMOIZED_CODE = pe._memoize(lambda: None).__code__


def _abstract(aval: core.AbstractValue) -> jax.ShapeDtypeStruct:
    """为规则物化生成形状签名，避免用具体样本固定切线或余切的数值。"""
    if not isinstance(aval, core.ShapedArray):
        raise UnsupportedJaxprError("derivative", f"unsupported rule abstract value {type(aval)}")
    # 当前重建方式只复制这些字段，其他抽象属性的透传仍需单独完善。
    return jax.ShapeDtypeStruct(
        aval.shape, aval.dtype, weak_type=aval.weak_type, sharding=aval.sharding
    )


def _differentiable(aval: core.AbstractValue) -> bool:
    """筛选浮点或复数抽象值，整数等叶子不分配动态求导种子。"""
    return jax.dtypes.issubdtype(cast(core.ShapedArray, aval).dtype, np.inexact)


def _activations(
    avals: Sequence[core.AbstractValue], symbolic: bool, *, include_float0: bool = False
) -> tuple[tuple[bool, ...], ...]:
    """列出需要物化的激活模式；每个模式是对全部叶子的一位布尔掩码。

    普通模式只有一个全激活组合；symbolic 模式枚举非空子集，因为未激活
    叶子会作为 SymbolicZero 传入，用户规则可能据此走不同分支。include_float0
    让 VJP 输出余切把 float0 也视为可显式传入的叶子。
    """
    # 输入切线只激活可微叶子；输出余切需区别显式 float0 与 SymbolicZero，
    # 即使两者数值都是零，用户 bwd 也可以依据其类型选择不同规则。
    indices = [i for i, aval in enumerate(avals) if include_float0 or _differentiable(aval)]
    if not indices:
        return ()
    # k 个可显式传入的叶子最多形成 2**k - 1 个非空组合；先检查数量，再展开枚举。
    count = 2 ** len(indices) - 1 if symbolic else 1
    if count > MAX_RULE_CASES:
        raise ComparisonBudgetError(
            "derivative",
            f"symbolic-zero rule budget exceeded: {count} cases, limit {MAX_RULE_CASES}",
            stage=ComparisonStage.STRUCTURAL,
            requested=count,
            limit=MAX_RULE_CASES,
        )
    result = []
    patterns = product((False, True), repeat=len(indices)) if symbolic else [(True,) * len(indices)]
    for pattern in patterns:
        if any(pattern):
            active = dict(zip(indices, pattern, strict=True))
            result.append(tuple(active.get(i, False) for i in range(len(avals))))
    return tuple(result)


def _constant_bindings(
    constants: Sequence[Any], equation: core.JaxprEqn, parent: core.Jaxpr | None
) -> tuple[ConstantBinding, ...]:
    """把规则捕获的旧 tracer/数组还原到调用实参，避免把动态值当字面常量。

    返回项为（input，原调用操作数位置）或（constant，静态值）。数组按对象
    身份寻找原绑定，不因内容相等就把两份独立捕获合并成一条输入边。
    """
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
            # 记录调用操作数位置，而非当时 tracer 的值，让捕获随实际输入变化。
            result.append(("input", positions[source]))
        elif isinstance(value, core.Tracer):
            raise UnsupportedJaxprError(
                "derivative", "unreviewed captured tracer in derivative rule"
            )
        else:
            result.append(("constant", value))
    return tuple(result)


def _constant_values(
    bindings: Sequence[ConstantBinding], primals: Sequence[Any]
) -> tuple[Any, ...]:
    """在新追踪中解析捕获绑定：动态捕获取当前原值，静态捕获直接透传。"""
    return tuple(primals[value] if kind == "input" else value for kind, value in bindings)


def _check_rule_order(rules: object) -> None:
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


def _jvp_rules(
    params: Parameters, equation: core.JaxprEqn, parent: core.Jaxpr | None
) -> tuple[JvpRule, ...]:
    """逐激活模式物化 JVP，记录输入掩码、输出零掩码和实际规则图。

    规则 thunk 的输入布局与原调用不同，因此先还原捕获，再接上原值与切线。
    每个 connected 图都以原调用的完整输入作为前缀，方便比较真实绑定。
    """
    primal = params["call_jaxpr"]
    # num_consts 是调用输入中的捕获前缀；求导激活掩码只对应后面的普通参数。
    avals = primal.in_avals[params["num_consts"] :]
    result = []
    for active in _activations(avals, params["symbolic_zeros"]):
        # JAX thunk 接收“是否为符号零”，与这里 active 的含义相反。
        graph, constants, out_zeros = params["jvp_jaxpr_fun"].call_wrapped(*(not a for a in active))
        bindings = _constant_bindings(constants, equation, parent)
        n = len(primal.invars)
        # 规则图输入必须是 [捕获常量, 普通原值, 活跃切线]；数量不符说明锁定
        # 版本的 thunk 接口假设已变化，继续按位置拼接会错配绑定。
        if len(graph.invars) != len(constants) + len(avals) + sum(active):
            raise InvalidJaxprError("derivative", "invalid custom JVP rule input count")

        def connected(
            *xs: Any,
            graph: core.Jaxpr = graph,
            bindings: Sequence[ConstantBinding] = bindings,
            n: int = n,
        ) -> tuple[Any, ...]:
            """用本次输入重建 thunk 所需的捕获前缀，并执行物化后的规则。"""
            # 包装图接收 [完整 primal 实参, 活跃切线]，再按规则接口补入捕获值。
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
        # 规则会省略符号零输出；out_zeros 也是接口语义，必须和实际输出图一起比较。
        result.append((active, tuple(out_zeros), jax.make_jaxpr(connected)(*inputs)))
    return tuple(result)


def _vjp_rules(
    params: Parameters, equation: core.JaxprEqn, parent: core.Jaxpr | None
) -> tuple[VjpRule, ...]:
    """连接 VJP 的前向残差与反向消费者，按输入和余切激活模式收集规则图。

    单独比较 fwd/bwd 函数体会遗漏残差错绑；这里以 connected 图中的实际
    数据边表达两者关系，同时保留输出树和残差树作为接口信息。
    """
    primal = params["call_jaxpr"]
    # nc 是 call_jaxpr 输入中的捕获前缀长度，不是该图 constvars 的长度。
    nc = params["num_consts"]
    activations = _activations(primal.in_avals[nc:], params["symbolic_zeros"])
    # 没有可微输出时 _activations 返回空；VJP 仍要物化一次反向规则
    # （整数余切按显式零数组传入），因此下面用全零模式兜底而不是跳过 bwd。
    cotangents = _activations(
        primal.out_avals, params["symbolic_zeros"], include_float0=params["symbolic_zeros"]
    ) or ((False,) * len(primal.out_avals),)
    # VJP 规则同时受输入激活模式和输出余切模式影响，需要验证两者的组合。
    if len(activations) * len(cotangents) > MAX_RULE_CASES:
        raise ComparisonBudgetError(
            "derivative",
            "combined input/cotangent symbolic-zero rule budget exceeded",
            stage=ComparisonStage.STRUCTURAL,
            requested=len(activations) * len(cotangents),
            limit=MAX_RULE_CASES,
        )
    result = []
    for active in activations:
        thunk = params["fwd_jaxpr_thunk"]
        # JAX 的 memoized thunk 只缓存图，不恢复 out_trees 的辅助 store。
        # 不同 symbolic 模式可产生不同 residual 树；再次物化必须重新执行
        # 已审查的原 thunk，否则会把上一个模式的透传表配给当前图。
        # 只接受由 _memoize 包裹、且没有附加 transform/params 的 WrappedFun；
        # 其他包装会改变 thunk 的输入布局，也无法从闭包取到原始规则函数。
        if (
            getattr(thunk.f, "__code__", None) is not _MEMOIZED_CODE
            or thunk.transforms
            or thunk.params
        ):
            raise UnsupportedJaxprError("derivative", "unreviewed custom VJP forward thunk wrapper")
        uncached = getclosurevars(thunk.f).nonlocals["fn"]
        # 绕过缓存直接调用原函数时，仍要保持 _memoize 缓存未命中路径的无 trace
        # 环境，否则规则图会捕获本次 make_jaxpr 的 trace。
        with core.set_current_trace(None):
            graph, constants = uncached(*active)
        bindings = _constant_bindings(constants, equation, parent)
        if len(graph.invars) != len(constants) + len(primal.invars) - nc:
            raise InvalidJaxprError("derivative", "invalid custom VJP forward input count")
        out_tree, residual_tree, forwards = params["out_trees"]()
        # forwards 每项对应一个逻辑残差槽：整数表示直接透传某个 primal 输入，
        # None 表示从 fwd 的实际输出中依次取值；二者共同组成 bwd 接收的残差。
        forwards = tuple(forwards)
        if len(forwards) != residual_tree.num_leaves or out_tree.num_leaves != len(primal.outvars):
            raise InvalidJaxprError("derivative", "invalid custom VJP result or residual interface")
        if any(
            f is not None and (type(f) is not int or not 0 <= f < len(primal.invars))
            for f in forwards
        ):
            raise InvalidJaxprError("derivative", "invalid residual input forwarding")
        # 物化 fwd 的输出为 [非透传残差, primal 输出]，残差总数不一定等于 nr。
        nr = sum(f is None for f in forwards)
        if len(graph.outvars) != nr + len(primal.outvars):
            raise InvalidJaxprError("derivative", "invalid custom VJP forward result count")

        for ct_active in cotangents:
            # 两个边界显式保留。返回槽位与消费者一起比较，不能单独比较 bwd
            # 后遗漏 fwd 透传输入给 residual 的路径。
            @partial(jax.jit, keep_unused=True)
            def fwd(
                *xs: Any, graph: core.Jaxpr = graph, bindings: Sequence[ConstantBinding] = bindings
            ) -> tuple[Any, ...]:
                """保留前向调用边界及全部输入，输出非透传残差和原值结果。

                keep_unused 让未被规则使用的输入仍留在接口中，捕获与残差才能
                按原调用位置连接，而不是被 DCE 掉后丢失绑定证据。
                """
                return tuple(
                    core.eval_jaxpr(graph, graph.consts, *_constant_values(bindings, xs), *xs[nc:])
                )

            if not primal.outvars:
                # 空 pytree 的 VJP 执行 fwd，但 pullback 不调用 bwd；保留 fwd
                # 边界中的全部残差计算，使隐藏的有序操作仍能被后续校验发现。
                def forward_only(
                    *xs: Any, fwd: Callable[..., tuple[Any, ...]] = fwd, nr: int = nr
                ) -> tuple[Any, ...]:
                    """空输出 pytree 的 VJP 只保留前向区域，不实例化反向规则。

                    pullback 不会被调用；切掉残差前缀后返回原值输出，隐藏在前向
                    区域里的有序操作仍由后续校验检查。
                    """
                    return fwd(*xs)[nr:]

                inputs = tuple(_abstract(aval) for aval in primal.in_avals)
                forward_graph = jax.make_jaxpr(forward_only)(*inputs)
                result.append((active, ct_active, out_tree, residual_tree, forward_graph))
                continue

            nr_full = len(forwards)

            def backward(
                *xs: Any,
                ct_active: tuple[bool, ...] = ct_active,
                params: Parameters = params,
                nr_full: int = nr_full,
            ) -> tuple[Any, ...]:
                """补回省略的输出余切，调用用户反向规则并物化返回的显式零。"""
                # 普通 VJP 必须实例化整数余切；symbolic 模式才把未显式传入的
                # 余切补成 SymbolicZero。此处与锁定 JAX 的反向调用约定保持一致。
                residuals, nonzero_cts = xs[:nr_full], iter(xs[nr_full:])
                cts = tuple(
                    next(nonzero_cts)
                    if nz
                    else cd.SymbolicZero(aval.to_ct_aval())
                    if params["symbolic_zeros"]
                    else ad_util.zeros_like_aval(aval.to_ct_aval())
                    for nz, aval in zip(ct_active, primal.out_avals, strict=True)
                )
                gradients, logs = params["bwd"].call_wrapped(*residuals, *cts)
                if logs is not None:
                    raise UnsupportedJaxprError(
                        "derivative", "custom VJP backward logs require an effect adapter"
                    )
                # None/Zero 返回值按 JAX 的实际 cotangent 语义显式物化，保留形状类型。
                return tuple(ad_util.instantiate(gradient) for gradient in gradients)

            # 透传残差从原调用输入取类型，非透传残差按 fwd 输出顺序消费类型。
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
            # bwd 还可能直接捕获调用输入；先建立紧凑的捕获前缀，再把原位置映射过去。
            # 这些边独立于 residual，不能把捕获值冻结成某次追踪的常量。
            captured_inputs = tuple(
                sorted({value for kind, value in backward_bindings if kind == "input"})
            )
            local_bindings = tuple(
                (kind, captured_inputs.index(value) if kind == "input" else value)
                for kind, value in backward_bindings
            )
            # 将 bwd 的 constvars 转成普通输入，包装器才能显式提供每个捕获值。
            backward_graph = pe.convert_constvars_jaxpr(backward_graph)

            capture_count = len(captured_inputs)

            @partial(jax.jit, keep_unused=True)
            def bwd(
                *xs: Any,
                graph: core.Jaxpr = backward_graph,
                bindings: Sequence[ConstantBinding] = cast(
                    tuple[ConstantBinding, ...], local_bindings
                ),
                count: int = capture_count,
            ) -> tuple[Any, ...]:
                """以紧凑捕获前缀、残差和余切调用反向图，保留其独立区域边界。

                keep_unused 与 fwd 同理：未被本模式使用的捕获或种子仍保留在
                接口中，避免未用输入被 DCE 掉后接口位置无法逐项比较。
                """
                return tuple(
                    core.eval_jaxpr(
                        graph, graph.consts, *_constant_values(bindings, xs[:count]), *xs[count:]
                    )
                )

            def connected(
                *xs: Any,
                fwd: Callable[..., tuple[Any, ...]] = fwd,
                bwd: Callable[..., tuple[Any, ...]] = bwd,
                forwards: tuple[int | None, ...] = forwards,
                nr: int = nr,
                captured_inputs: tuple[int, ...] = captured_inputs,
            ) -> tuple[Any, ...]:
                """在同一图中连接前向、残差透传与反向，输出原值及反向规则的结果。"""
                primals, seeds = xs[: len(primal.invars)], xs[len(primal.invars) :]
                forwarded = fwd(*primals)
                residual_values = iter(forwarded[:nr])
                # 连接实际 fwd 输出、透传输入和 bwd 捕获；分别比较两个函数体会漏掉接错线。
                residuals = tuple(
                    next(residual_values) if f is None else primals[f] for f in forwards
                )
                gradients = bwd(*(primals[i] for i in captured_inputs), *residuals, *seeds)
                return (*forwarded[nr:], *gradients)

            inputs = tuple(_abstract(aval) for aval in primal.in_avals) + seeds
            connected_graph = jax.make_jaxpr(connected)(*inputs)
            result.append((active, ct_active, out_tree, residual_tree, connected_graph))
    return tuple(result)
