"""带函数接口和追踪上下文的比较入口。

一次报告只覆盖给定输入签名、静态参数与显式请求的变换模式。
JVP/VJP 的种子作为动态参数追踪，不能用一个固定梯度投影代替完整规则。
调用方负责提供纯函数；本模块不分析追踪之外的任意 Python 行为。
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jaxlib
import numpy as np
from jax._src import core
from jax._src.interpreters import ad

from .canonical import Comparison, Issue, Status, _difference, compare_jaxprs
from .errors import JaxprError, UnsupportedJaxprError
from .semantics import Encoder, Frozen

MODES = ("forward", "jvp", "vjp")

# 与类型、精度和求导实现有关的配置；升级锁定版本时随 IR 适配一并审核。
CONTEXT_FLAGS = (
    "jax_enable_x64",
    "jax_explicit_x64_dtypes",
    "jax_default_matmul_precision",
    "jax_default_device",
    "jax_numpy_dtype_promotion",
    "jax_numpy_rank_promotion",
    "jax_default_prng_impl",
    "jax_threefry_partitionable",
    "jax_custom_jvp3",
    "jax_custom_vjp3",
    "jax_softmax_custom_jvp",
    "jax_use_shardy_partitioner",
    "jax_remove_size_one_mesh_axis_from_type",
    "jax_enable_custom_prng",
)


def _encoder():
    def unsupported(_, path):
        raise UnsupportedJaxprError(path, "graphs are not valid function contract metadata")

    return Encoder(unsupported, lambda _: None)


@dataclass(frozen=True)
class FunctionSignature:
    """展平前的接口、静态参数与变换范围；抽象值仍由 jaxpr 完整比较。"""

    input_tree: Frozen
    static_arguments: Frozen
    output_tree: Frozen
    differentiated_leaves: tuple[int, ...]
    transforms: tuple[str, ...]


@dataclass(frozen=True)
class FunctionTrace:
    mode: str
    signature: FunctionSignature
    context: Frozen
    jaxpr: core.Jaxpr = field(compare=False, repr=False)
    abstract_inputs: tuple = field(compare=False, repr=False)


@dataclass(frozen=True)
class ModeComparison:
    mode: str
    status: Status
    comparison: Comparison | None = None
    issues: tuple[Issue, ...] = ()
    difference: str | None = None
    failure_kind: str | None = None
    left: FunctionTrace | None = field(default=None, compare=False, repr=False)
    right: FunctionTrace | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class FunctionComparison:
    status: Status
    modes: tuple[ModeComparison, ...]
    unverified_modes: tuple[str, ...]
    cpu_compiled: bool


def _context():
    encoder = _encoder()
    values = jax.config.values
    return encoder.encode(
        (
            jax.__version__,
            jaxlib.__version__,
            tuple(jax.devices()),
            jax.sharding.get_abstract_mesh(),
            tuple((name, values[name]) for name in CONTEXT_FLAGS if name in values),
        ),
        "context",
    )


def _inexact(value):
    dtype = getattr(core.typeof(value), "dtype", None)
    return dtype is not None and jax.dtypes.issubdtype(dtype, np.inexact)


def _abstract(value, *, cotangent=False):
    aval = core.typeof(value)
    if cotangent:
        aval = aval.to_ct_aval()
    if not isinstance(aval, core.ShapedArray):
        raise UnsupportedJaxprError(
            "signature", f"unsupported function input abstract type {type(aval)}"
        )
    return jax.ShapeDtypeStruct(
        aval.shape, aval.dtype, weak_type=aval.weak_type, sharding=aval.sharding
    )


def _prepare(function, args, kwargs, static_argnums, static_argnames, argnums):
    """分离静态位置与动态 pytree，重建原始调用，不改变对外的位置顺序。"""
    signature = inspect.signature(function)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    positional = [
        name
        for name, param in signature.parameters.items()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)
    ]
    static = set()
    for index in static_argnums:
        if type(index) is not int or not -len(args) <= index < len(args):
            raise ValueError("static_argnums contains an invalid positional index")
        static.add(index % len(args))
    for name in static_argnames:
        if name not in bound.arguments:
            raise ValueError(f"unknown static argument {name!r}")
        if name in positional and positional.index(name) < len(args):
            static.add(positional.index(name))
    static_names = set(static_argnames)
    dynamic_positions = tuple(i for i in range(len(args)) if i not in static)
    dynamic_args = tuple(args[i] for i in dynamic_positions)
    dynamic_kwargs = {k: v for k, v in kwargs.items() if k not in static_names}
    static_values = tuple(("arg", i, args[i]) for i in sorted(static)) + tuple(
        ("keyword", name, bound.arguments[name])
        for name in sorted(static_names)
        if name not in positional or positional.index(name) >= len(args)
    )
    leaves, tree = jax.tree.flatten((dynamic_args, dynamic_kwargs))

    def run(*flat):
        values, keywords = jax.tree.unflatten(tree, flat)
        positions = dict(zip(dynamic_positions, values, strict=True))
        positional_args = [positions.get(i, value) for i, value in enumerate(args)]
        return function(*positional_args, **{**kwargs, **keywords})

    # argnums 按原函数位置指定；未指定时对全部动态浮点/复数叶子求导。
    if argnums is None:
        differentiated = tuple(i for i, value in enumerate(leaves) if _inexact(value))
    else:
        selected = set()
        for index in argnums:
            if type(index) is not int or not -len(args) <= index < len(args):
                raise ValueError("argnums contains an invalid positional index")
            index %= len(args)
            if index in static:
                raise ValueError("a static argument cannot also be differentiated")
            selected.add(index)
        offset = 0
        differentiated = []
        for index, value in zip(dynamic_positions, dynamic_args, strict=True):
            part = jax.tree.leaves(value)
            if index in selected:
                differentiated.extend(offset + i for i, leaf in enumerate(part) if _inexact(leaf))
            offset += len(part)
        differentiated = tuple(differentiated)
    return run, tuple(_abstract(v) for v in leaves), tree, static_values, differentiated


def trace_function(
    function: Callable,
    *args,
    kwargs: Mapping | None = None,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    mode: str = "forward",
    argnums: Sequence[int] | None = None,
    transforms: Sequence[str] = (),
) -> FunctionTrace:
    """追踪普通前向或一阶 JVP/VJP，返回可独立比较的函数契约与图。

    transforms 中的 jit/vmap 按给定顺序作用于原函数，然后应用请求的求导
    模式；不同顺序属于不同范围。vmap 使用默认 axis 0，高阶或其他映射
    策略可由调用方显式包装函数，并作为另外的追踪对象验证。
    """
    if mode not in MODES:
        raise ValueError(f"unsupported comparison mode {mode!r}")
    context = _context()
    run, abstract, in_tree, statics, differentiated = _prepare(
        function, args, dict(kwargs or {}), static_argnums, static_argnames, argnums
    )
    for transform in transforms:
        if transform == "jit":
            run = jax.jit(run)
        elif transform == "vmap":
            run = jax.vmap(run)
        else:
            raise ValueError(f"unsupported transformation {transform!r}")
    mode_inputs = abstract
    if mode != "forward":
        if mode == "jvp":
            seeds = tuple(_abstract(abstract[i], cotangent=True) for i in differentiated)
        else:
            output = jax.eval_shape(run, *abstract)
            output_leaves, output_tree = jax.tree.flatten(output)
            seeds = tuple(_abstract(v, cotangent=True) for v in output_leaves)

        def with_derivative(*flat):
            primals, seed_values = flat[: len(abstract)], flat[len(abstract) :]
            selected = tuple(primals[i] for i in differentiated)

            def selected_function(*values):
                full = list(primals)
                for index, value in zip(differentiated, values, strict=True):
                    full[index] = value
                return run(*full)

            if mode == "jvp":
                return jax.jvp(selected_function, selected, seed_values)
            primal, pullback = jax.vjp(selected_function, *selected)
            return primal, pullback(jax.tree.unflatten(output_tree, seed_values))

        mode_run = with_derivative
        mode_inputs = (*abstract, *seeds)
    else:
        mode_run = run
    graph, output_shape = jax.make_jaxpr(mode_run, return_shape=True)(*mode_inputs)
    # custom_vjp 的非法 JVP 可能先追踪出 custom_lin，直到执行时才抛错。
    # 这是 JAX 自身拒绝的变换，不应记录成比较器尚未支持一个普通原语。
    pending = [graph]
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, core.Jaxpr):
            for equation in value.eqns:
                if equation.primitive is ad.custom_lin_p:
                    raise TypeError("cannot apply forward-mode JVP to a custom_vjp function")
                pending.extend(equation.params.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif type(value) is dict:
            pending.extend(value.values())
    encoder = _encoder()
    interface = FunctionSignature(
        encoder.encode(in_tree, "signature.inputs"),
        encoder.encode(statics, "signature.static"),
        encoder.encode(jax.tree.structure(output_shape), "signature.outputs"),
        differentiated if mode != "forward" else (),
        tuple(transforms),
    )
    return FunctionTrace(mode, interface, context, graph, tuple(mode_inputs))


def compare_function_traces(left: FunctionTrace, right: FunctionTrace) -> ModeComparison:
    """先核对完整函数范围，再比较图；不能把不同上下文的追踪结果混为一谈。"""
    if left.mode != right.mode or left.context != right.context:
        return ModeComparison(
            left.mode,
            Status.UNKNOWN,
            issues=(Issue("context", "trace contexts or comparison modes differ"),),
            failure_kind="context_mismatch",
            left=left,
            right=right,
        )
    difference = _difference(left.signature, right.signature)
    if difference:
        return ModeComparison(
            left.mode,
            Status.DIFFERENT,
            difference=difference.replace("graph", "signature", 1),
            failure_kind="contract_difference",
            left=left,
            right=right,
        )
    # 求导模式已经通过公开变换展开为实际一阶程序；内部普通调用只比较其
    # 前向语义，不能从一次一阶检查继续外推尚未请求的高阶规则。
    result = compare_jaxprs(left.jaxpr, right.jaxpr, derivative_scope="primal")
    return ModeComparison(
        left.mode,
        result.status,
        comparison=result,
        issues=result.issues,
        difference=result.difference,
        failure_kind="unsupported" if result.status == Status.UNKNOWN else None,
        left=left,
        right=right,
    )


def compare_functions(
    left: Callable,
    right: Callable,
    *args,
    kwargs: Mapping | None = None,
    right_args: tuple | None = None,
    right_kwargs: Mapping | None = None,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    modes: Sequence[str] = ("forward",),
    argnums: Sequence[int] | None = None,
    transforms: Sequence[str] = (),
    check_cpu: bool = False,
) -> FunctionComparison:
    """对每个请求模式分别追踪、检查契约并比较，未请求模式显式列出。

    check_cpu 额外编译两侧物化图，用于区别 CPU 编译拒绝和图不支持；
    数值采样由案例运行器另行执行，不将某个样本相等作为结构结论。
    """
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("modes must contain distinct forward/jvp/vjp entries")
    results = []
    compiled = check_cpu
    for mode in modes:
        traces, issues = [], []
        failure_kind = None
        for side, function, positional, keywords in (
            ("left", left, args, kwargs),
            (
                "right",
                right,
                args if right_args is None else right_args,
                kwargs if right_kwargs is None else right_kwargs,
            ),
        ):
            try:
                traced = trace_function(
                    function,
                    *positional,
                    kwargs=keywords,
                    static_argnums=static_argnums,
                    static_argnames=static_argnames,
                    mode=mode,
                    argnums=argnums,
                    transforms=transforms,
                )
                traces.append(traced)
            except (
                JaxprError,
                TypeError,
                ValueError,
                NotImplementedError,
                RuntimeError,
                AssertionError,
            ) as exc:
                issues.append(Issue(f"{side}.{mode}.trace", str(exc)))
                failure_kind = "unsupported" if isinstance(exc, JaxprError) else "trace_rejected"
                traces.append(None)
                compiled = False
                continue
            if check_cpu:
                try:

                    def evaluate(*flat, graph=traced.jaxpr):
                        return core.eval_jaxpr(graph, graph.consts, *flat)

                    jax.jit(evaluate).lower(*traced.abstract_inputs).compile()
                except (
                    TypeError,
                    ValueError,
                    NotImplementedError,
                    RuntimeError,
                    AssertionError,
                ) as exc:
                    issues.append(Issue(f"{side}.{mode}.compile", str(exc)))
                    failure_kind = "cpu_compile_rejected"
                    compiled = False
        if issues:
            results.append(
                ModeComparison(
                    mode,
                    Status.UNKNOWN,
                    issues=tuple(issues),
                    failure_kind=failure_kind,
                    left=traces[0],
                    right=traces[1],
                )
            )
        else:
            results.append(compare_function_traces(*traces))
    statuses = {result.status for result in results}
    status = (
        Status.UNKNOWN
        if Status.UNKNOWN in statuses
        else Status.DIFFERENT
        if Status.DIFFERENT in statuses
        else Status.EQUIVALENT
    )
    return FunctionComparison(
        status, tuple(results), tuple(mode for mode in MODES if mode not in modes), compiled
    )
