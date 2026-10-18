"""函数入口：把 Python 函数追踪成 jaxpr 快照，再比较这些快照。

一次比较只覆盖给定的参数、静态参数、变换序列和显式请求的模式。
求导模式会把种子也当成普通输入来追踪，因此覆盖任意种子，而不是某个固定梯度。
调用方要保证传入纯函数；本模块不分析追踪范围之外的 Python 行为。

本文件只负责“追踪”和“组织比较”：规范化与图比较在 core/，代数证明在 algebra/。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Never

import jax
import jaxlib
import numpy as np
from jax._src import core
from jax._src.interpreters import ad
from jax._src.layout import AutoLayout, Format

from .algebra import AlgebraComparison, compare_algebraic_jaxprs
from .algebra.budget import Budget, validate_timeout
from .core._typing import TreeDef
from .core.canonical import Comparison, Issue, Status, compare_jaxprs, difference
from .core.semantics import Encoder, Frozen, encoded_equal
from .errors import (
    ComparisonContractError,
    ComparisonError,
    ComparisonFailureKind,
    CompilationError,
    TraceError,
    UnsupportedJaxprError,
)

# 顺序固定，unverified_modes 按这个顺序列出，报告才稳定。
MODES = ("forward", "jvp", "vjp")

# 会影响图和求导规则的 JAX 配置项；升级 JAX 版本时要和 IR 适配一起审查。
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


def _encoder() -> Encoder:
    """创建函数契约用的编码器：契约里出现 jaxpr 就直接报错。

    契约只存 pytree 结构、静态值、变换顺序和抽象值，图由 jaxpr 比较负责。
    这里出现图说明追踪边界已经丢了，不能当成可忽略的额外字段。
    """

    def unsupported(_: core.Jaxpr, path: str) -> Never:
        """契约里混进图时，报一个带路径的不支持错误。"""
        raise UnsupportedJaxprError(path, "graphs are not valid function contract metadata")

    return Encoder(unsupported, lambda _: None)


@dataclass(frozen=True)
class FunctionSignature:
    """函数接口长什么样：pytree 结构、静态参数、变换顺序和抽象值。

    具体计算仍由 jaxpr 比较，这里只描述接口。
    """

    # 输入 pytree 的结构；每个叶子的具体类型由图里的抽象值给出。
    input_tree: Frozen
    # 静态参数连位置（或关键字名）一起记，避免“只影响 Python 分支”的改动被漏掉。
    static_arguments: Frozen
    output_tree: Frozen
    # 求导针对展平后的第几个输入，不是原函数的 argnums；种子不算在内。
    differentiated_leaves: tuple[int, ...]
    # 按应用顺序记录；先 jit 后 vmap 和先 vmap 后 jit 是两种不同的追踪范围。
    transforms: tuple[str, ...]
    # 追踪前的完整抽象值，避免重新追踪时默认属性被丢掉。
    input_avals: Frozen = ()
    output_avals: Frozen = ()


@dataclass(frozen=True)
class FunctionTrace:
    """一个模式的追踪结果：契约、环境，以及带常量绑定的物化图。

    jaxpr 和 abstract_inputs 不参与 dataclass 自动生成的相等比较。
    判断图是否等价要用 compare_function_traces，比较两个快照对象没有意义。
    """

    mode: str
    signature: FunctionSignature
    context: Frozen
    jaxpr: core.Jaxpr = field(compare=False, repr=False)
    # 物化图的全部动态输入签名；求导模式还要加上种子，供可选的编译检查使用。
    abstract_inputs: tuple[jax.ShapeDtypeStruct, ...] = field(compare=False, repr=False)


@dataclass(frozen=True)
class ModeComparison:
    """一个模式（前向/JVP/VJP）的结论，以及两侧的追踪证据。"""

    mode: str
    status: Status
    # 结构比较的报告，留着方便定位差异来自哪一步改写。
    comparison: Comparison | None = None
    issues: tuple[Issue, ...] = ()
    # 已确认的差异位置：结构差异的路径，或数学不等价的输出序号。
    difference: str | None = None
    left: FunctionTrace | None = field(default=None, compare=False, repr=False)
    right: FunctionTrace | None = field(default=None, compare=False, repr=False)
    # 只有 auto 模式下结构比较得出 different 才填；structural 模式始终是 None。
    algebra: AlgebraComparison | None = None
    # 等价是怎么来的：structural（整张图相同）或 algebraic（独立代数证明）。
    equivalence_kind: str | None = None


@dataclass(frozen=True)
class FunctionComparison:
    """所有请求模式的汇总结果，同时列出这次没有检查的模式。"""

    status: Status
    # 每个请求模式一项，顺序和请求顺序一致；中途失败的模式不会出现在这里。
    modes: tuple[ModeComparison, ...]
    # 这次没请求的模式，按固定顺序列出；它们没有结论，不能推断。
    unverified_modes: tuple[str, ...]
    # 只有请求了编译、且每个模式两侧都编译成功才是 True；它不代表跑过数值验证。
    cpu_compiled: bool
    # 整体结论的等价来源：structural 或 algebraic；整体不等价时为 None。
    equivalence_kind: str | None = None


def _context() -> Frozen:
    """记下 JAX/jaxlib 版本、设备、当前 mesh 和受审查的配置项。

    这些都会影响生成的图和求导规则：任何一项不同，两侧的图就不能直接比。
    所以它作为 context 参与比较，而不只是诊断信息。
    """
    # 每次追踪都重新记录环境：版本、设备或精度配置不同的图，结论不能直接合并。
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


def _inexact(value: Any) -> bool:
    """按抽象 dtype 判断是不是浮点/复数叶子，不读具体数值。"""
    dtype = getattr(core.typeof(value), "dtype", None)
    return dtype is not None and jax.dtypes.issubdtype(dtype, np.inexact)


def _abstract(value: Any, *, cotangent: bool = False) -> jax.ShapeDtypeStruct:
    """把一个输入值变成 make_jaxpr 用的抽象签名；要余切时先转成余切类型。

    只看类型不看数值。构造完再和原始抽象值比一遍，确认 sharding、布局等
    属性都保住了；保不住就抛 UnsupportedJaxprError，而不是悄悄放过去。
    """
    aval = core.typeof(value)
    if cotangent:
        aval = aval.to_ct_aval()
    if not isinstance(aval, core.ShapedArray):
        raise UnsupportedJaxprError(
            "signature", f"unsupported function input abstract type {type(aval)}"
        )
    # 不要拿 aval 的默认 sharding 兜底：那样会从 memory_kind 重置 memory_space，
    # 把 Host 内存这类属性丢掉。
    sharding = value.sharding if isinstance(value, jax.ShapeDtypeStruct) else aval.sharding
    if aval.layout is not AutoLayout:
        sharding = Format(aval.layout, sharding)
    result = jax.ShapeDtypeStruct(
        aval.shape,
        aval.dtype,
        weak_type=aval.weak_type,
        sharding=sharding,
        manual_axis_type=aval.manual_axis_type,
        _memory_space=aval.memory_space,
    )
    encoder = _encoder()
    if not encoded_equal(
        encoder.encode(aval, "signature.original"),
        encoder.encode(core.typeof(result), "signature.abstract"),
    ):
        raise UnsupportedJaxprError("signature", "cannot faithfully preserve input abstract value")
    return result


def _prepare(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    static_argnums: Sequence[int],
    static_argnames: Sequence[str],
    argnums: Sequence[int] | None,
) -> tuple[
    Callable[..., Any],
    tuple[jax.ShapeDtypeStruct, ...],
    TreeDef,
    tuple[tuple[str, int | str, Any], ...],
    tuple[int, ...],
]:
    """把一次调用拆成“静态参数 + 动态 pytree”，并准备一个能还原原始调用的函数。

    返回：还原调用的 run、动态输入的抽象签名、输入 pytree、静态值、可微叶子编号。
    后续 JVP/VJP 只用这些叶子编号，不再把原参数位置当成展平位置。
    static_argnums/static_argnames/argnums 越界，或 argnums 指向静态参数时抛 ValueError。
    """
    # 先按函数签名绑定参数并补上默认值，这样没显式传的静态关键字也能记录下来。
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
        # 负数下标先转正，再和按名字找到的静态参数合并，避免同一参数处理两次。
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
    # 静态参数不进图，但可能影响 Python 分支，所以位置和值都要记下来。
    static_values = tuple(("arg", i, args[i]) for i in sorted(static)) + tuple(
        ("keyword", name, bound.arguments[name])
        for name in sorted(static_names)
        if name not in positional or positional.index(name) >= len(args)
    )
    # jaxpr 只看展平后的叶子，所以另外保存 pytree，才能区分 tuple/list/dict 这类接口差异。
    leaves, tree = jax.tree.flatten((dynamic_args, dynamic_kwargs))

    def run(*flat: Any) -> Any:
        """把展平的动态叶子放回原来的位置和容器；静态参数直接用闭包里的值。"""
        values, keywords = jax.tree.unflatten(tree, flat)
        positions = dict(zip(dynamic_positions, values, strict=True))
        positional_args = [positions.get(i, value) for i, value in enumerate(args)]
        return function(*positional_args, **{**kwargs, **keywords})

    # argnums 用的是原函数参数位置；不指定就对所有动态的浮点/复数叶子求导。
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
        # 一个参数可能拆成多个叶子，用累计偏移把 argnums 换成展平后的下标。
        for index, value in zip(dynamic_positions, dynamic_args, strict=True):
            part = jax.tree.leaves(value)
            if index in selected:
                differentiated.extend(offset + i for i, leaf in enumerate(part) if _inexact(leaf))
            offset += len(part)
        differentiated = tuple(differentiated)
    return run, tuple(_abstract(v) for v in leaves), tree, static_values, differentiated


def trace_function(
    function: Callable[..., Any],
    *args: Any,
    kwargs: Mapping[str, Any] | None = None,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    mode: str = "forward",
    argnums: Sequence[int] | None = None,
    transforms: Sequence[str] = (),
) -> FunctionTrace:
    """追踪前向、JVP 或 VJP，返回一份可以直接拿去比较的函数快照。

    transforms 里的 jit/vmap 按给定顺序套在函数上，再应用求导模式；
    顺序不同就是不同的追踪范围。vmap 固定用 axis 0，高阶或其它映射
    策略由调用方自己包一层再追踪。
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
        # JVP 的种子对应被求导的输入，VJP 的种子对应全部输出。
        # 种子也当成普通输入，这样比较覆盖任意种子，而不是某一个固定投影。
        if mode == "jvp":
            seeds = tuple(_abstract(abstract[i], cotangent=True) for i in differentiated)
        else:
            output = jax.eval_shape(run, *abstract)
            output_leaves, output_tree = jax.tree.flatten(output)
            seeds = tuple(_abstract(v, cotangent=True) for v in output_leaves)

        def with_derivative(*flat: Any) -> Any:
            """把原值和种子分开，把求导这一步本身也追踪成可比较的程序。"""
            primals, seed_values = flat[: len(abstract)], flat[len(abstract) :]
            selected = tuple(primals[i] for i in differentiated)

            def selected_function(*values: Any) -> Any:
                """只把被求导的输入换成 AD 变量，其它输入原样传给 run。"""
                # 没被求导的输入仍是外层图的动态输入，这里只是暂时当常数用。
                full = list(primals)
                for index, value in zip(differentiated, values, strict=True):
                    full[index] = value
                return run(*full)

            if mode == "jvp":
                return jax.jvp(selected_function, selected, seed_values)
            # primal 和 pullback 来自同一次 vjp 调用，前向残差与反向计算之间的连接会留在图里。
            primal, pullback = jax.vjp(selected_function, *selected)
            return primal, pullback(jax.tree.unflatten(output_tree, seed_values))

        mode_run = with_derivative
        mode_inputs = (*abstract, *seeds)
    else:
        mode_run = run
    # return_shape 顺带拿到输出 pytree：图输出叶子相同，不代表 Python 返回结构相同。
    graph, output_shape = jax.make_jaxpr(mode_run, return_shape=True)(*mode_inputs)
    # 求导模式返回 (原值, 切线或余切)，所以输出树记的是变换后的真实接口。
    # custom_vjp 不能做 forward-mode JVP：JAX 会先追出 custom_lin，而不是当场报错。
    # 这里主动拦下，免得被误记成“比较器不支持某个原语”。

    # 参数里可能嵌套 call_jaxpr 或规则图，所以要递归翻 tuple/list/dict。
    # 按对象身份去重：既避免重复扫描共享子图，也避免参数成环时死循环。
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
        tuple(encoder.encode(core.typeof(value), "signature.aval") for value in mode_inputs),
        tuple(encoder.encode(value.aval, "signature.output_aval") for value in graph.outvars),
    )
    return FunctionTrace(mode, interface, context, graph, tuple(mode_inputs))


def compare_function_traces(
    left: FunctionTrace,
    right: FunctionTrace,
    *,
    equivalence: str = "auto",
    algebra_budget: int = 10000,
    algebra_timeout: float = 10.0,
) -> ModeComparison:
    """先核对两侧契约是否一致，再比较图，返回这个模式的结论。

    模式、环境（context）和函数签名必须完全一致，否则抛 ComparisonContractError：
    前提不同的两张图没法比，不能先比图再补说明。equivalence="auto" 时，
    只有结构比较得出 different 才继续做代数证明。图比较固定用
    derivative_scope="primal"，因为求导程序已经通过公开变换展开过了，
    不能从这次检查外推还没请求的高阶规则。比较中的 ComparisonError
    会补上模式前缀后重新抛出。
    """
    if equivalence not in ("auto", "structural"):
        raise ValueError("equivalence must be auto or structural")
    # 不管哪种模式都先校验预算和超时：参数错了就在入口报，而不是因为
    # structural 模式不查代数就悄悄跳过。
    Budget(algebra_budget)
    validate_timeout(algebra_timeout)
    if left.mode != right.mode or left.context != right.context:
        raise ComparisonContractError(
            f"{left.mode}.context",
            "trace contexts or comparison modes differ",
            failure_kind=ComparisonFailureKind.CONTEXT_MISMATCH,
        )
    difference_path = difference(left.signature, right.signature)
    if difference_path:
        raise ComparisonContractError(
            f"{left.mode}.{difference_path.replace('graph', 'signature', 1)}",
            "function contracts differ",
        )
    # 求导已经展开成一阶程序了，这里只比较展开后的前向语义；
    # 一次一阶检查推不出还没请求的高阶规则。
    try:
        result = compare_jaxprs(left.jaxpr, right.jaxpr, derivative_scope="primal")
        algebra = None
        status, issues, difference_path = result.status, result.issues, result.difference
        kind = "structural" if status == Status.EQUIVALENT else None
        if equivalence == "auto" and status == Status.DIFFERENT:
            algebra = compare_algebraic_jaxprs(
                left.jaxpr,
                right.jaxpr,
                algebra_budget=algebra_budget,
                algebra_timeout=algebra_timeout,
            )
            status, issues = algebra.status, algebra.issues
            kind = algebra.equivalence_kind
            if status == Status.EQUIVALENT:
                difference_path = None
            elif status == Status.INEQUIVALENT:
                difference_path = f"algebra.outputs[{algebra.output}]"
    except ComparisonError as exc:
        raise exc.at(left.mode) from exc
    return ModeComparison(
        left.mode,
        status,
        comparison=result,
        issues=issues,
        difference=difference_path,
        left=left,
        right=right,
        algebra=algebra,
        equivalence_kind=kind,
    )


def compare_functions(
    left: Callable[..., Any],
    right: Callable[..., Any],
    *args: Any,
    kwargs: Mapping[str, Any] | None = None,
    right_args: tuple[Any, ...] | None = None,
    right_kwargs: Mapping[str, Any] | None = None,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
    modes: Sequence[str] = ("forward",),
    argnums: Sequence[int] | None = None,
    transforms: Sequence[str] = (),
    check_cpu: bool = False,
    equivalence: str = "auto",
    algebra_budget: int = 10000,
    algebra_timeout: float = 10.0,
) -> FunctionComparison:
    """对每个请求的模式分别追踪、核对契约、比较，最后汇总成一份报告。

    - equivalence="auto"：结构不同时再试理想实数范围内的代数证明；"structural" 只看结构。
    - algebra_budget / algebra_timeout：限制每个模式的代数查询次数和符号 worker 超时；
      浮点交换律之类的警告留在 algebra 报告里，不会丢。
    - check_cpu：额外编译两侧的物化图，用来区分“CPU 后端不接受这张图”和“图本身不支持”；
      它不做数值采样，样本相等也不能当成结构结论。
    - right_args / right_kwargs 传 None 表示右侧复用左侧的调用签名；传空容器表示右侧确实没有参数。
    - 没请求的模式会列在 unverified_modes 里：它们没有结论，不能当成等价。
    - 任一请求模式失败会立即抛出分类异常，全部跑完才汇总。
    - 汇总时 inequivalent 表示至少一个模式有数学不等价依据，各模式结论保留在 modes 里，
      不会用求导模式的差异去推前向结论。
    """
    modes = tuple(modes)
    if equivalence not in ("auto", "structural"):
        raise ValueError("equivalence must be auto or structural")
    Budget(algebra_budget)
    validate_timeout(algebra_timeout)
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("modes must contain distinct forward/jvp/vjp entries")
    results = []
    # 只有请求了检查、且追踪和编译都成功才保持 True；图不同也可能编译成功。
    compiled = check_cpu
    for mode in modes:
        traces = []
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
            except ComparisonError as exc:
                raise exc.at(f"{side}.{mode}") from exc
            except Exception as exc:
                raise TraceError(f"{side}.{mode}.trace", f"{type(exc).__name__}: {exc}") from exc
            if check_cpu:
                try:

                    def evaluate(*flat: Any, graph: core.Jaxpr = traced.jaxpr) -> list[Any]:
                        """直接执行物化图和它绑定的常量，只用来让 JIT 检查后端能不能接受。"""
                        return core.eval_jaxpr(graph, graph.consts, *flat)

                    # 只做 lower/compile，看后端接不接受，不执行编译产物。
                    jax.jit(evaluate).lower(*traced.abstract_inputs).compile()
                except Exception as exc:
                    raise CompilationError(
                        f"{side}.{mode}.compile", f"{type(exc).__name__}: {exc}"
                    ) from exc
        results.append(
            compare_function_traces(
                *traces,
                equivalence=equivalence,
                algebra_budget=algebra_budget,
                algebra_timeout=algebra_timeout,
            )
        )
    # 汇总取最保守的结论：任一模式 inequivalent 就整体 inequivalent；
    # 否则任一模式 different 就整体 different。不能因为某个模式等价就忽略别的差异。
    statuses = {result.status for result in results}
    status = (
        Status.INEQUIVALENT
        if Status.INEQUIVALENT in statuses
        else Status.DIFFERENT
        if Status.DIFFERENT in statuses
        else Status.EQUIVALENT
    )
    kind = None
    if status == Status.EQUIVALENT:
        # 全部模式都等价才给来源：只要有一个模式靠代数证明就记 algebraic，
        # 全部靠结构相同才是 structural。
        kind = (
            "algebraic" if any(r.equivalence_kind == "algebraic" for r in results) else "structural"
        )
    return FunctionComparison(
        status, tuple(results), tuple(mode for mode in MODES if mode not in modes), compiled, kind
    )
