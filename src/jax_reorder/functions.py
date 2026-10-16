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
from jax._src.layout import AutoLayout, Format

from .algebra import AlgebraComparison, compare_algebraic_jaxprs, validate_timeout
from .algebra_ir import Budget
from .canonical import Comparison, Issue, Status, _difference, compare_jaxprs
from .errors import JaxprError, UnsupportedJaxprError
from .semantics import Encoder, Frozen, encoded_equal

# 固定模式顺序也用于报告 unverified_modes，避免遗漏项随集合遍历顺序变化。
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
    """创建函数契约专用编码器；元数据中的嵌套图不在这个入口解释。"""

    def unsupported(_, path):
        """把误放进接口元数据的图转成可定位的不支持错误。"""
        raise UnsupportedJaxprError(path, "graphs are not valid function contract metadata")

    return Encoder(unsupported, lambda _: None)


@dataclass(frozen=True)
class FunctionSignature:
    """展平前的接口、静态参数与变换范围；抽象值仍由 jaxpr 完整比较。"""

    # tree 保存容器形状和辅助数据，动态叶子的类型由实际追踪图表示。
    input_tree: Frozen
    # 静态值带原始位置或关键字名称，防止图中未用到的静态接口变化被遗漏。
    static_arguments: Frozen
    output_tree: Frozen
    # 编号针对展平后的动态输入叶子，不是原函数的 argnums，也不包含切线/余切种子。
    differentiated_leaves: tuple[int, ...]
    # 保存应用顺序，例如先 jit 后 vmap 与先 vmap 后 jit 是不同的追踪范围。
    transforms: tuple[str, ...]
    # 保留追踪前全部抽象属性，不能依赖重新追踪后可能被丢弃的默认值。
    input_avals: Frozen = ()
    output_avals: Frozen = ()


@dataclass(frozen=True)
class FunctionTrace:
    """一个模式的追踪快照，包含函数契约、环境及带常量绑定的物化图。

    jaxpr 和 abstract_inputs 用于后续比较或编译，不参与数据类默认相等判断。
    图是否等价应通过 compare_function_traces 检查，不能比较快照对象代替。
    """

    mode: str
    signature: FunctionSignature
    context: Frozen
    jaxpr: core.Jaxpr = field(compare=False, repr=False)
    # 物化图的全部动态输入签名；求导模式还包括种子，供可选的编译检查使用。
    abstract_inputs: tuple = field(compare=False, repr=False)


@dataclass(frozen=True)
class ModeComparison:
    """单个前向或求导模式的结论，并保留失败阶段与两侧追踪证据。"""

    mode: str
    status: Status
    # 契约或追踪阶段已经失败时，尚未进入底层图比较，此字段保持 None。
    comparison: Comparison | None = None
    issues: tuple[Issue, ...] = ()
    # difference 定位已确认的差异；issues 解释未完成检查，二者含义不同。
    difference: str | None = None
    # 区分上下文不一致、契约差异、追踪拒绝、不支持语义及 CPU 编译拒绝。
    failure_kind: str | None = None
    left: FunctionTrace | None = field(default=None, compare=False, repr=False)
    right: FunctionTrace | None = field(default=None, compare=False, repr=False)
    algebra: AlgebraComparison | None = None
    equivalence_kind: str | None = None


@dataclass(frozen=True)
class FunctionComparison:
    """汇总全部请求模式，同时列出本次没有检查的模式。"""

    status: Status
    modes: tuple[ModeComparison, ...]
    unverified_modes: tuple[str, ...]
    # 仅在请求编译且各模式两侧都编译成功时为 True，不表示已执行数值验证。
    cpu_compiled: bool
    equivalence_kind: str | None = None


def _context():
    """编码版本、设备、当前 mesh 和已列入审查范围的配置快照。"""
    # 每次追踪重新记录环境；版本、设备或精度配置不同的图不能直接合并比较结论。
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
    """根据抽象 dtype 识别浮点和复数叶子，不读取样本数值。"""
    dtype = getattr(core.typeof(value), "dtype", None)
    return dtype is not None and jax.dtypes.issubdtype(dtype, np.inexact)


def _abstract(value, *, cotangent=False):
    """把输入转换为追踪签名；请求余切时先采用对应的余切抽象类型。"""
    aval = core.typeof(value)
    if cotangent:
        aval = aval.to_ct_aval()
    if not isinstance(aval, core.ShapedArray):
        raise UnsupportedJaxprError(
            "signature", f"unsupported function input abstract type {type(aval)}"
        )
    # 未指定 sharding 不能替换成 aval 的默认 NamedSharding，否则锁定版本会
    # 从 memory_kind 重置 memory_space，丢失 Host 等属性。
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


def _prepare(function, args, kwargs, static_argnums, static_argnames, argnums):
    """分离静态位置与动态 pytree，重建原始调用，不改变对外的位置顺序。

    返回展平调用器、动态叶子的抽象签名、输入树、静态值及可微叶子索引。
    后续 JVP/VJP 只使用这些叶子索引，不再把原参数位置当作展平位置。
    """
    # 先按 Python 签名核对参数；补入默认值后，省略的静态关键字也能被记录。
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
        # 负数位置归一化后与按名称找到的位置合并，同一参数只剔除一次。
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
    # 静态参数可能只影响 Python 分支、不会成为图输入，仍需记录其位置和完整值。
    static_values = tuple(("arg", i, args[i]) for i in sorted(static)) + tuple(
        ("keyword", name, bound.arguments[name])
        for name in sorted(static_names)
        if name not in positional or positional.index(name) >= len(args)
    )
    # jaxpr 只看到展平的叶子；单独保留 tree，才能区分 tuple/list/dict 等接口差异。
    leaves, tree = jax.tree.flatten((dynamic_args, dynamic_kwargs))

    def run(*flat):
        """把动态叶子还原到原始容器与位置，静态参数继续使用闭包中的值。"""
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
        # 一个原始参数可能包含多个叶子，用累计偏移把 argnums 转成展平输入索引。
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
        # JVP 的种子对应选中的输入叶子；VJP 的种子对应全部输出叶子。
        # 都作为动态参数追踪，使比较覆盖任意种子，而不是某个固定投影的结果。
        if mode == "jvp":
            seeds = tuple(_abstract(abstract[i], cotangent=True) for i in differentiated)
        else:
            output = jax.eval_shape(run, *abstract)
            output_leaves, output_tree = jax.tree.flatten(output)
            seeds = tuple(_abstract(v, cotangent=True) for v in output_leaves)

        def with_derivative(*flat):
            """拆开原值与种子，把一阶变换本身追踪成可比较的程序。"""
            primals, seed_values = flat[: len(abstract)], flat[len(abstract) :]
            selected = tuple(primals[i] for i in differentiated)

            def selected_function(*values):
                """只替换选中的可微输入，其余动态输入由外层追踪继续提供。"""
                # 未被选中求导的输入仍是外层图的动态输入，仅在本次 AD 中视为固定。
                full = list(primals)
                for index, value in zip(differentiated, values, strict=True):
                    full[index] = value
                return run(*full)

            if mode == "jvp":
                return jax.jvp(selected_function, selected, seed_values)
            # 同一次调用产生 primal 和 pullback，保留前向残差与反向计算的实际连接。
            primal, pullback = jax.vjp(selected_function, *selected)
            return primal, pullback(jax.tree.unflatten(output_tree, seed_values))

        mode_run = with_derivative
        mode_inputs = (*abstract, *seeds)
    else:
        mode_run = run
    # return_shape 同时提供输出 pytree；图输出叶子相同不代表 Python 返回结构相同。
    graph, output_shape = jax.make_jaxpr(mode_run, return_shape=True)(*mode_inputs)
    # 求导模式返回（前向值，切线或余切），因此输出树也记录变换后的完整接口。
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
        tuple(encoder.encode(core.typeof(value), "signature.aval") for value in mode_inputs),
        tuple(encoder.encode(value.aval, "signature.output_aval") for value in graph.outvars),
    )
    return FunctionTrace(mode, interface, context, graph, tuple(mode_inputs))


def compare_function_traces(
    left: FunctionTrace,
    right: FunctionTrace,
    *,
    equivalence="auto",
    algebra_budget=10000,
    algebra_timeout=10.0,
) -> ModeComparison:
    """先核对完整函数范围，再比较图；不能把不同上下文的追踪结果混为一谈。"""
    if equivalence not in ("auto", "structural"):
        raise ValueError("equivalence must be auto or structural")
    Budget(algebra_budget)
    validate_timeout(algebra_timeout)
    # 上下文不同意味着比较前提不一致，归为 unknown；同一前提下的接口差异才报 different。
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
    algebra = None
    status, issues, difference = result.status, result.issues, result.difference
    kind = "structural" if status == Status.EQUIVALENT else None
    failure = "unsupported" if status == Status.UNKNOWN else None
    if equivalence == "auto" and status == Status.DIFFERENT:
        algebra = compare_algebraic_jaxprs(
            left.jaxpr, right.jaxpr, algebra_budget=algebra_budget, algebra_timeout=algebra_timeout
        )
        status, issues = algebra.status, algebra.issues
        kind = algebra.equivalence_kind
        failure = "algebra_incomplete" if status == Status.UNKNOWN else None
        if status == Status.EQUIVALENT:
            difference = None
        elif status == Status.INEQUIVALENT:
            difference = f"algebra.outputs[{algebra.output}]"
    return ModeComparison(
        left.mode,
        status,
        comparison=result,
        issues=issues,
        difference=difference,
        failure_kind=failure,
        left=left,
        right=right,
        algebra=algebra,
        equivalence_kind=kind,
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
    equivalence: str = "auto",
    algebra_budget: int = 10000,
    algebra_timeout: float = 10.0,
) -> FunctionComparison:
    """对每个请求模式分别追踪、检查契约并比较，未请求模式显式列出。

    默认 auto 在结构不同时尝试理想实数代数证明；structural 保留原结构结果。
    algebra_budget 限制每个模式的一次数学查询；algebra_timeout 限制其符号 worker 秒数。
    浮点警告保存在 algebra 报告中。

    check_cpu 额外编译两侧物化图，用于区别 CPU 编译拒绝和图不支持；
    数值采样由案例运行器另行执行，不将某个样本相等作为结构结论。

    right_args / right_kwargs 为 None 时复用左侧的调用签名；显式传入空容器
    则表示右侧确实使用空参数。各模式保留独立结论，汇总优先级为
    inequivalent、unknown、different、equivalent。inequivalent 表示至少一个请求模式
    有数学不等价证据，具体模式保留在 modes 中，不从求导差异外推前向结论。
    """
    modes = tuple(modes)
    if equivalence not in ("auto", "structural"):
        raise ValueError("equivalence must be auto or structural")
    Budget(algebra_budget)
    validate_timeout(algebra_timeout)
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("modes must contain distinct forward/jvp/vjp entries")
    results = []
    # 仅在请求检查且所有追踪、编译均成功时保留 True；图不同也可能编译成功。
    compiled = check_cpu
    for mode in modes:
        # 各模式独立记录两侧的失败证据；某个模式失败后仍检查其余请求模式。
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
                # 保持 traces 的左右位置固定，即使一侧失败，另一侧仍能单独留下证据。
                traces.append(None)
                compiled = False
                continue
            if check_cpu:
                try:

                    def evaluate(*flat, graph=traced.jaxpr):
                        """直接执行物化图及其常量绑定，供 JIT 检查后端可接受性。"""
                        return core.eval_jaxpr(graph, graph.consts, *flat)

                    # lower/compile 检查后端能否接受物化图，不实际调用编译产物。
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
            results.append(
                compare_function_traces(
                    *traces,
                    equivalence=equivalence,
                    algebra_budget=algebra_budget,
                    algebra_timeout=algebra_timeout,
                )
            )
    # 一个请求模式的完整反例即可否定全部请求模式等价；其余未知模式仍逐项报告。
    statuses = {result.status for result in results}
    status = (
        Status.INEQUIVALENT
        if Status.INEQUIVALENT in statuses
        else Status.UNKNOWN
        if Status.UNKNOWN in statuses
        else Status.DIFFERENT
        if Status.DIFFERENT in statuses
        else Status.EQUIVALENT
    )
    kind = None
    if status == Status.EQUIVALENT:
        kind = (
            "algebraic" if any(r.equivalence_kind == "algebraic" for r in results) else "structural"
        )
    return FunctionComparison(
        status, tuple(results), tuple(mode for mode in MODES if mode not in modes), compiled, kind
    )
