"""函数级契约与前向/JVP/VJP 范围的独立验收。

覆盖：
- 正例：原始与换序函数在 forward/jvp/vjp 三种模式与 5 种 transform 组合下 equivalent，
  且 cpu_compiled 为真、未请求模式进入 unverified_modes；静态参数（按位置/名称）、关键字
  默认值、argnums 子集、整数叶子、抽象 ShapeDtypeStruct 输入均可比较。
- 契约差异与反例：tuple/list 输出树、字典键、展平前的输入 pytree、静态参数值、shape/
  dtype/weak_type、追踪环境、不同 custom JVP 规则都产生契约差异或 inequivalent；动态余切
  覆盖每个输出而非只比较其和。
- 拒绝路径：不可追踪的函数抛 TraceError（trace_rejected），CPU lowering 失败抛
  CompilationError（cpu_compile_rejected），custom_vjp 上的 jvp 不适用，非法 modes 抛
  ValueError；某个模式的失败不能被其他模式的负结论掩盖。

前置条件：CPU 平台；check_cpu=True 的用例会实际 jit 编译。这里是有限模式与变换顺序的
证据，不覆盖高阶求导或任意 JAX 变换组合，也不做数值采样。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import ComparisonContractError, CompilationError, Status, TraceError
from jax_reorder.functions import compare_function_traces, compare_functions, trace_function


def original(x, y):
    """基准函数：两个互不依赖的逐元素调用，输出为带键字典。"""
    a = lax.sin(x)
    b = lax.cos(y)
    return {"a": a, "b": b}


def reordered(x, y):
    """正例：只交换两个独立调用的定义顺序，字典输出顺序不变。"""
    b = lax.cos(y)
    a = lax.sin(x)
    return {"a": a, "b": b}


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
@pytest.mark.parametrize("transforms", [(), ("jit",), ("vmap",), ("vmap", "jit"), ("jit", "vmap")])
def test_declared_function_modes(mode, transforms):
    """每种请求模式与每种 transform 顺序都应识别换序正例；未请求模式显式列出。"""
    x, y = jnp.array([0.7, 1.2]), jnp.array([-0.3, 0.8])
    result = compare_functions(
        original, reordered, x, y, modes=(mode,), transforms=transforms, check_cpu=True
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.cpu_compiled
    # 只比较请求的模式，其余模式必须进入 unverified_modes 而不是被当作已验证。
    assert result.unverified_modes == tuple(m for m in ("forward", "jvp", "vjp") if m != mode)
    assert result.modes[0].comparison.correspondence is not None
    assert result.modes[0].left.signature.transforms == transforms


def test_tuple_list_and_dictionary_key_contracts():
    """输出 pytree 的类型和键属于契约：tuple 与 list 不互通，字典键不同也要拒绝。"""
    x = jnp.float32(1)

    def tuple_fn(x):
        return x, x + 1

    def list_fn(x):
        return [x, x + 1]

    with pytest.raises(ComparisonContractError) as error:
        compare_functions(tuple_fn, list_fn, x)
    result = error.value
    assert "output_tree" in result.path
    with pytest.raises(ComparisonContractError) as error:
        compare_functions(lambda x: {"a": x}, lambda x: {"b": x}, x)
    result = error.value


def test_input_pytrees_are_compared_before_flattening():
    """输入树在展平前比较：tuple 与 list 展平后叶子相同，但接口不同。"""

    def first(pair):
        return pair[0] - pair[1]

    values = (jnp.float32(1), jnp.float32(2))
    with pytest.raises(ComparisonContractError) as error:
        compare_functions(first, first, values, right_args=(list(values),))
    result = error.value
    assert "input_tree" in result.path


@pytest.mark.parametrize("by_name", [False, True])
def test_static_arguments_and_non_differentiable_flags(by_name):
    """静态参数可按位置或名称声明；两侧静态值不同属于契约差异。"""

    def first(x, flag):
        return lax.sin(x) if flag else lax.cos(x)

    options = {"static_argnames": ("flag",)} if by_name else {"static_argnums": (1,)}
    result = compare_functions(
        first, first, jnp.float32(1), True, modes=("forward", "jvp", "vjp"), **options
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.modes[1].left.signature.differentiated_leaves == (0,)
    with pytest.raises(ComparisonContractError) as error:
        compare_functions(
            first, first, jnp.float32(1), True, right_args=(jnp.float32(1), False), **options
        )
    result = error.value
    assert "static_arguments" in result.path


def test_keywords_static_defaults_and_differentiated_argument_subset():
    """关键字默认值、static_argnames 与 argnums 子集共同决定被比较的接口。"""

    def first(x, y, *, sign=True):
        value = x * y
        return value if sign else -value

    result = compare_functions(
        first,
        first,
        jnp.float32(2),
        jnp.float32(3),
        static_argnames=("sign",),
        argnums=(0,),
        modes=("forward", "jvp", "vjp"),
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.modes[2].left.signature.differentiated_leaves == (0,)
    graph = result.modes[2].left.jaxpr
    # VJP 图的实际动态输入是 x、y 和标量输出的余切向量，按此顺序执行核对。
    actual = core.eval_jaxpr(graph, graph.consts, jnp.float32(2), jnp.float32(3), jnp.float32(7))
    np.testing.assert_array_equal(actual, [6, 21])
    # 两侧关键字静态值不同（sign=True/False），必须作为契约差异拒绝。
    with pytest.raises(ComparisonContractError):
        compare_functions(
            first,
            first,
            jnp.float32(2),
            jnp.float32(3),
            kwargs={"sign": True},
            right_kwargs={"sign": False},
            static_argnames=("sign",),
        )


def test_integer_leaves_remain_inputs_but_are_not_differentiated():
    """整数叶子仍是普通动态输入，只是不参与默认求导。"""

    def first(values):
        return values["x"] * values["count"].astype(jnp.float32)

    data = {"x": jnp.float32(2), "count": jnp.int32(3)}
    result = compare_functions(first, first, data, modes=("jvp", "vjp"))
    assert result.status == Status.EQUIVALENT, result
    assert result.modes[0].left.signature.differentiated_leaves == (1,)


def test_dynamic_cotangents_cover_each_output_not_only_their_sum():
    """VJP 对每个输出使用独立余切向量；只比较余切之和会漏掉输出槽位交换。"""

    def first(x):
        return (2 * x, 3 * x)

    def wrong(x):
        return (3 * x, 2 * x)

    result = compare_functions(
        first, wrong, jnp.float32(2), modes=("vjp",), equivalence="structural"
    )
    assert result.status == Status.DIFFERENT, result
    left, right = result.modes[0].left, result.modes[0].right
    # 一份原始输入加两个独立余切向量，不能退化成 grad(sum(...))。
    assert len(left.jaxpr.invars) == len(right.jaxpr.invars) == 3

    def derivative(trace, ca, cb):
        """按给定两个余切向量执行 pullback，取最后一个输出作为 VJP 结果。"""
        return core.eval_jaxpr(
            trace.jaxpr, trace.jaxpr.consts, jnp.float32(2), jnp.float32(ca), jnp.float32(cb)
        )[-1]

    # 两个余切相等时两侧结果相同；权重不同时立即暴露差异。
    assert derivative(left, 1, 1) == derivative(right, 1, 1)
    assert derivative(left, 1, 0) != derivative(right, 1, 0)


def test_environment_is_captured_and_does_not_share_stale_results():
    """追踪环境（默认 matmul 精度）参与比较；不同环境下不能复用旧追踪结果。"""

    with jax.default_matmul_precision("default"):
        first = trace_function(lax.sin, jnp.float32(1))
    with jax.default_matmul_precision("highest"):
        second = trace_function(lax.sin, jnp.float32(1))
    with pytest.raises(ComparisonContractError) as error:
        compare_function_traces(first, second)
    result = error.value
    assert result.failure_kind == "context_mismatch"


def test_abstract_inputs_shape_dtype_and_weak_type():
    """抽象输入只需 ShapeDtypeStruct；shape、dtype 与 weak_type 都是契约的一部分。"""
    x = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    assert compare_functions(lax.neg, lax.neg, x).status == Status.EQUIVALENT
    # 形状或 dtype 不同即接口不同，不能只看两者都是 float32 数组。
    for changed in (
        jax.ShapeDtypeStruct((3, 2), jnp.float32),
        jax.ShapeDtypeStruct((2, 3), jnp.bfloat16),
    ):
        with pytest.raises(ComparisonContractError):
            compare_functions(lax.neg, lax.neg, x, right_args=(changed,))
    # Python float 是 weak type，与显式 float32 数组的抽象属性不同。
    with pytest.raises(ComparisonContractError):
        compare_functions(lax.neg, lax.neg, 1.0, right_args=(jnp.float32(1),))


def test_trace_errors_are_not_graph_differences():
    """追踪失败必须与“图不同”区分：抛 TraceError 并保留原始异常链。"""

    def invalid(x):
        return x if x > 0 else -x

    with pytest.raises(TraceError) as error:
        compare_functions(invalid, invalid, jnp.float32(1))
    result = error.value
    assert result.failure_kind == "trace_rejected"
    assert result.path == "left.forward.trace"
    # __cause__ 指向 JAX 抛出的原始异常，便于定位而不是只看到项目包装。
    assert result.__cause__ is not None


def test_cpu_compilation_errors_have_their_own_category(monkeypatch):
    """CPU lowering 失败单独分类为 cpu_compile_rejected，不与图差异或追踪失败混淆。"""

    def reject(*args, **kwargs):
        raise RuntimeError("example CPU lowering rejection")

    # 替换 jax.jit 模拟后端拒绝，避免依赖真实平台特定失败。
    monkeypatch.setattr(jax, "jit", reject)
    with pytest.raises(CompilationError) as error:
        compare_functions(lax.neg, lax.neg, jnp.float32(1), check_cpu=True)
    result = error.value
    assert result.failure_kind == "cpu_compile_rejected"
    assert result.path == "left.forward.compile"


def test_inapplicable_jvp_raises_after_successful_forward():
    """custom_vjp 上求 JVP 是 JAX 不适用的组合：前向成功后仍要报告 jvp 失败。"""

    @jax.custom_vjp
    def first(x):
        return x * x

    def fwd(x):
        return x * x, x

    def bwd(x, c):
        return (2 * x * c,)

    first.defvjp(fwd, bwd)
    # 请求 forward 与 jvp 两个模式：forward 能完成，但 jvp 在 custom_vjp 上不可用。
    with pytest.raises(TraceError) as error:
        compare_functions(first, first, jnp.float32(1), modes=("forward", "jvp"))
    result = error.value
    assert result.path == "left.jvp.trace"
    assert result.failure_kind == "trace_rejected"


@pytest.mark.parametrize("modes", [(), ("forward", "forward"), ("grad",)])
def test_invalid_mode_specification_is_rejected(modes):
    """空模式、重复模式与未支持的模式名都在入口处被拒绝。"""
    with pytest.raises(ValueError, match="modes"):
        compare_functions(lax.neg, lax.neg, jnp.float32(1), modes=modes)


def test_mathematical_inequivalence_retains_structural_report_and_output_evidence():
    """数学不等价保留结构差异与输出证据：结构层判 different，代数层给出假查询。"""
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    result = compare_functions(lambda x: x, lambda x: 2 * x, x)
    assert result.status == Status.INEQUIVALENT
    # 数学路径不使用结构对应关系，因此 equivalence_kind 为 None。
    assert result.equivalence_kind is None
    mode = result.modes[0]
    assert mode.comparison.status == Status.DIFFERENT
    assert mode.difference == "algebra.outputs[0]"
    assert mode.algebra.correspondence is None
    assert mode.algebra.backend_result.output == 0
    # 至少有一条查询返回 False，说明负结论有实际查询记录而不是默认值。
    assert any(query.answer is False for query in mode.algebra.backend_result.queries)


def test_inequivalence_cannot_hide_another_requested_mode_failure(monkeypatch):
    """一个模式的不等价不能掩盖另一个模式的失败：后者必须抛异常。"""
    import jax_reorder.functions as functions

    original_trace = functions.trace_function

    def reject_jvp(*args, **kwargs):
        """只让 jvp 追踪失败，forward 保持原实现。"""
        if kwargs.get("mode") == "jvp":
            raise NotImplementedError("unavailable JVP")
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(functions, "trace_function", reject_jvp)
    # forward 已经判 inequivalent，但 jvp 追踪失败仍必须优先抛出 TraceError。
    with pytest.raises(TraceError, match="left.jvp.trace") as error:
        functions.compare_functions(
            lambda x: x,
            lambda x: 2 * x,
            jax.ShapeDtypeStruct((), jnp.float32),
            modes=("forward", "jvp"),
        )
    assert isinstance(error.value.__cause__, NotImplementedError)


def test_derivative_inequivalence_does_not_change_the_forward_verdict():
    """前向相同但 JVP 规则不同：整体判 inequivalent，且两种结论分别保留。"""

    @jax.custom_jvp
    def first(x):
        return x

    @first.defjvp
    def first_rule(primals, tangents):
        return primals[0], tangents[0]

    @jax.custom_jvp
    def second(x):
        return x

    @second.defjvp
    def second_rule(primals, tangents):
        # 前向值相同，但切向量映射是 2*t，与 first 的 t 不同。
        return primals[0], 2 * tangents[0]

    result = compare_functions(
        first, second, jax.ShapeDtypeStruct((), jnp.float32), modes=("forward", "jvp")
    )
    assert result.status == Status.INEQUIVALENT
    forward, jvp = result.modes
    # 前向模式没有差异；差异只落在 jvp 模式的输出公式上。
    assert forward.status == Status.EQUIVALENT and forward.difference is None
    assert jvp.status == Status.INEQUIVALENT
    assert jvp.algebra.backend_result.output == 1
