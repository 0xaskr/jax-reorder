"""自定义规则内容、残差连接、一阶作用域及公开 AD 变换的交叉验证。

覆盖：
- 正例：custom_vjp/custom_jvp 的重新排序规则、透传残差、不同显式形状下判 equivalent，且
  规范化图执行结果与原图一致、幂等、方向对称；规则在 jit/vmap 等声明变换内也保持。
- 反例：错误残差、错误前向、错误切向量系数、额外 schema 字段、静态不可微参数、整数
  pytree 叶子、混合输出/全离散输出/空输出、symbolic float0 组合、不同残差树、闭包动态与
  常量捕获、latent 有序规则，都在对应模式（forward/jvp/vjp）被判出差异。
- 拒绝路径：动态 while 的反向被 JAX 拒绝（TraceError，trace_rejected）；非法反向接口、
  规则预算耗尽（ComparisonBudgetError）、只在规则中捕获的动态值（未显式展开时拒绝）都
  显式报错，不回退成忽略。
- 前置条件：锁定 JAX 0.11.2 的 custom_derivatives 私有实现；CPU 平台；只覆盖一阶规则。
- 固定案例不证明任意高阶求导或所有 symbolic 组合。
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core
from jax._src import custom_derivatives as cd

from examples import custom_derivative_cases as cases
from jax_reorder import (
    ComparisonBudgetError,
    Status,
    TraceError,
    UnsupportedJaxprError,
    canonicalize,
    compare_jaxprs,
)
from jax_reorder.derivatives import semantic_parameters
from jax_reorder.functions import compare_functions


@pytest.mark.parametrize(
    "original,candidate",
    [
        (cases.vjp_original, cases.vjp_reordered),
        (cases.forwarded_residual, cases.forwarded_reordered),
        (cases.jvp_original, cases.jvp_reordered),
    ],
)
@pytest.mark.parametrize("shape", [(), (2, 3)])
def test_direct_custom_call_matches_rules_and_preserves_execution(original, candidate, shape):
    args = (jnp.full(shape, 0.7), jnp.full(shape, -0.3))
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (original, candidate))
    snapshots = str(a), str(b)
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    assert snapshots == (str(a), str(b))
    assert compare_jaxprs(b, a).status == Status.EQUIVALENT
    for raw, normalized in ((a, result.left), (b, result.right)):
        core.check_jaxpr(normalized.jaxpr)
        assert canonicalize(normalized.jaxpr).graph == normalized.graph
        for expected, actual in zip(
            core.eval_jaxpr(raw, raw.consts, *args),
            core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, *args),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("candidate", [cases.vjp_wrong_residual, cases.vjp_wrong_fwd])
def test_forward_and_gradient_scope_have_distinct_results(candidate):
    args = (jnp.float32(0.7), jnp.float32(-0.3))
    result = compare_functions(
        cases.vjp_original,
        candidate,
        *args,
        modes=("forward", "vjp"),
        check_cpu=True,
        equivalence="structural",
    )
    assert result.modes[0].status == Status.EQUIVALENT, result
    assert result.modes[1].status == Status.DIFFERENT, result
    assert result.status == Status.DIFFERENT
    graphs = [jax.make_jaxpr(fn)(*args) for fn in (cases.vjp_original, candidate)]
    assert compare_jaxprs(*graphs).status == Status.DIFFERENT
    ly, left = jax.vjp(cases.vjp_original, *args)
    ry, right = jax.vjp(candidate, *args)
    seeds = (jnp.float32(1.3), jnp.float32(-0.2))
    assert any(
        not np.array_equal(a, b)
        for a, b in zip((*ly, *left(seeds)), (*ry, *right(seeds)), strict=True)
    )


def test_custom_jvp_wrong_coefficient_is_found_in_each_derivative_mode():
    result = compare_functions(
        cases.jvp_original,
        cases.jvp_wrong_gradient,
        jnp.float32(0.7),
        jnp.float32(-0.3),
        modes=("forward", "jvp", "vjp"),
        equivalence="structural",
    )
    assert [m.status for m in result.modes] == [
        Status.EQUIVALENT,
        Status.DIFFERENT,
        Status.DIFFERENT,
    ]


@pytest.mark.parametrize("function", [cases.symbolic_jvp, cases.symbolic_vjp])
def test_symbolic_zero_input_activation_modes_are_materialized(function):
    graph = jax.make_jaxpr(function)(jnp.float32(0.7), jnp.float32(-0.3))
    params = semantic_parameters(graph.eqns[0])
    assert {rule[0] for rule in params["rules"]} == {(False, True), (True, False), (True, True)}
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.EQUIVALENT, result
    assert (
        compare_functions(
            function, function, jnp.float32(0.7), jnp.float32(-0.3), modes=("vjp",), argnums=(0,)
        ).status
        == Status.EQUIVALENT
    )


@pytest.mark.parametrize("transform", ["jit", "vmap", "cond", "scan"])
@pytest.mark.parametrize(
    "pair", [(cases.vjp_original, cases.vjp_reordered), (cases.jvp_original, cases.jvp_reordered)]
)
def test_custom_rules_inside_declared_transformations(transform, pair):
    def wrap(function):
        if transform == "jit":
            return jax.jit(function)
        if transform == "vmap":
            return jax.vmap(function)
        if transform == "cond":
            return lambda x, y: lax.cond(x[0] > 0, function, function, x, y)

        def scan(x, y):
            def body(carry, item):
                value = function(x, y)
                leaves = jax.tree.leaves(value)
                return carry + leaves[0], value

            return lax.scan(body, jnp.zeros_like(x), jnp.arange(2))

        return scan

    args = jnp.array([0.7, 1.2]), jnp.array([-0.3, 0.4])
    result = compare_functions(
        wrap(pair[0]), wrap(pair[1]), *args, modes=("forward", "vjp"), check_cpu=True
    )
    assert result.status == Status.EQUIVALENT, result


def test_custom_schema_addition_is_not_ignored():
    graph = jax.make_jaxpr(cases.jvp_original)(jnp.float32(0.7), jnp.float32(-0.3))
    eq = graph.eqns[0]
    assert eq.primitive is cd.custom_jvp_call_p
    bad = graph.replace(eqns=[eq.replace(params={**eq.params, "new_rule_flag": True})])
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(bad, bad)
    result = error.value
    assert "schema changed" in result.reason


def test_static_nondifferentiable_argument_and_rule_primal_call():
    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def scaled(scale, x, y):
        return scale * (x * y)

    def fwd(scale, x, y):
        return scale * (x * y), (x, y)

    def bwd(scale, residuals, cotangent):
        x, y = residuals
        return (cotangent * scale) * y, (cotangent * scale) * x

    scaled.defvjp(fwd, bwd)
    args = 3, jnp.float32(0.7), jnp.float32(-0.3)
    result = compare_functions(
        scaled, scaled, *args, static_argnums=(0,), modes=("forward", "vjp"), check_cpu=True
    )
    assert result.status == Status.EQUIVALENT, result
    graph = jax.make_jaxpr(scaled, static_argnums=(0,))(*args)
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT

    @jax.custom_jvp
    def square(x):
        return x * x

    @square.defjvp
    def rule(primals, tangents):
        (x,), (t,) = primals, tangents
        # 注册规则调用自身的普通前向是合法模式，不应无限递归展开高阶规则。
        return square(x), (2 * x) * t

    graph = jax.make_jaxpr(square)(jnp.float32(1))
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT


def test_integer_pytree_leaf_and_explicit_zero_cotangent():
    @jax.custom_vjp
    def scaled(data):
        return {"out": data["x"] * data["count"].astype(jnp.float32)}

    def fwd(data):
        return {"out": data["x"] * data["count"].astype(jnp.float32)}, data

    def bwd(data, cotangent):
        return ({"x": cotangent["out"] * data["count"].astype(jnp.float32), "count": None},)

    scaled.defvjp(fwd, bwd)
    data = {"x": jnp.array([0.7, -0.3]), "count": jnp.int32(3)}
    graph = jax.make_jaxpr(scaled)(data)
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.EQUIVALENT, result
    assert (
        compare_functions(scaled, scaled, data, modes=("forward", "vjp"), check_cpu=True).status
        == Status.EQUIVALENT
    )
    gradient = jax.grad(lambda x: scaled({**data, "x": x})["out"].sum())(data["x"])
    np.testing.assert_array_equal(gradient, np.full((2,), 3.0))


def mixed_output_vjp(scale, dtype, symbolic):
    @jax.custom_vjp
    def function(x):
        return x * x, jnp.zeros_like(x, dtype=dtype)

    def fwd(x):
        x = x.value if symbolic else x
        return (x * x, jnp.zeros_like(x, dtype=dtype)), ()

    def bwd(residual, cotangents):
        real, discrete = cotangents
        if isinstance(real, cd.SymbolicZero):
            return (None,)
        return (real if isinstance(discrete, cd.SymbolicZero) else scale * real,)

    function.defvjp(fwd, bwd, symbolic_zeros=symbolic)
    return function


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_])
@pytest.mark.parametrize("shape", [(), (2,)])
def test_mixed_output_vjp_materializes_actual_float0_cotangents(symbolic, dtype, shape):
    left, right = (mixed_output_vjp(scale, dtype, symbolic) for scale in (2, 3))
    x = jnp.full(shape, 2, dtype=jnp.float32)
    seed = jnp.full(shape, 3, dtype=jnp.float32)
    zero = np.zeros(shape, dtype=jax.dtypes.float0)
    a, b = (jax.make_jaxpr(fn)(x) for fn in (left, right))
    actual = [jax.vjp(fn, x)[1]((seed, zero))[0] for fn in (left, right)]
    np.testing.assert_array_equal(actual[0], 2 * seed)
    np.testing.assert_array_equal(actual[1], 3 * seed)
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    assert compare_jaxprs(a, a).status == Status.EQUIVALENT
    assert (
        compare_functions(left, right, x, modes=("vjp",), equivalence="structural").status
        == Status.DIFFERENT
    )
    rules = semantic_parameters(a.eqns[0], parent=a)["rules"]
    if symbolic:
        assert {rule[1] for rule in rules} == {(False, True), (True, False), (True, True)}
    for _, active, _, _, graph in rules:
        explicit = tuple(
            value for enabled, value in zip(active, (seed, zero), strict=True) if enabled
        )
        gradient = core.eval_jaxpr(graph, graph.consts, x, *explicit)[-1]
        expected = 0 * seed if not active[0] else seed if symbolic and not active[1] else 2 * seed
        np.testing.assert_array_equal(gradient, expected)


def discrete_output_vjp(offset, scale, dtype, symbolic):
    @jax.custom_vjp
    def function(x):
        return x.astype(dtype)

    def fwd(x):
        x = x.value if symbolic else x
        return (x + offset).astype(dtype), (x, x + 1)

    def bwd(residual, cotangent):
        x, _ = residual
        return (jnp.full_like(x, scale),)

    function.defvjp(fwd, bwd, symbolic_zeros=symbolic)
    return function


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.bool_])
def test_discrete_output_vjp_keeps_forward_and_backward_rules(symbolic, dtype):
    functions = [
        discrete_output_vjp(offset, scale, dtype, symbolic)
        for offset, scale in ((0, 2), (1, 2), (0, 3))
    ]
    x = jnp.float32(0)
    graphs = [jax.make_jaxpr(fn)(x) for fn in functions]
    outputs, gradients = [], []
    for function in functions:
        value, pullback = jax.vjp(function, x)
        outputs.append(value)
        gradients.append(pullback(np.zeros((), dtype=jax.dtypes.float0))[0])
    assert outputs[0] != outputs[1]
    assert gradients[0] != gradients[2]
    assert compare_jaxprs(graphs[0], graphs[0]).status == Status.EQUIVALENT
    for changed in graphs[1:]:
        assert compare_jaxprs(graphs[0], changed).status == Status.DIFFERENT
    rules = semantic_parameters(graphs[0].eqns[0], parent=graphs[0])["rules"]
    assert len(rules) == 1
    _, active, _, _, graph = rules[0]
    inputs = (x, np.zeros((), dtype=jax.dtypes.float0)) if active[0] else (x,)
    value, gradient = core.eval_jaxpr(graph, graph.consts, *inputs)
    np.testing.assert_array_equal(value, outputs[0])
    np.testing.assert_array_equal(gradient, gradients[0])


@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_no_real_output_still_checks_latent_ordered_forward_rule(symbolic, empty):
    @jax.custom_vjp
    def function(x):
        return () if empty else x.astype(jnp.int32)

    def fwd(x):
        x = x.value if symbolic else x
        lax.dce_sink(x)
        return function(x), ()

    def bwd(residual, cotangent):
        return (None,)

    function.defvjp(fwd, bwd, symbolic_zeros=symbolic)
    graph = jax.make_jaxpr(function)(jnp.float32(2))
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "latent ordered" in result.reason


@pytest.mark.parametrize("symbolic", [False, True])
def test_empty_output_keeps_forward_without_calling_inactive_backward(symbolic):
    @jax.custom_vjp
    def function(x):
        return ()

    def fwd(x):
        x = x.value if symbolic else x
        return (), x + 1

    def bwd(residual, cotangent):
        raise AssertionError("JAX does not call the backward rule for an empty output tree")

    function.defvjp(fwd, bwd, symbolic_zeros=symbolic)
    x = jnp.float32(2)
    _, pullback = jax.vjp(function, x)
    np.testing.assert_array_equal(pullback(()), (0,))
    graph = jax.make_jaxpr(function)(x)
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT
    rules = semantic_parameters(graph.eqns[0], parent=graph)["rules"]
    assert len(rules) == 1
    forward = rules[0][-1]
    assert forward.eqns and not forward.outvars


def test_symbolic_float0_output_combinations_obey_rule_budget():
    @jax.custom_vjp
    def function(x):
        return tuple((x + i).astype(jnp.int32) for i in range(7))

    def fwd(x):
        return function(x.value), ()

    def bwd(residual, cotangents):
        return (None,)

    function.defvjp(fwd, bwd, symbolic_zeros=True)
    x = jnp.float32(2)
    _, pullback = jax.vjp(function, x)
    seeds = (np.zeros((), dtype=jax.dtypes.float0),) * 7
    np.testing.assert_array_equal(pullback(seeds), (0,))
    graph = jax.make_jaxpr(function)(x)
    with pytest.raises(ComparisonBudgetError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "127 cases" in result.reason
    assert "budget exceeded" in result.reason


def test_discrete_output_does_not_hide_invalid_backward_interface():
    @jax.custom_vjp
    def function(x):
        return x.astype(jnp.int32)

    def fwd(x):
        return function(x), ()

    def bwd(residual, cotangent):
        return ()

    function.defvjp(fwd, bwd)
    graph = jax.make_jaxpr(function)(jnp.float32(2))
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "rule tracing rejected" in result.reason


def test_invalid_backward_interface_is_not_accepted_as_forward_equivalence():
    @jax.custom_vjp
    def invalid(x, y):
        return x * y

    def fwd(x, y):
        return x * y, (x, y)

    def bwd(residuals, cotangent):
        return (cotangent * residuals[1],)

    invalid.defvjp(fwd, bwd)
    args = (jnp.float32(0.7), jnp.float32(-0.3))
    graph = jax.make_jaxpr(invalid)(*args)
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "rule tracing rejected" in result.reason
    with pytest.raises(TraceError, match="left.vjp.trace"):
        compare_functions(invalid, invalid, *args, modes=("forward", "vjp"))


def test_symbolic_zero_cotangents_are_checked_as_separate_modes():
    @jax.custom_vjp
    def function(x, y):
        return x * y, x + y

    def fwd(x, y):
        return (x.value * y.value, x.value + y.value), (x.value, y.value)

    def bwd(residuals, cotangents):
        x, y = residuals
        a, b = cotangents
        if isinstance(a, cd.SymbolicZero):
            return b, b
        if isinstance(b, cd.SymbolicZero):
            return a * y, a * x
        return a * y + b, a * x + b

    function.defvjp(fwd, bwd, symbolic_zeros=True)
    graph = jax.make_jaxpr(function)(jnp.float32(0.7), jnp.float32(-0.3))
    parameters = semantic_parameters(graph.eqns[0])
    assert len(parameters["rules"]) == 9
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.EQUIVALENT, result


def test_dynamic_while_reverse_mode_is_reported_as_jax_rejection():
    def loop(x, n):
        _, result = lax.while_loop(
            lambda state: state[0] < n,
            lambda state: (state[0] + 1, state[1] * x),
            (jnp.int32(0), x),
        )
        return result

    with pytest.raises(TraceError, match="left.vjp.trace"):
        compare_functions(loop, loop, jnp.float32(0.7), jnp.int32(2), modes=("forward", "vjp"))


def test_symbolic_modes_with_different_residual_trees_are_repeatable():
    @jax.custom_vjp
    def function(x, y):
        return x * y

    def fwd(x, y):
        residual = (x.value if y.perturbed else None, y.value if x.perturbed else None)
        return x.value * y.value, residual

    def bwd(residual, c):
        x, y = residual
        return None if y is None else c * y, None if x is None else c * x

    function.defvjp(fwd, bwd, symbolic_zeros=True)
    graph = jax.make_jaxpr(function)(jnp.float32(0.7), jnp.float32(-0.3))
    for _ in range(2):
        result = compare_jaxprs(graph, graph)
        assert result.status == Status.EQUIVALENT, result


def captured_vjp(scale, x):
    @jax.custom_vjp
    def function(x):
        return scale * x

    def fwd(x):
        return scale * x, ()

    def bwd(residual, cotangent):
        return (scale * cotangent,)

    function.defvjp(fwd, bwd)
    return function(x)


def captured_jvp(scale, x):
    @jax.custom_jvp
    def function(x):
        return scale * x

    @function.defjvp
    def rule(primals, tangents):
        (x,), (t,) = primals, tangents
        return scale * x, scale * t

    return function(x)


@pytest.mark.parametrize("function", [captured_jvp, captured_vjp])
def test_dynamic_and_constant_closure_values_remain_bound(function):
    args = jnp.float32(2), jnp.array([0.7, -0.3])
    dynamic = jax.make_jaxpr(function)(*args)
    result = compare_jaxprs(dynamic, dynamic)
    assert result.status == Status.EQUIVALENT, result
    result = compare_functions(
        function, function, *args, modes=("forward", "vjp"), argnums=(1,), check_cpu=True
    )
    assert result.status == Status.EQUIVALENT, result
    np.testing.assert_array_equal(
        jax.grad(lambda x: function(args[0], x).sum())(args[1]), np.array([2, 2], np.float32)
    )
    # 同一数组对象同时用于原始调用与规则捕获，不能当成互不关联的常量。
    graph = jax.make_jaxpr(lambda x: function(args[0], x))(args[1])
    assert graph.consts
    assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT


def test_normalized_custom_call_keeps_public_vjp_execution():
    args = jnp.float32(0.7), jnp.float32(-0.3)
    graph = jax.make_jaxpr(cases.vjp_original)(*args)
    normalized = canonicalize(graph).jaxpr

    def raw(*xs):
        return tuple(core.eval_jaxpr(graph, graph.consts, *xs))

    def changed(*xs):
        return tuple(core.eval_jaxpr(normalized, normalized.consts, *xs))

    left, left_pullback = jax.vjp(raw, *args)
    right, right_pullback = jax.vjp(changed, *args)
    seeds = jnp.float32(1.3), jnp.float32(-0.2)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(left_pullback(seeds), right_pullback(seeds))


def test_latent_ordered_rules_are_not_treated_as_pure_custom_calls():
    @jax.custom_jvp
    def f(x):
        return x * x

    @f.defjvp
    def rule(primals, tangents):
        (x,), (t,) = primals, tangents
        jax.lax.dce_sink(t)
        return x * x, (2 * x) * t

    graph = jax.make_jaxpr(jax.jit(f))(jnp.float32(1))
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "latent ordered" in result.reason
    result = compare_functions(f, f, jnp.float32(1), modes=("forward", "jvp"))
    assert result.status == Status.EQUIVALENT, result


def test_rule_only_dynamic_capture_is_reported_and_explicit_vjp_is_supported():
    def f(scale, x):
        @jax.custom_vjp
        def inner(x):
            return x * x

        def fwd(x):
            return x * x, scale

        def bwd(residual, cotangent):
            return (residual * cotangent,)

        inner.defvjp(fwd, bwd)
        return inner(x)

    args = jnp.float32(3), jnp.float32(2)
    graph = jax.make_jaxpr(f)(*args)
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "outside the custom call operands" in result.reason
    result = compare_functions(f, f, *args, modes=("forward", "vjp"), argnums=(1,))
    assert result.status == Status.EQUIVALENT, result
