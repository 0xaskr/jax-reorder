"""自定义规则内容、残差连接、一阶作用域及公开 AD 变换的交叉验证。"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core
from jax._src import custom_derivatives as cd

from examples import custom_derivative_cases as cases
from jax_reorder import Status, canonicalize, compare_jaxprs
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
    result = compare_jaxprs(bad, bad)
    assert result.status == Status.UNKNOWN
    assert "schema changed" in result.issues[0].reason


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
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.UNKNOWN
    assert "rule tracing rejected" in result.issues[0].reason
    result = compare_functions(invalid, invalid, *args, modes=("forward", "vjp"))
    assert result.modes[0].status == Status.EQUIVALENT
    assert result.modes[1].failure_kind == "trace_rejected"
    assert result.status == Status.UNKNOWN


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

    result = compare_functions(loop, loop, jnp.float32(0.7), jnp.int32(2), modes=("forward", "vjp"))
    assert result.modes[0].status == Status.EQUIVALENT
    assert result.modes[1].failure_kind == "trace_rejected"


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
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.UNKNOWN
    assert "latent ordered" in result.issues[0].reason
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
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.UNKNOWN
    assert "outside the custom call operands" in result.issues[0].reason
    result = compare_functions(f, f, *args, modes=("forward", "vjp"), argnums=(1,))
    assert result.status == Status.EQUIVALENT, result
