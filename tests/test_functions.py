"""函数级契约与前向/JVP/VJP 范围的独立验收。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import Status
from jax_reorder.functions import compare_function_traces, compare_functions, trace_function


def original(x, y):
    a = lax.sin(x)
    b = lax.cos(y)
    return {"a": a, "b": b}


def reordered(x, y):
    b = lax.cos(y)
    a = lax.sin(x)
    return {"a": a, "b": b}


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
@pytest.mark.parametrize("transforms", [(), ("jit",), ("vmap",), ("vmap", "jit"), ("jit", "vmap")])
def test_declared_function_modes(mode, transforms):
    x, y = jnp.array([0.7, 1.2]), jnp.array([-0.3, 0.8])
    result = compare_functions(
        original, reordered, x, y, modes=(mode,), transforms=transforms, check_cpu=True
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.cpu_compiled
    assert result.unverified_modes == tuple(m for m in ("forward", "jvp", "vjp") if m != mode)
    assert result.modes[0].comparison.correspondence is not None
    assert result.modes[0].left.signature.transforms == transforms


def test_tuple_list_and_dictionary_key_contracts():
    x = jnp.float32(1)

    def tuple_fn(x):
        return x, x + 1

    def list_fn(x):
        return [x, x + 1]

    result = compare_functions(tuple_fn, list_fn, x)
    assert result.status == Status.DIFFERENT
    assert "output_tree" in result.modes[0].difference
    result = compare_functions(lambda x: {"a": x}, lambda x: {"b": x}, x)
    assert result.status == Status.DIFFERENT


def test_input_pytrees_are_compared_before_flattening():
    def first(pair):
        return pair[0] - pair[1]

    values = (jnp.float32(1), jnp.float32(2))
    result = compare_functions(first, first, values, right_args=(list(values),))
    assert result.status == Status.DIFFERENT
    assert "input_tree" in result.modes[0].difference


@pytest.mark.parametrize("by_name", [False, True])
def test_static_arguments_and_non_differentiable_flags(by_name):
    def first(x, flag):
        return lax.sin(x) if flag else lax.cos(x)

    options = {"static_argnames": ("flag",)} if by_name else {"static_argnums": (1,)}
    result = compare_functions(
        first, first, jnp.float32(1), True, modes=("forward", "jvp", "vjp"), **options
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.modes[1].left.signature.differentiated_leaves == (0,)
    result = compare_functions(
        first, first, jnp.float32(1), True, right_args=(jnp.float32(1), False), **options
    )
    assert result.status == Status.DIFFERENT
    assert "static_arguments" in result.modes[0].difference


def test_keywords_static_defaults_and_differentiated_argument_subset():
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
    actual = core.eval_jaxpr(graph, graph.consts, jnp.float32(2), jnp.float32(3), jnp.float32(7))
    np.testing.assert_array_equal(actual, [6, 21])
    changed = compare_functions(
        first,
        first,
        jnp.float32(2),
        jnp.float32(3),
        kwargs={"sign": True},
        right_kwargs={"sign": False},
        static_argnames=("sign",),
    )
    assert changed.status == Status.DIFFERENT


def test_integer_leaves_remain_inputs_but_are_not_differentiated():
    def first(values):
        return values["x"] * values["count"].astype(jnp.float32)

    data = {"x": jnp.float32(2), "count": jnp.int32(3)}
    result = compare_functions(first, first, data, modes=("jvp", "vjp"))
    assert result.status == Status.EQUIVALENT, result
    assert result.modes[0].left.signature.differentiated_leaves == (1,)


def test_dynamic_cotangents_cover_each_output_not_only_their_sum():
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
        return core.eval_jaxpr(
            trace.jaxpr, trace.jaxpr.consts, jnp.float32(2), jnp.float32(ca), jnp.float32(cb)
        )[-1]

    assert derivative(left, 1, 1) == derivative(right, 1, 1)
    assert derivative(left, 1, 0) != derivative(right, 1, 0)


def test_environment_is_captured_and_does_not_share_stale_results():
    with jax.default_matmul_precision("default"):
        first = trace_function(lax.sin, jnp.float32(1))
    with jax.default_matmul_precision("highest"):
        second = trace_function(lax.sin, jnp.float32(1))
    result = compare_function_traces(first, second)
    assert result.status == Status.UNKNOWN
    assert result.failure_kind == "context_mismatch"


def test_abstract_inputs_shape_dtype_and_weak_type():
    x = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    assert compare_functions(lax.neg, lax.neg, x).status == Status.EQUIVALENT
    for changed in (
        jax.ShapeDtypeStruct((3, 2), jnp.float32),
        jax.ShapeDtypeStruct((2, 3), jnp.bfloat16),
    ):
        assert (
            compare_functions(lax.neg, lax.neg, x, right_args=(changed,)).status == Status.DIFFERENT
        )
    assert (
        compare_functions(lax.neg, lax.neg, 1.0, right_args=(jnp.float32(1),)).status
        == Status.DIFFERENT
    )


def test_trace_errors_are_not_graph_differences():
    def invalid(x):
        return x if x > 0 else -x

    result = compare_functions(invalid, invalid, jnp.float32(1))
    assert result.status == Status.UNKNOWN
    assert result.modes[0].failure_kind == "trace_rejected"
    assert len(result.modes[0].issues) == 2


def test_cpu_compilation_errors_have_their_own_category(monkeypatch):
    def reject(*args, **kwargs):
        raise RuntimeError("example CPU lowering rejection")

    monkeypatch.setattr(jax, "jit", reject)
    result = compare_functions(lax.neg, lax.neg, jnp.float32(1), check_cpu=True)
    assert result.status == Status.UNKNOWN
    assert result.modes[0].failure_kind == "cpu_compile_rejected"
    assert not result.cpu_compiled


def test_custom_vjp_forward_and_inapplicable_jvp_are_reported_separately():
    @jax.custom_vjp
    def first(x):
        return x * x

    def fwd(x):
        return x * x, x

    def bwd(x, c):
        return (2 * x * c,)

    first.defvjp(fwd, bwd)
    result = compare_functions(first, first, jnp.float32(1), modes=("forward", "jvp"))
    assert result.status == Status.UNKNOWN
    assert result.modes[0].status == Status.EQUIVALENT
    assert result.modes[0].failure_kind is None
    # custom_vjp 本身没有 JVP 规则，此项是 JAX 变换拒绝。
    assert result.modes[1].failure_kind == "trace_rejected"


@pytest.mark.parametrize("modes", [(), ("forward", "forward"), ("grad",)])
def test_invalid_mode_specification_is_rejected(modes):
    with pytest.raises(ValueError, match="modes"):
        compare_functions(lax.neg, lax.neg, jnp.float32(1), modes=modes)


def test_mathematical_inequivalence_retains_structural_report_and_output_evidence():
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    result = compare_functions(lambda x: x, lambda x: 2 * x, x)
    assert result.status == Status.INEQUIVALENT
    assert result.equivalence_kind is None
    mode = result.modes[0]
    assert mode.comparison.status == Status.DIFFERENT
    assert mode.failure_kind is None
    assert mode.difference == "algebra.outputs[0]"
    assert mode.algebra.proof is None
    assert mode.algebra.backend_result.output == 0
    assert any(query.answer is False for query in mode.algebra.backend_result.queries)


def test_proven_inequivalence_is_preserved_when_another_requested_mode_is_unknown(monkeypatch):
    import jax_reorder.functions as functions

    original_trace = functions.trace_function

    def reject_jvp(*args, **kwargs):
        if kwargs.get("mode") == "jvp":
            raise NotImplementedError("unavailable JVP")
        return original_trace(*args, **kwargs)

    monkeypatch.setattr(functions, "trace_function", reject_jvp)
    result = functions.compare_functions(
        lambda x: x,
        lambda x: 2 * x,
        jax.ShapeDtypeStruct((), jnp.float32),
        modes=("forward", "jvp"),
    )
    assert result.status == Status.INEQUIVALENT
    assert [m.status for m in result.modes] == [Status.INEQUIVALENT, Status.UNKNOWN]
    assert result.modes[1].failure_kind == "trace_rejected"
    assert result.unverified_modes == ("vjp",)


def test_derivative_inequivalence_does_not_change_the_forward_verdict():
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
        return primals[0], 2 * tangents[0]

    result = compare_functions(
        first, second, jax.ShapeDtypeStruct((), jnp.float32), modes=("forward", "jvp")
    )
    assert result.status == Status.INEQUIVALENT
    forward, jvp = result.modes
    assert forward.status == Status.EQUIVALENT and forward.difference is None
    assert jvp.status == Status.INEQUIVALENT
    assert jvp.algebra.backend_result.output == 1
