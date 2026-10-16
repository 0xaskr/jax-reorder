"""用跨算子组合、索引反例和原图证据验收标量语义 IR，禁止逐元素展开。"""

import jax
import jax.numpy as jnp
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
)


def compare(left, right, shape, dtype=jnp.float32, **options):
    """实际追踪两个程序，检查的是代数层而不是执行样本。"""
    value = jax.ShapeDtypeStruct(shape, dtype)
    a, b = jax.make_jaxpr(left)(value), jax.make_jaxpr(right)(value)
    return a, b, compare_algebraic_jaxprs(a, b, **options)


def equivalent(left, right, shape, dtype=jnp.float32):
    """必须走代数证明；结构快路不能代替本轮验收。"""
    a, b, result = compare(left, right, shape, dtype)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "algebraic", result
    assert result.backend_result is not None
    return result


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("shape", [(3, 4), (2, 1, 3), (0, 3), (3, 0), (), (1, 1)])
def test_full_reshape_sum_uses_index_bijection(shape, dtype):
    equivalent(jnp.sum, lambda x: jnp.sum(x.reshape(-1)), shape, dtype)


@pytest.mark.parametrize("shape", [(5,), (3,), (5, 3), (5, 1, 2), (0, 3), (1, 3)])
@pytest.mark.parametrize("cut", [0, 1, 2, 5])
def test_partition_domain_union_keeps_empty_and_singleton_blocks(shape, cut):
    equivalent(jnp.sum, lambda x: jnp.sum(x[:cut]) + jnp.sum(x[cut:]), shape)


@pytest.mark.parametrize("function", [lambda x: x, jnp.sin, jnp.abs, lambda x: x * x + 2 * x])
def test_reshape_flat_partition_composes_through_scalar_functions(function):
    equivalent(
        lambda x: jnp.sum(function(x)),
        lambda x: jnp.sum(function(x.reshape(-1)[:5])) + jnp.sum(function(x.reshape(-1)[5:])),
        (3, 4),
    )


@pytest.mark.parametrize("function", [jnp.sin, jnp.abs, lambda x: jnp.where(x > 1, x, -x)])
def test_shape_maps_commute_with_scalar_calls(function):
    equivalent(lambda x: function(x).reshape(2, 6), lambda x: function(x.reshape(2, 6)), (3, 4))
    equivalent(lambda x: function(x)[1:3, 1:], lambda x: function(x[1:3, 1:]), (4, 5))
    equivalent(lambda x: function(x).T, lambda x: function(x.T), (3, 4))


@pytest.mark.parametrize("axis", [0, 1])
def test_partial_axis_reduction_and_multidimensional_blocks(axis):
    equivalent(
        lambda x: jnp.sum(x, axis=axis),
        lambda x: (
            jnp.sum(x[:2], axis=0) + jnp.sum(x[2:], axis=0)
            if axis == 0
            else jnp.sum(x[:, :2], axis=1) + jnp.sum(x[:, 2:], axis=1)
        ),
        (5, 4),
    )
    equivalent(
        jnp.sum,
        lambda x: jnp.sum(x[:2, :1]) + jnp.sum(x[2:, :1]) + jnp.sum(x[:2, 1:]) + jnp.sum(x[2:, 1:]),
        (5, 4),
    )


def test_views_compose_without_source_operator_patterns():
    equivalent(lambda x: x, lambda x: x.reshape(2, 6).reshape(3, 4), (3, 4))
    equivalent(jnp.sum, lambda x: jnp.sum(x.T.reshape(2, 6)), (3, 4))
    equivalent(jnp.sum, lambda x: jnp.sum(x[:2].reshape(-1)) + jnp.sum(x[2:].reshape(-1)), (5, 4))
    equivalent(lambda x: x[1:5:2], lambda x: x[1:][::2], (5, 4))


@pytest.mark.parametrize(
    "wrong",
    [
        lambda x: jnp.sum(x[:2]) + jnp.sum(x[3:]),
        lambda x: jnp.sum(x[:3]) + jnp.sum(x[2:]),
        lambda x: jnp.sum(x[:2]) + jnp.sum(x[:2]),
        lambda x: jnp.sum(x[::2]),
        lambda x: jnp.sum(x.reshape(-1)[:5]) + jnp.sum(x.reshape(-1)[6:]),
    ],
)
def test_domain_gaps_overlap_repetition_and_stride_never_match_full_sum(wrong):
    a, b, result = compare(jnp.sum, wrong, (5, 4))
    assert result.status == Status.INEQUIVALENT, result
    assert result.backend_result is not None


def test_output_positions_and_shape_mappings_remain_observable():
    for wrong in (lambda x: x.T.reshape(3, 4), lambda x: x.reshape(4, 3).T):
        a, b, result = compare(lambda x: x, wrong, (3, 4))
        assert result.status == Status.INEQUIVALENT, result
        assert result.backend_result is not None
    _, _, result = compare(lambda x: jnp.sin(jnp.sum(x)), lambda x: jnp.sum(jnp.sin(x)), (5,))
    assert result.status == Status.INEQUIVALENT, result


def test_partial_function_domain_cannot_be_extended_by_prefix_cancellation():
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare(
            lambda x: jnp.sum(jnp.log(x[2:])),
            lambda x: jnp.sum(jnp.log(x)) - jnp.sum(jnp.log(x[:2])),
            (5,),
        )
    equivalent(lambda x: jnp.sum(jnp.log(x)), lambda x: jnp.sum(jnp.log(x.reshape(-1))), (3, 4))


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_reshape_and_scalar_scan_derivatives_are_checked_separately(mode):
    x = jnp.ones((5, 3))
    for left, right in (
        (jnp.sum, lambda x: jnp.sum(x.reshape(-1))),
        (
            lambda x: jax.lax.scan(lambda c, a: (c + jnp.sin(a), None), jnp.zeros(3), x)[0],
            lambda x: jnp.sum(jnp.sin(x), axis=0),
        ),
    ):
        result = compare_functions(left, right, x, modes=(mode,))
        assert result.status == Status.EQUIVALENT, result


def test_large_shapes_preserve_symbolic_work_and_obey_budget():
    reports = []
    for size in (20, 1000000):

        def left(x):
            return jnp.sum(jnp.sin(x))

        def right(x):
            return jnp.sum(jnp.sin(x[:7])) + jnp.sum(jnp.sin(x[7:]))

        _, _, result = compare(left, right, (size, 3))
        assert result.status == Status.EQUIVALENT, result
        reports.append(result)
        with pytest.raises(ComparisonBudgetError) as error:
            compare(left, right, (size, 3), algebra_budget=1)
        limited = error.value
        assert "budget" in limited.reason
    assert reports[0].work == reports[1].work


def test_broadcast_predicate_keeps_singleton_source_coordinate():
    _, _, wrong = compare(jnp.abs, lambda x: jnp.where(x[:1] > 0, x, -x), (5,))
    assert wrong.status == Status.INEQUIVALENT, wrong
    equivalent(
        lambda x: jnp.where(x[:1] > 0, x, -x),
        lambda x: jnp.where(jnp.broadcast_to(x[:1], x.shape) > 0, x, -x),
        (5,),
    )
