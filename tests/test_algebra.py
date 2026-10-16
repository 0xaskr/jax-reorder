"""索引代数的数学正例、不可误接受的连接与数值边界。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import (
    Status,
    compare_algebraic_jaxprs,
    compare_function_traces,
    compare_functions,
    compare_jaxprs,
    trace_function,
)


def signature(shape=(), dtype=jnp.float32):
    """只提供签名，避免测试依赖某个样本数值。"""
    return jax.ShapeDtypeStruct(shape, dtype)


def algebra(left, right, *args, **options):
    """从实际 JAX 追踪图调用底层代数入口。"""
    a, b = jax.make_jaxpr(left)(*args), jax.make_jaxpr(right)(*args)
    result = compare_algebraic_jaxprs(a, b, **options)
    return result


POLYNOMIALS = [
    (lambda a, b, c: a + b, lambda a, b, c: b + a),
    (lambda a, b, c: a * b, lambda a, b, c: b * a),
    (lambda a, b, c: (a + b) + c, lambda a, b, c: a + (c + b)),
    (lambda a, b, c: (a * b) * c, lambda a, b, c: a * (c * b)),
    (lambda a, b, c: a * (b + c), lambda a, b, c: a * b + a * c),
    (lambda a, b, c: (a + b) * (a - b), lambda a, b, c: a * a - b * b),
    (lambda a, b, c: (a + b) ** 2, lambda a, b, c: a * a + 2 * a * b + b * b),
    (lambda a, b, c: a + a + a, lambda a, b, c: 3 * a),
    (lambda a, b, c: lax.neg(lax.neg(a)), lambda a, b, c: a),
    (lambda a, b, c: a + 0, lambda a, b, c: a),
    (lambda a, b, c: a * 1, lambda a, b, c: a),
    (lambda a, b, c: a - a, lambda a, b, c: jnp.zeros_like(a)),
    (lambda a, b, c: a**0, lambda a, b, c: jnp.ones_like(a)),
    (lambda a, b, c: jnp.sin(a + b), lambda a, b, c: jnp.sin(b + a)),
    (lambda a, b, c: (a * b) + (a * b), lambda a, b, c: 2 * (a * b)),
]


@pytest.mark.parametrize("pair", POLYNOMIALS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_elementwise_polynomial_rules(pair, dtype):
    result = algebra(*pair, *(signature((2, 3), dtype) for _ in range(3)))
    assert result.status == Status.EQUIVALENT, result
    assert result.backend_result and result.assumptions and result.warnings


@pytest.mark.parametrize("shape", [(0,), (1,), (3,), (2, 1, 3)])
def test_polynomial_empty_unit_and_broadcast_dimensions(shape):
    result = algebra(
        lambda a, b: a * (b + 1), lambda a, b: a * b + a, signature(shape), signature(())
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("dims", [(2, 3, 4), (3, 3, 3), (2, 0, 4), (2, 1, 4), (0, 3, 4)])
def test_matmul_and_broadcast_reduce_are_the_same_index_expression(dims):
    m, k, n = dims
    result = algebra(
        lambda a, b: a @ b,
        lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1),
        signature((m, k)),
        signature((k, n)),
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("batch", [(), (2,), (1, 2)])
def test_matrix_chain_and_distributivity(dtype, batch):
    a, b, c = [signature((*batch, *shape), dtype) for shape in ((2, 3), (3, 4), (4, 5))]
    result = algebra(lambda a, b, c: (a @ b) @ c, lambda a, b, c: a @ (b @ c), a, b, c)
    assert result.status == Status.EQUIVALENT, result
    result = algebra(lambda a, b, c: a @ (b + c), lambda a, b, c: a @ b + a @ c, a, b, b)
    assert result.status == Status.EQUIVALENT, result
    result = algebra(lambda a, b, c: (a + b) @ c, lambda a, b, c: a @ c + b @ c, a, a, b)
    assert result.status == Status.EQUIVALENT, result


def test_batch_broadcast_transpose_and_einsum():
    a, b = signature((1, 2, 3)), signature((5, 3, 4))
    result = algebra(
        lambda a, b: a @ b,
        lambda a, b: jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2),
        a,
        b,
    )
    assert result.status == Status.EQUIVALENT, result
    a, b = signature((2, 3)), signature((3, 4))
    for other in (lambda a, b: (b.T @ a.T).T, lambda a, b: jnp.einsum("ik,kj->ij", a, b)):
        assert algebra(lambda a, b: a @ b, other, a, b).status == Status.EQUIVALENT


@pytest.mark.parametrize("shape", [(2, 3), (3, 3), (0, 3), (3, 0), (1, 1)])
def test_independent_sum_order_and_linearity(shape):
    a = signature(shape)
    result = algebra(
        lambda x: jnp.sum(jnp.sum(x, axis=0), axis=0),
        lambda x: jnp.sum(jnp.sum(x, axis=1), axis=0),
        a,
    )
    assert result.status == Status.EQUIVALENT, result
    result = algebra(lambda x, y: jnp.sum(x + y), lambda x, y: jnp.sum(x) + jnp.sum(y), a, a)
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("length", [0, 1, 7, 4096])
def test_unused_sum_index_contributes_its_extent(length):
    result = algebra(
        lambda x: jnp.sum(jnp.broadcast_to(x, (length,))), lambda x: x * length, signature()
    )
    assert result.status == Status.EQUIVALENT, result


def test_product_of_independent_sums_retains_all_cross_terms():
    x = signature((4,))
    result = algebra(
        lambda a, b: jnp.sum(a) * jnp.sum(b), lambda a, b: jnp.sum(a[:, None] * b[None, :]), x, x
    )
    assert result.status == Status.EQUIVALENT, result
    wrong = algebra(lambda a, b: jnp.sum(a) * jnp.sum(b), lambda a, b: jnp.sum(a * b), x, x)
    assert wrong.status == Status.INEQUIVALENT
    assert wrong.backend_result.output is not None


@pytest.mark.parametrize(
    "pair,shapes",
    [
        ((lambda a, b: a - b, lambda a, b: b - a), ((3,), (3,))),
        ((lambda a, b: a @ b, lambda a, b: b @ a), ((3, 3), (3, 3))),
        ((lambda a, b: jnp.sum(a, axis=0), lambda a, b: jnp.sum(a, axis=1)), ((3, 3), ())),
        ((lambda a, b: a + b[:, None], lambda a, b: a + b[None, :]), ((3, 3), (3,))),
        ((lambda a, b: (a, b), lambda a, b: (b, a)), ((3,), (3,))),
        ((lambda a, b: a @ b, lambda a, b: a @ b.T), ((3, 3), (3, 3))),
    ],
)
def test_wrong_index_connections_have_complete_output_negative_proofs(pair, shapes):
    result = algebra(*pair, *(signature(s) for s in shapes))
    assert result.status == Status.INEQUIVALENT, result
    assert result.correspondence is None
    assert result.backend_result.output is not None


def test_unsupported_maximum_difference_remains_unknown():
    result = algebra(
        lambda a, b: jnp.max(a + b),
        lambda a, b: jnp.max(a) + jnp.max(b),
        signature((3,)),
        signature((3,)),
    )
    assert result.status == Status.UNKNOWN
    assert result.correspondence is None


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.complex64])
def test_non_real_domains_do_not_gain_commutativity(dtype):
    result = algebra(
        lambda a, b: a + b, lambda a, b: b + a, signature((3,), dtype), signature((3,), dtype)
    )
    assert result.status == Status.UNKNOWN


def test_precision_cast_and_partial_function_boundaries():
    x = signature((3, 3))
    result = algebra(
        lambda a, b: lax.dot_general(a, b, (((1,), (0,)), ((), ())), precision=lax.Precision.HIGH),
        lambda a, b: a @ b,
        x,
        x,
    )
    assert result.status == Status.UNKNOWN
    result = algebra(lambda a: a.astype(jnp.bfloat16).astype(jnp.float32), lambda a: a, x)
    assert result.status == Status.UNKNOWN
    for partial in (jnp.log, jnp.sqrt, lambda x: 1 / x):
        for left in (
            lambda x, partial=partial: 0 * partial(x),
            lambda x, partial=partial: partial(x) - partial(x),
        ):
            result = algebra(left, lambda x: jnp.zeros_like(x), signature((3,)))
            assert result.status == Status.UNKNOWN


def test_dead_pure_work_is_ignored_only_after_complete_semantic_check():
    def unused(x):
        _ = x * x + jnp.sin(x)
        return x + 1

    result = algebra(unused, lambda x: 1 + x, signature((3,)))
    assert result.status == Status.EQUIVALENT, result

    def unsafe(x):
        jax.debug.print("value {}", x)
        return x + 1

    assert algebra(unsafe, lambda x: x + 1, signature()).status == Status.UNKNOWN
    assert (
        algebra(lambda x: x + jnp.inf, lambda x: jnp.inf + x, signature()).status == Status.UNKNOWN
    )


def test_default_auto_retains_structural_report_and_explicit_mode():
    a, b, c = (signature((3,)) for _ in range(3))
    left, right = POLYNOMIALS[4]
    auto = compare_functions(left, right, a, b, c, modes=("forward", "jvp", "vjp"), check_cpu=True)
    assert auto.status == Status.EQUIVALENT, auto
    assert auto.equivalence_kind == "algebraic" and auto.cpu_compiled
    assert all(
        m.algebra and m.algebra.backend_result and m.comparison.status == Status.DIFFERENT
        for m in auto.modes
    )
    structural = compare_functions(left, right, a, b, c, equivalence="structural")
    assert structural.status == Status.DIFFERENT and structural.modes[0].algebra is None
    same = compare_functions(left, left, a, b, c)
    assert same.equivalence_kind == "structural" and same.modes[0].algebra is None
    graph = jax.make_jaxpr(left)(a, b, c)
    assert compare_jaxprs(graph, jax.make_jaxpr(right)(a, b, c)).status == Status.DIFFERENT


def test_signature_and_environment_mismatch_do_not_enter_algebra():
    x = signature((2,))
    assert (
        compare_functions(lambda x: x, lambda x: x, x, right_args=(signature((3,)),)).status
        == Status.DIFFERENT
    )
    left = trace_function(lambda x: x, x)
    right = replace(left, context=("changed",))
    report = compare_function_traces(left, right)
    assert report.status == Status.UNKNOWN and report.algebra is None


def test_input_abstract_memory_and_manual_axis_are_not_lost():
    from jax_reorder.functions import _abstract, _encoder

    mesh = jax.sharding.AbstractMesh((1,), ("x",), axis_types=(jax.sharding.AxisType.Manual,))
    for value in (
        jax.ShapeDtypeStruct((2,), jnp.float32, _memory_space=core.MemorySpace.Host),
        jax.ShapeDtypeStruct(
            (2,),
            jnp.float32,
            sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
            manual_axis_type=core.ManualAxisType(varying=frozenset({"x"})),
        ),
    ):
        original = core.typeof(value)
        reproduced = core.typeof(_abstract(value))
        assert _encoder().encode(original, "input") == _encoder().encode(reproduced, "input")


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        compare_functions(lambda x: x, lambda x: x, signature(), algebra_budget=budget)


def test_budget_is_deterministic_and_large_axes_are_symbolic():
    f, g = POLYNOMIALS[6]
    x = signature((4,))
    a = algebra(f, g, x, x, x, algebra_budget=1)
    b = algebra(g, f, x, x, x, algebra_budget=1)
    assert a.status == b.status == Status.UNKNOWN
    assert "budget" in a.issues[0].reason
    assert a == algebra(f, g, x, x, x, algebra_budget=1)
    reports = []
    for k in (3, 4096):
        reports.append(
            algebra(
                lambda a, b: a @ b,
                lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1),
                signature((2, k)),
                signature((k, 4)),
            )
        )
    assert all(r.status == Status.EQUIVALENT for r in reports)
    assert reports[0].work == reports[1].work


def test_real_equivalence_keeps_floating_point_witnesses_and_warning():
    def f(a, b, c):
        return (a + b) + c

    def g(a, b, c):
        return a + (b + c)

    args = tuple(jnp.float32(x) for x in (33554432, -33554432, 1))
    report = compare_functions(f, g, *args)
    assert report.status == Status.EQUIVALENT
    assert f(*args) != g(*args)
    assert "bitwise" in report.modes[0].algebra.warnings[0]
    cases = [
        (lambda x: (x + x) - x, lambda x: x, jnp.finfo(jnp.float32).max),
        (lambda x: x * x - x * x, lambda x: jnp.zeros_like(x), jnp.finfo(jnp.float32).max),
        (lambda x: x + 0, lambda x: x, -0.0),
    ]
    for left, right, value in cases:
        x = jnp.float32(value)
        result = compare_functions(left, right, x)
        assert result.status == Status.EQUIVALENT, result
        assert np.asarray(left(x)).tobytes() != np.asarray(right(x)).tobytes()
