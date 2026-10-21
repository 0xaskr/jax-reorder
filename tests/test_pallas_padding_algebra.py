"""Allocated input padding has a separate, complete-output initialization duty."""

from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
)
from jax_reorder.algebra._typing import Equation
from jax_reorder.algebra.adapter.pallas import project
from jax_reorder.algebra.budget import Budget
from jax_reorder.algebra.expression import SymbolicTensor, coordinates
from jax_reorder.algebra.symbolic import SymbolicBuilder
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    NormalizedEvidence,
    _same_rewrite,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_padding_cases import native_tail, padded_consumer, padded_reduction
from tests.pallas_projection_oracle import at_point
from tests.test_pallas_padding_resources import padded_resources


def trace(function, *, rank=2, size=5):
    shape = (2, size) if rank == 2 else (size,)
    return jax.make_jaxpr(function)(jax.ShapeDtypeStruct(shape, jnp.float32))


def compare(function, reference, *, rank=2, size=5):
    return compare_algebraic_jaxprs(
        trace(function, rank=rank, size=size),
        trace(reference, rank=rank, size=size),
        algebra_budget=100000,
    )


@pytest.mark.parametrize("rank", [1, 2])
@pytest.mark.parametrize("kind,other", [("sum", 0), ("sum", 7), ("mean", 0), ("max", 0)])
def test_mask_before_reduction_preserves_the_original_reduction_semantics(rank, kind, other):
    def reference(x):
        if kind == "max":
            return jnp.maximum(jnp.max(x, axis=-1, keepdims=True), 0)
        result = jnp.sum(x, axis=-1, keepdims=True) + 3 * other
        return result / 8 if kind == "mean" else result

    result = compare(padded_reduction(kind=kind, other=other, rank=rank), reference, rank=rank)
    assert result.status is Status.EQUIVALENT
    assert "pallas.input-padding-validity:v1" in result.rules
    assert any(p and p.deferred_padding for p in result.normalizations)


@pytest.mark.parametrize("kind", ["sum", "mean", "max"])
def test_padding_is_not_removed_from_mean_max_or_nonzero_other(kind):
    size = 1 if kind == "max" else 5
    function = padded_reduction(kind=kind, other=7 if kind == "sum" else 0, limit=size)

    def wrong_reference(x):
        if kind == "mean":
            return jnp.mean(x, axis=-1, keepdims=True)
        if kind == "max":
            return jnp.max(x, axis=-1, keepdims=True)
        return jnp.sum(x, axis=-1, keepdims=True)

    assert compare(function, wrong_reference, size=size).status is Status.INEQUIVALENT


@pytest.mark.parametrize("kind", ["raw", "cancel", "zero", "post_mask"])
def test_undefined_lanes_cannot_escape_or_be_erased_by_arithmetic(kind):
    with pytest.raises(MathematicalUndecidedError, match="memory initialization"):
        compare(padded_reduction(kind=kind), lambda x: jnp.sum(x, axis=-1, keepdims=True))


def test_wrong_mask_that_includes_undefined_lanes_is_not_a_mathematical_difference():
    with pytest.raises(MathematicalUndecidedError, match="memory initialization"):
        compare(padded_reduction(limit=6), lambda x: jnp.sum(x, axis=-1, keepdims=True))


def test_wrong_mask_that_discards_valid_lanes_has_a_complete_output_counterexample():
    assert (
        compare(padded_reduction(limit=4), lambda x: jnp.sum(x, axis=-1, keepdims=True)).status
        is Status.INEQUIVALENT
    )


def test_two_different_graphs_with_identical_undefined_domains_cannot_match():
    with pytest.raises(MathematicalUndecidedError, match="memory initialization"):
        compare(padded_reduction(kind="cancel"), padded_reduction(kind="zero"))


def test_initialization_is_independent_of_partial_mathematical_definedness():
    result = compare(
        padded_reduction(kind="log"), lambda x: jnp.sum(jnp.log(x), axis=-1, keepdims=True)
    )
    assert result.status is Status.EQUIVALENT
    graph, _ = project(trace(padded_reduction(kind="log")).jaxpr, Budget(100000))
    (value,) = SymbolicBuilder(100000).graph(graph)
    assert value.defined != sp.true
    assert value.memory_validity == sp.true


def test_deferred_projection_is_replayable_but_not_a_normalized_equality_certificate():
    source = trace(padded_reduction())
    view = adapt(source).view
    complete = normalize_view(view, budget=AnalysisBudget(100000))
    x = np.arange(10, dtype=np.float32).reshape(2, 5)
    padded = np.pad(x, ((0, 0), (0, 3)), constant_values=12345)
    for row in range(2):
        assert at_point(complete.projection.outputs[0], (padded,), (row, 0)) == x[row].sum()
    verify_normalized_evidence(
        view, view, NormalizedEvidence(complete, complete, ((0, 0),)), budget=AnalysisBudget(100000)
    )
    proof = normalize_view(view, budget=AnalysisBudget(100000), defer_padding=True)
    replay = normalize_view(view, budget=AnalysisBudget(100000), defer_padding=True)
    assert _same_rewrite(proof, replay)
    assert not _same_rewrite(proof, replace(replay, deferred_padding=False))
    with pytest.raises(CorrespondenceError, match="algebra initialization proof"):
        verify_normalized_evidence(
            view,
            view,
            NormalizedEvidence(proof, replay, ((0, 0),)),
            budget=AnalysisBudget(100000),
        )


@pytest.mark.parametrize("mode", ["input_dma", "uninitialized", "source_write", "missing_wait"])
def test_deferred_padding_does_not_bypass_dma_or_uninitialized_scratch(mode):
    fn, shape, _ = padded_resources(mode=mode)
    source = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))
    with pytest.raises(UnsupportedJaxprError):
        normalize_view(adapt(source).view, budget=AnalysisBudget(100000), defer_padding=True)


def test_outer_jax_postprocessing_does_not_launder_partial_memory_values():
    kernel = padded_reduction(kind="raw")
    with pytest.raises(UnsupportedJaxprError, match="outer postprocessing"):
        compare(lambda x: kernel(x) + 1, lambda x: jnp.sum(x, axis=-1, keepdims=True) + 1)
    builder = SymbolicBuilder(10000)
    partial = SymbolicTensor((2,), np.dtype("float32"), sp.Symbol("x"), memory_validity=sp.false)
    eq = cast(
        Equation,
        SimpleNamespace(
            primitive=SimpleNamespace(name="copy"),
            params={},
            invars=(),
            outvars=(SimpleNamespace(aval=SimpleNamespace(shape=(2,), dtype=np.dtype("float32"))),),
        ),
    )
    with pytest.raises(UnsupportedJaxprError, match="ordinary JAX boundary"):
        builder.operation(eq, [partial], 0)


@pytest.mark.parametrize("kind,other", [("sum", 0), ("sum", 7), ("mean", 0), ("max", 0)])
def test_cpu_interpret_matches_numpy_expected_values(kind, other):
    # Negative values distinguish max(valid) from max(valid, zero padding).
    x = -np.arange(1, 11, dtype=np.float32).reshape(2, 5)
    actual = padded_reduction(kind=kind, other=other, interpret=True)(x)
    padded = np.pad(x, ((0, 0), (0, 3)), constant_values=other)
    reduce = {"sum": np.sum, "mean": np.mean, "max": np.max}[kind]
    np.testing.assert_array_equal(actual, reduce(padded, axis=-1, keepdims=True))


@pytest.mark.parametrize("kind,other", [("sum", 0), ("sum", 7), ("mean", 0), ("max", 0)])
def test_native_sized_tail_cpu_values(kind, other):
    x = -np.arange(1, 8 * 384 + 1, dtype=np.float32).reshape(8, 384)
    first, tail = x[:, :256], np.pad(x[:, 256:], ((0, 0), (0, 128)), constant_values=other)
    reduce = {"sum": np.sum, "mean": np.mean, "max": np.max}[kind]
    expected = np.concatenate(
        [np.broadcast_to(reduce(v, axis=-1, keepdims=True), (8, 128)) for v in (first, tail)],
        axis=-1,
    )
    np.testing.assert_array_equal(native_tail(kind=kind, other=other, interpret=True)(x), expected)


@pytest.mark.parametrize("kind,other", [("sum", 0), ("sum", 7), ("mean", 0)])
def test_native_sized_tail_complete_symbolic_outputs(kind, other):
    x = -np.arange(1, 8 * 384 + 1, dtype=np.float32).reshape(8, 384)
    first, tail = x[:, :256], np.pad(x[:, 256:], ((0, 0), (0, 128)), constant_values=other)
    reduce = np.mean if kind == "mean" else np.sum
    expected = np.concatenate(
        [np.broadcast_to(reduce(v, axis=-1, keepdims=True), (8, 128)) for v in (first, tail)],
        axis=-1,
    )
    source = jax.make_jaxpr(native_tail(kind=kind, other=other))(
        jax.ShapeDtypeStruct((8, 384), jnp.float32)
    )
    graph, proof = project(source.jaxpr, Budget(100000))
    (formula,) = SymbolicBuilder(100000).graph(graph)
    assert proof.deferred_padding
    assert formula.memory_validity == sp.true
    # Independent NumPy expected values also check the lowered complete output
    # formulas, including the first and last row of both output windows.
    for point in ((0, 0), (0, 128), (7, 127), (7, 255)):
        expression, _ = formula.read(point)
        expression = expression.replace(
            lambda node: isinstance(node, sp.Sum),
            lambda node: sp.Add(
                *(
                    node.function.xreplace({node.limits[0][0]: i})
                    for i in range(int(node.limits[0][1]), int(node.limits[0][2]) + 1)
                )
            ),
        )
        reads = expression.atoms(sp.Indexed)
        assert all(str(read.base.label) == "x0" for read in reads)
        evaluated = expression.xreplace(
            {read: float(x[tuple(map(int, read.indices))]) for read in reads}
        )
        assert float(evaluated) == expected[point]


@pytest.mark.parametrize("kind,other", [("sum", 0), ("sum", 7), ("mean", 0), ("mean", 7)])
def test_native_sized_masked_sum_domains_have_complete_public_cas_proofs(kind, other):
    def reference(x):
        first = jnp.sum(x[:, :256], axis=-1, keepdims=True)
        tail = jnp.sum(x[:, 256:], axis=-1, keepdims=True) + 128 * other
        if kind == "mean":
            first, tail = first / 256, tail / 256
        return jnp.concatenate(
            [jnp.broadcast_to(value, (8, 128)) for value in (first, tail)], axis=1
        )

    signature = jax.ShapeDtypeStruct((8, 384), jnp.float32)
    source = jax.make_jaxpr(native_tail(kind=kind, other=other))(signature)
    expected = jax.make_jaxpr(reference)(signature)
    result = compare_algebraic_jaxprs(source, expected, algebra_budget=100000)
    assert result.status is Status.EQUIVALENT
    assert "pallas.masked-sum-domains:v1" in result.rules


def test_native_mean_cannot_shrink_its_original_denominator():
    def wrong_reference(x):
        return jnp.concatenate(
            [
                jnp.broadcast_to(jnp.mean(part, axis=-1, keepdims=True), (8, 128))
                for part in (x[:, :256], x[:, 256:])
            ],
            axis=1,
        )

    signature = jax.ShapeDtypeStruct((8, 384), jnp.float32)
    result = compare_algebraic_jaxprs(
        jax.make_jaxpr(native_tail(kind="mean"))(signature),
        jax.make_jaxpr(wrong_reference)(signature),
        algebra_budget=100000,
    )
    assert result.status is Status.INEQUIVALENT
    assert any(
        query.answer is False and query.scope.startswith("output_coordinate=")
        for query in result.backend_result.queries
    )
    x = np.ones((8, 384), np.float32)
    actual = np.asarray(native_tail(kind="mean", interpret=True)(x))
    np.testing.assert_array_equal(actual[:, :128], 1)
    np.testing.assert_array_equal(actual[:, 128:], 0.5)
    # The wrong denominator averages 128 valid ones to 1 instead of dividing
    # the original 256-lane masked sum by 256.
    np.testing.assert_array_equal(np.mean(x[:, 256:], axis=-1), 1)


@pytest.mark.parametrize("limit", [383, 385])
def test_native_sized_wrong_mask_has_an_independent_value_or_initialization_witness(limit):
    signature = jax.ShapeDtypeStruct((8, 384), jnp.float32)
    wrong = jax.make_jaxpr(native_tail(limit=limit))(signature)
    graph, _ = project(wrong.jaxpr, Budget(100000))
    (formula,) = SymbolicBuilder(100000).graph(graph)
    if limit == 385:
        # One definite invalid output defeats a total-initialization obligation;
        # this is not a sample-based proof that the other lanes are valid.
        at_tail = formula.memory_validity.xreplace(
            dict(zip(coordinates(formula.shape), map(sp.Integer, (0, 128)), strict=True))
        )
        assert at_tail.doit() == sp.false
    else:
        x = np.ones((8, 384), np.float32)
        actual = np.asarray(native_tail(limit=limit, interpret=True)(x))
        np.testing.assert_array_equal(actual[:, :128], 256)
        np.testing.assert_array_equal(actual[:, 128:], 127)
        assert x[0, 256:].sum() == 128


@pytest.mark.parametrize("mode", ["dot", "raw_dot", "reverse_select", "undefined_predicate"])
def test_padding_validity_survives_contraction_and_predicate_selection(mode):
    def reference(x):
        return x @ x.T if mode in ("dot", "raw_dot") else jnp.sum(x, axis=1, keepdims=True)

    if mode in ("raw_dot", "undefined_predicate"):
        with pytest.raises(MathematicalUndecidedError, match="memory initialization"):
            compare(padded_consumer(mode=mode), reference)
    else:
        assert compare(padded_consumer(mode=mode), reference).status is Status.EQUIVALENT


def reference(kind="sum", other=0, *, include_padding=True):
    def run(x):
        values = []
        for position, part in enumerate((x[:, :256], x[:, 256:])):
            reduce = jnp.max if kind == "max" else jnp.sum
            value = reduce(part, axis=-1, keepdims=True)
            if position and include_padding:
                value = jnp.maximum(value, other) if kind == "max" else value + 128 * other
            if kind == "mean":
                value = value / 256
            values.append(jnp.broadcast_to(value, (8, 128)))
        return jnp.concatenate(values, axis=1)

    return run


def native_trace(function):
    return jax.make_jaxpr(function)(jax.ShapeDtypeStruct((8, 384), jnp.float32))


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_missing_valid_lane_has_a_complete_output_counterexample(kind):
    result = compare_algebraic_jaxprs(
        native_trace(native_tail(kind=kind, limit=383)),
        native_trace(reference(kind)),
        algebra_budget=100000,
    )
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


@pytest.mark.parametrize("kind", ["sum", "max"])
def test_one_uninitialized_lane_is_rejected_before_value_comparison(kind):
    with pytest.raises(MathematicalUndecidedError, match="memory initialization") as error:
        compare_algebraic_jaxprs(
            native_trace(native_tail(kind=kind, limit=385)),
            native_trace(reference(kind)),
            algebra_budget=100000,
        )
    assert error.value.path == "algebra.pallas.initialization"


@pytest.mark.parametrize("other", [0, 7, -3])
def test_large_max_proves_all_outputs_and_replays_source_projection(other):
    source, expected = (
        native_trace(native_tail(kind="max", other=other)),
        native_trace(reference("max", other)),
    )
    result = compare_algebraic_jaxprs(source, expected, algebra_budget=100000)
    assert result.status is Status.EQUIVALENT
    assert result.backend_result is not None
    assert all(q.answer is True for q in result.backend_result.queries)
    replay = normalize_view(adapt(source).view, budget=AnalysisBudget(100000), defer_padding=True)
    assert _same_rewrite(result.normalizations[0], replay)
    # CPU numerical evidence is independent of the complete symbolic proof.
    x = -np.arange(1, 8 * 384 + 1, dtype=np.float32).reshape(8, 384)
    first = np.max(x[:, :256], axis=-1, keepdims=True)
    tail = np.maximum(np.max(x[:, 256:], axis=-1, keepdims=True), other)
    oracle = np.concatenate([np.broadcast_to(v, (8, 128)) for v in (first, tail)], axis=1)
    np.testing.assert_array_equal(native_tail(kind="max", other=other, interpret=True)(x), oracle)


def test_large_max_cannot_drop_padding_for_negative_inputs():
    result = compare_algebraic_jaxprs(
        native_trace(native_tail(kind="max")),
        native_trace(reference("max", include_padding=False)),
        algebra_budget=100000,
    )
    assert result.status is Status.INEQUIVALENT
    assert any("exact rational extremum witness" in q.method for q in result.backend_result.queries)


def test_large_max_still_respects_analysis_budget():
    with pytest.raises(ComparisonBudgetError):
        compare_algebraic_jaxprs(
            native_trace(native_tail(kind="max")),
            native_trace(reference("max")),
            algebra_budget=128,
        )


def test_large_row_extrema_on_different_reduction_intervals_cannot_match():
    signature = jax.ShapeDtypeStruct((4, 40), jnp.float32)
    left = jax.make_jaxpr(lambda x: jnp.sum(jnp.max(x[:2], axis=1)))(signature)
    right = jax.make_jaxpr(lambda x: jnp.sum(jnp.max(x[2:], axis=1)))(signature)
    try:
        result = compare_algebraic_jaxprs(left, right, algebra_budget=100000)
    except MathematicalUndecidedError:
        pass
    else:
        assert result.status is Status.INEQUIVALENT
    x = np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], (4, 40))
    assert np.max(x[:2], axis=1).sum() == 1
    assert np.max(x[2:], axis=1).sum() == 5
