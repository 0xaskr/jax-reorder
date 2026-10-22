"""Real serial-grid kernels expose gaps that simplified pointwise cases miss."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax.experimental import pallas as pl

from examples.pallas_kernels import CASES, upstream
from jax_reorder import (
    ComparisonBudgetError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.call_bindings import extract_call
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.tensor_memory import functionalize_tensor_call
from tests.pallas_oracle import run_worker
from tests.pallas_projection_oracle import at_point
from tests.pallas_upstream_state_cases import (
    FAMILIES,
    function,
    inputs,
    reference,
    trace,
    verify_sources,
)


def test_upstream_modules_match_pinned_github_sources():
    verify_sources()


@pytest.mark.parametrize("family", FAMILIES)
def test_upstream_original_is_comparable_at_both_structural_entries(family):
    source = trace(family)
    assert (
        compare_jaxprs(source, source, include_derivative_rules=False).status is Status.EQUIVALENT
    )
    result = compare_pallas_kernels(source, source)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result)


@pytest.mark.parametrize("family", FAMILIES)
def test_upstream_complete_state_projection_matches_independent_numerical_oracle(family):
    source = trace(family)
    result = compare_pallas_kernels(source, source, check_protocol=True, analysis_budget=30000)
    verify_pallas_kernel_comparison(source, source, result, analysis_budget=30000)
    projection = result.left_protocol.projection
    for seed in (7, 23):
        args = inputs(family, seed)
        expected = jax.tree.leaves(reference(family, args))
        for value, output in zip(projection.outputs, expected, strict=True):
            output = np.asarray(output)
            points = [
                tuple(0 for _ in output.shape),
                tuple(n - 1 for n in output.shape),
                tuple(n // 2 for n in output.shape),
            ]
            for point in points:
                actual = at_point(value, args, point)
                np.testing.assert_allclose(
                    actual, output[point], rtol=2e-4, atol=1 if output.dtype.kind in "iu" else 2e-4
                )


@pytest.mark.parametrize("family,mode,equivalent", CASES)
def test_existing_open_source_mutations_also_pass_through_state_projection(
    family, mode, equivalent
):
    proof = normalize_view(adapt(trace(family, mode)).view, budget=AnalysisBudget())
    args = inputs(family, 23)
    outputs = jax.tree.leaves(reference(family, args))
    matches = []
    for value, output in zip(proof.projection.outputs, outputs, strict=True):
        output = np.asarray(output)
        for point in [tuple(0 for _ in output.shape), tuple(n - 1 for n in output.shape)]:
            actual = at_point(value, args, point)
            matches.append(
                np.allclose(
                    actual, output[point], rtol=2e-4, atol=1 if output.dtype.kind in "iu" else 2e-4
                )
            )
    assert all(matches) == equivalent


@pytest.mark.parametrize(
    "target,scope", [("jax", "function"), ("retile", "function"), ("retile", "device_local")]
)
def test_upstream_serial_matmul_positive_algebra(target, scope):
    right = (
        jax.make_jaxpr(lambda a, b: a @ b)(*inputs("jax_matmul"))
        if target == "jax"
        else trace("jax_matmul", block_k=256)
    )
    left = trace("jax_matmul")
    if scope == "device_local":
        assert (
            compare_pallas_kernels(left, right, equivalence="semantic").status is Status.DIFFERENT
        )
    result = compare_algebraic_jaxprs(left, right, scope=scope)
    assert result.status is Status.EQUIVALENT
    assert result.backend_result is not None and result.warnings


def test_original_attention_moves_kv_iteration_between_grid_and_kernel_loop():
    left, right = (trace("jax_attention", block_k=k) for k in (128, 256))
    assert compare_pallas_kernels(left, right, equivalence="semantic").status is Status.EQUIVALENT
    assert compare_algebraic_jaxprs(left, right, scope="device_local").status is Status.EQUIVALENT
    a, b = (adapt(source).view for source in (left, right))
    result = compare_program_views(a, b)
    assert result.status is Status.EQUIVALENT and result.equivalence_kind == "normalized"
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())
    # The 256-wide call uses two actual dynamic slices (starts 0 and 128).
    projection = normalize_view(b, budget=AnalysisBudget()).projection
    args = inputs("jax_attention", 23)
    expected = np.asarray(reference("jax_attention", args))
    for point in [(0, 0, 0, 0), (0, 0, 64, 64), (0, 0, 127, 127)]:
        np.testing.assert_allclose(
            at_point(projection.outputs[0], args, point), expected[point], rtol=2e-4, atol=2e-4
        )


@pytest.mark.parametrize("scope", ["function", "device_local"])
@pytest.mark.parametrize("mode", ["reset_each", "overwrite", "double_state"])
def test_upstream_serial_matmul_mutation_has_complete_output_counterexample(mode, scope):
    left, right = trace("jax_matmul"), trace("jax_matmul", mode)
    if scope == "device_local":
        assert (
            compare_pallas_kernels(left, right, equivalence="semantic").status is Status.DIFFERENT
        )
    result = compare_algebraic_jaxprs(left, right, scope=scope)
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_upstream_matmul_missing_initialization_is_rejected():
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(adapt(trace("jax_matmul", "missing_init")).view, budget=AnalysisBudget())


def test_upstream_matmul_repeated_parallel_writers_are_rejected():
    call = extract_call(adapt(trace("jax_matmul")).view, budget=AnalysisBudget()).operation.call
    call = replace(call, grid=(*call.grid[:2], replace(call.grid[2], semantics="parallel")))
    with pytest.raises(UnsupportedJaxprError, match="trailing serial"):
        functionalize_tensor_call(call, budget=AnalysisBudget())


def test_upstream_serial_grid_is_budgeted_before_replay():
    source = jax.make_jaxpr(function("jax_matmul"))(
        jax.ShapeDtypeStruct((16, 128 * 2**20), jnp.float32),
        jax.ShapeDtypeStruct((128 * 2**20, 256), jnp.float32),
    )
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    with pytest.raises(ComparisonBudgetError, match="serial_grid"):
        functionalize_tensor_call(call, budget=AnalysisBudget(128))


def test_real_rope_lane_gather_out_of_bounds_promise_is_checked():
    source = trace("rope")
    outer = source.eqns[0]
    kernel = outer.params["jaxpr"]
    eqns = [
        e.replace(invars=[e.invars[0], core.Literal(np.int32(128), e.invars[1].aval)])
        if e.primitive.name == "xor"
        else e
        for e in kernel.eqns
    ]
    changed = source.replace(
        eqns=[outer.replace(params={**outer.params, "jaxpr": kernel.replace(eqns=eqns)})]
    )
    core.check_jaxpr(changed)
    assert compare_pallas_kernels(changed, changed).status is Status.EQUIVALENT
    with pytest.raises(UnsupportedJaxprError, match="in-bounds promise"):
        normalize_view(adapt(changed).view, budget=AnalysisBudget())


@pytest.mark.parametrize("mode", ["and", "or", "not", "bool_cast"])
def test_upstream_lane_gather_keeps_integer_bitwise_and_boolean_operations_distinct(mode):
    gather = upstream(0)["_lane_gather"]

    def indices(lane):
        return {
            "and": lambda: lane & 1,
            "or": lambda: lane | 1,
            "not": lambda: ~(-lane - 1),
            "bool_cast": lambda: lane.astype(bool).astype(jnp.int32),
        }[mode]()

    def kernel(x, y):
        lane = jnp.maximum(jnp.arange(128), 0)
        y[...] = gather(x[...], indices(lane))

    signature = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    source = jax.make_jaxpr(pl.pallas_call(kernel, out_shape=signature))(signature)
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget())
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    expected = x[3, np.asarray(indices(jnp.arange(128)))]
    actual = [at_point(proof.projection.outputs[0], (x,), (3, i)) for i in range(128)]
    np.testing.assert_array_equal(actual, expected)


def test_open_source_call_cost_metadata_is_preserved():
    source = trace("rope")
    outer = source.eqns[0]
    cost = pl.CostEstimate(17, 19, 23, remote_bytes_transferred=29)
    changed = source.replace(eqns=[outer.replace(params={**outer.params, "cost_estimate": cost})])
    call = extract_call(adapt(changed).view, budget=AnalysisBudget()).operation.call
    assert call.cost_estimate == (17, 19, 23, 29)
    assert compare_pallas_kernels(source, changed).status is Status.DIFFERENT


@pytest.mark.parametrize("kind", ["jvp", "vjp"])
def test_open_source_custom_derivative_forward_wrapper_keeps_captured_arguments(kind):
    kernel = function("jax_matmul")

    def wrapper(scale):
        def outer(x, y):
            capture = y * scale
            if kind == "jvp":

                @jax.custom_jvp
                def inner(a):
                    return kernel(a, capture)

                @inner.defjvp
                def rule(primals, tangents):
                    return inner(primals[0]), kernel(tangents[0], capture)
            else:

                @jax.custom_vjp
                def inner(a):
                    return kernel(a, capture)

                inner.defvjp(lambda a: (inner(a), ()), lambda _, g: (g @ capture.T,))
            return inner(x)

        return jax.make_jaxpr(outer)(*inputs("jax_matmul"))

    a, b = wrapper(jnp.float32(1)), wrapper(jnp.float32(2))
    assert compare_pallas_kernels(a, a).status is Status.EQUIVALENT
    assert compare_pallas_kernels(a, b).status is Status.DIFFERENT
    assert compare_algebraic_jaxprs(a, b).status is Status.INEQUIVALENT


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "family,mode,block_k",
    [(f, None, 128) for f in FAMILIES]
    + [("jax_matmul", None, 256), ("jax_attention", None, 256)]
    + [("jax_matmul", m, 128) for m in ("reset_each", "overwrite", "double_state")],
)
def test_upstream_kernel_actual_cpu_execution_and_mutation_witnesses(family, mode, block_k):
    expected = {"case": family, "mode": mode, "block_k": block_k, "status": "ok", "races": False}
    assert (
        run_worker(
            "tests.pallas_upstream_state_cases",
            family,
            mode,
            expected=expected,
            entry="execute",
            kwargs=dict(block_k=block_k),
        )
        == expected
    )


def lane_review_indices(lane, mode):
    return {
        "signed_div": lambda: jax.lax.div(lane - 63, jnp.int32(-3)) + 32,
        "signed_rem": lambda: jax.lax.rem(lane - 63, jnp.int32(-3)) + 4,
        "signed_sign": lambda: jnp.sign(lane - 64) + 1,
        "boolean": lambda: jnp.logical_xor(lane > 20, jnp.logical_not(lane < 80)).astype(jnp.int32),
        "min_max": lambda: jnp.minimum(jnp.maximum(lane - 32, 0), 63),
        "unsigned": lambda: (lane.astype(jnp.uint8) ^ jnp.uint8(1)).astype(jnp.int32),
        "overflow_add": lambda: jax.lax.rem(lane + jnp.int32(2**31 - 1), jnp.int32(128)),
        "overflow_div": lambda: (
            jax.lax.div(jnp.minimum(lane, jnp.int32(-(2**31))), jnp.int32(-1)) & 127
        ),
        "zero_divisor": lambda: jax.lax.div(lane, lane - lane),
        "out_of_bounds": lambda: lane + 1,
    }[mode]()


def lane_review_source(mode):
    gather = upstream(0)["_lane_gather"]

    def kernel(x, y):
        lane = jnp.maximum(jnp.arange(128), 0)
        y[...] = gather(x[...], lane_review_indices(lane, mode))

    signature = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    return jax.make_jaxpr(pl.pallas_call(kernel, out_shape=signature))(signature)


@pytest.mark.parametrize(
    "mode", ["signed_div", "signed_rem", "signed_sign", "boolean", "min_max", "unsigned"]
)
def test_vector_lane_proof_matches_signed_and_boolean_jax_semantics(mode):
    proof = normalize_view(adapt(lane_review_source(mode)).view, budget=AnalysisBudget())
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    expected = x[3, np.asarray(lane_review_indices(jnp.arange(128), mode))]
    actual = [at_point(proof.projection.outputs[0], (x,), (3, i)) for i in range(128)]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("overflow_add", "overflow"),
        ("overflow_div", "overflow"),
        ("zero_divisor", "zero divisor"),
        ("out_of_bounds", "in-bounds promise"),
    ],
)
def test_vector_lane_proof_preserves_intermediate_integer_and_address_rejections(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(adapt(lane_review_source(mode)).view, budget=AnalysisBudget())
