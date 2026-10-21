"""Public Pallas algebra bridge: full outputs, CAS evidence and refusal paths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax._src.state import types as st
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonBudgetError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_jaxprs,
)
from tests.test_pallas_semantic_slice import build, trace


def pointwise(fn, block=128, *, size=1024, inputs=1, outputs=1, interpret=False):
    sig = jax.ShapeDtypeStruct((size,), jnp.float32)
    spec = pl.BlockSpec((block,), lambda i: (i,))

    def kernel(*refs):
        values = fn(*(x[:] for x in refs[:inputs]))
        values = (values,) if outputs == 1 else values
        for ref, value in zip(refs[inputs:], values, strict=True):
            ref[:] = jnp.broadcast_to(value, (block,))

    return pl.pallas_call(
        kernel,
        out_shape=sig if outputs == 1 else (sig,) * outputs,
        grid=(size // block,),
        in_specs=(spec,) * inputs,
        out_specs=spec if outputs == 1 else (spec,) * outputs,
        interpret=interpret,
    )


def traced(fn, size=1024, inputs=1):
    return jax.make_jaxpr(fn)(*(jax.ShapeDtypeStruct((size,), jnp.float32),) * inputs)


def mathematical(a, b, **kwargs):
    assert compare_jaxprs(a, b).status is Status.DIFFERENT
    result = compare_algebraic_jaxprs(a, b, **kwargs)
    assert result.correspondence is None
    assert result.backend_result.backend == "sympy"
    assert result.backend_result.queries
    assert result.warnings and result.normalizations
    assert "blocked-memory-to-global-values:v2" in result.rules
    return result


@pytest.mark.parametrize("mode", ["add", "nonlinear"])
@pytest.mark.parametrize("reverse", [False, True])
def test_retiling_uses_complete_projection_and_trusted_cas(mode, reverse):
    result = mathematical(trace(128, mode=mode), trace(256, reverse, mode=mode))
    assert result.status is Status.EQUIVALENT
    assert all(p.projection.conditions for p in result.normalizations)


def test_tail_domain_is_symbolic_even_for_large_arrays():
    size = 2**24 + 13
    result = mathematical(
        trace(128, size=size, tail=True), trace(256, True, size=size, tail=True), algebra_budget=512
    )
    assert result.status is Status.EQUIVALENT
    assert result.work <= 512


@pytest.mark.parametrize("candidate", [lambda x: 2 * x + 2, lambda x: (x + 1) * 2])
def test_different_arithmetic_graphs_are_compared_after_memory_lowering(candidate):
    result = mathematical(
        traced(pointwise(lambda x: (x + 1) + (x + 1))), traced(pointwise(candidate, 256))
    )
    assert result.status is Status.EQUIVALENT


@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_pallas_and_jax_share_ordered_external_inputs(swap, wrapped):
    pallas = pointwise(lambda x, y: (x - y) + (x - y), inputs=2)

    def ordinary(x, y):
        return 2 * (x - y)

    a, b = (jax.jit(pallas), jax.jit(ordinary)) if wrapped else (pallas, ordinary)
    if swap:
        a, b = b, a
    result = mathematical(traced(a, inputs=2), traced(b, inputs=2))
    assert result.status is Status.EQUIVALENT
    assert sum(p is not None for p in result.normalizations) == 1


def test_complete_output_difference_has_a_concrete_coordinate_query():
    result = mathematical(trace(), trace(mode="add_two"))
    assert result.status is Status.INEQUIVALENT and result.output == 0
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_multiple_outputs_cannot_be_swapped_or_partially_matched():
    p = pointwise(lambda x: (x + 1, x * 2), outputs=2)
    a = traced(p)
    assert mathematical(a, traced(lambda x: (1 + x, 2 * x))).status is Status.EQUIVALENT
    result = mathematical(a, traced(lambda x: (1 + x, 3 * x)))
    assert result.status is Status.INEQUIVALENT and result.output == 1
    assert mathematical(a, traced(lambda x: (2 * x, 1 + x))).status is Status.INEQUIVALENT


def test_cast_and_nonfinite_values_do_not_acquire_real_algebra_identities():
    def fn(x):
        return x.astype(jnp.bfloat16).astype(jnp.float32)

    assert mathematical(traced(pointwise(fn)), traced(fn)).status is Status.EQUIVALENT
    with pytest.raises(MathematicalUndecidedError):
        mathematical(traced(pointwise(fn)), traced(lambda x: x))
    with pytest.raises(MathematicalUndecidedError):
        mathematical(
            traced(pointwise(lambda x: jnp.log(x) - jnp.log(x))),
            traced(lambda x: jnp.zeros_like(x)),
        )


@pytest.mark.parametrize("mode", ["uninitialized", "duplicate", "input_write", "partial"])
def test_effects_and_unsafe_memory_cannot_be_erased_by_algebra(mode):
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_algebraic_jaxprs(trace(mode=mode), trace())
    assert error.value.stage == "algebra"
    assert error.value.failure_kind == "unsupported_conversion"


def test_retained_unused_effect_is_checked_even_if_frontend_normally_eliminates_it():
    graph = trace()
    outer = graph.eqns[0]
    kernel = outer.params["jaxpr"]
    unused = kernel.eqns[0].replace(
        invars=[kernel.invars[1]],
        outvars=[core.DropVar(kernel.eqns[0].outvars[0].aval)],
        effects={st.ReadEffect(kernel.invars[1])},
    )
    kernel = kernel.replace(eqns=[unused, *kernel.eqns], effects=kernel.effects | unused.effects)
    changed = graph.replace(eqns=[outer.replace(params={**outer.params, "jaxpr": kernel})])
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        compare_algebraic_jaxprs(changed, graph)


def test_program_id_used_as_data_is_not_erased_by_retiling():
    result = mathematical(trace(128, mode="program_id"), trace(256, mode="program_id"))
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_equal_read_sets_do_not_erase_the_coordinate_to_value_mapping():
    result = mathematical(trace(256), trace(256, in_mapping=lambda p: (3 - p,)))
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


def test_regular_semaphore_is_quiescent_between_independent_programs():
    def kernel(x, y, sem):
        pl.semaphore_signal(sem, 1)
        pl.semaphore_wait(sem, 1)
        y[:] = x[:] + 1

    sig = jax.ShapeDtypeStruct((1024,), jnp.float32)
    spec = pl.BlockSpec((128,), lambda i: (i,))
    fn = pl.pallas_call(
        kernel,
        out_shape=sig,
        grid=(8,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.REGULAR,),
    )
    a = traced(fn)
    assert compare_jaxprs(a, a).status is Status.EQUIVALENT
    result = compare_algebraic_jaxprs(a, trace())
    assert result.status is Status.EQUIVALENT
    assert "local-control-memory-to-values:v1" in result.rules


def test_local_dma_projection_retains_progress_assumption_and_wait_evidence():
    from tests.test_pallas_dma import copy, function, relayed

    for fn in (function(copy), function(relayed, arrays=1)):
        result = mathematical(traced(fn, size=128), traced(lambda x: x, size=128))
        assert result.status is Status.EQUIVALENT
        assert any("eventually completes" in text for text in result.assumptions)
        assert result.normalizations[0].projection.communication
        assert result.normalizations[0].projection.waits


def test_split_dma_preserves_piecewise_output_coordinates():
    from tests.test_pallas_dma import function

    def kernel(x, y, semaphore):
        tpu.async_copy(x.at[128:], y.at[:128], semaphore).wait()
        tpu.async_copy(x.at[:128], y.at[128:], semaphore).wait()

    a = traced(function(kernel, size=256, block=256), size=256)
    b = traced(lambda x: jnp.concatenate((x[128:], x[:128])), size=256)
    result = mathematical(a, b)
    assert result.status is Status.EQUIVALENT
    assert result.normalizations[0].projection.outputs[0].operation == "select_n"
    assert mathematical(a, traced(lambda x: x, size=256)).status is Status.INEQUIVALENT


def test_public_auto_mode_and_cpu_oracle():
    a, b = pointwise(lambda x: (x + 1) * 2), pointwise(lambda x: 2 * x + 2, 256)
    result = compare_functions(a, b, jax.ShapeDtypeStruct((1024,), jnp.float32))
    assert result.status is Status.EQUIVALENT and result.equivalence_kind == "algebraic"
    x = jnp.linspace(-1, 1, 1024)
    for block, formula in ((128, lambda x: (x + 1) * 2), (256, lambda x: 2 * x + 2)):
        actual = pointwise(formula, block, interpret=True)(x)
        np.testing.assert_allclose(actual, 2 * np.asarray(x) + 2, rtol=2e-6, atol=2e-6)


def test_analysis_and_cas_share_the_public_algebra_budget():
    with pytest.raises(ComparisonBudgetError) as error:
        compare_algebraic_jaxprs(trace(128), trace(256), algebra_budget=2)
    assert error.value.stage == "algebra"


def test_nondefault_compiler_options_are_not_dropped():
    with pytest.raises(UnsupportedJaxprError):
        compare_algebraic_jaxprs(
            traced(build(128, compiler_params=tpu.CompilerParams(flags={"test_flag": True}))),
            trace(256),
        )


def test_shard_local_projection_cannot_be_used_as_global_outputs():
    from tests.test_pallas_sharding import sharded

    with pytest.raises(UnsupportedJaxprError, match="global shard assembly") as error:
        compare_algebraic_jaxprs(traced(sharded(128), size=2048), traced(sharded(256), size=2048))
    assert error.value.stage == "algebra"
    assert error.value.failure_kind == "unsupported_conversion"
