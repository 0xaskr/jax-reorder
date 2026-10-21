"""Conditional core schedules and mapping-independent interleaved grids."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    CoreExecutionBinding,
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.call_bindings import extract_call
from jax_reorder.core.coordinate_partition import equal_on_output_domain
from jax_reorder.core.core_execution import validate_binding
from jax_reorder.core.grid_execution import functionalize_banked_grid
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import compare_program_views, normalize_view
from jax_reorder.core.tensor_memory import functionalize_tensor_call
from tests.pallas_projection_oracle import at_point
from tests.pallas_scheduling_cases import banked_grid, contiguous_suffix, interleaved, routed


def trace(function, shape):
    return jax.make_jaxpr(function)(jax.ShapeDtypeStruct(shape, jnp.float32))


BINDING = CoreExecutionBinding((((0,),), ((1,),)))


@pytest.mark.parametrize("dma", [False, True])
def test_interleaved_independent_scratch_and_dma_match_grid_axis_reordering(dma):
    left = trace(interleaved(dma=dma), (16, 384))
    right = trace(interleaved(reorder=True, dma=dma), (16, 384))
    result = compare_program_views(adapt(left).view, adapt(right).view)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    proof = normalize_view(adapt(left).view, budget=AnalysisBudget()).projection
    x = np.arange(16 * 384, dtype=np.float32).reshape(16, 384)
    expected = x + np.repeat(np.arange(3), 128)[None, :]
    for row in (0, 7, 8, 15):
        for col in (0, 127, 128, 255, 256, 383):
            assert at_point(proof.outputs[0], (x,), (row, col)) == expected[row, col]
    if not dma:
        np.testing.assert_array_equal(jax.jit(interleaved(interpret=True))(x), expected)


def test_interleaved_changed_computation_is_not_equal():
    left = adapt(trace(interleaved(), (16, 384))).view
    right = adapt(trace(interleaved(delta=1), (16, 384))).view
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(left, right)


def test_interleaved_cannot_privatize_call_scratch_across_parallel_programs():
    source = trace(interleaved(persistent=True), (16, 384))
    with pytest.raises(UnsupportedJaxprError, match="owners overlap"):
        compare_pallas_kernels(source, source, check_protocol=True, execution_cores=1)


@pytest.mark.parametrize("mode", ["forward", "self"])
def test_explicit_two_core_binding_proves_targets_and_complete_output(mode):
    source = trace(routed(mode=mode), (8, 256))
    with pytest.raises(UnsupportedJaxprError, match="core target binding"):
        compare_pallas_kernels(source, source, check_protocol=True)
    result = compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)
    assert result.status is Status.EQUIVALENT
    assert result.execution == BINDING
    assert "does not enforce" in result.assumptions[-1]
    protocol = result.left_execution.protocol
    assert len(protocol.counters) == (1 if mode == "forward" else 2)
    assert {p.core for p in result.left_execution.programs} == {0, 1}
    verify_pallas_kernel_comparison(source, source, result)
    x = np.arange(8 * 256, dtype=np.float32).reshape(8, 256)
    for row in (0, 7):
        for col in (0, 127, 128, 255):
            assert (
                at_point(result.left_protocol.projection.outputs[0], (x,), (row, col))
                == x[row, col] * 2
            )


def test_binding_preserves_same_core_counter_across_distinct_programs():
    source = trace(routed(target=0), (8, 256))
    binding = CoreExecutionBinding((((0,), (1,)),))
    result = compare_pallas_kernels(source, source, check_protocol=True, execution=binding)
    verify_pallas_kernel_comparison(source, source, result)
    assert len(result.left_execution.protocol.counters) == 1


def test_swapping_core_assignment_changes_protocol_validity():
    source = trace(routed(), (8, 256))
    compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)
    swapped = CoreExecutionBinding((((1,),), ((0,),)))
    with pytest.raises(UnsupportedJaxprError, match="no consumer|conserved"):
        compare_pallas_kernels(source, source, check_protocol=True, execution=swapped)


@pytest.mark.parametrize(
    "mode,target,reason",
    [
        ("forward", 2, "unbound"),
        ("wrong_count", 1, "conserved"),
        ("cycle", 1, "deadlock|cycle|progress|unsatisfied"),
        ("read", 1, "signal/wait"),
        ("data_branch", 1, "statically resolved"),
    ],
)
def test_invalid_or_unreviewed_core_protocol_is_refused(mode, target, reason):
    source = trace(routed(mode=mode, target=target), (8, 256))
    with pytest.raises(UnsupportedJaxprError, match=reason):
        compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)


def test_explicit_binding_does_not_invent_shared_scratch_values():
    source = trace(routed(scratch=True), (8, 256))
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)


@pytest.mark.parametrize(
    "programs,reason",
    [
        ((((0,),),), "cover"),
        ((((0,),), ((0,),)), "repeats"),
        ((((0,),), ((2,),)), "out-of-range"),
        ((((1,), (0,)),), "lexicographic"),
    ],
)
def test_binding_must_cover_grid_once_in_reviewed_order(programs, reason):
    source = trace(routed(), (8, 256))
    with pytest.raises(UnsupportedJaxprError, match=reason):
        compare_pallas_kernels(
            source, source, check_protocol=True, execution=CoreExecutionBinding(programs)
        )


def test_arbitrary_axis_must_not_be_partitioned_across_cores():
    source = trace(routed(serial=True), (8, 256))
    with pytest.raises(UnsupportedJaxprError, match="one core"):
        compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)


@pytest.mark.parametrize(
    "field", ["assumptions", "binding", "protocol", "programs", "outputs", "missing"]
)
def test_core_execution_evidence_tampering_is_refused(field):
    source = trace(routed(), (8, 256))
    result = compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)
    evidence = result.left_execution
    if field == "assumptions":
        result = replace(result, assumptions=())
    elif field == "binding":
        result = replace(
            result, left_execution=replace(evidence, binding=CoreExecutionBinding((((0,), (1,)),)))
        )
    elif field == "protocol":
        result = replace(
            result, left_execution=replace(evidence, protocol=replace(evidence.protocol, edges=()))
        )
    elif field == "programs":
        result = replace(result, left_execution=replace(evidence, programs=evidence.programs[:-1]))
    elif field == "outputs":
        program = evidence.programs[0]
        program = replace(program, projection=replace(program.projection, outputs=()))
        result = replace(
            result, left_execution=replace(evidence, programs=(program, *evidence.programs[1:]))
        )
    else:
        result = replace(result, left_execution=None)
    with pytest.raises(CorrespondenceError):
        verify_pallas_kernel_comparison(source, source, result)


def test_core_grid_is_charged_before_enumeration():
    source = trace(routed(size=2**20), (8, 128 * 2**20))
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    with pytest.raises(ComparisonBudgetError, match="core_execution.grid"):
        validate_binding(BINDING, call, AnalysisBudget(128))


def test_binding_is_not_an_implicit_configuration_option():
    source = trace(routed(), (8, 256))
    for kwargs in ({}, {"check_protocol": True, "execution_cores": 1}):
        with pytest.raises(ValueError, match="execution requires"):
            compare_pallas_kernels(source, source, execution=BINDING, **kwargs)


@pytest.mark.parametrize("mode", ["base", "dma"])
def test_actual_disjoint_banks_preserve_serial_state_without_assuming_private_scratch(mode):
    source = trace(banked_grid(mode=mode), (16, 128))
    target = trace(banked_grid(reorder=True, mode=mode), (16, 128))
    with pytest.raises(UnsupportedJaxprError, match="owner-to-core stability is unbound"):
        normalize_view(adapt(source).view, budget=AnalysisBudget())
    result = compare_pallas_kernels(source, source, check_protocol=True, execution_cores=1)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result)
    proof = result.left_protocol.projection
    expected = normalize_view(adapt(target).view, budget=AnalysisBudget()).projection
    assert equal_on_output_domain(
        proof.outputs[0], expected.outputs[0], (16, 384), AnalysisBudget()
    )
    assert any("actual call-lifetime scratch" in condition for condition in proof.conditions)
    x = np.arange(16 * 128, dtype=np.float32).reshape(16, 128)
    for row in (0, 7, 8, 15):
        for a, increment in enumerate((1, 3, 6)):
            for col in (0, 127):
                assert (
                    at_point(proof.outputs[0], (x,), (row, 128 * a + col))
                    == x[row, col] + increment
                )


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("overlap", "owners overlap"),
        ("missing_init", "uninitialized tensor output/scratch"),
        ("pending", "before completion wait"),
    ],
)
def test_banked_interleaving_rejects_shared_uninitialized_or_pending_resources(mode, reason):
    source = trace(banked_grid(mode=mode), (16, 128))
    with pytest.raises(UnsupportedJaxprError, match=reason):
        compare_pallas_kernels(source, source, check_protocol=True, execution_cores=1)


def test_banked_owner_expansion_is_budgeted():
    source = trace(banked_grid(owners=2**18), (8 * 2**18, 128))
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    # Finite address refinement may exhaust the same budget before the bank
    # fallback. Both entry paths must charge before enumerating owners.
    with pytest.raises(ComparisonBudgetError, match="semantic analysis budget exhausted"):
        functionalize_tensor_call(call, budget=AnalysisBudget(500), execution_cores=1)
    with pytest.raises(ComparisonBudgetError, match="interleaved_banks.owners"):
        functionalize_banked_grid(
            call,
            budget=AnalysisBudget(500),
            coordinates=None,
            axis_ranges=None,
            execution_cores=1,
            project=lambda *_: pytest.fail("owner projection ran before budget validation"),
        )


def test_large_interleaved_independent_grid_retains_symbolic_parallel_owner():
    source = trace(interleaved(owners=2**20), (8 * 2**20, 384))
    budget = AnalysisBudget(2000)
    proof = normalize_view(adapt(source).view, budget=budget)
    assert proof.projection.outputs
    assert budget.used < 2000


@pytest.mark.parametrize("programs", [(), [], (((True,),), ((1,),)), (([0],), ((1,),))])
def test_malformed_core_binding_is_rejected(programs):
    source = trace(routed(), (8, 256))
    with pytest.raises((ValueError, UnsupportedJaxprError)):
        compare_pallas_kernels(
            source, source, check_protocol=True, execution=CoreExecutionBinding(programs)
        )


def test_binding_field_tampering_cannot_reuse_a_previous_topology_proof():
    source = trace(routed(mode="self"), (8, 256))
    result = compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)
    result = replace(result, execution=CoreExecutionBinding((((1,),), ((0,),))))
    with pytest.raises((CorrespondenceError, UnsupportedJaxprError)):
        verify_pallas_kernel_comparison(source, source, result)


def test_multicore_binding_requires_native_explicit_signal_target():
    source = trace(routed(target=None), (8, 256))
    with pytest.raises(UnsupportedJaxprError, match="explicit signal target"):
        compare_pallas_kernels(source, source, check_protocol=True, execution=BINDING)


def test_contiguous_serial_suffix_is_independent_of_interleaved_outer_grid_order():
    a = adapt(trace(contiguous_suffix(), (16, 768))).view
    b = adapt(trace(contiguous_suffix(reorder=True), (16, 768))).view
    result = compare_program_views(a, b)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
