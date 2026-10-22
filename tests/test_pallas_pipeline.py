"""Real JAX double buffers: complete value proofs, hazards, replay and CPU runs."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_pallas_kernels,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.pallas_oracle import run_worker
from tests.pallas_pipeline_cases import (
    environment,
    inputs,
    serial_grid_function,
    trace,
    verify_sources,
)
from tests.pallas_projection_oracle import at_point


def projection(family="copy", mode=None, **kwargs):
    with environment():
        return normalize_view(
            adapt(trace(family, mode, **kwargs)).view, budget=AnalysisBudget(100000)
        ).projection


def test_pipeline_sources_match_fixed_github_revision():
    verify_sources()


@pytest.mark.parametrize("family", ["copy", "matmul"])
@pytest.mark.parametrize(
    "variant", ["early_wait", "four_buffers", "synchronous", "different_wait_descriptor"]
)
def test_real_pipeline_schedule_changes_have_equal_complete_outputs(family, variant):
    options = (
        {"buffers": 4}
        if variant == "four_buffers"
        else {"synchronous": True}
        if variant == "synchronous"
        else {"mode": variant}
    )
    with environment():
        left, right = adapt(trace(family)).view, adapt(trace(family, **options)).view
        result = compare_program_views(left, right, analysis_budget=100000)
        assert result.status is Status.EQUIVALENT
        assert result.equivalence_kind == "normalized"
        verify_normalized_evidence(left, right, result.evidence, budget=AnalysisBudget(100000))
        transfers = result.evidence.left.projection.communication
        assert any(
            "iteration" in t.wait.path and t.wait.path != t.transfer.start.path for t in transfers
        )
        assert any(t.transfer.generation > 1 for t in transfers)


@pytest.mark.parametrize("family", ["copy", "matmul"])
def test_real_pipeline_projection_matches_numpy_across_all_output_tiles(family):
    result = projection(family)
    for seed in (7, 23):
        args = inputs(family, seed)
        expected = (
            np.asarray(args[0]) if family == "copy" else np.asarray(args[0]) @ np.asarray(args[1])
        )
        for row in range(4):
            for col in range(4):
                point = (row * 128 + 17, col * 128 + 53)
                np.testing.assert_allclose(
                    at_point(result.outputs[0], args, point), expected[point], rtol=3e-4, atol=3e-5
                )


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("missing_input_wait", "destination accessed before completion wait"),
        ("wrong_wait_slot", "destination accessed before completion wait"),
        ("reuse_inflight_slot", "destination accessed before completion wait"),
        ("missing_output_wait", "source overwritten before completion wait"),
        ("missing_final_wait", "scope exits with DMA"),
    ],
)
def test_real_pipeline_bad_schedules_are_rejected_for_the_actual_hazard(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        projection(mode=mode)


def test_wrong_dma_source_is_legal_communication_but_changes_the_output():
    result = projection(mode="wrong_block")
    assert len(result.communication) == 32
    for seed in (7, 23):
        args = inputs("copy", seed)
        for point in [(0, 0), (144, 255), (511, 511)]:
            mapped = (point[0], (point[1] + 128) % 512)
            assert at_point(result.outputs[0], args, point) == args[0][mapped]
        assert at_point(result.outputs[0], args, (0, 0)) != args[0][0, 0]


@pytest.mark.parametrize(
    "scope,family,mode,status,budget",
    [
        (scope, "copy", mode, status, 10000)
        for scope in ("function", "device_local")
        for mode, status in (
            ("early_wait", Status.EQUIVALENT),
            ("wrong_block", Status.INEQUIVALENT),
        )
    ]
    + [("device_local", "matmul", "early_wait", Status.EQUIVALENT, 100000)],
)
def test_real_pipeline_complete_values_reach_public_algebra_entry(
    scope, family, mode, status, budget
):
    with environment():
        left, right = trace(family), trace(family, mode=mode)
        if scope == "device_local":
            expected = Status.EQUIVALENT if status is Status.EQUIVALENT else Status.DIFFERENT
            options = {"analysis_budget": budget} if budget != 10000 else {}
            assert (
                compare_pallas_kernels(left, right, equivalence="semantic", **options).status
                is expected
            )
        options = {"algebra_budget": budget} if budget != 10000 else {}
        result = compare_algebraic_jaxprs(left, right, scope=scope, **options)
    assert result.status is status
    assert result.backend_result is not None or (
        scope == "device_local" and status is Status.EQUIVALENT
    )
    if status is Status.INEQUIVALENT:
        assert any(
            q.answer is False and q.scope.startswith("output_coordinate=")
            for q in result.backend_result.queries
        )


def test_pipeline_replay_rejects_changed_dma_generation():
    with environment():
        left, right = adapt(trace()).view, adapt(trace(mode="early_wait")).view
        result = compare_program_views(left, right, analysis_budget=30000)
        evidence = result.evidence
        transfer, *rest = evidence.left.projection.communication
        corrupt = replace(transfer, transfer=replace(transfer.transfer, generation=999))
        invalid = replace(
            evidence,
            left=replace(
                evidence.left,
                projection=replace(evidence.left.projection, communication=(corrupt, *rest)),
            ),
        )
        with pytest.raises(CorrespondenceError):
            verify_normalized_evidence(left, right, invalid, budget=AnalysisBudget(30000))


def test_pipeline_analysis_budget_rejects_without_unbounded_replay():
    with environment(), pytest.raises(ComparisonBudgetError):
        normalize_view(adapt(trace()).view, budget=AnalysisBudget(100))


def grid_trace(**kwargs):
    return jax.make_jaxpr(serial_grid_function(**kwargs))(
        jax.ShapeDtypeStruct((32, 128), jnp.float32)
    )


def test_dma_can_remain_pending_between_serial_grid_programs():
    result = normalize_view(adapt(grid_trace()).view, budget=AnalysisBudget()).projection
    x = np.arange(32 * 128, dtype=np.float32).reshape(32, 128)
    for point in [(0, 0), (8, 40), (23, 127)]:
        assert at_point(result.outputs[0], (x,), point) == x[point] + 1
    assert len(result.communication) == 6
    assert any(
        ".grid(0,)" in t.transfer.start.path and ".grid(1,)" in t.wait.path
        for t in result.communication
    )


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"staged": True}, "input_staging_boundary"),
        ({"overwrite": True}, "destination accessed before completion wait"),
    ],
)
def test_serial_grid_does_not_ignore_staging_reload_or_slot_reuse(options, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(adapt(grid_trace(**options)).view, budget=AnalysisBudget())


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("alias", "shared HBM version proof"),
        ("direct_read", "ANY storage requires an explicit DMA"),
        ("data_index", "static integer"),
        ("outside_slot", "out of bounds"),
    ],
)
def test_new_dma_admission_keeps_unproved_memory_cases_explicit(mode, reason):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu

    def body(x, y, buffers, sems):
        if mode == "direct_read":
            y[...] = x[...]
            return
        index = (
            x[0, 0].astype(jnp.int32)
            if mode == "data_index"
            else jnp.int32(2 if mode == "outside_slot" else 0)
        )
        tpu.async_copy(x, buffers.at[index], sems.at[index]).wait()
        tpu.async_copy(buffers.at[index], y, sems.at[index]).wait()

    typ = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    spec = pl.BlockSpec(memory_space=pl.ANY) if mode in ("alias", "direct_read") else pl.BlockSpec()
    f = pl.pallas_call(
        body,
        out_shape=typ,
        in_specs=[spec],
        out_specs=spec,
        input_output_aliases={0: 0} if mode == "alias" else {},
        scratch_shapes=(tpu.VMEM((2, 8, 128), jnp.float32), tpu.SemaphoreType.DMA((2,))),
    )
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(adapt(jax.make_jaxpr(f)(typ)).view, budget=AnalysisBudget())


@pytest.mark.parametrize("mode", ["balanced", "pending_count", "parallel"])
def test_local_allocation_scope_preserves_regular_semaphore_obligations(mode):
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu

    def body(x, y):
        def scoped(sem):
            pl.semaphore_signal(sem, 1)
            if mode != "pending_count":
                pl.semaphore_wait(sem, 1)
            y[...] = x[...]

        pl.run_scoped(scoped, tpu.SemaphoreType.REGULAR)

    spec = pl.BlockSpec((8, 128), lambda i: (i, 0))
    typ = jax.ShapeDtypeStruct((16, 128) if mode == "parallel" else (8, 128), jnp.float32)
    f = pl.pallas_call(
        body,
        out_shape=typ,
        grid=(2 if mode == "parallel" else 1,),
        in_specs=[spec],
        out_specs=spec,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
    )
    source = adapt(jax.make_jaxpr(f)(typ)).view
    if mode == "balanced":
        result = normalize_view(source, budget=AnalysisBudget()).projection
        x = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
        assert at_point(result.outputs[0], (x,), (7, 127)) == x[7, 127]
    else:
        reason = "unconsumed semaphore count" if mode == "pending_count" else "core target binding"
        with pytest.raises(UnsupportedJaxprError, match=reason):
            normalize_view(source, budget=AnalysisBudget())


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "family,mode,buffers,synchronous",
    [
        ("copy", None, 2, False),
        ("copy", "early_wait", 2, False),
        ("copy", None, 4, False),
        ("copy", None, 2, True),
        ("copy", "wrong_block", 2, False),
        ("copy", "different_wait_descriptor", 2, False),
        ("copy", "missing_input_wait", 2, False),
        ("matmul", None, 2, False),
        ("matmul", "early_wait", 2, False),
    ],
)
def test_original_pipeline_schedules_execute_in_official_cpu_interpreter(
    family, mode, buffers, synchronous
):
    run_worker(
        "tests.pallas_pipeline_cases",
        family,
        mode,
        buffers,
        synchronous,
        expected=dict(
            case=family,
            mode=mode,
            buffers=buffers,
            synchronous=synchronous,
            status="ok",
            races=False,
        ),
        entry="execute",
        devices=1,
        startup_timeout=120,
        execution_timeout=60,
    )


@pytest.mark.pallas_communication
def test_serial_grid_dma_executes_with_official_cpu_interpreter():
    run_worker(
        "tests.pallas_pipeline_cases",
        expected=dict(case="serial_grid", status="ok", races=False),
        entry="execute_grid",
        devices=1,
        startup_timeout=120,
        execution_timeout=60,
    )
