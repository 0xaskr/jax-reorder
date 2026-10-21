"""Cross-window scratch state, explicit output DMA revisits and invalid staging."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonBudgetError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
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
from tests.pallas_grid_state_cases import inputs, reference, trace, upstream_kernel, verify_sources
from tests.pallas_oracle import run_worker
from tests.pallas_projection_oracle import at_point

POSITIVES = [
    ("scratch", None),
    ("scratch", "history"),
    ("scratch", "banks"),
    ("dma", None),
    ("dma", "revisit"),
    ("dma", "reverse"),
]


def test_original_grid_state_source_matches_pinned_github_manifest():
    verify_sources()


@pytest.mark.parametrize("family,mode", POSITIVES)
def test_grid_state_complete_projection_and_resource_evidence_replay(family, mode):
    source = trace(family, mode)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result)
    projection = result.left_protocol.projection
    # Includes the first and last element of every output window/state slot.
    for seed in (7, 23):
        args = inputs(family, mode, seed)
        expected = np.asarray(reference(family, mode, args))
        rows = range(expected.shape[0])
        for i in rows:
            for tail in [
                tuple(0 for _ in expected.shape[1:]),
                tuple(n - 1 for n in expected.shape[1:]),
            ]:
                point = (i, *tail)
                np.testing.assert_equal(
                    at_point(projection.outputs[0], args, point), expected[point]
                )
    if family == "dma":
        transfers = projection.communication
        assert len(transfers) == (2 if mode is None else 6)
        if mode is not None:
            # Initialized output, not an aliased input, supplies the later DMA.
            assert any(t.transfer.source == 1 for t in transfers)


@pytest.mark.parametrize(
    "family,mode", [("scratch", "history"), ("scratch", "banks"), ("dma", "revisit")]
)
def test_complete_grid_state_output_matches_jax_reference(family, mode):
    pure = jax.make_jaxpr(
        (lambda x: jnp.stack([2 * x[0], 2 * x[1]]))
        if family == "dma"
        else lambda x: reference(family, mode, (x,))
    )(*inputs(family, mode))
    result = compare_algebraic_jaxprs(trace(family, mode), pure)
    assert result.status is Status.EQUIVALENT


def test_independent_explicit_hbm_output_updates_can_change_order():
    a, b = trace("dma", "revisit"), trace("dma", "reverse")
    av, bv = adapt(a).view, adapt(b).view
    result = compare_program_views(av, bv)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    verify_normalized_evidence(av, bv, result.evidence, budget=AnalysisBudget())
    assert compare_algebraic_jaxprs(a, b).status is Status.EQUIVALENT


def test_flat_broadcast_reference_does_not_invent_a_piecewise_read_identity():
    # The core proves both explicit output reads. The backend currently cannot
    # identify input[0]/input[1] pieces with input[output:0] for this flat form.
    pure = jax.make_jaxpr(lambda x: x * 2)(*inputs("dma", "revisit"))
    with pytest.raises(MathematicalUndecidedError, match="output identity"):
        compare_algebraic_jaxprs(trace("dma", "revisit"), pure)


@pytest.mark.parametrize(
    "family,base,mode",
    [("scratch", "banks", "reset_each"), ("dma", "revisit", "wrong_destination")],
)
def test_state_or_destination_mutation_has_complete_output_counterexample(family, base, mode):
    result = compare_algebraic_jaxprs(trace(family, base), trace(family, mode))
    assert result.status is Status.INEQUIVALENT
    assert any(
        q.answer is False and q.scope.startswith("output_coordinate=")
        for q in result.backend_result.queries
    )


@pytest.mark.parametrize(
    "family,mode,reason",
    [
        ("scratch", "missing_init", "uninitialized"),
        ("scratch", "wrong_slot", "uninitialized"),
        ("dma", "missing_wait", "before completion wait"),
    ],
)
def test_invalid_state_reuse_is_refused(family, mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(adapt(trace(family, mode)).view, budget=AnalysisBudget())


def test_serial_scratch_cannot_be_proved_using_parallel_program_order():
    call = extract_call(
        adapt(trace("scratch", "history")).view, budget=AnalysisBudget()
    ).operation.call
    call = replace(call, grid=(replace(call.grid[0], semantics="parallel"),))
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        functionalize_tensor_call(call, budget=AnalysisBudget())


def test_rank_one_scratch_reuses_the_same_serial_window_interpreter():
    f = pl.pallas_call(
        upstream_kernel("scratch", "history"),
        out_shape=jax.ShapeDtypeStruct((384,), jnp.float32),
        grid=(3,),
        in_specs=[pl.BlockSpec((128,), lambda i: (0,))],
        out_specs=pl.BlockSpec((128,), lambda i: (i,)),
        scratch_shapes=[tpu.VMEM((128,), jnp.float32)],
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((128,), jnp.float32))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    args = (np.arange(128, dtype=np.float32),)
    for i in range(3):
        for lane in (0, 127):
            assert (
                at_point(result.left_protocol.projection.outputs[0], args, (128 * i + lane,))
                == lane + i + 1
            )


def test_rank_one_tensor_coordinates_do_not_use_the_legacy_scalar_witness():
    def source(offset):
        def kernel(x, y, scratch):
            p = pl.program_id(0)

            @pl.when(p == 0)
            def initialize():
                scratch[...] = jax.lax.iota(jnp.int32, 128) + offset

            y[...] = scratch[...] + p * 128

        spec = pl.BlockSpec((128,), lambda p: (p,))
        f = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((256,), jnp.int32),
            grid=(2,),
            in_specs=[spec],
            out_specs=spec,
            scratch_shapes=[tpu.VMEM((128,), jnp.int32)],
        )
        return adapt(jax.make_jaxpr(f)(jax.ShapeDtypeStruct((256,), jnp.int32))).view

    a, b = source(0), source(1)
    for offset, view in enumerate((a, b)):
        value = normalize_view(view, budget=AnalysisBudget()).projection.outputs[0]
        assert at_point(value, (np.zeros(256, dtype=np.int32),), (200,)) == 200 + offset
    with pytest.raises(MathematicalUndecidedError, match="mathematical backend"):
        compare_program_views(a, b)


def prefix_call(*, reverse=False, interleaved=False):
    def body(x, y, scratch):
        a, b = pl.program_id(1), pl.program_id(2)

        @pl.when((a == 0) & (b == 0))
        def initialize():
            scratch[...] = x[...]

        scratch[...] += 1
        y[...] = scratch[...]

    return jax.make_jaxpr(
        pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((2, 16, 384), jnp.float32),
            grid=(2, 2, 3),
            in_specs=[pl.BlockSpec((1, 8, 128), lambda p, a, b: (p, 0, 0))],
            out_specs=pl.BlockSpec(
                (1, 8, 128), lambda p, a, b: (p, 1 - a if reverse else a, 2 - b if reverse else b)
            ),
            scratch_shapes=[tpu.VMEM((1, 8, 128), jnp.float32)],
            compiler_params=tpu.CompilerParams(
                dimension_semantics=("arbitrary", "parallel", "arbitrary")
                if interleaved
                else ("parallel", "arbitrary", "arbitrary")
            ),
        )
    )(jax.ShapeDtypeStruct((2, 8, 128), jnp.float32))


@pytest.mark.parametrize("reverse", [False, True])
def test_parallel_prefix_keeps_separate_serial_state_and_exact_window_coordinates(reverse):
    source = prefix_call(reverse=reverse)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    args = (np.arange(2 * 8 * 128, dtype=np.float32).reshape(2, 8, 128),)
    value = result.left_protocol.projection.outputs[0]
    for p in range(2):
        for a in range(2):
            for b in range(3):
                aa, bb = (1 - a, 2 - b) if reverse else (a, b)
                for row, col in [(0, 0), (7, 127)]:
                    actual = at_point(value, args, (p, aa * 8 + row, bb * 128 + col))
                    assert actual == args[0][p, row, col] + a * 3 + b + 1


def test_interleaved_serial_parallel_scratch_lifetime_is_refused():
    with pytest.raises(UnsupportedJaxprError, match="parallel prefix and serial suffix"):
        normalize_view(adapt(prefix_call(interleaved=True)).view, budget=AnalysisBudget())


def test_cross_window_serial_replay_has_a_budget():
    call = extract_call(
        adapt(trace("scratch", "history")).view, budget=AnalysisBudget()
    ).operation.call
    call = replace(call, grid=(replace(call.grid[0], size=2**20),))
    # Keep the output ownership contract valid while increasing program count.
    block = call.blocks[-1]
    block = replace(block, array=replace(block.array, shape=(8 * 2**20, 128)))
    call = replace(call, blocks=(*call.blocks[:-1], block))
    with pytest.raises(ComparisonBudgetError, match="serial_grid"):
        functionalize_tensor_call(call, budget=AnalysisBudget(128))


def test_output_window_is_committed_after_its_entire_reduction_suffix():
    def kernel(x, y, scratch):
        i, k = pl.program_id(0), pl.program_id(1)

        @pl.when((i == 0) & (k == 0))
        def initialize():
            scratch[...] = x[...]

        scratch[...] += 1

        @pl.when(k == 1)
        def finish():
            y[...] = scratch[...]

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((24, 128), jnp.float32),
        grid=(3, 2),
        in_specs=[pl.BlockSpec((8, 128), lambda i, k: (0, 0))],
        out_specs=pl.BlockSpec((8, 128), lambda i, k: (i, 0)),
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32)],
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((8, 128), jnp.float32))
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget()).projection
    args = (np.arange(8 * 128, dtype=np.float32).reshape(8, 128),)
    for i in range(3):
        assert at_point(proof.outputs[0], args, (8 * i + 7, 127)) == args[0][7, 127] + 2 * (i + 1)


def boundary_call(mode):
    def body(x, y, scratch, sem):
        i = pl.program_id(0)

        @pl.when(i == 0)
        def first():
            scratch[0] = x[...]
            if mode == "stale_output":
                y[...] = x[...]
            elif mode == "output_destination":
                tpu.make_async_copy(scratch.at[0], y, sem).start()
            else:
                y[...] = x[...]
                source = y if mode == "output_source" else scratch.at[0]
                tpu.make_async_copy(source, scratch.at[1], sem).start()

        @pl.when(i == 1)
        def second():
            if mode == "stale_output":
                y[...] += 1
            elif mode == "output_destination":
                tpu.make_async_copy(scratch.at[0], y, sem).wait()
            else:
                source = y if mode == "output_source" else scratch.at[0]
                tpu.make_async_copy(source, scratch.at[1], sem).wait()
                y[...] = scratch[1]

    return jax.make_jaxpr(
        pl.pallas_call(
            body,
            out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
            grid=(2,),
            in_specs=[pl.BlockSpec((8, 128), lambda i: (0, 0))],
            out_specs=pl.BlockSpec((8, 128), lambda i: (i, 0)),
            scratch_shapes=[tpu.VMEM((2, 8, 128), jnp.float32), tpu.SemaphoreType.DMA],
        )
    )(jax.ShapeDtypeStruct((8, 128), jnp.float32))


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("stale_output", "uninitialized"),
        ("output_source", "DMA source"),
        ("output_destination", "DMA destination"),
    ],
)
def test_output_staging_cannot_survive_a_window_change(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(adapt(boundary_call(mode)).view, budget=AnalysisBudget())


def test_pending_scratch_to_scratch_dma_can_cross_output_window_boundary():
    source = boundary_call("scratch")
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    projection = result.left_protocol.projection
    assert len(projection.communication) == 1
    args = (np.arange(8 * 128, dtype=np.float32).reshape(8, 128),)
    for row in (0, 7, 8, 15):
        assert at_point(projection.outputs[0], args, (row, 127)) == args[0][row % 8, 127]


def test_nonconsecutive_automatic_output_revisit_remains_outside_tpu_contract():
    def kernel(x, y, scratch):
        scratch[...] = x[...]
        y[...] = scratch[...]

    spec = pl.BlockSpec((8, 128), lambda k, i: (i, 0))
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 128), jnp.float32),
        grid=(2, 2),
        in_specs=[spec],
        out_specs=spec,
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32)],
    )
    with pytest.raises(UnsupportedJaxprError, match="trailing serial grid suffix"):
        normalize_view(
            adapt(jax.make_jaxpr(f)(jax.ShapeDtypeStruct((16, 128), jnp.float32))).view,
            budget=AnalysisBudget(),
        )


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "family,mode", POSITIVES + [("scratch", "reset_each"), ("dma", "wrong_destination")]
)
def test_upstream_and_ai_variants_execute_in_official_tpu_cpu_interpreter(family, mode):
    run_worker(
        "tests.pallas_grid_state_cases",
        family,
        mode,
        expected=dict(family=family, mode=mode, status="ok"),
        entry="execute",
        devices=1,
        execution_timeout=60,
    )
