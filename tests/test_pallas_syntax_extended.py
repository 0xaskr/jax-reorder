"""Finite syntax rules preserve coordinates, resource holes and operation order."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonBudgetError,
    Status,
    UnsupportedJaxprError,
    compare_pallas_kernels,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import compare_program_views, normalize_view
from tests.pallas_oracle import run_worker
from tests.pallas_projection_oracle import at_point


def function(mode="stride", *, wrong=False):
    spec = pl.BlockSpec((8, 128), lambda: (0, 0))
    scratch = (tpu.VMEM((8, 128), jnp.float32),) if mode == "hole" else ()
    if mode == "dma_hazard":
        scratch = (tpu.VMEM((8, 128), jnp.float32), tpu.SemaphoreType.DMA)

    def kernel(x, y, *resources):
        if mode in ("stride", "stride_explicit", "chain", "gap"):
            if mode == "chain":
                y.at[:8, :].at[::2, :][...] = x.at[:8, :].at[1::2, :][...]
                y.at[:8, :].at[1::2, :][...] = x.at[:8, :].at[::2, :][...]
            elif mode == "stride_explicit":
                for i in range(0, 8, 2):
                    y[i : i + 1, :] = x[i + 1 : i + 2, :]
                    y[i + 1 : i + 2, :] = x[i : i + 1, :]
            else:
                y[::2, :] = x[1::2, :]
                if mode != "gap":
                    y[1::2, :] = x[1::2, :] if wrong else x[::2, :]
        elif mode == "hole":
            resources[0][::2, :] = x[::2, :]
            y[...] = resources[0][...]
        elif mode == "dma_hazard":
            copy = tpu.async_copy(x, resources[0], resources[1])
            resources[0][1::2, :] = x[::2, :]
            copy.wait()
            y[...] = resources[0][...]
        elif mode in ("reshape", "transpose_reshape"):
            if mode == "reshape":
                y[...] = lax.reshape(x[...], (8, 128), dimensions=(0, 1) if wrong else (1, 0))
            else:
                y[...] = x[...].T.reshape((8, 128))
        else:
            raise AssertionError(mode)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=(),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=scratch,
        compiler_params=tpu.CompilerParams(),
    )


def trace(mode="stride", *, wrong=False):
    return jax.make_jaxpr(function(mode, wrong=wrong))(jax.ShapeDtypeStruct((8, 128), jnp.float32))


@pytest.mark.parametrize("mode", ["stride", "chain", "reshape"])
def test_extended_syntax_matches_explicit_coordinate_graph(mode):
    other = "transpose_reshape" if mode == "reshape" else "stride" if mode == "chain" else "chain"
    result = compare_program_views(adapt(trace(mode)).view, adapt(trace(other)).view)
    assert result.status is Status.EQUIVALENT
    assert result.equivalence_kind in ("normalized", "structural")


@pytest.mark.parametrize("mode", ["stride", "chain", "reshape"])
def test_extended_syntax_projection_matches_independent_numpy(mode):
    source = trace(mode)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    expected = x.T.reshape(x.shape) if mode == "reshape" else x[np.arange(8) ^ 1]
    for point in np.ndindex(x.shape):
        assert at_point(result.left_protocol.projection.outputs[0], (x,), point) == expected[point]


@pytest.mark.parametrize("mode", ["stride", "reshape"])
def test_wrong_mapping_does_not_match(mode):
    assert compare_pallas_kernels(trace(mode), trace(mode, wrong=True)).status is Status.DIFFERENT
    left = normalize_view(adapt(trace(mode)).view, budget=AnalysisBudget()).projection
    right = normalize_view(adapt(trace(mode, wrong=True)).view, budget=AnalysisBudget()).projection
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    assert at_point(left.outputs[0], (x,), (1, 0)) != at_point(right.outputs[0], (x,), (1, 0))


@pytest.mark.parametrize("mode", ["gap", "hole", "dma_hazard"])
def test_strided_holes_and_inflight_dma_are_not_erased(mode):
    with pytest.raises(UnsupportedJaxprError, match="uninitialized|in.flight|pending|DMA"):
        normalize_view(adapt(trace(mode)).view, budget=AnalysisBudget())


def test_strided_expansion_charges_budget():
    with pytest.raises(ComparisonBudgetError):
        normalize_view(adapt(trace()).view, budget=AnalysisBudget(40))


def test_strided_cartesian_product_is_charged_before_region_allocation():
    def kernel(x, y):
        y[...] = jnp.zeros((64, 128), jnp.float32)
        y[::2, ::2] = x[::2, ::2]

    spec = pl.BlockSpec((64, 128), lambda: (0, 0))
    fn = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((64, 128), jnp.float32),
        grid=(),
        in_specs=[spec],
        out_specs=spec,
    )
    source = jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((64, 128), jnp.float32))
    # Each axis's list fits, but their 32 * 64 singleton rectangles do not.
    # The failure must occur at the product's precharge, not while storing it.
    with pytest.raises(ComparisonBudgetError, match="access_regions"):
        normalize_view(adapt(source).view, budget=AnalysisBudget(512))


def execute(mode):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = tpu.InterpretParams(
        detect_races=True,
        dma_execution_mode="on_wait",
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
    )
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    with tpu.force_tpu_interpret_mode(params):
        compiled = jax.jit(function(mode)).lower(x).compile()
    print("READY", flush=True)
    expected = x.T.reshape(x.shape) if mode == "reshape" else x[np.arange(8) ^ 1]
    np.testing.assert_array_equal(jax.block_until_ready(compiled(x)), expected)
    jax.effects_barrier()
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=mode, status="ok")), flush=True)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["stride", "chain", "reshape"])
def test_extended_syntax_matches_official_cpu_execution(mode):
    run_worker(
        "tests.test_pallas_syntax_extended",
        mode,
        expected=dict(case=mode, status="ok"),
        entry="execute",
        devices=1,
        execution_timeout=60,
    )


def coordinate_function(mode="index", *, wrong=False, groups=2):
    spec = pl.BlockSpec((8, 128), lambda i, j: (i, j))

    def kernel(x, y):
        p = pl.program_id(0)
        if mode == "index":
            y[...] = jnp.broadcast_to(x[p + (8 if wrong else 0), :], y.shape)
        elif mode == "dynamic_slice":
            v = x[pl.ds(p, 4, 2), :]
            y[:4, :] = v
            y[4:, :] = v
        elif mode == "dynamic_write":
            y[...] = x[...]
            y[p, :] = x[0, :]
        else:
            y[...] = x[...]
            count = (
                jnp.minimum(jnp.maximum(x[0, 0].astype(jnp.int32), 0), 2 if wrong else 3)
                if mode == "bounded"
                else p + (2 if wrong else 1)
            )

            def condition(i):
                if mode == "unbounded":
                    return x[0, 0] > i
                if mode == "infinite":
                    return jnp.bool_(True)
                return i < count

            def body(i):
                y[...] = y[...] + (i + 1)
                return i + 1

            final = lax.while_loop(condition, body, jnp.int32(0))
            y[...] = y[...] + final

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8 * groups, 256), jnp.float32),
        grid=(groups, 2),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )


def coordinate_trace(mode="index", *, wrong=False, groups=2):
    return jax.make_jaxpr(coordinate_function(mode, wrong=wrong, groups=groups))(
        jax.ShapeDtypeStruct((8 * groups, 256), jnp.float32)
    )


def coordinate_expected(x, mode, *, wrong=False):
    expected = np.empty_like(x)
    for i in range(x.shape[0] // 8):
        for j in range(2):
            block = x[i * 8 : (i + 1) * 8, j * 128 : (j + 1) * 128]
            if mode == "index":
                value = np.broadcast_to(block[i], block.shape)
            elif mode == "dynamic_slice":
                value = np.tile(block[i : i + 8 : 2], (2, 1))
            else:
                count = (
                    np.clip(int(block[0, 0]), 0, 2 if wrong else 3)
                    if mode == "bounded"
                    else i + (2 if wrong else 1)
                )
                value = block + count * (count + 1) // 2 + count
            expected[i * 8 : (i + 1) * 8, j * 128 : (j + 1) * 128] = value
    return expected


@pytest.mark.parametrize("mode", ["index", "dynamic_slice", "while", "bounded"])
def test_finite_program_domains_preserve_dynamic_values(mode):
    result = compare_pallas_kernels(
        coordinate_trace(mode), coordinate_trace(mode), check_protocol=True
    )
    x = np.arange(16 * 256, dtype=np.float32).reshape((16, 256))
    expected = coordinate_expected(x, mode)
    for row in range(16):
        for col in (0, 127, 128, 255):
            assert (
                at_point(result.left_protocol.projection.outputs[0], (x,), (row, col))
                == expected[row, col]
            )


def test_wrong_while_bound_preserves_a_concrete_difference():
    x = np.arange(16 * 256, dtype=np.float32).reshape((16, 256))
    values = []
    for wrong in (False, True):
        result = normalize_view(
            adapt(coordinate_trace("while", wrong=wrong)).view, budget=AnalysisBudget()
        )
        values.append(at_point(result.projection.outputs[0], (x,), (0, 0)))
    assert values[0] != values[1]


@pytest.mark.parametrize("mode", ["index", "unbounded", "infinite"])
def test_unproved_dynamic_resources_still_fail(mode):
    with pytest.raises(
        ComparisonBudgetError if mode == "infinite" else UnsupportedJaxprError,
        match="bounds|static integer|finite carry|budget",
    ):
        normalize_view(
            adapt(coordinate_trace(mode, wrong=mode == "index")).view,
            budget=AnalysisBudget(200),
        )


def test_dynamic_loop_domain_is_budgeted_before_branch_allocation():
    with pytest.raises(ComparisonBudgetError, match="while_domain"):
        normalize_view(
            adapt(coordinate_trace("while", groups=10000)).view, budget=AnalysisBudget(1000)
        )


def execute_coordinate(mode):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = tpu.InterpretParams(
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
    )
    x = np.arange(16 * 256, dtype=np.float32).reshape((16, 256))
    with tpu.force_tpu_interpret_mode(params):
        compiled = jax.jit(coordinate_function(mode)).lower(x).compile()
    print("READY", flush=True)
    np.testing.assert_array_equal(jax.block_until_ready(compiled(x)), coordinate_expected(x, mode))
    jax.effects_barrier()
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=mode, status="ok")), flush=True)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["index", "dynamic_slice", "while", "bounded"])
def test_coordinate_syntax_matches_official_cpu_execution(mode):
    run_worker(
        "tests.test_pallas_syntax_extended",
        mode,
        expected=dict(case=mode, status="ok"),
        entry="execute_coordinate",
        devices=1,
        execution_timeout=60,
    )


def atomic_function(mode):
    from jax.experimental.pallas import triton

    spec = pl.BlockSpec((8, 128), lambda: (0, 0))

    def kernel(x, y):
        y[...] = x[...]
        incoming = jnp.ones((8, 128), jnp.float32)
        if mode == "atomic":
            old = triton.atomic_add(y, (slice(None), slice(None)), incoming)
            y[...] = y[...] + old
        else:
            jax.ref.addupdate(y, (slice(None), slice(None)), incoming)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=(),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(),
    )


@pytest.mark.parametrize("mode", ["atomic", "addupdate"])
def test_non_tpu_atomic_is_rejected_without_dropping_effects(mode):
    from jax._src.pallas.mosaic import lowering
    from jax._src.pallas.triton.primitives import atomic_rmw_p
    from jax._src.state.primitives import addupdate_p

    primitive = atomic_rmw_p if mode == "atomic" else addupdate_p
    assert not any(primitive in rules for rules in lowering.lowering_rules.values())
    source = jax.make_jaxpr(atomic_function(mode))(jax.ShapeDtypeStruct((8, 128), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="primitive|effect"):
        compare_pallas_kernels(source, source, check_protocol=True)


def execute_atomic_rejection(mode):
    expected = "does not support int16" if mode == "atomic" else "must be called on `Ref`"
    print("READY", flush=True)
    with tpu.force_tpu_interpret_mode(tpu.InterpretParams()):
        with pytest.raises(ValueError, match=expected):
            jax.jit(atomic_function(mode)).lower(
                jax.ShapeDtypeStruct((8, 128), jnp.float32)
            ).compile()
    print("RESULT " + json.dumps(dict(case=mode, status="rejected_by_tpu")), flush=True)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["atomic", "addupdate"])
def test_locked_tpu_interpreter_does_not_execute_gpu_atomic(mode):
    run_worker(
        "tests.test_pallas_syntax_extended",
        mode,
        expected=dict(case=mode, status="rejected_by_tpu"),
        entry="execute_atomic_rejection",
        devices=1,
        execution_timeout=60,
    )


@pytest.mark.parametrize("count", [-5, 0, 1, 2, 3, 7])
def test_bounded_while_uses_each_input_dependent_trip_count(count):
    proof = normalize_view(adapt(coordinate_trace("bounded")).view, budget=AnalysisBudget())
    x = np.full((16, 256), count, np.float32)
    expected = coordinate_expected(x, "bounded")
    assert at_point(proof.projection.outputs[0], (x,), (9, 129)) == expected[9, 129]


def test_one_dimensional_strided_ref_uses_the_same_exact_region_rule():
    spec = pl.BlockSpec((128,), lambda i: (i,))

    def kernel(x, y):
        y[::2] = x[1::2]
        y[1::2] = x[::2]

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
        grid=(2,),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((256,), jnp.float32))
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget(100000))
    x = np.arange(256, dtype=np.float32)
    for i in range(256):
        assert at_point(proof.projection.outputs[0], (x,), (i,)) == x[i ^ 1]


def bounded_resource_function(mode):
    spec = pl.BlockSpec((8, 128), lambda: (0, 0))
    scratch = (
        (tpu.VMEM((8, 128), jnp.float32), tpu.SemaphoreType.DMA)
        if mode.startswith("dma")
        else (tpu.SemaphoreType.REGULAR,)
    )

    def kernel(x, y, *resources):
        y[...] = x[...]
        count = jnp.minimum(jnp.maximum(x[0, 0].astype(jnp.int32), 0), 3)

        def body(i):
            if mode.startswith("dma"):
                copy = tpu.async_copy(x, resources[0], resources[1])
                if mode != "dma_pending":
                    copy.wait()
                    y[...] = y[...] + resources[0][...]
            else:
                pl.semaphore_signal(resources[0], 1)
                y[...] = y[...] + 1
                if mode != "counter_unconsumed":
                    pl.semaphore_wait(resources[0], 1)
            return i + 1

        lax.while_loop(lambda i: i < count, body, jnp.int32(0))
        if mode == "dma_post":
            tpu.async_copy(y, resources[0], resources[1]).wait()
            y[...] = resources[0][...] + 1

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=(),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=scratch,
        compiler_params=tpu.CompilerParams(),
    )


@pytest.mark.parametrize("mode", ["dma", "counter", "dma_pending", "counter_unconsumed"])
def test_bounded_loop_preserves_every_resource_obligation(mode):
    source = jax.make_jaxpr(bounded_resource_function(mode))(
        jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )
    if mode in ("dma_pending", "counter_unconsumed"):
        with pytest.raises(UnsupportedJaxprError, match="outstanding DMA|counter states"):
            compare_pallas_kernels(source, source, check_protocol=True)
    else:
        result = compare_pallas_kernels(source, source, check_protocol=True)
        for count in range(4):
            x = np.full((8, 128), count, np.float32)
            actual = at_point(result.left_protocol.projection.outputs[0], (x,), (5, 17))
            expected = count * (count + 1) if mode == "dma" else 2 * count
            assert actual == expected


def test_bounded_dma_cases_have_distinct_evidence_ids_and_origins():
    source = jax.make_jaxpr(bounded_resource_function("dma_post"))(
        jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )
    proof = compare_pallas_kernels(source, source, check_protocol=True).left_protocol.projection
    records = tuple((t.transfer.id, t.transfer.start.path) for t in proof.communication)
    assert len(records) == len({i for i, _ in records}) == len({p for _, p in records}) == 7
    assert all(".case[" in path for _, path in records[:-1])
    assert ".case[" not in records[-1][1]
    assert {producer for w in proof.waits for producer in w.producers} == {i for i, _ in records}
    for count in range(4):
        x = np.full((8, 128), count, np.float32)
        assert at_point(proof.outputs[0], (x,), (3, 91)) == count * (count + 1) + 1


@pytest.mark.parametrize(
    "mode", ["stride", "chain", "transpose_reshape", "index", "dynamic_slice", "while", "bounded"]
)
def test_new_native_tpu_syntax_lowers_offline_to_verified_stablehlo(mode):
    from tests.pallas_pipeline_cases import environment

    plain = mode in ("stride", "chain", "transpose_reshape")
    with environment():
        build = function(mode) if plain else coordinate_function(mode)
        signature = jax.ShapeDtypeStruct((8, 128) if plain else (16, 256), jnp.float32)
        lowered = jax.jit(build).trace(signature).lower(lowering_platforms=("tpu",))
        assert lowered.compiler_ir("stablehlo").operation.verify()


@pytest.mark.parametrize("mode", ["atomic", "addupdate", "reshape"])
def test_backend_unimplemented_syntax_is_distinguished_from_cpu_interpretation(mode):
    from tests.pallas_pipeline_cases import environment

    with environment():
        build = function(mode) if mode == "reshape" else atomic_function(mode)
        with pytest.raises(NotImplementedError, match="^$" if mode == "reshape" else "primitive"):
            jax.jit(build).trace(jax.ShapeDtypeStruct((8, 128), jnp.float32)).lower(
                lowering_platforms=("tpu",)
            )
