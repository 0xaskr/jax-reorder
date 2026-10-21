"""Device-local entry, bounded state proofs and adversarial resource cases."""

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import normalize_view
from tests.pallas_oracle import run_worker
from tests.pallas_protocol_cases import local_regular_function
from tests.pallas_remote_cases import projection_array, remote_graph
from tests.test_pallas_dma import function as dma_function
from tests.test_pallas_local_scope import (
    conditional_graph,
    host_preprocessed_function,
    loop_kernel,
    normalize,
    stateful_function,
    trace,
)
from tests.test_pallas_sharding import device_mesh


def projected(source, size=128):
    proof = normalize(source)
    x = np.arange(1, size + 1, dtype=np.float32)
    return projection_array(proof.projection.outputs[0], (x,), size), x


@pytest.mark.parametrize("kind", ["data", "read"])
def test_local_evidence_separates_structure_and_protocol_and_rejects_tampering(kind):
    source = conditional_graph(kind)
    structure = compare_pallas_kernels(source, source)
    assert structure.status is Status.EQUIVALENT and not structure.protocol_checked
    assert structure.left_protocol is None
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    proof = result.left_protocol
    forged = replace(
        result, left_protocol=replace(proof, projection=replace(proof.projection, outputs=()))
    )
    with pytest.raises(CorrespondenceError, match="protocol evidence"):
        verify_pallas_kernel_comparison(source, source, forged)
    with pytest.raises(CorrespondenceError, match="source bindings"):
        verify_pallas_kernel_comparison(source, conditional_graph(kind, True), result)


def test_local_scope_does_not_claim_equivalence_of_outer_postprocessing():
    function = stateful_function("scratch", 1)
    a, b = trace(function), trace(lambda x: function(x) + 2)
    assert compare_jaxprs(a, b).status is Status.DIFFERENT
    result = compare_pallas_kernels(a, b)
    assert result.status is Status.EQUIVALENT and result.scope == "device_local"
    verify_pallas_kernel_comparison(a, b, result)
    with pytest.raises(UnsupportedJaxprError, match="postprocessing"):
        normalize(b)


def test_outer_input_order_and_constant_contents_remain_in_the_local_contract():
    def function(x, y):
        def kernel(a, b, out):
            out[:] = a[:] - b[:]

        spec = pl.BlockSpec((128,), lambda p: (p,))
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
            grid=(1,),
            in_specs=(spec, spec),
            out_specs=spec,
        )(x, y)

    sig = jax.ShapeDtypeStruct((128,), jnp.float32)
    a = jax.make_jaxpr(function)(sig, sig)
    b = jax.make_jaxpr(lambda x, y: function(y, x))(sig, sig)
    assert compare_pallas_kernels(a, b).status is Status.DIFFERENT
    a, b = (trace(host_preprocessed_function("numpy_argument", n)) for n in (3, 4))
    assert compare_pallas_kernels(a, b).status is Status.DIFFERENT


def test_local_structure_reuses_dependency_preserving_canonical_order():
    def function(reverse):
        def kernel(x, y):
            value = x[:]
            if reverse:
                cosine, sine = jnp.cos(value), jnp.sin(value)
            else:
                sine, cosine = jnp.sin(value), jnp.cos(value)
            y[:] = sine + cosine

        return dma_function(kernel, semaphores=0)

    a, b = trace(function(False)), trace(function(True))
    assert compare_jaxprs(a, b).status is Status.EQUIVALENT
    result = compare_pallas_kernels(a, b, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(a, b, result)


@pytest.mark.parametrize("mode", ["missing_wait", "uninitialized", "counter_mismatch"])
def test_every_data_branch_must_satisfy_resource_obligations(mode):
    def kernel(x, y, sem):
        def good():
            pl.semaphore_signal(sem, 1)
            pl.semaphore_wait(sem, 1)
            y[:] = x[:]

        def bad():
            if mode != "uninitialized":
                pl.semaphore_signal(sem, 1)
            if mode == "missing_wait":
                pl.semaphore_wait(sem, 2)
            if mode != "uninitialized":
                y[:] = x[:] + 1

        lax.cond(x[0] > 0, good, bad)

    source = trace(local_regular_function(kernel))
    assert compare_pallas_kernels(source, source).status is Status.EQUIVALENT
    reason = {
        "missing_wait": "unsatisfied",
        "uninitialized": "uninitialized",
        "counter_mismatch": "different counter states",
    }[mode]
    with pytest.raises(UnsupportedJaxprError, match=reason):
        compare_pallas_kernels(source, source, check_protocol=True)


@pytest.mark.parametrize("infinite", [False, True])
def test_while_requires_a_finite_proof_and_respects_analysis_budget(infinite):
    def kernel(x, y, sem):
        y[:] = x[:]

        def body(i):
            pl.semaphore_signal(sem, 1)
            pl.semaphore_wait(sem, 1)
            return i + 1

        lax.while_loop(lambda i: jnp.bool_(True) if infinite else x[0] > i, body, jnp.int32(0))

    source = trace(local_regular_function(kernel))
    with pytest.raises(
        ComparisonBudgetError if infinite else UnsupportedJaxprError, match="budget|finite carry"
    ):
        normalize_view(adapt(source).view, budget=AnalysisBudget(80))


def test_scan_carry_and_reverse_scalar_sequence_are_not_erased():
    def kernel(x, y, sem):
        def body(total, item):
            pl.semaphore_signal(sem, 1)
            pl.semaphore_wait(sem, 1)
            return total + item, total + item

        total, history = lax.scan(body, jnp.float32(0), x[:], reverse=True)
        y[:] = history + total

    actual, x = projected(trace(local_regular_function(kernel)))
    np.testing.assert_array_equal(actual, np.cumsum(x[::-1])[::-1] + x.sum())


def test_scalar_reads_use_the_requested_offset_not_the_output_lane():
    def kernel(x, y, sem):
        pl.semaphore_signal(sem, 1)
        pl.semaphore_wait(sem, 1)
        y[:] = x[:] + x[7]

    actual, x = projected(trace(local_regular_function(kernel)))
    np.testing.assert_array_equal(actual, x + x[7])


def test_python_remainder_route_keeps_the_jax_pure_wrapper_and_target_binding():
    def kernel(x, y, send, receive):
        peer = (lax.axis_index("d") + 1) % 2
        tpu.async_remote_copy(x, y, send, receive, (peer,)).wait()

    source = remote_graph(kernel)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.protocol_checked
    verify_pallas_kernel_comparison(source, source, result)
    actual, x = projected(source, 256)
    np.testing.assert_array_equal(actual, np.roll(x, 128))


def test_dma_read_cannot_be_used_as_a_stable_condition_while_in_flight():
    def kernel(x, y, sem):
        copy = tpu.make_async_copy(x, y, sem)
        copy.start()
        flag = pl.semaphore_read(sem) == 0
        copy.wait()
        lax.cond(flag, lambda: None, lambda: None)
        y[:] = y[:] + flag

    source = trace(dma_function(kernel))
    with pytest.raises(UnsupportedJaxprError, match="scheduling|stable zero"):
        compare_pallas_kernels(source, source, check_protocol=True)


@pytest.mark.parametrize("mode", ["scratch_carry", "unfinished_dma", "residual_counter"])
def test_cross_program_resource_lifetimes_are_checked(mode):
    def kernel(x, y, resource):
        if mode == "scratch_carry":
            lax.cond(
                pl.program_id(0) == 0, lambda: resource.__setitem__(slice(None), x[:]), lambda: None
            )
            y[:] = resource[:]
        elif mode == "unfinished_dma":
            tpu.make_async_copy(x, y, resource).start()
        else:
            pl.semaphore_signal(resource, 1)
            y[:] = x[:]

    if mode == "residual_counter":
        spec = pl.BlockSpec((128,), lambda p: (p,))
        function = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
            grid=(2,),
            in_specs=(spec,),
            out_specs=spec,
            scratch_shapes=(tpu.SemaphoreType.REGULAR,),
        )
    else:
        function = dma_function(
            kernel,
            programs=2,
            size=256,
            arrays=int(mode == "scratch_carry"),
            semaphores=int(mode != "scratch_carry"),
        )
    source = trace(function, 256)
    if mode == "scratch_carry":
        from tests.pallas_projection_oracle import at_point

        proof = normalize(source)
        x = np.arange(256, dtype=np.float32)
        for point in (0, 127, 128, 255):
            assert at_point(proof.projection.outputs[0], (x,), (point,)) == x[point % 128]
        return
    with pytest.raises(
        UnsupportedJaxprError, match="uninitialized|outstanding|unconsumed|conserved|unfinished"
    ):
        normalize(source)


def aliased_function(*, reverse=False, read_output_first=False, interpret=False, parallel=False):
    def kernel(x, y):
        if read_output_first:
            y[:] = y[:] + x[:]
        else:
            y[:] = x[:] + 1
            y[:] = y[:] + x[:]

    source_spec = pl.BlockSpec((128,), lambda p: (1 - p,) if reverse else (p,))
    destination_spec = pl.BlockSpec((128,), lambda p: (p,))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
        grid=(2,),
        in_specs=(source_spec,),
        out_specs=destination_spec,
        input_output_aliases={0: 0},
        interpret=interpret,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)) if parallel else None,
    )


def test_blocked_alias_keeps_staging_inputs_and_output_versions_separate():
    actual, x = projected(trace(aliased_function(), 256), 256)
    np.testing.assert_array_equal(actual, 2 * x + 1)


@pytest.mark.parametrize("mode", ["mapping", "initialization"])
def test_alias_requires_matching_blocks_and_initialized_local_outputs(mode):
    source = trace(
        aliased_function(reverse=mode == "mapping", read_output_first=mode == "initialization"), 256
    )
    with pytest.raises(UnsupportedJaxprError, match="alias requires|uninitialized"):
        normalize(source)


def test_alias_cannot_hide_repeated_parent_bindings_with_different_block_maps():
    def kernel(x, other, y):
        y[:] = x[:] + other[:]

    forward = pl.BlockSpec((128,), lambda p: (p,))
    backward = pl.BlockSpec((128,), lambda p: (1 - p,))
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
        grid=(2,),
        in_specs=(forward, backward),
        out_specs=forward,
        input_output_aliases={0: 0},
        compiler_params=tpu.CompilerParams(),
    )
    source = trace(lambda x: call(x, x), 256)
    assert compare_pallas_kernels(source, source).status is Status.EQUIVALENT
    with pytest.raises(UnsupportedJaxprError, match="repeats an outer input"):
        compare_pallas_kernels(source, source, check_protocol=True)
    with pytest.raises(UnsupportedJaxprError, match="repeats an outer input"):
        normalize(source)


@pytest.mark.parametrize("kind", ["numpy", "data", "fori", "while"])
def test_new_complete_projections_reach_the_public_mathematical_entry(kind):
    if kind == "numpy":
        source, reference = (
            trace(host_preprocessed_function("numpy_argument")),
            trace(lambda x: x + 3),
        )
    elif kind == "data":
        source, reference = (
            conditional_graph("data"),
            trace(lambda x: jnp.where(x[0] > 0, x + 1, x + 2)),
        )
    else:
        source = trace(local_regular_function(loop_kernel(kind)))
        reference = trace(lambda x: x + 2)
    result = compare_algebraic_jaxprs(source, reference)
    assert result.status is Status.EQUIVALENT and result.normalizations


def test_shared_outer_expression_dag_does_not_expand_exponentially():
    function = stateful_function("scratch", 1)

    def wrapper(x):
        for _ in range(50):
            x = x * x
        return function(x)

    source = trace(wrapper)
    result = compare_pallas_kernels(source, source, check_protocol=True, analysis_budget=512)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result, analysis_budget=512)


def test_zero_dimensional_scalar_prefetch_keeps_the_outer_device_binding():
    def function(offset):
        def local(x):
            def kernel(d, a, b):
                b[:] = a[:] + d[()]

            spec = pl.BlockSpec((128,), lambda p, d: (p,))
            return pl.pallas_call(
                kernel,
                out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
                grid_spec=tpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=1, grid=(1,), in_specs=(spec,), out_specs=spec
                ),
            )(lax.axis_index("d") + offset, x)

        return jax.shard_map(
            local,
            mesh=device_mesh(),
            in_specs=jax.sharding.PartitionSpec("d"),
            out_specs=jax.sharding.PartitionSpec("d"),
            check_vma=False,
        )

    source = trace(function(0), 256)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(source, source, result)
    assert compare_pallas_kernels(source, trace(function(1), 256)).status is Status.DIFFERENT


def test_array_constant_keeps_all_elements_and_global_block_coordinates():
    constant = np.arange(256, dtype=np.float32) ** 2

    def function(x):
        def kernel(a, c, out):
            out[:] = a[:] + c[:]

        spec = pl.BlockSpec((128,), lambda p: (p,))
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
            grid=(2,),
            in_specs=(spec, spec),
            out_specs=spec,
        )(x, constant)

    actual, x = projected(trace(function, 256), 256)
    np.testing.assert_array_equal(actual, x + constant)


@pytest.mark.parametrize("kind", ["scratch", "dma", "semaphore"])
def test_many_independent_programs_do_not_enumerate_the_grid_or_elements(kind):
    size = 2**24
    source = trace(stateful_function(kind, size // 128), size)
    budget = AnalysisBudget(256)
    proof = normalize_view(adapt(source).view, budget=budget)
    assert budget.used < 256 and proof.projection.domains[0].size == size


def parallel_function(kind, *, interpret=False):
    if kind == "alias":
        return aliased_function(interpret=interpret, parallel=True)

    def kernel(x, y, resource):
        if kind == "scratch":
            resource[:] = x[:] + 1
            y[:] = resource[:]
        elif kind == "dma":
            tpu.async_copy(x, y, resource).wait()
        else:
            pl.semaphore_signal(resource, 1)
            pl.semaphore_wait(resource, 1)
            y[:] = x[:]

    resource = (
        tpu.VMEM((128,), jnp.float32)
        if kind == "scratch"
        else (tpu.SemaphoreType.DMA if kind == "dma" else tpu.SemaphoreType.REGULAR)
    )
    spec = pl.BlockSpec((128,), lambda p: (p,))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
        grid=(2,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(resource,),
        interpret=interpret,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel",)),
    )


@pytest.mark.parametrize("kind", ["alias", "scratch", "dma"])
def test_parallel_programs_have_the_same_independent_resource_obligations(kind):
    actual, x = projected(trace(parallel_function(kind), 256), 256)
    expected = 2 * x + 1 if kind == "alias" else x + int(kind == "scratch")
    np.testing.assert_array_equal(actual, expected)


def test_parallel_regular_signal_does_not_assume_the_target_is_the_current_core():
    source = trace(parallel_function("semaphore"), 256)
    assert compare_pallas_kernels(source, source).status is Status.EQUIVALENT
    with pytest.raises(UnsupportedJaxprError, match="core target binding"):
        normalize(source)


def execute_parallel_oracle(kind, mode):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = tpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=2,
    )
    function = parallel_function(kind, interpret=params)
    x = np.arange(1, 257, dtype=np.float32)
    compiled = jax.jit(function).lower(jax.ShapeDtypeStruct(x.shape, x.dtype)).compile()
    print("READY", flush=True)
    actual = jax.block_until_ready(compiled(x))
    jax.effects_barrier()
    expected = 2 * x + 1 if kind == "alias" else x + int(kind == "scratch")
    np.testing.assert_array_equal(actual, expected)
    assert interpreter.races is not None and not interpreter.races.races_found
    print(
        "RESULT " + json.dumps({"case": kind, "mode": mode, "status": "ok", "races": False}),
        flush=True,
    )


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "kind,mode",
    [(k, "on_wait") for k in ("alias", "scratch", "dma")] + [("dma", "eager")],
)
def test_two_core_parallel_resource_and_alias_oracle(kind, mode):
    expected = {"case": kind, "mode": mode, "status": "ok", "races": False}
    assert (
        run_worker(
            "tests.test_pallas_local_entry",
            kind,
            mode,
            expected=expected,
            entry="execute_parallel_oracle",
        )
        == expected
    )
