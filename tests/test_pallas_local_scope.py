"""Device-local scope probes: graph evidence, protocol refusals and CPU oracles.

All source graphs use real native tracing. A structural match deliberately does
not assert that communication is valid; normalization is checked independently.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import (
    InvalidJaxprError,
    Status,
    UnsupportedJaxprError,
    compare_jaxprs,
    compare_pallas_kernels,
    verify_correspondence,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import compare_program_views, normalize_view
from tests.pallas_oracle import run_worker
from tests.pallas_protocol_cases import local_regular_function
from tests.pallas_remote_cases import projection_array, remote_graph, request_reply
from tests.test_pallas_dma import function as dma_function
from tests.test_pallas_semantic_slice import trace as ordinary_trace
from tests.test_pallas_sharding import device_mesh, sharded


def trace(function, size=128):
    return jax.make_jaxpr(function)(jax.ShapeDtypeStruct((size,), jnp.float32))


def normalize(source):
    return normalize_view(adapt(source).view, budget=AnalysisBudget())


def assert_structure(left, right):
    result = compare_jaxprs(left, right)
    assert result.status is Status.EQUIVALENT
    verify_correspondence(left, right, result.correspondence)


def inner_kernel(source):
    return source.eqns[0].params["jaxpr"].eqns[0].params["jaxpr"]


def prefetched_coordinate(offset=0):
    spec = pl.BlockSpec((128,), lambda p, d: (p,))

    def local(x):
        coordinate = lax.broadcast_in_dim(lax.axis_index("d") + offset, (1,), ())

        def kernel(d, a, b):
            b[:] = a[:] + d[0]

        function = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
            grid_spec=tpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1, grid=(1,), in_specs=(spec,), out_specs=spec
            ),
        )
        return function(coordinate, x)

    return trace(
        jax.shard_map(
            local, mesh=device_mesh(), in_specs=P("d"), out_specs=P("d"), check_vma=False
        ),
        256,
    )


def two_axis_graph(wrong=False):
    mesh = Mesh(np.array(jax.devices("cpu")[:4]).reshape((2, 2)), ("row", "col"))
    spec = pl.BlockSpec((128,), lambda p: (p,))

    def kernel(x, y, send, receive):
        row, col = lax.axis_index("row"), lax.axis_index("col")
        target = (row, lax.rem((row if wrong else col) + 1, jnp.int32(2)))
        tpu.async_remote_copy(x, y, send, receive, target).wait()

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.DMA,) * 2,
    )
    return trace(
        jax.shard_map(
            function,
            mesh=mesh,
            in_specs=P(("row", "col")),
            out_specs=P(("row", "col")),
            check_vma=False,
        ),
        512,
    )


def conditional_kernel(kind, flip=False):
    def kernel(x, y, sem):
        if kind == "data":
            predicate = x[0] > 0
        elif kind == "read":
            pl.semaphore_signal(sem, 1)
            predicate = pl.semaphore_read(sem) == 1
            pl.semaphore_wait(sem, 1)
        else:
            predicate = lax.axis_index("d") == 0
        if flip:
            predicate = lax.bitwise_not(predicate)

        def action(amount):
            def run():
                pl.semaphore_signal(sem, amount)
                pl.semaphore_wait(sem, amount)
                y[:] = x[:] + amount

            return run

        lax.cond(predicate, action(1), action(2))

    return kernel


def conditional_graph(kind, flip=False):
    kernel = conditional_kernel(kind, flip)
    if kind == "device":
        return remote_graph(kernel, regular=1, semaphores=0)
    return trace(local_regular_function(kernel))


def loop_kernel(kind, count=2):
    def kernel(x, y, sem):
        y[:] = x[:]

        def body(i):
            pl.semaphore_signal(sem, 1)
            pl.semaphore_wait(sem, 1)
            y[:] = y[:] + 1
            return i + 1

        if kind == "while":
            lax.while_loop(lambda i: i < count, body, jnp.int32(0))
        else:
            lax.fori_loop(0, count, lambda i, _: (body(i), None)[1], None)

    return kernel


def route_graph(wrap):
    def kernel(x, y, send, receive):
        target = lax.axis_index("d") + 1
        if wrap:
            target = lax.rem(target, jnp.int32(2))
        tpu.async_remote_copy(x, y, send, receive, (target,)).wait()

    return remote_graph(kernel)


def stateful_function(kind, programs, *, interpret=False):
    def kernel(x, y, *scratch):
        (resource,) = scratch
        if kind == "scratch":
            resource[:] = x[:] + 1
            y[:] = resource[:]
        elif kind == "dma":
            tpu.async_copy(x, y, resource).wait()
        else:
            pl.semaphore_signal(resource, 1)
            pl.semaphore_wait(resource, 1)
            y[:] = x[:]

    if kind in ("scratch", "dma"):
        return dma_function(
            kernel,
            programs=programs,
            size=128 * programs,
            arrays=int(kind == "scratch"),
            semaphores=int(kind == "dma"),
            interpret=interpret,
        )
    spec = pl.BlockSpec((128,), lambda p: (p,))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128 * programs,), jnp.float32),
        grid=(programs,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.REGULAR,),
        interpret=interpret,
    )


def host_preprocessed_function(kind, offset=3, *, interpret=False):
    spec = pl.BlockSpec((128,), lambda p: (p,))

    def function(x):
        if kind == "python":
            constant = sum((1, offset - 1))
        elif kind == "numpy_scalar":
            constant = np.sum(np.array([1, offset - 1], dtype=np.float32))
        else:
            constant = np.full((128,), offset, dtype=np.float32)
        if kind == "numpy_argument":

            def kernel(a, c, b):
                b[:] = a[:] + c[:]

            inputs, specs = (x, constant), (spec, spec)
        else:

            def kernel(a, b):
                b[:] = a[:] + constant

            inputs, specs = (x,), (spec,)
        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
            grid=(1,),
            in_specs=specs,
            out_specs=spec,
            interpret=interpret,
        )(*inputs)

    return function


@pytest.mark.parametrize("kind", ["python", "numpy_scalar"])
def test_host_scalar_preprocessing_becomes_a_kernel_literal(kind):
    source = trace(host_preprocessed_function(kind))
    assert [eqn.primitive.name for eqn in source.eqns] == ["pallas_call"]
    assert not source.consts
    assert_structure(source, source)
    assert (
        compare_jaxprs(source, trace(host_preprocessed_function(kind, 4))).status
        is Status.DIFFERENT
    )
    proof = normalize(source)
    x = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(projection_array(proof.projection.outputs[0], (x,), 128), x + 3)
    np.testing.assert_array_equal(host_preprocessed_function(kind, interpret=True)(x), x + 3)


def test_host_array_argument_preserves_outer_constant_binding():
    source = trace(host_preprocessed_function("numpy_argument"))
    assert [eqn.primitive.name for eqn in source.eqns] == ["pallas_call"]
    assert len(source.consts) == 1
    assert_structure(source, source)
    changed = trace(host_preprocessed_function("numpy_argument", 4))
    assert compare_jaxprs(source, changed).status is Status.DIFFERENT
    proof = normalize(source)
    x = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(projection_array(proof.projection.outputs[0], (x,), 128), x + 3)
    np.testing.assert_array_equal(
        host_preprocessed_function("numpy_argument", interpret=True)(x), x + 3
    )


def test_host_array_capture_is_rejected_by_the_jax_frontend():
    with pytest.raises(ValueError, match="captures constants .* You should pass them as inputs"):
        trace(host_preprocessed_function("numpy_capture"))


def test_outer_coordinate_binding_is_preserved_by_the_local_entry():
    source = prefetched_coordinate()
    assert_structure(source, prefetched_coordinate())
    assert compare_jaxprs(source, prefetched_coordinate(1)).status is Status.DIFFERENT
    result = compare_pallas_kernels(source, prefetched_coordinate(), check_protocol=True)
    assert result.status is Status.EQUIVALENT and result.protocol_checked
    verify_pallas_kernel_comparison(source, prefetched_coordinate(), result)
    assert compare_pallas_kernels(source, prefetched_coordinate(1)).status is Status.DIFFERENT


def test_outer_dataflow_is_composed_without_scalar_prefetch():
    function = stateful_function("dma", 1)
    source = trace(lambda x: function(x + 1))
    assert_structure(source, source)
    proof = normalize(source)
    x = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(projection_array(proof.projection.outputs[0], (x,), 128), x + 1)


def test_extracting_a_kernel_requires_context_only_when_its_graph_depends_on_it():
    left, right = (
        trace(sharded(128, size=256, mesh=device_mesh(reverse)), 256) for reverse in (False, True)
    )
    assert compare_jaxprs(left, right).status is Status.DIFFERENT
    local = compare_pallas_kernels(left, right, check_protocol=True)
    assert local.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(left, right, local)
    assert_structure(inner_kernel(left), inner_kernel(right))
    dependent = remote_graph()
    assert_structure(dependent, dependent)
    with pytest.raises(InvalidJaxprError, match="unbound axis name: d"):
        compare_jaxprs(inner_kernel(dependent), inner_kernel(dependent))


def check_two_axis_mesh():
    source = two_axis_graph()
    assert_structure(source, two_axis_graph())
    assert compare_jaxprs(source, two_axis_graph(True)).status is Status.DIFFERENT
    result = compare_pallas_kernels(source, two_axis_graph())
    assert result.status is Status.EQUIVALENT and not result.protocol_checked
    verify_pallas_kernel_comparison(source, two_axis_graph(), result)
    assert compare_pallas_kernels(source, two_axis_graph(True)).status is Status.DIFFERENT
    view = adapt(source).view
    axes = view.root.operations[0].shard_map.mesh.axes
    assert [(name, size) for name, size, _ in axes] == [("row", 2), ("col", 2)]
    with pytest.raises(UnsupportedJaxprError, match="requires one concrete mesh axis"):
        normalize_view(view, budget=AnalysisBudget())


def test_two_nontrivial_mesh_axes_preserve_route_binding_without_claiming_protocol_support():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.test_pallas_local_scope import check_two_axis_mesh; check_two_axis_mesh()",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=4",
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("flip", [False, True])
def test_device_guard_preserves_branch_values_and_balanced_counters(flip):
    source = conditional_graph("device", flip)
    assert_structure(source, source)
    assert compare_jaxprs(source, conditional_graph("device", not flip)).status is Status.DIFFERENT
    proof = normalize(source)
    x = np.arange(256, dtype=np.float32)
    increments = (2, 1) if flip else (1, 2)
    expected = x + np.repeat(increments, 128)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 256), expected
    )


@pytest.mark.parametrize("kind", ["data", "read"])
def test_dynamic_conditions_and_stable_reads_preserve_complete_branch_values(kind):
    source = conditional_graph(kind)
    assert_structure(source, source)
    assert compare_jaxprs(source, conditional_graph(kind, True)).status is Status.DIFFERENT
    proof = normalize(source)
    for sign in (1, -1):
        x = sign * np.arange(1, 129, dtype=np.float32)
        expected = x + (1 if kind == "read" or sign == 1 else 2)
        np.testing.assert_array_equal(
            projection_array(proof.projection.outputs[0], (x,), 128), expected
        )


@pytest.mark.parametrize("kind", ["fori", "while"])
def test_bounded_effectful_loops_preserve_counters_and_sequential_memory_versions(kind):
    source = trace(local_regular_function(loop_kernel(kind)))
    changed = trace(local_regular_function(loop_kernel(kind, 3)))
    assert_structure(source, source)
    assert compare_jaxprs(source, changed).status is Status.DIFFERENT
    proof = normalize(source)
    x = np.arange(128, dtype=np.float32)
    np.testing.assert_array_equal(projection_array(proof.projection.outputs[0], (x,), 128), x + 2)


def test_wrapped_route_is_valid_but_last_device_exposes_missing_modulo():
    valid, invalid = route_graph(True), route_graph(False)
    assert compare_jaxprs(valid, invalid).status is Status.DIFFERENT
    assert_structure(invalid, invalid)
    x = np.arange(256, dtype=np.float32)
    proof = normalize(valid)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), 256), np.roll(x, 128)
    )
    with pytest.raises(UnsupportedJaxprError, match="outside the mesh"):
        normalize(invalid)


@pytest.mark.parametrize(
    "case,reason", [("cycle", "wait dependency cycle"), ("missing", "conserved")]
)
def test_matching_invalid_protocols_do_not_establish_validity(case, reason):
    def missing_signal(x, y, sem):
        pl.semaphore_wait(sem, 1)
        y[:] = x[:]

    source = (
        remote_graph(request_reply(delayed=True), semaphores=4)
        if case == "cycle"
        else trace(local_regular_function(missing_signal))
    )
    assert_structure(source, source)
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize(source)


@pytest.mark.parametrize("kind", ["scratch", "dma", "semaphore"])
@pytest.mark.parametrize("programs", [1, 2])
def test_independent_program_resources_are_initialized_and_quiescent(kind, programs):
    source = trace(stateful_function(kind, programs), 128 * programs)
    assert_structure(source, source)
    proof = normalize(source)
    x = np.arange(128 * programs, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), x.size), x + int(kind == "scratch")
    )


def test_plain_multi_program_retiling_remains_supported_while_alias_is_explicit():
    source = ordinary_trace(size=256)
    retiled = ordinary_trace(256, size=256)
    assert (
        compare_program_views(adapt(source).view, adapt(retiled).view).equivalence_kind
        == "normalized"
    )
    aliased = ordinary_trace(size=256, aliases={0: 0})
    assert_structure(aliased, aliased)
    assert compare_jaxprs(source, aliased).status is Status.DIFFERENT
    proof = normalize(aliased)
    x = np.arange(256, dtype=np.float32)
    np.testing.assert_array_equal(projection_array(proof.projection.outputs[0], (x,), 256), x + 1)


def execute_local_oracle(case, mode):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = tpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
        allow_hbm_allocation_in_run_scoped=False,
    )
    if case == "data":
        function = local_regular_function(conditional_kernel("data"), interpret=params)
        x = np.arange(1, 129, dtype=np.float32)
        samples = [(x, x + 1), (-x, -x + 2)]
    elif case in ("fori", "while"):
        function = local_regular_function(loop_kernel(case), interpret=params)
        x = np.arange(128, dtype=np.float32)
        samples = [(x, x + 2)]
    else:
        function = stateful_function(case, 2, interpret=params)
        x = np.arange(256, dtype=np.float32)
        samples = [(x, x + int(case == "scratch"))]
    compiled = jax.jit(function).lower(jax.ShapeDtypeStruct(x.shape, x.dtype)).compile()
    print("READY", flush=True)
    for x, expected in samples:
        actual = jax.block_until_ready(compiled(x))
        jax.effects_barrier()
        np.testing.assert_array_equal(actual, expected)
        assert interpreter.races is not None and not interpreter.races.races_found
    print(
        "RESULT " + json.dumps({"case": case, "mode": mode, "status": "ok", "races": False}),
        flush=True,
    )


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case,mode",
    [(case, "on_wait") for case in ("data", "fori", "while", "scratch", "dma", "semaphore")]
    + [("dma", "eager")],
)
def test_valid_unproved_cases_against_isolated_official_cpu_interpreter(case, mode):
    expected = {"case": case, "mode": mode, "status": "ok", "races": False}
    assert (
        run_worker(
            "tests.test_pallas_local_scope",
            case,
            mode,
            expected=expected,
            entry="execute_local_oracle",
        )
        == expected
    )
