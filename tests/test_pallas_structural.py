"""Public Pallas structural comparison, separate from memory/protocol proofs.

These are CPU abstract traces. No test executes a potentially blocking kernel,
and structural evidence does not claim TPU compilation or protocol validity.
"""

import subprocess
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax._src import core
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from jax_reorder import (
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_functions,
    compare_jaxprs,
    verify_correspondence,
)
from tests.test_pallas_semantic_slice import build, trace


def program(kernel, *, shape=(8, 128), scratch=(), grid=(), **kwargs):
    signature = jax.ShapeDtypeStruct(shape, jnp.float32)
    fn = pl.pallas_call(kernel, out_shape=signature, grid=grid, scratch_shapes=scratch, **kwargs)
    return fn, jax.make_jaxpr(fn)(signature)


def kernel_graph(graph):
    return graph.eqns[0].params["jaxpr"]


def assert_equivalent(a, b):
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT
    assert result.correspondence is not None
    verify_correspondence(a, b, result.correspondence)
    core.check_jaxpr(result.left.jaxpr)
    return result


def test_public_structural_and_auto_function_fast_path():
    signature = jax.ShapeDtypeStruct((1024,), jnp.float32)
    for mode in ("structural", "auto"):
        result = compare_functions(build(128), build(128), signature, equivalence=mode)
        assert result.status == Status.EQUIVALENT
    assert_equivalent(trace(), trace())


def test_retiling_is_structural_difference_without_running_memory_analysis(monkeypatch):
    from jax_reorder.core import normalization

    def forbidden(*args, **kwargs):
        pytest.fail("structural comparison must not invoke protocol/value normalization")

    monkeypatch.setattr(normalization, "normalize_view", forbidden)
    assert compare_jaxprs(trace(128), trace(256)).status == Status.DIFFERENT
    assert compare_jaxprs(trace(128), trace(128, True)).status == Status.DIFFERENT


def test_auto_mode_proves_retiling_through_the_algebra_backend():
    result = compare_functions(build(128), build(256), jax.ShapeDtypeStruct((1024,), jnp.float32))
    assert result.status is Status.EQUIVALENT
    assert result.modes[0].algebra.backend_result.backend == "sympy"


def arithmetic(reverse, subtract=False):
    def kernel(x, y):
        v = x[...]
        if reverse:
            b, a = jnp.cos(v), jnp.sin(v)
        else:
            a, b = jnp.sin(v), jnp.cos(v)
        y[...] = b - a if subtract else a + b

    return kernel


def test_independent_pure_operations_reorder_but_operands_remain_ordered():
    a = program(arithmetic(False))[1]
    b = program(arithmetic(True))[1]
    result = assert_equivalent(a, b)
    assert result.correspondence.equations[0].children
    assert compare_jaxprs(a, program(arithmetic(True, True))[1]).status == Status.DIFFERENT


@pytest.mark.parametrize("mode", ["partial", "dynamic", "reshape", "transpose", "dot", "reduce"])
def test_multidimensional_memory_and_tensor_operations(mode):
    def kernel(x, y):
        if mode == "partial":
            y[:4, :] = x[:4, :]
            y[4:, :] = x[4:, :] + 1
        elif mode == "dynamic":
            i = pl.program_id(0)
            y.at[pl.ds(i, 4), :][...] = x.at[pl.ds(i, 4), :][...] + 1
        elif mode == "reshape":
            y[...] = x[...].reshape((1, 8, 128)).reshape((8, 128))
        elif mode == "transpose":
            y[...] = jnp.transpose(jnp.transpose(x[...]))
        elif mode == "dot":
            v = x[...]
            y[...] = jnp.dot(jnp.dot(v, v.T), v)
        else:
            y[...] = jnp.broadcast_to(jnp.sum(x[...], axis=0), (8, 128))

    _, graph = program(kernel, grid=(1,))
    assert_equivalent(graph, graph)


@pytest.mark.parametrize("mode", ["cond", "fori", "while", "scan"])
def test_control_flow_preserves_ref_bindings_and_nested_order(mode):
    def kernel(x, y):
        y[...] = x[...]

        def update(i):
            y[...] = y[...] + 1
            return i + 1

        if mode == "cond":
            lax.cond(pl.program_id(0) == 0, update, lambda i: i, 0)
        elif mode == "fori":
            lax.fori_loop(0, 3, lambda i, c: update(c), 0)
        elif mode == "while":
            lax.while_loop(lambda i: i < 3, update, 0)
        else:
            lax.scan(lambda c, _: (update(c), None), 0, None, length=3)

    _, graph = program(kernel, grid=(1,))
    assert_equivalent(graph, graph)


def sem_kernel(*, split=False, wait=2, index=0, read=False):
    def kernel(x, y, sems):
        s = sems.at[index]
        if split:
            pl.semaphore_signal(s, 1)
            pl.semaphore_signal(s, 1)
        else:
            pl.semaphore_signal(s, 2)
        count = pl.semaphore_read(s) if read else 0
        pl.semaphore_wait(s, wait)
        y[...] = x[...] + count

    return kernel


def sem_graph(**options):
    return program(sem_kernel(**options), scratch=(pltpu.SemaphoreType.REGULAR((2,)),))[1]


def test_semaphore_arrays_and_counter_read_are_structural_nodes():
    assert_equivalent(sem_graph(read=True), sem_graph(read=True))
    for different in (sem_graph(index=1), sem_graph(wait=1), sem_graph(split=True)):
        assert compare_jaxprs(sem_graph(), different).status == Status.DIFFERENT


def test_no_runtime_counter_or_deadlock_proof_is_required_for_structural_identity():
    # The graph intentionally waits for more than it signals; never execute it.
    # Matching its structure is not a declaration that the program completes.
    graph = sem_graph(wait=3)
    assert_equivalent(graph, graph)


def dma_kernel(early):
    def kernel(x, y, temp, sem):
        v = x[...]
        dma = pltpu.make_async_copy(x, temp, sem)
        if early:
            dma.start()
        a = jnp.sin(v)
        if not early:
            dma.start()
        dma.wait()
        y[...] = temp[...] + a

    return kernel


def test_dma_can_commute_with_pure_work_without_pairing_or_fusing_nodes():
    scratch = (pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.DMA)
    a = program(dma_kernel(True), scratch=scratch)[1]
    b = program(dma_kernel(False), scratch=scratch)[1]
    result = assert_equivalent(a, b)
    primitives = [e.primitive.name for e in kernel_graph(result.left.jaxpr).eqns]
    assert primitives.count("dma_start") == primitives.count("dma_wait") == 1


def test_memory_and_synchronization_order_cannot_be_forged_in_correspondence():
    graph = sem_graph(split=True)
    result = assert_equivalent(graph, graph)
    kernel = kernel_graph(graph)
    positions = [i for i, e in enumerate(kernel.eqns) if e.primitive is pp.semaphore_signal_p]
    assert len(positions) == 2
    root = result.correspondence
    key, child = next(c for c in root.equations[0].children if c[0] == "params.jaxpr")
    eqs = list(child.equations)
    i, j = positions
    eqs[i], eqs[j] = replace(eqs[i], right=j), replace(eqs[j], right=i)
    altered = replace(child, equations=tuple(eqs))
    outer = replace(
        root.equations[0],
        children=tuple((k, altered if k == key else c) for k, c in root.equations[0].children),
    )
    with pytest.raises(CorrespondenceError, match="order"):
        verify_correspondence(graph, graph, replace(root, equations=(outer,)))


def test_run_scoped_resources_and_prng_state():
    def kernel(x, y):
        def body(temp, sem):
            pltpu.prng_seed(42)
            pl.semaphore_signal(sem, 1)
            temp[...] = x[...] + pltpu.prng_random_bits((8, 128)).astype(jnp.float32)
            pl.semaphore_wait(sem, 1)
            y[...] = temp[...]

        pl.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.REGULAR)

    graph = program(kernel)[1]
    assert_equivalent(graph, graph)


def test_unknown_pallas_parameter_remains_a_rejection():
    graph = trace()
    eq = graph.eqns[0]
    graph = graph.replace(eqns=[eq.replace(params={**eq.params, "new_semantics": True})])
    with pytest.raises(UnsupportedJaxprError, match="schema"):
        canonicalize(graph)


@pytest.mark.parametrize("scoped", [False, True])
def test_private_allocations_can_rename_without_changing_resource_uses(scoped):
    def build(reverse):
        def body(x, y, first, second):
            temp, sem = (second, first) if reverse else (first, second)
            pl.semaphore_signal(sem, 1)
            temp[...] = x[...]
            pl.semaphore_wait(sem, 1)
            y[...] = temp[...]

        scratch = (pltpu.VMEM((8, 128), jnp.float32), pltpu.SemaphoreType.REGULAR)
        if reverse:
            scratch = scratch[::-1]
        if scoped:

            def kernel(x, y):
                pl.run_scoped(lambda a, b: body(x, y, a, b), *scratch)

            return program(kernel)[1]
        return program(body, scratch=scratch)[1]

    assert_equivalent(build(False), build(True))


def test_scoped_captures_follow_actual_operands_after_pure_reordering():
    def build(reverse, wrong=False):
        def kernel(x, y):
            v = x[...]
            a, b = jnp.sin(v), jnp.cos(v)

            def body():
                if reverse:
                    d, c = b * 2, a + 3
                else:
                    c, d = a + 3, b * 2
                y[...] = d - c if wrong else c - d

            pl.run_scoped(body)

        return program(kernel)[1]

    assert_equivalent(build(False), build(True))
    assert compare_jaxprs(build(False), build(True, True)).status == Status.DIFFERENT


@pytest.mark.parametrize(
    "mode",
    [
        "print",
        "check",
        "random",
        "pallas_random",
        "mxu",
        "masked",
        "bitcast",
        "roll",
        "round",
        "packing",
        "annotate",
        "addupdate",
        "barrier",
        "new_ref",
        "freeze",
    ],
)
def test_official_tpu_operation_families(mode):
    def kernel(x, y):
        value = x[...]
        if mode == "print":
            pl.debug_print("v={}", value[0, 0])
        elif mode == "check":
            pl.debug_check(jnp.all(value >= 0), "negative")
        elif mode in {"random", "pallas_random"}:
            key = jax.random.key(4)
            if mode == "pallas_random":
                key = pltpu.to_pallas_key(key)
            value = jax.random.uniform(key, value.shape)
        elif mode == "mxu":
            pltpu.matmul_push_rhs(jnp.zeros((128, 128), value.dtype), 0, 0)
            pltpu.matmul_lhs_fifo(value, 0, load_staged_rhs=0)
            value = pltpu.matmul_pop_fifo(value.shape, value.dtype, 0)
        elif mode == "masked":
            value = pltpu.load(x, mask=jnp.ones(value.shape, jnp.bool_))
        elif mode == "bitcast":
            value = pltpu.bitcast(pltpu.bitcast(value, jnp.int32), jnp.float32)
        elif mode == "roll":
            value = pltpu.roll(value, 1, axis=1)
        elif mode == "round":
            value = pltpu.stochastic_round(
                value, jnp.zeros(value.shape, jnp.uint32), target_dtype=jnp.bfloat16
            ).astype(jnp.float32)
        elif mode == "packing":
            packed = pltpu.pack_elementwise(
                (value.astype(jnp.int32), (value + 1).astype(jnp.int32)), packed_dtype=jnp.int16
            )
            value = pltpu.unpack_elementwise(
                packed, index=1, packed_dtype=jnp.int16, unpacked_dtype=jnp.int32
            ).astype(jnp.float32)
        elif mode == "annotate":
            value = pltpu.load(pltpu.annotate(x, no_store=True))
        elif mode == "addupdate":
            y[...] = value
            jax.ref.addupdate(y, (), value)
            return
        elif mode == "barrier":
            semaphore = pltpu.get_barrier_semaphore()
            pl.semaphore_signal(semaphore, 1)
            pl.semaphore_wait(semaphore, 1)
        elif mode in {"new_ref", "freeze"}:
            temp = pl.empty_ref_like(pltpu.VMEM(value.shape, value.dtype))
            temp[...] = value
            value = jax.ref.freeze(temp) if mode == "freeze" else temp[...]
        y[...] = value

    graph = program(kernel)[1]
    assert_equivalent(graph, graph)


def test_free_ref_lifetime_is_ordered_even_though_its_jax_effects_are_empty():
    def build(early):
        def kernel(x, y):
            temp = pl.empty_ref_like(pltpu.VMEM(x.shape, x.dtype))
            temp[...] = x[...]
            value = temp[...]
            if early:
                core.free_ref(temp)
            y[...] = value
            if not early:
                core.free_ref(temp)

        return program(kernel)[1]

    before, after = build(True), build(False)
    assert_equivalent(before, before)
    assert compare_jaxprs(before, after).status == Status.DIFFERENT


def test_access_assumption_is_preserved_as_semantic_metadata():
    def build(no_store):
        def kernel(x, y):
            y[...] = pltpu.load(pltpu.annotate(x, no_store=no_store))

        return program(kernel)[1]

    assert compare_jaxprs(build(False), build(True)).status == Status.DIFFERENT


@pytest.mark.parametrize("library_first", [True, False])
def test_pallas_prng_support_is_independent_of_import_order(library_first):
    imports = [
        "from jax_reorder import Status, compare_jaxprs",
        "from jax.experimental.pallas import tpu as pltpu",
    ]
    if not library_first:
        imports.reverse()
    code = (
        "\n".join(imports)
        + """
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
def kernel(x, y):
    key = pltpu.to_pallas_key(jax.random.key(4))
    y[...] = x[...] + jax.random.uniform(key, x.shape)
sig = jax.ShapeDtypeStruct((8, 128), jnp.float32)
graph = jax.make_jaxpr(pl.pallas_call(kernel, out_shape=sig))(sig)
assert compare_jaxprs(graph, graph).status == Status.EQUIVALENT
"""
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, timeout=30)


def test_remote_dma_conditional_waits_and_device_routes_are_structural():
    from tests.pallas_remote_cases import remote_graph, request_reply

    for graph in (remote_graph(), remote_graph(request_reply(), semaphores=4)):
        assert_equivalent(graph, graph)
    normal = remote_graph(request_reply(), semaphores=4)
    delayed = remote_graph(request_reply(delayed=True), semaphores=4)
    assert_equivalent(delayed, delayed)
    assert compare_jaxprs(normal, delayed).status == Status.DIFFERENT


def test_multidimensional_index_maps_scalar_prefetch_and_dynamic_grid():
    signature = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    index = jax.ShapeDtypeStruct((1,), jnp.int32)
    spec = pl.BlockSpec((8, 128), lambda i, index: (i + index[0], 0))

    def kernel(index, x, y):
        y[...] = x[...] + index[0]

    call = pl.pallas_call(
        kernel,
        out_shape=signature,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1, grid=(2,), in_specs=(spec,), out_specs=spec
        ),
    )
    graph = jax.make_jaxpr(call)(index, signature)
    assert_equivalent(graph, graph)

    def dynamic(n, x):
        spec = pl.BlockSpec((8, 128), lambda i: (i, 0))
        return pl.pallas_call(
            lambda a, b: b.__setitem__(Ellipsis, a[...]),
            out_shape=signature,
            grid=(n,),
            in_specs=(spec,),
            out_specs=spec,
        )(x)

    graph = jax.make_jaxpr(dynamic)(jax.ShapeDtypeStruct((), jnp.int32), signature)
    assert_equivalent(graph, graph)


def test_cpu_interpreter_confirms_reordered_scoped_kernel_is_executable():
    def kernel(x, y):
        def body(temp):
            temp[...] = jnp.sin(x[...])
            y[...] = temp[...] + 2

        pl.run_scoped(body, pltpu.VMEM((8, 128), jnp.float32))

    fn, graph = program(kernel, interpret=True)
    normalized = canonicalize(graph)
    assert canonicalize(normalized.jaxpr).graph == normalized.graph
    x = jnp.arange(1024, dtype=jnp.float32).reshape((8, 128))
    (actual,) = core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, x)
    assert jnp.array_equal(actual, fn(x))


def test_resource_bijection_cannot_merge_distinct_semaphores():
    def build(merge):
        def kernel(x, y, a, b):
            pl.semaphore_signal(a, 1)
            pl.semaphore_signal(b, 2)
            pl.semaphore_wait(a, 1)
            pl.semaphore_wait(a if merge else b, 2)
            y[...] = x[...]

        return program(kernel, scratch=(pltpu.SemaphoreType.REGULAR,) * 2)[1]

    assert compare_jaxprs(build(False), build(True)).status == Status.DIFFERENT


def test_unknown_identity_effect_and_opaque_parameter_fail_closed():
    from jax._src import effects

    class UnknownEffect(effects.Effect):
        pass

    original = trace()
    eq = original.eqns[0]
    bad = original.replace(eqns=[eq.replace(primitive=core.Primitive("pallas_call"))])
    with pytest.raises(UnsupportedJaxprError, match="primitive"):
        canonicalize(bad)
    bad = original.replace(effects={UnknownEffect()})
    with pytest.raises(UnsupportedJaxprError, match="effect"):
        canonicalize(bad)
    bad = original.replace(
        eqns=[eq.replace(params={**eq.params, "metadata": {"opaque": object()}})]
    )
    with pytest.raises(UnsupportedJaxprError, match="opaque"):
        canonicalize(bad)


def test_semaphore_observations_cannot_move_across_signal_or_wait():
    def observe(early):
        def kernel(x, y, sem):
            pl.semaphore_signal(sem, 3)
            if early:
                value = pl.semaphore_read(sem)
            pl.semaphore_wait(sem, 3)
            if not early:
                value = pl.semaphore_read(sem)
            y[...] = x[...] + value

        return program(kernel, scratch=(pltpu.SemaphoreType.REGULAR,))[1]

    a, b = observe(True), observe(False)
    assert compare_jaxprs(a, b).status is Status.DIFFERENT
    for graph in (a, b):
        before = [
            eq.primitive
            for eq in kernel_graph(graph).eqns
            if eq.primitive in (pp.semaphore_read_p, pp.semaphore_signal_p, pp.semaphore_wait_p)
        ]
        after = [
            eq.primitive
            for eq in kernel_graph(canonicalize(graph).jaxpr).eqns
            if eq.primitive in (pp.semaphore_read_p, pp.semaphore_signal_p, pp.semaphore_wait_p)
        ]
        assert after == before
