"""Finite ordinary semaphore slots retain their identities and lifetimes."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import Status, compare_pallas_kernels, verify_pallas_kernel_comparison
from jax_reorder.errors import UnsupportedJaxprError


def array_counter(*, mode="normal", scoped=False, serial=False, rank=1, observe=True):
    sem_shape = (2,) if rank == 1 else (2, 2)

    def body(x, y, *scratch):
        def compute(sem):
            def slot(i):
                return sem.at[i] if rank == 1 else sem.at[i, 1]

            if serial:
                i = pl.program_id(0) % 2
                pl.semaphore_signal(slot(i), 1)
                count = pl.semaphore_read(slot(i)) if observe else 1
                pl.semaphore_wait(slot(i), 1)
                y[...] = x[...] + count
                return
            for i in range(2):
                pl.semaphore_signal(slot(i), i + 1)
            first = pl.semaphore_read(slot(0)) if observe else 1
            second = pl.semaphore_read(slot(1)) if observe else 2
            pl.semaphore_wait(slot(0), 1)
            if mode == "wrong_slot":
                pl.semaphore_wait(slot(0), 2)
            elif mode != "leftover":
                pl.semaphore_wait(slot(1), 2)
            if mode == "out_of_bounds":
                pl.semaphore_signal(slot(jnp.int32(2)), 0)
            if mode == "data_index":
                index = x[0, 0].astype(jnp.int32)
                pl.semaphore_signal(slot(index), 0)
            y[...] = x[...] + first + second

        if scoped:
            pl.run_scoped(compute, tpu.SemaphoreType.REGULAR(sem_shape))
        else:
            compute(scratch[0])

    shape = (8, 512) if serial else (8, 128)
    spec = pl.BlockSpec((8, 128), lambda i: (0, i))
    fn = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(4,) if serial else (1,),
        in_specs=[spec],
        out_specs=spec,
        scratch_shapes=[] if scoped else [tpu.SemaphoreType.REGULAR(sem_shape)],
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
    return fn, jax.ShapeDtypeStruct(shape, jnp.float32)


def graph(**kwargs):
    fn, abstract = array_counter(**kwargs)
    return jax.make_jaxpr(fn)(abstract)


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("rank", [1, 2])
def test_regular_array_elements_preserve_count_and_scope(scoped, rank):
    closed = graph(scoped=scoped, rank=rank)
    result = compare_pallas_kernels(closed, closed, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(closed, closed, result)


def test_serial_program_coordinate_resolves_array_element():
    closed = graph(serial=True)
    assert compare_pallas_kernels(closed, closed, check_protocol=True).status is Status.EQUIVALENT


@pytest.mark.parametrize("mode", ["wrong_slot", "leftover", "out_of_bounds", "data_index"])
@pytest.mark.parametrize("scoped", [False, True])
def test_counter_slot_errors_are_not_erased(mode, scoped):
    closed = graph(mode=mode, scoped=scoped)
    with pytest.raises(UnsupportedJaxprError):
        compare_pallas_kernels(closed, closed, check_protocol=True)


def test_different_array_slots_change_structural_graph():
    good, bad = graph(), graph(mode="wrong_slot")
    assert compare_pallas_kernels(good, bad).status is Status.DIFFERENT


@pytest.mark.parametrize("loop", ["scan", "while"])
def test_remote_finite_loop_binds_dma_slices_and_replays(loop):
    from jax_reorder.core.adapters.jax import adapt
    from jax_reorder.core.index_domain import AnalysisBudget
    from jax_reorder.core.normalization import normalize_view
    from tests.pallas_remote_cases import remote_graph

    def kernel(x, y, send, receive):
        peer = 1 - jax.lax.axis_index("d")

        def step(i):
            tpu.async_remote_copy(
                x.at[pl.ds(i * 128, 128)], y.at[pl.ds(i * 128, 128)], send, receive, (peer,)
            ).wait()
            return i + 1

        if loop == "scan":
            jax.lax.fori_loop(0, 2, lambda i, _: step(i), jnp.int32(0))
        else:
            jax.lax.while_loop(lambda i: i < 2, step, jnp.int32(0))

    view = adapt(remote_graph(kernel, size=256)).view
    proof = normalize_view(view, budget=AnalysisBudget())
    assert len(proof.remote.transfers) == 4
    assert len({e.origin for e in proof.remote.protocol.protocol.events}) >= 12
    closed = remote_graph(kernel, size=256)
    result = compare_pallas_kernels(closed, closed, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(closed, closed, result)


def remote_array_function(
    *, mode="normal", regular=False, serial_grid=False, buffered=False, nesting=None, mesh=None
):
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    def kernel(x, y, *sems):
        peer = 1 - jax.lax.axis_index("d")
        if regular:
            (sem,) = sems

            def stage(i, _):
                pl.semaphore_signal(sem.at[i], 1, device_id=(peer,))
                pl.semaphore_wait(sem.at[(1 - i) if mode == "wrong_slot" else i], 1)
                return jnp.int32(0)

            jax.lax.fori_loop(0, 2, stage, jnp.int32(0))
            y[...] = x[...]
        else:
            send, receive = sems

            def start(i, _):
                tpu.async_remote_copy(
                    x.at[pl.ds(i * 128, 128)],
                    y.at[pl.ds((0 if mode == "overlap_destination" else i) * 128, 128)],
                    send.at[i],
                    receive.at[i],
                    (peer,),
                )
                return jnp.int32(0)

            if serial_grid:

                def launch():
                    i = (
                        jax.jit(lambda: pl.program_id(0) + pl.num_programs(0) - 2)()
                        if nesting == "jit"
                        else pl.program_id(0) + pl.num_programs(0) - 2
                    )

                    @pl.when(pl.program_id(0) == 0)
                    def first():
                        start(i, jnp.int32(0))

                    @pl.when(pl.program_id(0) == 1)
                    def second():
                        start(i, jnp.int32(0))

                    return jnp.int32(0)

                if nesting == "jit":
                    launch()
                elif nesting == "scan":
                    jax.lax.fori_loop(0, 1, lambda i, c: launch(), jnp.int32(0))
                elif nesting == "while":
                    jax.lax.while_loop(lambda i: i < 1, lambda i: launch() + 1, jnp.int32(0))
                else:
                    launch()
            else:
                jax.lax.fori_loop(0, 2, start, jnp.int32(0))

            def finish(i, _):
                transfer = tpu.make_async_remote_copy(
                    x.at[pl.ds(i * 128, 128)],
                    y.at[pl.ds(i * 128, 128)],
                    send.at[i],
                    receive.at[(1 - i) if mode == "wrong_slot" else i],
                    (peer,),
                )
                transfer.wait_send()
                if mode != "missing_wait":
                    transfer.wait_recv()
                return jnp.int32(0)

            if serial_grid:

                @pl.when(pl.program_id(0) == pl.num_programs(0) - 1)
                def finish_all():
                    jax.lax.fori_loop(0, 2, finish, jnp.int32(0))
            else:
                jax.lax.fori_loop(0, 2, finish, jnp.int32(0))

    spec = (
        pl.BlockSpec((256,), lambda i: (0,), memory_space=pl.ANY if not buffered else None)
        if serial_grid
        else pl.BlockSpec((256,), lambda i: (i,))
    )
    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
        grid=(2,) if serial_grid else (1,),
        in_specs=[spec],
        out_specs=spec,
        scratch_shapes=[tpu.SemaphoreType.REGULAR((2,))]
        if regular
        else [tpu.SemaphoreType.DMA((2,))] * 2,
    )
    mesh = Mesh(np.array(jax.devices("cpu")[:2]), ("d",)) if mesh is None else mesh
    return jax.shard_map(function, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False)


@pytest.mark.parametrize("regular", [False, True])
def test_remote_semaphore_arrays_keep_device_and_element_identity(regular):
    closed = jax.make_jaxpr(remote_array_function(regular=regular))(
        jax.ShapeDtypeStruct((512,), jnp.float32)
    )
    result = compare_pallas_kernels(closed, closed, check_protocol=True)
    assert result.status is Status.EQUIVALENT
    verify_pallas_kernel_comparison(closed, closed, result)


@pytest.mark.parametrize("mode,regular", [("missing_wait", False), ("wrong_slot", True)])
def test_remote_array_bad_wait_is_rejected(mode, regular):
    closed = jax.make_jaxpr(remote_array_function(mode=mode, regular=regular))(
        jax.ShapeDtypeStruct((512,), jnp.float32)
    )
    with pytest.raises(UnsupportedJaxprError):
        compare_pallas_kernels(closed, closed, check_protocol=True)


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case", ["array", "scoped", "serial", "remote_dma", "remote_regular", "remote_grid"]
)
@pytest.mark.parametrize("mode", ["on_wait", "eager"])
def test_official_cpu_interpreter_new_communication_cases(case, mode):
    from tests.pallas_oracle import run_worker

    result = run_worker(
        "tests.test_pallas_communication_extended",
        case,
        mode,
        entry="run_cpu",
        expected=dict(case=case, mode=mode, status="ok"),
    )
    assert result["status"] == "ok"


@pytest.mark.parametrize("nesting", [None, "jit", "scan", "while"])
def test_remote_serial_grid_carries_dma_on_persistent_hbm_without_barrier(nesting):
    from jax_reorder.core.adapters.jax import adapt
    from jax_reorder.core.index_domain import AnalysisBudget
    from jax_reorder.core.normalization import normalize_view

    closed = jax.make_jaxpr(remote_array_function(serial_grid=True, nesting=nesting))(
        jax.ShapeDtypeStruct((512,), jnp.float32)
    )
    proof = normalize_view(adapt(closed).view, budget=AnalysisBudget())
    assert len(proof.remote.transfers) == 4
    assert any(
        ".program[0]" in event.origin.path for event in proof.remote.protocol.protocol.events
    )
    assert any(
        ".program[1]" in event.origin.path for event in proof.remote.protocol.protocol.events
    )
    result = compare_pallas_kernels(closed, closed, check_protocol=True)
    verify_pallas_kernel_comparison(closed, closed, result)


def test_remote_grid_keeps_one_real_staging_window_for_a_constant_map():
    closed = jax.make_jaxpr(remote_array_function(serial_grid=True, buffered=True))(
        jax.ShapeDtypeStruct((512,), jnp.float32)
    )
    result = compare_pallas_kernels(closed, closed, check_protocol=True)
    proof = result.left_protocol
    phases = [event.origin.path for event in proof.remote.protocol.protocol.events]
    assert sum("input_write" in phase for phase in phases) == 2
    assert sum("output_write" in phase for phase in phases) == 2
    assert all(port.ref != port.hbm for port in proof.remote.protocol.io.ports)


def core_route_function(semantics):
    def kernel(x, y, sem):
        @pl.when(pl.program_id(0) == 0)
        def signal():
            pl.semaphore_signal(sem, 1, core_index=1)

        @pl.when(pl.program_id(0) == 1)
        def wait():
            pl.semaphore_wait(sem, 1)

        y[...] = x[...]

    spec = pl.BlockSpec((8, 128), lambda i: (0, i))
    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 256), jnp.float32),
        grid=(2,),
        in_specs=[spec],
        out_specs=spec,
        scratch_shapes=[tpu.SemaphoreType.REGULAR],
        compiler_params=tpu.CompilerParams(dimension_semantics=(semantics,)),
    )
    return function


@pytest.mark.parametrize("semantics", ["parallel", "core_parallel"])
def test_program_coordinate_is_not_a_core_route_binding(semantics):
    closed = jax.make_jaxpr(core_route_function(semantics))(
        jax.ShapeDtypeStruct((8, 256), jnp.float32)
    )
    with pytest.raises(UnsupportedJaxprError):
        compare_pallas_kernels(closed, closed, check_protocol=True)


def test_two_legal_core_assignments_have_different_counter_progress():
    from jax_reorder.core.index_domain import AnalysisBudget
    from jax_reorder.core.model import Origin
    from jax_reorder.core.protocol import (
        DeviceProgram,
        Instance,
        Semaphore,
        Signal,
        Wait,
        analyze_protocol,
    )

    # Participant IDs stand for physical cores in this minimal countermodel.
    # Logical program 0 signals physical core 1; logical program 1 waits locally.
    semaphores = tuple(Semaphore(Instance(core, 0), 0, "regular") for core in (0, 1))
    signal = Signal(Instance(1, 0), 1, Origin("program[0].signal(core=1)"))
    matching = (
        DeviceProgram(0, (signal,)),
        DeviceProgram(1, (Wait(Instance(1, 0), "regular", 1, Origin("program[1].wait")),)),
    )
    assert analyze_protocol(matching, semaphores, budget=AnalysisBudget()).counters
    swapped = (
        DeviceProgram(0, (Wait(Instance(0, 0), "regular", 1, Origin("program[1].wait")),)),
        DeviceProgram(1, (signal,)),
    )
    with pytest.raises(UnsupportedJaxprError):
        analyze_protocol(swapped, semaphores, budget=AnalysisBudget())


@pytest.mark.parametrize("serial_grid", [False, True])
@pytest.mark.parametrize("mode", ["overlap_destination", "missing_wait"])
def test_remote_pipeline_preserves_write_conflicts_and_completion(mode, serial_grid):
    closed = jax.make_jaxpr(remote_array_function(mode=mode, serial_grid=serial_grid))(
        jax.ShapeDtypeStruct((512,), jnp.float32)
    )
    with pytest.raises(UnsupportedJaxprError):
        compare_pallas_kernels(closed, closed, check_protocol=True)


AOT_CASES = (
    "array",
    "scoped",
    "serial",
    "remote_dma",
    "remote_regular",
    "remote_grid",
    "core_parallel",
)


def lower_communication_case(case):
    from jax._src.mesh import AbstractDevice
    from jax.sharding import AbstractMesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = AbstractMesh(
        (2,) if case.startswith("remote") else (),
        ("d",) if case.startswith("remote") else (),
        abstract_device=AbstractDevice("TPU v5", 1, "tpu"),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        if case.startswith("remote"):
            fn = remote_array_function(
                regular=case == "remote_regular", serial_grid=case == "remote_grid", mesh=mesh
            )
            spec = jax.ShapeDtypeStruct((512,), jnp.float32, sharding=NamedSharding(mesh, P("d")))
        elif case == "core_parallel":
            fn, spec = (
                core_route_function("core_parallel"),
                jax.ShapeDtypeStruct((8, 256), jnp.float32),
            )
        else:
            fn, spec = array_counter(scoped=case == "scoped", serial=case == "serial", rank=2)
        return jax.jit(fn).trace(spec).lower(lowering_platforms=("tpu",))


@pytest.mark.parametrize("case", AOT_CASES)
def test_new_communication_syntax_lowers_to_verified_tpu_ir(case):
    ir = lower_communication_case(case).compiler_ir("stablehlo")
    assert ir.operation.verify()
    assert "tpu_custom_call" in str(ir)


def run_cpu(case, mode):
    """Official TPU interpreter checks for newly admitted semaphore slots and loops."""
    import json

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = tpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    with tpu.force_tpu_interpret_mode(params):
        if case.startswith("remote"):
            regular = case == "remote_regular"
            function = remote_array_function(regular=regular, serial_grid=case == "remote_grid")
            x = np.arange(512, dtype=np.float32)
            expected = x if regular else np.roll(x, 256)
        else:
            function, abstract = array_counter(
                scoped=case == "scoped", serial=case == "serial", rank=2, observe=False
            )
            x = np.arange(np.prod(abstract.shape), dtype=np.float32).reshape(abstract.shape)
            expected = x + (1 if case == "serial" else 3)
        compiled = jax.jit(function).lower(x).compile()
        print("READY", flush=True)
        actual = jax.block_until_ready(compiled(x))
        jax.effects_barrier()
        np.testing.assert_array_equal(actual, expected)
        assert interpreter.races is not None and not interpreter.races.races_found
        print("RESULT " + json.dumps(dict(case=case, mode=mode, status="ok")), flush=True)
