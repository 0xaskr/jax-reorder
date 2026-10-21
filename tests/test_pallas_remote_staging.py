"""Automatic staging windows participate in finite remote DMA proofs."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import (
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_pallas_kernels,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import compare_program_views, normalize_view
from tests.pallas_remote_cases import projection_array


def staged_exchange(*, mode="normal", groups=2, multiple=False, mesh=None, size=None, width=128):
    input_size = groups * width * (2 if mode == "source_change" else 1)
    output_size = groups * width * (2 if mode == "output_change" else 1)
    if size is not None:
        input_size = output_size = size

    def kernel(x, y, *resources):
        total, send, receive, ack = resources if multiple else (None, *resources)
        p = pl.program_id(0)
        peer = 1 - jax.lax.axis_index("d")

        def transfer_phase(part, start):
            transfer = tpu.make_async_remote_copy(x.at[:part], y.at[:part], send, receive, (peer,))
            if start:
                transfer.start()
            else:
                if mode != "missing_send":
                    transfer.wait_send()
                if mode != "missing_receive":
                    transfer.wait_recv()

        def phase(start):
            if size is not None and size % width and mode != "unguarded_tail":
                jax.lax.cond(
                    p >= 2 * (groups - 1),
                    lambda: transfer_phase(size % width, start),
                    lambda: transfer_phase(width, start),
                )
            else:
                transfer_phase(width, start)

        @pl.when(p % 2 == 0)
        def begin():
            if mode != "no_ack":

                @pl.when(p > 0)
                def synchronize_completed_windows():
                    pl.semaphore_signal(ack, 1, device_id=(peer,))
                    pl.semaphore_wait(ack, 1)

            if multiple:

                @pl.when(p == 0)
                def initialize():
                    total[:] = jnp.zeros((128,), jnp.float32)

            phase(True)

        @pl.when(p % 2 == 1)
        def finish():
            phase(False)
            if multiple:
                total[:] += y[:]

    input_spec = pl.BlockSpec((width,), lambda p: (p if mode == "source_change" else p // 2,))
    output_spec = pl.BlockSpec(
        (width,),
        lambda p: (
            p
            if mode == "output_change"
            else groups - 1 - p // 2
            if mode == "wrong_block"
            else p // 2,
        ),
    )
    out_shape = jax.ShapeDtypeStruct((output_size,), jnp.float32)
    fn = pl.pallas_call(
        kernel,
        out_shape=(out_shape, jax.ShapeDtypeStruct((128,), jnp.float32)) if multiple else out_shape,
        grid=(groups * 2,),
        in_specs=[input_spec],
        out_specs=(output_spec, pl.BlockSpec((128,), lambda p: (0,))) if multiple else output_spec,
        scratch_shapes=[tpu.SemaphoreType.DMA, tpu.SemaphoreType.DMA, tpu.SemaphoreType.REGULAR],
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)) if mesh is None else mesh
    return jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=P("d"),
        out_specs=(P("d"), P("d")) if multiple else P("d"),
        check_vma=False,
    ), input_size * 2


def graph(**options):
    fn, size = staged_exchange(**options)
    return jax.make_jaxpr(fn)(jax.ShapeDtypeStruct((size,), jnp.float32))


@pytest.mark.parametrize("groups,multiple", [(1, False), (2, False), (3, False), (2, True)])
def test_remote_dma_crosses_programs_inside_each_buffer_window(groups, multiple):
    closed = graph(groups=groups, multiple=multiple)
    proof = normalize_view(adapt(closed).view, budget=AnalysisBudget(100000))
    x = np.arange(groups * 256, dtype=np.float32)
    expected = np.roll(x, groups * 128)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), x.size), expected
    )
    if multiple:
        total = expected.reshape(2, groups, 128).sum(axis=1).reshape(-1)
        np.testing.assert_array_equal(
            projection_array(proof.projection.outputs[1], (x,), 256), total
        )
    assert (
        compare_pallas_kernels(closed, closed, check_protocol=True, analysis_budget=100000).status
        is Status.EQUIVALENT
    )
    phases = [e.origin.path for e in proof.remote.protocol.protocol.events]
    assert sum("input_write" in path for path in phases) == groups * 2
    assert sum("output_write[1]" in path for path in phases) == groups * 2


@pytest.mark.parametrize(
    "mode", ["source_change", "output_change", "missing_send", "missing_receive", "no_ack"]
)
def test_real_staging_and_peer_order_conflicts_are_rejected(mode):
    with pytest.raises(UnsupportedJaxprError):
        normalize_view(adapt(graph(mode=mode)).view, budget=AnalysisBudget(100000))


def test_wrong_output_block_remains_a_value_difference():
    a, b = (adapt(graph(mode=mode)).view for mode in ("normal", "wrong_block"))
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(a, b, analysis_budget=100000)
    projections = [
        normalize_view(v, budget=AnalysisBudget(100000)).projection.outputs[0] for v in (a, b)
    ]
    x = np.arange(512, dtype=np.float32)
    normal, reversed_blocks = [projection_array(p, (x,), x.size) for p in projections]
    np.testing.assert_array_equal(normal, np.roll(x, 256))
    np.testing.assert_array_equal(
        reversed_blocks, np.roll(x, 256).reshape(2, 2, 128)[:, ::-1].reshape(-1)
    )
    assert not np.array_equal(normal, reversed_blocks)


def test_aligned_partial_tail_dma_never_reads_undefined_padding():
    proof = normalize_view(adapt(graph(width=256, size=384)).view, budget=AnalysisBudget(100000))
    x = np.arange(768, dtype=np.float32)
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), x.size), np.roll(x, 384)
    )
    assert sorted(sum(v.span.size for v in t.source_versions) for t in proof.remote.transfers) == [
        128,
        128,
        256,
        256,
    ]


def test_full_tail_dma_cannot_treat_undefined_input_padding_as_zero():
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(
            adapt(graph(width=256, size=384, mode="unguarded_tail")).view,
            budget=AnalysisBudget(100000),
        )


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "case",
    [
        "remote_windows",
        "remote_multi",
        "remote_tail",
        "remote_wrong_block",
        "local_fragment",
        "local_stale_tail",
        "local_reduce",
    ],
)
@pytest.mark.parametrize("mode", ["on_wait", "eager"])
def test_official_cpu_interpreter_staging_and_padding(case, mode):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.test_pallas_remote_staging",
        case,
        mode,
        entry="run_cpu",
        expected=dict(case=case, mode=mode, status="ok"),
    )


def test_large_uncovered_output_is_rejected_before_materializing_all_windows():
    with pytest.raises(UnsupportedJaxprError, match="leave a gap"):
        normalize_view(adapt(graph(groups=1, size=2**30)).view, budget=AnalysisBudget(1000))


def run_cpu(case, mode):
    """Run actual automatic windows through the official TPU CPU interpreter."""
    import json

    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    from tests.test_pallas_padding_resources import aligned_tail

    params = tpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    with tpu.force_tpu_interpret_mode(params):
        if case.startswith("remote"):
            options = dict(multiple=case == "remote_multi")
            if case == "remote_tail":
                options.update(width=256, size=384)
            if case == "remote_wrong_block":
                options.update(mode="wrong_block")
            function, size = staged_exchange(**options)
            x = np.arange(size, dtype=np.float32)
            first = np.roll(x, size // 2)
            expected = (
                (first, first.reshape(2, 2, 128).sum(axis=1).reshape(-1))
                if case == "remote_multi"
                else first
            )
            if case == "remote_wrong_block":
                expected = first.reshape(2, 2, 128)[:, ::-1].reshape(-1)
        else:
            reduce = case == "local_reduce"
            stale = case == "local_stale_tail"
            reduce |= stale
            function, shape, output_shape = aligned_tail(
                reduce=reduce, mode="stale_tail" if stale else "normal"
            )
            x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            tail = x[:, 128:256].sum(axis=-1, keepdims=True) if stale else 128 * 7
            expected = (
                np.broadcast_to(x.sum(axis=-1, keepdims=True) + tail, output_shape) if reduce else x
            )
        compiled = jax.jit(function).lower(x).compile()
    print("READY", flush=True)
    actual = jax.block_until_ready(compiled(x))
    jax.effects_barrier()
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, mode=mode, status="ok")), flush=True)
