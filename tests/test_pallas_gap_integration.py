"""Cross-feature checks for restored remote views and padded reduction domains."""

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
    compare_algebraic_jaxprs,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import normalize_view
from tests.pallas_composed_dma_cases import native_padded_view
from tests.pallas_remote_cases import projection_array


def restored_indexed_function(*, synchronize=True, wrong=False, mesh=None):
    def kernel(x, y, send, receive, ack):
        p = pl.program_id(0)
        peer = 1 - jax.lax.axis_index("d")
        if synchronize:

            @pl.when(p > 0)
            def acknowledge():
                pl.semaphore_signal(ack, 1, device_id=(peer,))
                pl.semaphore_wait(ack, 1)

        tpu.async_remote_copy(
            x.at[:].at[:128] if wrong else x.at[:].at[128:],
            y.at[:].at[:128],
            send,
            receive,
            (peer,),
        ).wait()

        @pl.when(p < 2)
        def finish_initial_visit():
            tpu.async_remote_copy(x.at[:].at[:128], y.at[:].at[128:], send, receive, (peer,)).wait()

    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((512,), jnp.float32),
        grid=(3,),
        in_specs=[pl.BlockSpec((256,), lambda p: (p,))],
        out_specs=pl.BlockSpec(
            (256,),
            lambda p: (p % 2,),
            pipeline_mode=pl.Buffered(1, revisit=pl.RevisitMode.ANY),
        ),
        scratch_shapes=[
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.DMA,
            tpu.SemaphoreType.REGULAR,
        ],
        compiler_params=tpu.CompilerParams(dimension_semantics=("arbitrary",)),
    )
    return jax.shard_map(
        function,
        mesh=Mesh(np.asarray(jax.devices("cpu")[:2]), ("d",)) if mesh is None else mesh,
        in_specs=P("d"),
        out_specs=P("d"),
        check_vma=False,
    )


def restored_projection(**options):
    source = jax.make_jaxpr(restored_indexed_function(**options))(
        jax.ShapeDtypeStruct((1536,), jnp.float32)
    )
    return normalize_view(adapt(source).view, budget=AnalysisBudget(100000))


@pytest.mark.parametrize("wrong", [False, True])
def test_restored_output_retains_old_half_after_transformed_remote_dma(wrong):
    proof = restored_projection(wrong=wrong)
    x = np.arange(1536, dtype=np.float32)
    source = x.reshape(2, 3, 256)[::-1]
    expected = np.empty((2, 2, 256), np.float32)
    for p in range(3):
        expected[:, p % 2, :128] = source[:, p, :128] if wrong else source[:, p, 128:]
        if p < 2:
            expected[:, p, 128:] = source[:, p, :128]
    np.testing.assert_array_equal(
        projection_array(proof.projection.outputs[0], (x,), expected.size), expected.reshape(-1)
    )
    assert proof.remote.protocol.io.ports[1].restore_on_revisit
    assert any(
        command.transfer.destination_to_source is not None
        for program in proof.remote.protocol.programs
        for command in program.commands
        if hasattr(command, "transfer")
    )


def test_restoration_and_transforms_still_need_peer_synchronization():
    with pytest.raises(UnsupportedJaxprError):
        restored_projection(synchronize=False)


@pytest.mark.parametrize("limit", [383, 384, 385])
def test_native_padded_reshape_and_interval_cas_preserve_full_domain(limit):
    shape = jax.ShapeDtypeStruct((16, 384), jnp.float32)
    source = jax.make_jaxpr(native_padded_view(limit=limit))(shape)
    reference = jax.make_jaxpr(
        lambda x: jnp.broadcast_to(jnp.sum(x, axis=1, keepdims=True), (16, 128))
    )(shape)
    if limit == 385:
        with pytest.raises(MathematicalUndecidedError, match="memory initialization"):
            compare_algebraic_jaxprs(source, reference, algebra_budget=100000)
    else:
        result = compare_algebraic_jaxprs(source, reference, algebra_budget=100000)
        assert result.status is (Status.EQUIVALENT if limit == 384 else Status.INEQUIVALENT)
