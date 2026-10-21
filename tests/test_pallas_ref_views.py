"""Complete Ref views keep one storage object and exact coordinate bijections."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import Status, UnsupportedJaxprError, compare_pallas_kernels
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import normalize_view
from tests.pallas_oracle import run_worker
from tests.pallas_pipeline_cases import environment
from tests.pallas_projection_oracle import at_point


def function(mode="reshape_read", *, wrong=False, generic_interpret=False):
    spec = pl.BlockSpec((8, 128), lambda: (0, 0))
    scratch = (
        (tpu.VMEM((8, 128), jnp.float32), tpu.SemaphoreType.DMA) if mode.startswith("dma") else ()
    )

    def kernel(x, y, *resources):
        if mode == "reshape_read":
            y[...] = x.reshape((1, 8, 128))[...].reshape((8, 128)) + 1
        elif mode == "reshape_write":
            y.reshape((1, 8, 128))[...] = (x[...] + 1).reshape((1, 8, 128))
            y[...] = y[...] * 2
        elif mode == "transpose":
            y.transpose((1, 0))[...] = x.transpose((1, 0))[...] + 1
        elif mode == "chain":
            y[...] = (
                x.reshape((2, 4, 128))
                .transpose((0, 1, 2) if wrong else (1, 0, 2))[...]
                .reshape((8, 128))
            )
        elif mode == "chain_write":
            y.reshape((2, 4, 128)).transpose((1, 0, 2))[...] = x[...].reshape((4, 2, 128))
        elif mode == "partial":
            y[...] = x.reshape((1, 8, 128))[0, :, :] + 1
        else:
            copy = tpu.async_copy(x, resources[0], resources[1])
            if mode == "dma_write":
                resources[0].reshape((1, 8, 128))[...] = jnp.zeros((1, 8, 128), jnp.float32)
            if mode == "dma_safe":
                copy.wait()
            y[...] = resources[0].reshape((1, 8, 128))[...].reshape((8, 128))
            if mode == "dma_read":
                copy.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=(),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=scratch,
        compiler_params=tpu.CompilerParams(),
        interpret=generic_interpret,
    )


def trace(mode="reshape_read", *, wrong=False):
    return jax.make_jaxpr(function(mode, wrong=wrong))(jax.ShapeDtypeStruct((8, 128), jnp.float32))


def expected(x, mode):
    if mode in ("reshape_read", "transpose", "partial"):
        return x + 1
    if mode == "reshape_write":
        return 2 * (x + 1)
    if mode == "chain":
        return x.reshape((2, 4, 128)).transpose((1, 0, 2)).reshape(x.shape)
    if mode == "chain_write":
        return x.reshape((4, 2, 128)).transpose((1, 0, 2)).reshape(x.shape)
    return x


@pytest.mark.parametrize(
    "mode",
    ["reshape_read", "reshape_write", "transpose", "chain", "chain_write", "dma_safe", "partial"],
)
def test_full_ref_views_match_numpy_and_reuse_storage(mode):
    source = trace(mode)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    output = expected(x, mode)
    for point in np.ndindex(x.shape):
        assert at_point(result.left_protocol.projection.outputs[0], (x,), point) == output[point]


def test_wrong_ref_permutation_preserves_observable_difference():
    assert (
        compare_pallas_kernels(trace("chain"), trace("chain", wrong=True)).status
        is Status.DIFFERENT
    )
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    values = [
        normalize_view(adapt(trace("chain", wrong=wrong)).view, budget=AnalysisBudget()).projection
        for wrong in (False, True)
    ]
    assert at_point(values[0].outputs[0], (x,), (1, 0)) != at_point(
        values[1].outputs[0], (x,), (1, 0)
    )


@pytest.mark.parametrize("mode", ["dma_read", "dma_write"])
def test_partial_views_and_dma_hazards_remain_explicit(mode):
    with pytest.raises(UnsupportedJaxprError, match="full static views|DMA"):
        compare_pallas_kernels(trace(mode), trace(mode), check_protocol=True)


def execute(mode):
    x = np.arange(1024, dtype=np.float32).reshape((8, 128))
    compiled = jax.jit(function(mode, generic_interpret=True)).lower(x).compile()
    print("READY", flush=True)
    np.testing.assert_array_equal(jax.block_until_ready(compiled(x)), expected(x, mode))
    jax.effects_barrier()
    print("RESULT " + json.dumps(dict(case=mode, status="ok")), flush=True)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("mode", ["reshape_read", "reshape_write", "dma_safe"])
def test_full_reshape_views_match_official_generic_cpu_execution(mode):
    run_worker(
        "tests.test_pallas_ref_views",
        mode,
        expected=dict(case=mode, status="ok"),
        entry="execute",
        devices=1,
        execution_timeout=60,
    )


@pytest.mark.parametrize("mode", ["reshape_read", "reshape_write", "dma_safe"])
def test_static_reshape_ref_lowers_offline_to_verified_tpu_stablehlo(mode):
    with environment():
        lowered = (
            jax.jit(function(mode))
            .trace(jax.ShapeDtypeStruct((8, 128), jnp.float32))
            .lower(lowering_platforms=("tpu",))
        )
        assert lowered.compiler_ir("stablehlo").operation.verify()


def test_ref_transpose_backend_limit_is_not_confused_with_cpu_interpretation():
    with environment(), pytest.raises(NotImplementedError, match="Unsupported transform"):
        jax.jit(function("transpose")).trace(jax.ShapeDtypeStruct((8, 128), jnp.float32)).lower(
            lowering_platforms=("tpu",)
        )


@pytest.mark.parametrize("mode", ["transpose", "chain"])
def test_ref_transpose_generic_interpreter_limit_is_explicit(mode):
    with pytest.raises(NotImplementedError, match="TransposeTransform"):
        jax.jit(function(mode, generic_interpret=True)).lower(
            jax.ShapeDtypeStruct((8, 128), jnp.float32)
        )
