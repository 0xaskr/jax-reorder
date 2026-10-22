"""Complete pinned JAX paged-attention calls with explicit local routing facts."""

import functools
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.paged_attention import paged_attention_kernel


def signatures(*, dynamic_routing=False, bounded_pages=False):
    values = tuple(
        jax.ShapeDtypeStruct(shape, jnp.float32)
        for shape in ((1, 8, 128), (1, 8, 16, 128), (1, 8, 16, 128))
    )
    return values + (
        (jax.ShapeDtypeStruct((1,), jnp.int32), jax.ShapeDtypeStruct((1, 8), jnp.int32))
        if dynamic_routing
        else (jax.ShapeDtypeStruct((1, 8), jnp.int32),)
        if bounded_pages
        else ()
    )


def function(
    *, inline=False, wrong=False, dynamic_routing=False, bounded_pages=False, wrong_page=False
):
    original = functools.partial(
        paged_attention_kernel.paged_attention,
        pages_per_compute_block=8,
        megacore_mode=None,
        inline_seq_dim=inline,
    )

    def call(q, k, v, *routing):
        if dynamic_routing:
            lengths, pages = routing
        else:
            lengths = jnp.full((1,), 128, jnp.int32)
            pages = routing[0] if bounded_pages else jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        if bounded_pages:
            pages = pages & 7
        if wrong_page:
            k = jnp.concatenate((k[:, 1:], k[:, :1]), axis=1)
        return original(q, k, 2 * v if wrong else v, lengths, pages)

    return call


def trace(**options):
    return jax.make_jaxpr(function(**options))(
        *signatures(
            dynamic_routing=options.get("dynamic_routing", False),
            bounded_pages=options.get("bounded_pages", False),
        )
    )


def inputs():
    return tuple(
        np.random.default_rng(i + 7).normal(0, 0.1, spec.shape).astype(np.float32)
        for i, spec in enumerate(signatures())
    )


def page_tables():
    return (
        np.arange(7, -1, -1, dtype=np.int32),
        np.array([8, -1, 2, 2, 0, 15, 4, 4], np.int32),
    )


def reference(values, pages, *, wrong=False, wrong_page=False):
    """Independent NumPy formulas for all three complete kernel outputs."""
    q, k, v = values
    if wrong_page:
        k = np.concatenate((k[:, 1:], k[:, :1]), axis=1)
    # The upstream copy descriptor casts both K and V to bfloat16, including
    # non-quantized inputs. Keep that cast in the independent numerical oracle.
    k = k.astype(jnp.bfloat16).astype(np.float32)
    v = (2 * v if wrong else v).astype(jnp.bfloat16).astype(np.float32)
    keys = k[0, pages & 7].reshape(128, 128)
    values_at_pages = v[0, pages & 7].reshape(128, 128)
    scores = q[0] @ keys.T
    maximum = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - maximum)
    total = weights.sum(axis=-1, keepdims=True)
    output = weights / total @ values_at_pages
    return output[None], maximum[None], total[None]


def execute(*, inline=False, wrong=False, bounded_pages=False):
    """Official CPU interpretation against a separate full attention oracle."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
    from jax.experimental.pallas import tpu

    specs = signatures(bounded_pages=bounded_pages)
    values = inputs()
    params = tpu.InterpretParams(
        detect_races=True,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    with tpu.force_tpu_interpret_mode(params):
        compiled = (
            jax.jit(function(inline=inline, wrong=wrong, bounded_pages=bounded_pages))
            .lower(*specs)
            .compile()
        )
    print("READY", flush=True)
    tables = page_tables() if bounded_pages else (np.arange(8, dtype=np.int32),)
    for table in tables:
        expected = reference(values, table, wrong=wrong)[0]
        arguments = values + (table.reshape(1, 8),) if bounded_pages else values
        actual = jax.block_until_ready(compiled(*arguments))
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        assert interpreter.races is not None and not interpreter.races.races_found
    print(
        "RESULT "
        + json.dumps(
            dict(inline=inline, wrong=wrong, bounded_pages=bounded_pages, status="ok", races=False)
        ),
        flush=True,
    )
