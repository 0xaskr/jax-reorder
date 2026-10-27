"""Pinned Keras JAX functions, independent AI candidates, and NumPy references.

Only complete original definitions are extracted. Splash calls retain the real
installed JAX TPU kernel; CPU/JAX references never replace that Pallas path.
"""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from tests.upstream_tpu.sources import load_definitions

REVISION = "cfebb2af8a202fcb31b8b86766f44eabe6295a7a"
BATCHNORM_CASES = (
    ("reordered", True, True),
    ("factored", False, True),
    ("wrong_offset", False, False),
)
ATTENTION_CASES = (
    ("direct", True, True),
    ("matmul", False, True),
    ("wrong_value", False, False),
    ("wrong_scale", False, False),
)


@cache
def originals():
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    return load_definitions(
        "keras",
        "nn",
        (
            "batch_normalization",
            "_apply_masks",
            "_dot_product_attention_core",
            "wrap_flash_attention",
        ),
        {
            "jax": jax,
            "jnp": jnp,
            "splash_attention_kernel": splash_attention_kernel,
            "splash_attention_mask": splash_attention_mask,
        },
    )


def batchnorm_function(variant=None, *, axis=-1):
    if variant is None:
        original = originals()["batch_normalization"]

        def baseline(x, mean, variance, offset, scale):
            return original(x, mean, variance, axis, offset, scale, epsilon=0.125)

        return baseline
    if variant not in {row[0] for row in BATCHNORM_CASES}:
        raise ValueError(variant)

    def candidate(x, mean, variance, offset, scale):
        shape = [1] * x.ndim
        shape[axis] = mean.shape[0]
        # These independent reshapes move before the reciprocal square root.
        scale = jnp.reshape(scale, shape)
        offset = jnp.reshape(offset, shape)
        variance = jnp.reshape(variance, shape)
        mean = jnp.reshape(mean, shape)
        inv = jax.lax.rsqrt(variance + 0.125) * scale
        if variant == "factored":
            return (x - mean) * inv + offset
        residual = -mean * inv
        residual = residual - offset if variant == "wrong_offset" else residual + offset
        return jnp.add(x * inv, residual)

    return candidate


def batchnorm_inputs(seed, *, axis=-1):
    rng = np.random.default_rng(seed)
    shape = (2, 3, 4)
    channels = shape[axis]
    arrays = (
        rng.normal(size=shape),
        rng.uniform(-0.8, 0.8, channels),
        rng.uniform(0.2, 2.0, channels),
        rng.uniform(0.2, 0.9, channels),
        rng.uniform(0.4, 1.6, channels),
    )
    return tuple(jnp.asarray(value, dtype=jnp.float32) for value in arrays)


def batchnorm_reference(args, *, axis=-1):
    x, mean, variance, offset, scale = (np.asarray(value, dtype=np.float64) for value in args)
    shape = [1] * x.ndim
    shape[axis] = mean.shape[0]
    mean, variance, offset, scale = (
        value.reshape(shape) for value in (mean, variance, offset, scale)
    )
    return (x - mean) * scale / np.sqrt(variance + 0.125) + offset


def attention_function(variant=None, *, is_causal=False):
    if variant is None:
        original = originals()["_dot_product_attention_core"]

        def baseline(query, key, value, bias, mask):
            return original(query, key, value, bias, mask, is_causal, scale=0.5)

        return baseline
    if variant not in {row[0] for row in ATTENTION_CASES}:
        raise ValueError(variant)
    apply_masks = originals()["_apply_masks"]

    def candidate(query, key, value, bias, mask):
        dtype = jnp.promote_types(query.dtype, jnp.float32)
        if variant == "matmul":
            q = jnp.transpose(query, (0, 2, 1, 3))
            k = jnp.transpose(key, (0, 2, 3, 1))
            logits = jnp.matmul(q, k, preferred_element_type=dtype)
        else:
            logits = jnp.einsum("BTNH,BSNH->BNTS", query, key, preferred_element_type=dtype)
        logits = logits * jnp.array(0.75 if variant == "wrong_scale" else 0.5, dtype=logits.dtype)
        logits = (logits + bias).astype(logits.dtype)
        logits = apply_masks(logits, mask, is_causal).astype(jnp.float32)
        probabilities = jax.nn.softmax(logits, axis=-1).astype(key.dtype)
        if variant == "matmul":
            values = jnp.transpose(value, (0, 2, 1, 3))
            return jnp.transpose(jnp.matmul(probabilities, values), (0, 2, 1, 3))
        values = key if variant == "wrong_value" else value
        return jnp.einsum("BNTS,BSNH->BTNH", probabilities, values)

    return candidate


def attention_inputs(seed):
    rng = np.random.default_rng(seed)
    query = jnp.asarray(rng.normal(size=(1, 2, 1, 2)), dtype=jnp.float32)
    key = jnp.asarray(rng.normal(size=(1, 3, 1, 2)), dtype=jnp.float32)
    value = jnp.asarray(rng.uniform(-1.0, 2.0, size=(1, 3, 1, 2)), dtype=jnp.float32)
    bias = jnp.asarray(rng.uniform(-0.2, 0.3, size=(1, 1, 2, 3)), dtype=jnp.float32)
    mask = jnp.asarray([[[[True, False, True], [True, True, False]]]])
    return query, key, value, bias, mask


def attention_reference(args, *, is_causal=False):
    query, key, value, bias, mask = (np.asarray(value) for value in args)
    q = query.astype(np.float64).transpose(0, 2, 1, 3)
    k = key.astype(np.float64).transpose(0, 2, 3, 1)
    logits = q @ k * 0.5 + bias
    allowed = np.broadcast_to(mask, logits.shape).copy()
    if is_causal:
        allowed &= np.tril(np.ones(logits.shape[-2:], dtype=bool))
    # Keras uses a finite sentinel; preserve its all-masked-row behavior.
    sentinel = np.asarray(-0.7 * np.finfo(np.float32).max, dtype=np.float32)
    logits = np.where(allowed, logits, sentinel)
    weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return (weights @ value.astype(np.float64).transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)


def splash_function(variant=None):
    if variant not in (None, "direct", "wrong_value"):
        raise ValueError(variant)
    original = originals()["wrap_flash_attention"]

    def call(query, key, value):
        if variant is None:
            return original(query, key, value, decoder_segment_ids=None)
        from jax.experimental.pallas.ops.tpu.splash_attention import (
            splash_attention_kernel,
            splash_attention_mask,
        )

        # AI wrapper reconstructs the real kernel with the same static contract.
        heads, length = query.shape[1:3]
        causal = splash_attention_mask.CausalMask(shape=(length, length))
        masks = splash_attention_mask.MultiHeadMask(masks=(causal,) * heads)
        kernel = splash_attention_kernel.make_splash_mha(
            mask=masks, head_shards=1, q_seq_shards=1, attn_logits_soft_cap=None
        )
        values = key if variant == "wrong_value" else value
        return jax.vmap(kernel)(query, key, values, segment_ids=None)

    return call


def splash_signatures():
    return tuple(jax.ShapeDtypeStruct((1, 1, 128, 128), jnp.float32) for _ in range(3))


def splash_interpreter_evidence():
    """Execute original Pallas operations under JAX's official TPU interpreter."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
    from jax.experimental.pallas import tpu as pltpu

    rng = np.random.default_rng(7)
    args = tuple(
        jnp.asarray(
            rng.normal(scale=0.1 if i < 2 else 0.5, size=(1, 1, 128, 128)), dtype=jnp.float32
        )
        for i in range(3)
    )
    query, key, value = (np.asarray(array, dtype=np.float64) for array in args)
    logits = query @ key.swapaxes(-1, -2)
    logits = np.where(np.tril(np.ones((128, 128), dtype=bool)), logits, -np.inf)
    probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    reference = probabilities @ value
    params = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    evidence = {"execution": "official_tpu_interpreter_on_cpu", "shape": list(reference.shape)}
    for variant in (None, "direct", "wrong_value"):
        with pltpu.force_tpu_interpret_mode(params):
            compiled = jax.jit(splash_function(variant)).lower(*args).compile()
        actual = np.asarray(jax.block_until_ready(compiled(*args)))
        jax.effects_barrier()
        assert np.isfinite(actual).all()
        assert interpreter.races is not None and not interpreter.races.races_found
        error = float(np.max(np.abs(actual - reference)))
        evidence["original" if variant is None else variant] = error
        if variant == "wrong_value":
            assert error > 0.1
        else:
            np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-6)
    return evidence
