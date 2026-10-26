"""Pinned upstream Pallas kernels and locally authored comparison candidates.

Only the selected complete function definitions are loaded from verified source;
the enclosing projects, hardware detection, and autotuners are not imported.
Candidates below are handwritten AI variants, never substituted for the baseline.
Run this module for the structural result matrix; pytest also executes CPU interpret.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl

from jax_reorder import Status, compare_jaxprs

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/pallas_kernels"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def source(index):
    entry = MANIFEST["files"][index]
    data = (ROOT / entry["snapshot"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    assert (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        == entry["git_blob"]
    )
    return data.decode()


@functools.cache
def upstream(index):
    names = (
        {"dtype_max", "_lane_gather", "_cos_sin_lanes", "_rotate_gptj", "_rope_body"}
        if index == 0
        else {"_flash_attention_kernel_single_batch_single_step"}
    )
    definitions = [
        node
        for node in ast.parse(source(index)).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in definitions} == names
    namespace = dict(
        jax=jax,
        jnp=jnp,
        lax=lax,
        pl=pl,
        LANE=128,
        NUM_LANES=128,
        TRANS_B_DIM_NUMBERS=(((1,), (1,)), ((), ())),
    )
    module = ast.Module(body=definitions, type_ignores=[])
    # Postponed annotations avoid importing unrelated upstream dependencies.
    import __future__

    exec(
        compile(
            module,
            MANIFEST["files"][index]["path"],
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def rope_candidate(x_ref, cos_sin_ref, *out_refs, mode, quantized):
    helpers = upstream(0)
    x = x_ref[...].astype(jnp.float32)
    # Positive: rotate pure data before the next memory read. The read/write
    # sequence is unchanged, and the upstream helpers remain intact.
    rotated = helpers["_rotate_gptj"](x)
    if mode == "wrong_rotation":
        rotated = -rotated
    cos, sin = helpers["_cos_sin_lanes"](cos_sin_ref[...], rotary_dim=64, inverse=False)
    y = x * cos - rotated * sin if mode == "wrong_sign" else x * cos + rotated * sin
    if mode == "unused_pure":
        jnp.sin(x)
    if mode == "identity_mul":
        y *= 1.0
    if not quantized:
        x_ref[...] = y.astype(jnp.float32)
        return
    q_ref, scale_ref = out_refs
    qmax = helpers["dtype_max"](jnp.int8)
    abs_max = jnp.max(jnp.abs(y), axis=-1, keepdims=True)
    scale = abs_max * qmax if mode == "wrong_scale" else abs_max / qmax
    scale_inv = jnp.where(abs_max == 0.0, 0.0, 1.0 / scale)
    q_ref[...] = (y * scale_inv).astype(jnp.int8)
    scale_ref[...] = scale.reshape(scale_ref.shape)


def attention_candidate(q_ref, k_ref, v_ref, out_ref, *, mode):
    q, k = q_ref[0, 0], k_ref[0, 0]
    # Positive: the V read crosses only pure work; Q/K/V read order is retained.
    value = (k_ref if mode == "wrong_value" else v_ref)[0, 0]
    scores = lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
    scores *= 0.75 if mode == "wrong_scale" else 0.5
    row = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
    row += pl.program_id(2) * 128
    col = lax.broadcasted_iota(jnp.int32, (128, 128), 1)
    mask = col < row if mode == "wrong_mask" else col <= row
    scores += jnp.where(mask, 0.0, -1.0e6)
    maximum = jnp.max(scores, axis=1)[:, None]
    weights = jnp.exp(scores - maximum)
    total = jnp.sum(weights, axis=0 if mode == "wrong_axis" else 1)[:, None]
    weights /= total
    if mode == "unused_pure":
        jnp.sin(q)
    if mode == "identity_mul":
        weights *= 1.0
    out_ref[0, 0] = lax.dot(
        weights.astype(value.dtype), value, preferred_element_type=jnp.float32
    ).astype(out_ref.dtype)


# Every expected result is fixed before comparison. A structural difference is
# deliberately separate from whether this fixture has a numerical witness.
CASES = (
    ("rope", "reorder", True),
    ("rope", "wrong_sign", False),
    ("rope", "wrong_rotation", False),
    ("rope", "unused_pure", True),
    ("rope", "identity_mul", True),
    ("quantized_rope", "reorder", True),
    ("quantized_rope", "wrong_sign", False),
    ("quantized_rope", "wrong_scale", False),
    ("quantized_rope", "unused_pure", True),
    ("quantized_rope", "identity_mul", True),
    ("attention", "reorder", True),
    ("attention", "wrong_scale", False),
    ("attention", "wrong_mask", False),
    ("attention", "wrong_axis", False),
    ("attention", "wrong_value", False),
    ("attention", "unused_pure", True),
    ("attention", "identity_mul", True),
)


def expected_status(mode):
    # Pallas itself removes the unused sin while tracing, before this library
    # receives the graph. Multiplication by one survives that source pass.
    return Status.EQUIVALENT if mode in {"reorder", "unused_pure"} else Status.DIFFERENT


def inputs(family, seed):
    rng = np.random.default_rng(seed)
    if family == "attention":
        return tuple(jnp.asarray(rng.normal(size=(1, 1, 128, 128)), jnp.float32) for _ in range(3))
    angles = rng.uniform(-1, 1, size=(8, 32)).astype(np.float32)
    return (
        jnp.asarray(rng.normal(size=(8, 128)), jnp.float32),
        jnp.asarray(np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)),
    )


def function(family, mode=None, *, interpret=False):
    if family == "attention":
        original = upstream(1)["_flash_attention_kernel_single_batch_single_step"]

        def kernel(q, k, v, out):
            if mode is None:
                original(
                    (0, 0),
                    q,
                    k,
                    v,
                    None,
                    None,
                    None,
                    out,
                    causal=True,
                    sm_scale=0.5,
                    block_k=128,
                    kv_seq_len=128,
                    mask_value=-1.0e6,
                )
            else:
                attention_candidate(q, k, v, out, mode=mode)

        return pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1, 1, 128, 128), jnp.float32),
            grid=(1, 1, 1),
            interpret=interpret,
        )
    quantized = family == "quantized_rope"
    original = upstream(0)["_rope_body"]

    def kernel(x, cos_sin, *out):
        outputs = out if quantized else ()
        if not quantized:
            # The upstream body updates its Ref in place. Pallas input/output
            # aliases share HBM, not the input and output staging Refs on TPU.
            out[0][...] = x[...]
            x = out[0]
        if mode is None:
            original(
                x,
                cos_sin,
                *outputs,
                rotary_dim=64,
                inverse=False,
                out_dtype=jnp.dtype(jnp.float32),
                quant_dtype=jnp.dtype(jnp.int8) if quantized else None,
            )
        else:
            rope_candidate(x, cos_sin, *outputs, mode=mode, quantized=quantized)

    out_shape = (
        (jax.ShapeDtypeStruct((8, 128), jnp.int8), jax.ShapeDtypeStruct((8, 1), jnp.float32))
        if quantized
        else jax.ShapeDtypeStruct((8, 128), jnp.float32)
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        interpret=interpret,
        input_output_aliases={} if quantized else {0: 0},
    )


def oracle(family, args):
    arrays = tuple(np.asarray(x) for x in args)
    if family == "attention":
        q, k, v = arrays
        scores = q @ k.swapaxes(-1, -2) * np.float32(0.5)
        scores += np.where(
            np.arange(128)[None, :] <= np.arange(128)[:, None], np.float32(0), np.float32(-1.0e6)
        )
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        return weights / weights.sum(axis=-1, keepdims=True) @ v
    x, cos_sin = arrays
    cos, sin = np.split(cos_sin, 2, axis=-1)
    tail = x[:, -64:]
    rotated = np.stack((-tail[:, 1::2], tail[:, ::2]), axis=-1).reshape(tail.shape)
    value = np.concatenate(
        (x[:, :64], tail * cos.repeat(2, axis=-1) + rotated * sin.repeat(2, axis=-1)), axis=-1
    )
    if family == "rope":
        return value
    scale = np.abs(value).max(axis=-1, keepdims=True) / np.float32(127)
    inverse = np.divide(np.float32(1), scale, out=np.zeros_like(scale), where=scale != 0)
    return (value * inverse).astype(np.int8), scale


def main():
    results = []
    for family, mode, numerical_equal in CASES:
        args = inputs(family, 7)
        left, right = (jax.make_jaxpr(function(family, variant))(*args) for variant in (None, mode))
        result = compare_jaxprs(left, right, include_derivative_rules=False)
        expected = expected_status(mode)
        assert result.status == expected, (family, mode, result)
        results.append(
            dict(
                family=family,
                candidate=mode,
                structural=result.status.value,
                numerical_equal_expected=numerical_equal,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
