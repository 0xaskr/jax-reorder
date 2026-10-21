"""Source-bounded integer addresses and finite loop-carried scalar state."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

SHAPE = (8, 128)


def function(mode="mod", *, wrong=False, interpret=False, initialize_first=False):
    def kernel(x, y):
        if initialize_first:
            y[...] = x[...]
        raw = x[0, 0]
        if mode in ("and", "oob_and"):
            index = raw & (15 if mode == "oob_and" else 1 if wrong else 3)
        elif mode == "mod":
            index = raw % (3 if wrong else 4)
        elif mode in ("div", "lax_div"):
            clipped = jnp.minimum(jnp.maximum(raw, 0), 15)
            divisor = 5 if wrong else 4
            index = clipped // divisor if mode == "div" else lax.div(clipped, jnp.int32(divisor))
        elif mode == "negative_div":
            index = lax.div(jnp.minimum(jnp.maximum(raw, -15), 0), jnp.int32(-4))
        elif mode in ("wrong_guard", "signed_rem"):
            remainder = lax.rem(raw, jnp.int32(4))
            index = (
                jnp.where(remainder != 0, remainder + 4, remainder)
                if mode == "wrong_guard"
                else remainder
            )
        elif mode == "zero_divisor":
            index = lax.div(raw, jnp.int32(0))
        elif mode == "overflow":
            index = lax.div(raw, jnp.int32(-1))
        elif mode == "unbounded":
            index = raw
        else:
            index = raw & 3
        if mode != "missing_init" and not initialize_first:
            y[...] = x[...]
        if mode in ("carry", "alias_carry", "no_progress", "missing_init"):

            def condition(carry):
                return carry[1] > 0 if mode == "no_progress" else (carry[0] < 4) & (carry[1] > 0)

            def body(carry):
                y[...] += carry[2] if mode == "alias_carry" else 1
                return (
                    carry[0] + 1,
                    carry[1] - (0 if mode == "no_progress" else 2 if wrong else 1),
                    carry[2],
                )

            lax.while_loop(condition, body, (jnp.int32(0), index, index))
        else:
            y[index, :] = x[0, :] + 10

    spec = pl.BlockSpec(SHAPE, lambda: (0, 0))
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(SHAPE, jnp.int32),
        grid=(),
        in_specs=(spec,),
        out_specs=spec,
        interpret=interpret,
        compiler_params=tpu.CompilerParams(),
    )


def graph(mode="mod", **options):
    return jax.make_jaxpr(function(mode, **options))(jax.ShapeDtypeStruct(SHAPE, jnp.int32))


def inputs(raw):
    x = np.arange(np.prod(SHAPE), dtype=np.int32).reshape(SHAPE)
    x[0, 0] = raw
    return x


def expected(x, mode="mod", *, wrong=False):
    raw = int(x[0, 0])
    result = x.copy()
    if mode in ("carry", "alias_carry"):
        count = raw & 3
        result += ((count + 1) // 2 if wrong else count) * (count if mode == "alias_carry" else 1)
        return result
    if mode == "and":
        index = raw & (1 if wrong else 3)
    elif mode == "mod":
        index = raw % (3 if wrong else 4)
    elif mode in ("div", "lax_div"):
        index = min(max(raw, 0), 15) // (5 if wrong else 4)
    elif mode == "negative_div":
        index = abs(min(max(raw, -15), 0)) // 4
    elif mode == "wrong_guard":
        remainder = (abs(raw) % 4) * (-1 if raw < 0 else 1)
        index = remainder + 4 if remainder else remainder
    else:
        raise ValueError(mode)
    result[index] = x[0] + np.int32(10)
    return result


def layout_function(*, hint=True, wrong=False, interpret=False, reverse_loads=False, **options):
    def kernel(x, z, y):
        if reverse_loads:
            right, left = z[...], x[...]
        else:
            left, right = x[...], z[...]
        value = lax.dot_general(
            left,
            right,
            (((1 if wrong else 0,), (0,)), ((), ())),
            precision=lax.Precision.HIGHEST,
        )
        y[...] = value

    typ = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    spec = pl.BlockSpec((128, 128), lambda: (0, 0))
    return pl.pallas_call(
        kernel,
        out_shape=typ,
        grid=(),
        in_specs=(spec, spec),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(fuse_transposed_lhs_in_matmul=hint, **options),
        interpret=interpret,
    )
