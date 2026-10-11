"""固定大程序的资源检查；只扩大输入程序规模，不用程序生成改写候选。"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_reorder import Status, compare_jaxprs


def long_first(x, y):
    for _ in range(6000):
        x = lax.neg(x)
    side = lax.cos(y)
    return x, side


def long_second(x, y):
    side = lax.cos(y)
    for _ in range(6000):
        x = lax.neg(x)
    return x, side


def wide_first(x, y):
    values = tuple(lax.add(x, jnp.float32(i)) for i in range(512))
    side = lax.cos(y)
    return values, side


def wide_second(x, y):
    side = lax.cos(y)
    values = tuple(lax.add(x, jnp.float32(i)) for i in range(512))
    return values, side


def test_long_chain_with_independent_branch():
    args = jnp.float32(0.7), jnp.float32(-0.3)
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (long_first, long_second))
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    assert max(result.left.depths) == 6000
    np.testing.assert_array_equal(long_first(*args), long_second(*args))


def test_high_fan_out_and_many_outputs():
    args = jnp.float32(0.7), jnp.float32(-0.3)
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (wide_first, wide_second))
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    assert len(result.left.graph.outputs) == 513
    actual = wide_first(*args)
    np.testing.assert_array_equal(actual[0], np.arange(512, dtype=np.float32) + np.float32(0.7))
