"""纯 JIT 组合及 VJP 私有残差变化；非默认调用和部分函数仍保留边界。"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from jax_reorder import Status, compare_functions


def broadcast_sum(x, w):
    return jnp.sum(w[..., None] * x[:, None, :], axis=0)


def matrix_sum(x, w):
    return w.T @ x


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_default_jit_composes_different_private_residual_interfaces(mode):
    args = jax.ShapeDtypeStruct((4, 3), jnp.float32), jax.ShapeDtypeStruct((4, 2), jnp.float32)
    result = compare_functions(jax.jit(broadcast_sum), jax.jit(matrix_sum), *args, modes=(mode,))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues
    for report in result.modes:
        assert report.algebra and report.algebra.backend_result
        assert report.algebra.correspondence is None


def test_default_jit_keeps_constants_and_actual_bindings():
    weights = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    x = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    left = jax.jit(lambda x: broadcast_sum(x, weights))

    def right(x):
        return matrix_sum(x, weights)

    assert compare_functions(left, right, x).status == Status.EQUIVALENT

    def wrong(x):
        return matrix_sum(x, weights + 1)

    assert compare_functions(left, wrong, x).status != Status.EQUIVALENT


@pytest.mark.parametrize("metadata", ["donation", "inline", "keep_unused"])
def test_jit_composition_keeps_non_default_execution_metadata(metadata):
    settings = {
        "donation": {"donate_argnums": (0,)},
        "inline": {"inline": True},
        "keep_unused": {"keep_unused": True},
    }
    left = jax.jit(lambda x: x + x, **settings[metadata])

    def right(x):
        return 2 * x

    assert compare_functions(left, right, jnp.ones(3)).status != Status.EQUIVALENT


def test_unused_partial_call_argument_cannot_disappear():
    wrapped = jax.jit(lambda x, unused: x + x)

    def left(x):
        return wrapped(x, jnp.log(x))

    def right(x):
        return 2 * x

    assert compare_functions(left, right, jnp.ones(3)).status != Status.EQUIVALENT


def test_unused_partial_call_output_cannot_disappear():
    wrapped = jax.jit(jnp.log)

    def left(x):
        wrapped(x)
        return x + x

    assert compare_functions(left, lambda x: 2 * x, jnp.ones(3)).status != Status.EQUIVALENT


def test_jit_composition_retains_nested_loop_barriers():
    def body(x, length):
        return lax.while_loop(lambda c: c[0] < length, lambda c: (c[0] + 1, c[1] + x), (0, x))[1]

    x = jnp.ones(2)
    left = jax.jit(lambda x: body(x, 3) + x)

    def right(x):
        return x + body(x, 3)

    assert compare_functions(left, right, x).status == Status.EQUIVALENT

    def wrong(x):
        return x + body(x, 4)

    assert compare_functions(left, wrong, x).status != Status.EQUIVALENT
