"""共同纯函数的完整语义键、原始端口映射及证明复核。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from jax_reorder import CorrespondenceError, Status, compare_functions, verify_algebraic_proof


def normalization(iterations=20, offset=0.1):
    @jax.jit
    def run(x):
        y = jnp.exp(x)
        for _ in range(iterations):
            y = y / (jnp.sum(y, axis=-1, keepdims=True) + offset)
            y = y / (jnp.sum(y, axis=-2, keepdims=True) + offset)
        return y

    return run


def test_common_nonlinear_call_is_not_expanded_and_proof_is_independent(monkeypatch):
    normalize = normalization()

    def left(x, a, b):
        y = normalize(x)
        return y * a + y * b

    def right(x, a, b):
        return normalize(x) * (a + b)

    x = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 12
    result = compare_functions(left, right, x, jnp.float32(2), jnp.float32(3))
    assert result.status == Status.EQUIVALENT, result.modes[0].issues
    mode = result.modes[0]
    assert "common_jit" in mode.algebra.rules
    assert mode.algebra.work < 2000
    from jax_reorder.algebra_normalize import PolynomialNormalizer

    def forbidden(*args, **kwargs):
        raise AssertionError("proof must not call the normalizer")

    monkeypatch.setattr(PolynomialNormalizer, "normalize", forbidden)
    verify_algebraic_proof(mode.left.jaxpr, mode.right.jaxpr, mode.algebra.proof)
    proof = mode.algebra.proof
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(
            mode.left.jaxpr, mode.right.jaxpr, replace(proof, left=replace(proof.left, roots=()))
        )


@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_common_call_original_ports_are_restored_after_region_reordering(mode):
    @jax.jit
    def pair(x, y):
        return jnp.sin(x + 2 * y), jnp.cos(x - 3 * y)

    def left(x, y, a):
        u, v = pair(x, y)
        return a * u + v

    def right(x, y, a):
        u, v = pair(x, y)
        return v + u * a

    def wrong(x, y, a):
        u, v = pair(x, y)
        return u + v * a

    args = (jnp.arange(3, dtype=jnp.float32), jnp.ones(3), jnp.float32(2))
    assert compare_functions(left, right, *args, modes=(mode,)).status == Status.EQUIVALENT
    assert compare_functions(left, wrong, *args, modes=(mode,)).status != Status.EQUIVALENT


def test_common_call_cannot_hide_changed_constant_or_missing_partial_evaluation():
    first, changed = normalization(2, 0.1), normalization(2, 0.2)

    def left(x):
        return first(x) + first(x)

    def wrong(x):
        return 2 * changed(x)

    assert compare_functions(left, wrong, jnp.ones((2, 3))).status != Status.EQUIVALENT
    log = jax.jit(jnp.log)

    def partial(x):
        return 0 * log(x)

    assert compare_functions(partial, jnp.zeros_like, jnp.ones(3)).status != Status.EQUIVALENT
