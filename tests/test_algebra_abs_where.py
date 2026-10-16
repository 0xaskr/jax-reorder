"""绝对值分段恒等式、零点导数、真实绑定和证明复核。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jax_reorder import (
    CorrespondenceError,
    Status,
    compare_algebraic_jaxprs,
    compare_functions,
    verify_algebraic_proof,
)
from jax_reorder.algebra_normalize import PolynomialNormalizer


def _graphs(left, right, shape=(3,), dtype=jnp.float32):
    """使用抽象签名证明全部动态输入，而不是只比较给定样本。"""
    argument = jax.ShapeDtypeStruct(shape, dtype)
    return jax.make_jaxpr(left)(argument), jax.make_jaxpr(right)(argument)


VARIANTS = (
    lambda x: jnp.where(x > 0, x, -x),
    lambda x: jnp.where(x >= 0, x, -x),
    lambda x: jnp.where(0 < x, x, -x),
    lambda x: jnp.where(0 <= x, x, -x),
    lambda x: jnp.where(x < 0, -x, x),
    lambda x: jnp.where(x <= 0, -x, x),
    lambda x: jnp.where(0 > x, -x, x),
    lambda x: jnp.where(0 >= x, -x, x),
)


@pytest.mark.parametrize("rewritten", VARIANTS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_absolute_matches_both_half_axis_spellings(rewritten, dtype):
    left, right = _graphs(jnp.abs, rewritten, dtype=dtype)
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "algebraic"
    assert "select_n" in result.rules
    assert result.warnings
    assert result.backend_result is not None


@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 0, 3), (2, 1, 3)])
def test_absolute_scalar_empty_and_unit_axes(shape):
    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, VARIANTS[0], shape))
    assert result.status == Status.EQUIVALENT, result


def test_boolean_broadcast_tracks_the_same_element_as_both_branches():
    def left(x):
        return jnp.abs(jnp.broadcast_to(x, (2, 3)))

    def right(x):
        return jnp.where(x > 0, jnp.broadcast_to(x, (2, 3)), -x)

    result = compare_algebraic_jaxprs(*_graphs(left, right))
    assert result.status == Status.EQUIVALENT, result


def test_direct_select_and_transparently_named_wrapper_are_supported():
    def select_only(predicate, negative, positive):
        return lax.select(predicate, positive, negative)

    def rewritten(x):
        return jax.jit(select_only)(x > 0, -x, x)

    for right in (lambda x: lax.select(x > 0, x, -x), rewritten):
        result = compare_algebraic_jaxprs(*_graphs(jnp.abs, right))
        assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize(
    "rewritten",
    [
        lambda x: jnp.where(x > 1, x, -x),
        lambda x: jnp.where(x > -1, x, -x),
        lambda x: jnp.where(x > 0, -x, x),
        lambda x: jnp.where(x > 0, x, x),
        lambda x: jnp.where(x > 0, x, -2 * x),
        lambda x: jnp.where(x > 0, x + 1, -x),
        lambda x: jnp.where(x == 0, x, -x),
        lambda x: jnp.where(x != 0, x, -x),
    ],
)
def test_wrong_threshold_comparison_and_branches_are_not_absolute(rewritten):
    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, rewritten))
    assert result.status == Status.INEQUIVALENT
    assert result.proof is None


def test_comparison_cannot_be_borrowed_from_another_input():
    value = jax.ShapeDtypeStruct((3,), jnp.float32)
    left = jax.make_jaxpr(lambda x, y: jnp.abs(x))(value, value)
    right = jax.make_jaxpr(lambda x, y: jnp.where(y > 0, x, -x))(value, value)
    assert compare_algebraic_jaxprs(left, right).status == Status.INEQUIVALENT


def test_condition_broadcast_cannot_swap_equal_length_axes():
    value = jax.ShapeDtypeStruct((2,), jnp.float32)
    left = jax.make_jaxpr(lambda x: jnp.abs(jnp.broadcast_to(x, (2, 2))))(value)
    right = jax.make_jaxpr(lambda x: jnp.where(x[:, None] > 0, x[None, :], -x[None, :]))(value)
    assert compare_algebraic_jaxprs(left, right).status == Status.INEQUIVALENT


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_strict_positive_selection_does_not_hide_zero_derivative_difference(mode):
    result = compare_functions(jnp.abs, VARIANTS[0], jnp.zeros(3), modes=("forward", mode))
    assert result.modes[0].status == Status.EQUIVALENT
    assert result.modes[1].status == Status.INEQUIVALENT
    assert result.status == Status.INEQUIVALENT
    # 浮点抽样仅作为反例证据，等价结论仍由抽象图证明决定。
    _, absolute = jax.jvp(jnp.abs, (jnp.float32(0),), (jnp.float32(1),))
    _, rewritten = jax.jvp(VARIANTS[0], (jnp.float32(0),), (jnp.float32(1),))
    assert absolute == 1 and rewritten == -1


def test_non_strict_positive_selection_preserves_jax_first_derivatives():
    result = compare_functions(
        jnp.abs,
        VARIANTS[1],
        jnp.array([-2.0, -0.0, 0.0, 3.0]),
        modes=("forward", "jvp", "vjp"),
        check_cpu=True,
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.cpu_compiled


def test_signed_zero_is_a_real_equality_with_explicit_float_warning():
    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, VARIANTS[0], ()))
    assert result.status == Status.EQUIVALENT, result
    assert any("signed zero" in warning for warning in result.warnings)
    assert not np.signbit(np.asarray(jnp.abs(jnp.float32(0))))
    assert np.signbit(np.asarray(VARIANTS[0](jnp.float32(0))))


def test_proof_verifier_rechecks_piecewise_sources_without_normalizing(monkeypatch, legacy_algebra):
    left, right = _graphs(jnp.abs, VARIANTS[0])
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.EQUIVALENT, result

    def forbidden(*args, **kwargs):
        raise AssertionError("proof verification must not normalize")

    monkeypatch.setattr(PolynomialNormalizer, "normalize", forbidden)
    verify_algebraic_proof(left, right, result.proof)
    _, different = _graphs(jnp.abs, lambda x: jnp.where(x > 1, x, -x))
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(left, different, result.proof)
    changed = replace(result.proof, right=replace(result.proof.right, graph=different))
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(left, right, changed)


@pytest.mark.parametrize("metadata", ["donation", "inline", "compiler_options"])
def test_select_wrapper_keeps_non_default_call_metadata(metadata):
    options = {
        "donation": {"donate_argnums": (1,)},
        "inline": {"inline": True},
        "compiler_options": {},
    }[metadata]

    @jax.jit(**options)
    def wrapped(predicate, negative, positive):
        return lax.select(predicate, positive, negative)

    left, right = _graphs(jnp.abs, lambda x: wrapped(x > 0, -x, x))
    if metadata == "compiler_options":
        # JAX 的 Python API 禁止嵌套 compiler_options；直接检查 IR 参数保护。
        graph = right.jaxpr
        equations = list(graph.eqns)
        node = equations[-1]
        equations[-1] = node.replace(
            params={**node.params, "compiler_options_kvs": (("xla_cpu_enable_fast_math", False),)}
        )
        right = graph.replace(eqns=equations)
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.UNKNOWN


def test_wrapper_name_does_not_authorize_other_instructions():
    @jax.jit
    def _where(predicate, negative, positive):
        return lax.select(predicate, positive, negative) + 1

    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, lambda x: _where(x > 0, -x, x)))
    assert result.status == Status.INEQUIVALENT


def test_non_real_input_does_not_acquire_the_real_piecewise_rule():
    left, right = _graphs(jnp.abs, VARIANTS[0], dtype=jnp.int32)
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.UNKNOWN
    assert "select_absolute" not in result.rules
