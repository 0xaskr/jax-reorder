"""遮蔽 softmax 的有限输出事实仅由完整源图建立；全部检查只 tracing。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jax import lax
from jax._src import compiler, dispatch

from jax_reorder import Status, compare_algebraic_jaxprs
from jax_reorder.algebra_symbolic import SymbolicBuilder
from jax_reorder.algebra_transport import freeze


@pytest.fixture(autouse=True)
def no_array_execution(monkeypatch):
    jax.devices()

    def forbidden(*args, **kwargs):
        raise AssertionError("this test must only trace and reason symbolically")

    monkeypatch.setattr(compiler, "backend_compile_and_load", forbidden)
    monkeypatch.setattr(dispatch, "apply_primitive", forbidden)


def signature(shape, dtype=jnp.float32):
    return jax.ShapeDtypeStruct(shape, dtype)


def guarded(x, mask):
    return jnp.where(mask, jax.nn.softmax(jnp.where(mask, x, -jnp.inf)), 0.0)


def compare_pv(probabilities, shape=(2, 3), dtype=jnp.float32):
    def apply(x, mask, values, rewrite):
        p = probabilities(x, mask)
        if rewrite:
            return jnp.sum(p[:, :, None] * values[None, :, :], axis=1)
        return p @ values

    args = signature(shape, dtype), signature(shape, jnp.bool_), signature((shape[1], 4))
    graphs = tuple(
        jax.make_jaxpr(lambda x, m, v, rewrite=rewrite: apply(x, m, v, rewrite))(*args)
        for rewrite in (False, True)
    )
    return compare_algebraic_jaxprs(*graphs), graphs


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("nested", [False, True])
def test_guarded_softmax_proves_pv_with_source_conditions(dtype, nested):
    operation = jax.jit(jax.jit(guarded)) if nested else guarded
    result, graphs = compare_pv(lambda x, m: operation(x.astype(jnp.float32), m), dtype=dtype)
    assert result.status == Status.EQUIVALENT, result
    assert "guarded_softmax_finite" in result.rules
    assert result.backend_result is not None


def test_guarded_output_is_finite_but_retains_partial_internal_path():
    graph = jax.make_jaxpr(guarded)(signature((2, 3)), signature((2, 3), jnp.bool_))
    builder = SymbolicBuilder(100_000)
    output = builder.graph(freeze(graph))[0]
    assert output.source.finite
    assert output.finite
    assert output.defined != sp.true


def test_all_masked_rows_still_prove_guarded_pv():
    result, _ = compare_pv(lambda x, m: guarded(x, jnp.zeros_like(m)))
    assert result.status == Status.EQUIVALENT, result
    assert "guarded_softmax_finite" in result.rules


@pytest.mark.parametrize(
    "failure", ["guard", "mask_mapping", "sum_axis", "max_axis", "max_mapping"]
)
def test_guard_and_reduction_axis_mapping_must_match(failure):
    def probabilities(x, mask):
        inner_mask = mask.T if failure == "mask_mapping" else mask
        masked = jnp.where(inner_mask, x, -jnp.inf)
        maximum = jnp.max(masked, axis=0 if failure == "max_axis" else 1)
        maximum = maximum[None, :] if failure == "max_mapping" else maximum[:, None]
        numerator = jnp.exp(masked - maximum)
        denominator = jnp.sum(numerator, axis=0 if failure == "sum_axis" else 1, keepdims=True)
        return jnp.where(~mask if failure == "guard" else mask, numerator / denominator, 0.0)

    result, _ = compare_pv(probabilities, shape=(2, 2))
    assert result.status == Status.UNKNOWN, result
    assert "guarded_softmax_finite" not in result.rules


@pytest.mark.parametrize(
    "transform",
    [
        lambda x: x + -jnp.inf,
        lambda x: x / 0.0,
        lambda x: x.astype(jnp.float16).astype(jnp.float32),
    ],
)
def test_logits_require_proven_finiteness(transform):
    result, _ = compare_pv(lambda x, m: guarded(transform(x), m))
    assert result.status == Status.UNKNOWN, result
    assert "guarded_softmax_finite" not in result.rules


def test_complex_tanh_poles_do_not_inherit_the_real_finiteness_rule():
    result, _ = compare_pv(lambda x, m: guarded(jnp.abs(jnp.tanh(x)), m), dtype=jnp.complex64)
    assert result.status == Status.UNKNOWN, result
    assert "guarded_softmax_finite" not in result.rules


@pytest.mark.parametrize("subtract", [False, True])
def test_complex_tanh_poles_keep_opaque_computation_partial(subtract):
    def left(z):
        magnitude = jnp.abs(jnp.tanh(z))
        return jnp.real(z) + (magnitude - magnitude if subtract else 0 * magnitude)

    arg = signature((2,), jnp.complex64)
    graphs = jax.make_jaxpr(left)(arg), jax.make_jaxpr(jnp.real)(arg)
    result = compare_algebraic_jaxprs(*graphs)
    assert result.status == Status.UNKNOWN, result


@pytest.mark.parametrize("subtract", [False, True])
def test_common_jit_cannot_hide_complex_tanh_poles(subtract):
    operation = jax.jit(jnp.tanh)

    def left(z):
        p = operation(z)
        magnitude = jnp.abs(p)
        return jnp.real(z) + (magnitude - magnitude if subtract else 0 * magnitude), p

    def right(z):
        return jnp.real(z), operation(z)

    arg = signature((2,), jnp.complex64)
    graphs = jax.make_jaxpr(left)(arg), jax.make_jaxpr(right)(arg)
    result = compare_algebraic_jaxprs(*graphs)
    assert result.status == Status.UNKNOWN, result
    assert result.backend_result.failure_kind == "unsupported_conversion"


@pytest.mark.parametrize("hidden", [False, True])
def test_complex_tanh_feedback_does_not_assume_finite_next_carry(hidden):
    operation = jax.jit(jnp.tanh) if hidden else jnp.tanh

    def run(z, erase):
        def body(state, _):
            value, observed = state
            increment = 0.0 if erase else 0 * jnp.abs(value)
            return (operation(value), observed + increment), None

        return lax.scan(body, (z, jnp.real(z)), None, length=2)[0]

    arg = signature((2,), jnp.complex64)
    graphs = tuple(
        jax.make_jaxpr(lambda z, erase=erase: run(z, erase))(arg) for erase in (False, True)
    )
    result = compare_algebraic_jaxprs(*graphs)
    assert result.status == Status.UNKNOWN, result
    assert result.issues[0].path == "algebra.dtype"


@pytest.mark.parametrize("dtype", [jnp.float8_e4m3fn, jnp.float8_e4m3fnuz])
def test_padding_conversion_through_dtype_without_infinity_is_not_preserved(dtype):
    def probabilities(x, mask):
        # Broadcasting keeps both conversions in the abstract SSA graph; a
        # scalar-only cast could be folded by tracing into a NaN literal.
        padding = jnp.full_like(x, -jnp.inf).astype(dtype).astype(jnp.float32)
        return jnp.where(mask, jax.nn.softmax(jnp.where(mask, x, padding)), 0.0)

    graph = jax.make_jaxpr(probabilities)(signature((2, 3)), signature((2, 3), jnp.bool_))
    assert any(
        eq.primitive is lax.convert_element_type_p and eq.params["new_dtype"] == dtype
        for eq in graph.jaxpr.eqns
    )
    builder = SymbolicBuilder(100_000)
    output = builder.graph(freeze(graph))[0]
    assert not output.source.finite
    assert "guarded_softmax_finite" not in builder.rules
    result, _ = compare_pv(probabilities)
    assert result.status == Status.UNKNOWN, result
    assert "guarded_softmax_finite" not in result.rules


@pytest.mark.parametrize("nonfinite", [complex(-np.inf, 0), complex(np.nan, 0)])
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("subtract", [False, True])
def test_nonfinite_complex_constant_cannot_escape_through_absolute_value(
    nonfinite, captured, subtract
):
    value = np.asarray([nonfinite, nonfinite], np.complex64) if captured else nonfinite

    def left(x):
        magnitude = jnp.abs(jnp.asarray(value, jnp.complex64))
        return x + (magnitude - magnitude if subtract else 0 * magnitude)

    graphs = jax.make_jaxpr(left)(signature((2,))), jax.make_jaxpr(lambda x: x)(signature((2,)))
    result = compare_algebraic_jaxprs(*graphs)
    assert result.status == Status.UNKNOWN, result
    assert result.issues[0].path == "algebra.constant"


@pytest.mark.parametrize("fill", [None, -jnp.inf, jnp.inf, jnp.nan])
def test_dynamic_gather_nonfinite_fill_cannot_be_cleared(fill):
    def probabilities(x, index, mask):
        logits = jnp.take(x, index, axis=1, mode="fill", fill_value=fill)
        return guarded(logits, mask)

    graph = jax.make_jaxpr(probabilities)(
        signature((2, 5)), signature((3,), jnp.int32), signature((2, 3), jnp.bool_)
    )
    builder = SymbolicBuilder(100_000)
    output = builder.graph(freeze(graph))[0]
    assert not output.source.finite
    assert "guarded_softmax_finite" not in builder.rules


def test_scan_constants_and_slices_keep_finiteness_but_carry_does_not():
    def run(x, mask, values, rewrite, unsafe):
        def body(carry, row):
            logits = carry[0] if unsafe else row
            p = guarded(logits, mask)
            pv = jnp.sum(p[:, :, None] * values[None, :, :], axis=1) if rewrite else p @ values
            return (jnp.full_like(logits, -jnp.inf), pv), None

        initial = (x, jnp.zeros((x.shape[0], values.shape[1])))
        return lax.scan(body, initial, jnp.stack((x, x)), length=2)[0][1]

    args = signature((2, 3)), signature((2, 3), jnp.bool_), signature((3, 4))
    for unsafe in (False, True):
        graphs = tuple(
            jax.make_jaxpr(
                lambda x, m, v, rewrite=rewrite, unsafe=unsafe: run(x, m, v, rewrite, unsafe)
            )(*args)
            for rewrite in (False, True)
        )
        result = compare_algebraic_jaxprs(*graphs)
        assert result.status == (Status.UNKNOWN if unsafe else Status.EQUIVALENT), result
        if not unsafe:
            assert "guarded_softmax_finite" in result.rules
        else:
            # The first iteration may be finite; it does not authorize the next carry.
            assert result.backend_result.failure_kind == "condition_undecided"


def test_large_axis_does_not_increase_finiteness_work():
    work = []
    for size in (3, 1_000_000):
        graph = jax.make_jaxpr(guarded)(signature((2, size)), signature((2, size), jnp.bool_))
        builder = SymbolicBuilder(100_000)
        assert builder.graph(freeze(graph))[0].source.finite
        work.append(builder.budget.used)
    assert work[0] == work[1]


def test_internal_token_is_rejected_without_attribute_error():
    def left(x):
        lax.create_token(x)
        return x + x

    graphs = jax.make_jaxpr(left)(signature(())), jax.make_jaxpr(lambda x: 2 * x)(signature(()))
    result = compare_algebraic_jaxprs(*graphs)
    assert result.status == Status.UNKNOWN
    assert result.issues[0].path == "algebra.shape"
