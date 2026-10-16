"""负无穷 mask 的图级支持：有限分支可改写，非有限路径不可消项。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jax_reorder import Status, compare_algebraic_jaxprs, verify_algebraic_proof
from jax_reorder.errors import CorrespondenceError


def signature(shape=(), dtype=jnp.float32):
    return jax.ShapeDtypeStruct(shape, dtype)


def compare(left, right, *args):
    graphs = jax.make_jaxpr(left)(*args), jax.make_jaxpr(right)(*args)
    return compare_algebraic_jaxprs(*graphs), graphs


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("operation", [lambda x: x, jax.nn.softmax, lambda x: lax.top_k(x, 2)])
def test_finite_rewrite_under_unchanged_mask_and_nonlinearity(dtype, operation):
    result, graphs = compare(
        lambda x, m: operation(jnp.where(m, x + x, -jnp.inf)),
        lambda x, m: operation(jnp.where(m, 2 * x, -jnp.inf)),
        signature((3,), dtype),
        signature((3,), jnp.bool_),
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "algebraic"
    assert result.warnings
    assert "negative_infinity_constant" in result.rules
    assert result.backend_result is not None


def test_matmul_scale_before_mask_and_softmax():
    def attention(q, k, m, early):
        logits = (q * 0.5) @ k.T if early else (q @ k.T) * 0.5
        return jax.nn.softmax(jnp.where(m, logits, -jnp.inf), axis=-1)

    result, _ = compare(
        lambda q, k, m: attention(q, k, m, False),
        lambda q, k, m: attention(q, k, m, True),
        signature((2, 4)),
        signature((3, 4)),
        signature((2, 3), jnp.bool_),
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize(
    "candidate",
    [
        lambda x, m: jnp.where(~m, 2 * x, -jnp.inf),
        lambda x, m: jnp.where(m, 2 * x, -1e9),
        lambda x, m: 2 * x,
    ],
)
def test_mask_predicate_branch_and_constant_are_not_ignored(candidate):
    result, _ = compare(
        lambda x, m: jnp.where(m, x + x, -jnp.inf),
        candidate,
        signature((3,)),
        signature((3,), jnp.bool_),
    )
    assert result.status == Status.UNKNOWN
    assert result.issues[0].path == "algebra.outputs"


@pytest.mark.parametrize(
    "operation",
    [
        lambda y: y - y,
        lambda y: 0 * y,
        lambda y: (y - y) * 0,
        lambda y: jnp.sum(y) - jnp.sum(y),
        lambda y: (y @ y) - (y @ y),
        lambda y: jnp.sin(y) - jnp.sin(y),
        lambda y: jnp.exp(y - y) * 0,
    ],
)
def test_nonfinite_arithmetic_cannot_be_cancelled(operation):
    def left(x, m):
        return operation(jnp.where(m, x, -jnp.inf))

    def right(x, m):
        return jnp.zeros_like(operation(x))

    result, _ = compare(left, right, signature((3,)), signature((3,), jnp.bool_))
    assert result.status == Status.UNKNOWN, result


def test_nonfinite_power_is_not_erased_by_real_polynomial_rule():
    result, _ = compare(
        lambda x: (x - jnp.inf) ** 0,
        lambda x: jnp.ones_like(x),
        signature(),
    )
    # 正无穷输入仍拒绝展开；另一路检查负无穷经过幂运算的保守边界。
    assert result.status == Status.UNKNOWN
    result, _ = compare(lambda x: (x + -jnp.inf) ** 0, lambda x: jnp.ones_like(x), signature())
    assert result.status == Status.UNKNOWN


@pytest.mark.parametrize("shape", [(), (3,), (0,)])
@pytest.mark.parametrize("constant", [np.inf, np.nan])
def test_positive_infinity_and_nan_literals_remain_unsupported(shape, constant):
    result, _ = compare(lambda x: x + constant, lambda x: constant + x, signature(shape))
    assert result.status == Status.UNKNOWN
    assert result.issues[0].path == "algebra.constant"


def test_captured_array_keeps_finite_elements_and_negative_infinity_exact():
    a = np.array([-np.inf, 1, 2], dtype=np.float32)
    b = np.array([-np.inf, 1, 3], dtype=np.float32)

    def left(x):
        return x + x + a

    result, graphs = compare(left, lambda x: 2 * x + a, signature((3,)))
    assert result.status == Status.EQUIVALENT, result
    assert result.backend_result is not None
    result, _ = compare(left, lambda x: 2 * x + b, signature((3,)))
    assert result.status == Status.UNKNOWN


@pytest.mark.parametrize("hidden", [True, False])
def test_common_jit_does_not_hide_nonfinite_constants_or_arguments(hidden):
    @jax.jit
    def core(x):
        return x + -jnp.inf if hidden else x

    def left(x):
        y = core(x if hidden else x + -jnp.inf)
        return y - y, y

    def right(x):
        y = core(x if hidden else x + -jnp.inf)
        return jnp.zeros_like(y), y

    result, _ = compare(left, right, signature())
    assert result.status == Status.UNKNOWN
    assert "jit" in result.rules
    assert result.issues[0].path == "algebra.outputs"


@pytest.mark.parametrize("fill", [-np.inf, np.inf, np.nan])
def test_nonfinite_gather_parameter_cannot_hide_inside_common_jit(fill):
    def left(x, i):
        y = jnp.take(x, i, mode="fill", fill_value=fill)
        return y - y, y

    def right(x, i):
        y = jnp.take(x, i, mode="fill", fill_value=fill)
        return jnp.zeros_like(y), y

    result, _ = compare(left, right, signature((3,)), signature((2,), jnp.int32))
    assert result.status == Status.UNKNOWN
    assert result.backend_result.failure_kind in (
        "condition_undecided",
        "unsupported_conversion",
        "cas_undecided",
    )


@pytest.mark.parametrize("initial_inf", [True, False])
def test_scan_carry_tracks_nonfinite_initial_values_and_gather_fill(initial_inf):
    def run(x, index, erase):
        def body(carry, _):
            value, observed = carry
            cancelled = jnp.zeros_like(value) if erase else value - value
            next_value = jnp.take(x, index, mode="fill", fill_value=-np.inf)
            return (next_value, observed + cancelled), None

        initial = jnp.full(index.shape, -np.inf) if initial_inf else jnp.zeros(index.shape)
        result, _ = lax.scan(body, (initial, jnp.zeros_like(initial)), None, length=2)
        return result[1]

    result, _ = compare(
        lambda x, i: run(x, i, False),
        lambda x, i: run(x, i, True),
        signature((3,)),
        signature((2,), jnp.int32),
    )
    assert result.status == Status.UNKNOWN
    assert result.backend_result.failure_kind in (
        "condition_undecided",
        "unsupported_conversion",
        "cas_undecided",
    )


@pytest.mark.parametrize("loop", ["scan", "while"])
def test_later_iterations_do_not_assume_finite_carry(loop):
    def run(x, erase):
        def body(state):
            i, value, observed = state
            cancelled = jnp.zeros_like(value) if erase else value - value
            return i + 1, jnp.full_like(value, -jnp.inf), observed + cancelled

        initial = (jnp.int32(0), x, jnp.zeros_like(x))
        if loop == "scan":
            state, _ = lax.scan(lambda state, _: (body(state), None), initial, None, length=2)
        else:
            state = lax.while_loop(lambda state: state[0] < 2, body, initial)
        return state[2]

    result, _ = compare(lambda x: run(x, False), lambda x: run(x, True), signature())
    assert result.status == Status.UNKNOWN, result
    assert result.issues[0].path != "algebra.constant"


def test_nonfinite_scan_cannot_be_replaced_by_independent_sum():
    def left(x):
        return lax.scan(lambda c, _: (c + -jnp.inf, None), x, None, length=2)[0]

    result, _ = compare(left, lambda x: x + 2 * -jnp.inf, signature())
    assert result.status == Status.UNKNOWN
    assert "scan:independent_accumulation" not in result.rules


def test_conditionally_returned_nonfinite_value_cannot_be_cancelled():
    def left(x, predicate):
        y = lax.cond(predicate, lambda x: x, lambda x: x + -jnp.inf, x)
        return y - y

    result, _ = compare(
        left, lambda x, predicate: jnp.zeros_like(x), signature(), signature((), jnp.bool_)
    )
    assert result.status == Status.UNKNOWN


def test_all_masked_softmax_nan_path_and_final_guard_are_preserved():
    def attention(x, m, guarded):
        p = jax.nn.softmax(jnp.where(m, x, -jnp.inf))
        return jnp.where(m, p, 0.0) if guarded else p

    result, _ = compare(
        lambda x, m: attention(x, m, True),
        lambda x, m: attention(x, m, False),
        signature((3,)),
        signature((3,), jnp.bool_),
    )
    assert result.status == Status.UNKNOWN


def test_nonfinite_atom_proof_cannot_be_forged_as_total(legacy_algebra):
    result, graphs = compare(
        lambda x, m: jnp.where(m, x + x, -jnp.inf),
        lambda x, m: jnp.where(m, 2 * x, -jnp.inf),
        signature((3,)),
        signature((3,), jnp.bool_),
    )
    assert result.status == Status.EQUIVALENT
    steps = list(result.proof.steps)
    index = next(i for i, step in enumerate(steps) if not step.expression.total)
    step = steps[index]
    steps[index] = replace(step, expression=replace(step.expression, total=True))
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(*graphs, replace(result.proof, steps=tuple(steps)))
