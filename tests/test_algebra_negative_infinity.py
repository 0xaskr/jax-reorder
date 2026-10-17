"""负无穷 mask 的图级支持：有限分支可改写，非有限路径不可消项。

覆盖：
- 正例：在 mask 与非线性算子（直通、softmax、top-k）不变时，``x+x`` 与 ``2*x`` 的有限
  分支改写等价；矩阵乘法中 0.5 的缩放位置可移动；命中 ``negative_infinity_constant`` 规则
  并带浮点警告；捕获数组保留有限元素与 ``-inf`` 的精确值。
- 反例：否定谓词、分支写成 ``-1e9``、直接返回 ``2*x`` 都拒绝；非有限的加减乘除与幂运算、
  共同 JIT 内的常量/参数、gather 非有限 fill、scan carry 与后续迭代、条件返回的非有限值、
  全 mask softmax 的 NaN 路径与最终 guard 都不能被消项或交换。
- 拒绝：正无穷与 NaN 字面量在多种 shape 下仍不支持（UnsupportedJaxprError）。

前置条件：CPU 平台与 SymPy 1.14；结论是实数解释下的局部证据，不承诺 IEEE 浮点逐位等价。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jax_reorder import (
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
)


def signature(shape=(), dtype=jnp.float32):
    """构造抽象输入签名，使结论覆盖全部取值而不是某个样本。"""
    return jax.ShapeDtypeStruct(shape, dtype)


def compare(left, right, *args):
    """追踪两侧函数并返回（代数比较结果，两张图），便于同时检查规则与图。"""
    graphs = jax.make_jaxpr(left)(*args), jax.make_jaxpr(right)(*args)
    return compare_algebraic_jaxprs(*graphs), graphs


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("operation", [lambda x: x, jax.nn.softmax, lambda x: lax.top_k(x, 2)])
def test_finite_rewrite_under_unchanged_mask_and_nonlinearity(dtype, operation):
    """mask 与非线性算子不变时，只在有限分支内做代数改写应成立。"""
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
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)) as error:
        compare(
            lambda x, m: jnp.where(m, x + x, -jnp.inf),
            candidate,
            signature((3,)),
            signature((3,), jnp.bool_),
        )
    result = error.value
    assert result.path == "algebra.outputs"


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

    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare(left, right, signature((3,)), signature((3,), jnp.bool_))


def test_nonfinite_power_is_not_erased_by_real_polynomial_rule():
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare(
            lambda x: (x - jnp.inf) ** 0,
            lambda x: jnp.ones_like(x),
            signature(),
        )
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare(lambda x: (x + -jnp.inf) ** 0, lambda x: jnp.ones_like(x), signature())


@pytest.mark.parametrize("shape", [(), (3,), (0,)])
@pytest.mark.parametrize("constant", [np.inf, np.nan])
def test_positive_infinity_and_nan_literals_remain_unsupported(shape, constant):
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)) as error:
        compare(lambda x: x + constant, lambda x: constant + x, signature(shape))
    result = error.value
    assert result.path == "algebra.constant"


def test_captured_array_keeps_finite_elements_and_negative_infinity_exact():
    a = np.array([-np.inf, 1, 2], dtype=np.float32)
    b = np.array([-np.inf, 1, 3], dtype=np.float32)

    def left(x):
        return x + x + a

    result, graphs = compare(left, lambda x: 2 * x + a, signature((3,)))
    assert result.status == Status.EQUIVALENT, result
    assert result.backend_result is not None
    with pytest.raises((MathematicalUndecidedError, ComparisonTimeoutError)):
        compare(left, lambda x: 2 * x + b, signature((3,)))


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

    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)) as error:
        compare(left, right, signature())
    result = error.value
    assert "jit" in result.diagnostics["rules"]
    assert result.path == "algebra.outputs"


@pytest.mark.parametrize("fill", [-np.inf, np.inf, np.nan])
def test_nonfinite_gather_parameter_cannot_hide_inside_common_jit(fill):
    def left(x, i):
        y = jnp.take(x, i, mode="fill", fill_value=fill)
        return y - y, y

    def right(x, i):
        y = jnp.take(x, i, mode="fill", fill_value=fill)
        return jnp.zeros_like(y), y

    with pytest.raises(MathematicalUndecidedError) as error:
        compare(left, right, signature((3,)), signature((2,), jnp.int32))
    result = error.value
    assert result.failure_kind in (
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

    with pytest.raises(MathematicalUndecidedError) as error:
        compare(
            lambda x, i: run(x, i, False),
            lambda x, i: run(x, i, True),
            signature((3,)),
            signature((2,), jnp.int32),
        )
    result = error.value
    assert result.failure_kind in (
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

    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)) as error:
        compare(lambda x: run(x, False), lambda x: run(x, True), signature())
    result = error.value
    assert result.path != "algebra.constant"


def test_nonfinite_scan_cannot_be_replaced_by_independent_sum():
    def left(x):
        return lax.scan(lambda c, _: (c + -jnp.inf, None), x, None, length=2)[0]

    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)) as error:
        compare(left, lambda x: x + 2 * -jnp.inf, signature())
    result = error.value
    assert "scan:independent_accumulation" not in result.diagnostics["rules"]


def test_conditionally_returned_nonfinite_value_cannot_be_cancelled():
    def left(x, predicate):
        y = lax.cond(predicate, lambda x: x, lambda x: x + -jnp.inf, x)
        return y - y

    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare(left, lambda x, predicate: jnp.zeros_like(x), signature(), signature((), jnp.bool_))


def test_all_masked_softmax_nan_path_and_final_guard_are_preserved():
    def attention(x, m, guarded):
        p = jax.nn.softmax(jnp.where(m, x, -jnp.inf))
        return jnp.where(m, p, 0.0) if guarded else p

    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare(
            lambda x, m: attention(x, m, True),
            lambda x, m: attention(x, m, False),
            signature((3,)),
            signature((3,), jnp.bool_),
        )
