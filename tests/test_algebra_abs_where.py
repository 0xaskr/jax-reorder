"""绝对值分段恒等式、零点导数、真实绑定和证明复核。

覆盖：
- 正例：``abs(x)`` 与 8 种 where 写法（``>``/``>=``，正负分支互换，常量在左/在右）在三种
  浮点精度下等价；标量、零长度与单位轴；布尔广播与两条分支引用同一元素；直接 select 与
  透明 jit 包装；非严格比较保留 JAX 的一阶导数。
- 反例：错误的阈值（1、-1）、分支互换、两分支相同、非 abs 的线性分支、+1 偏移、``==``/
  ``!=`` 谓词都判 inequivalent；条件谓词不能借用另一个输入；等长轴之间的广播交换判不同。
- 拒绝与未决：包装器的非默认调用元数据（donation/inline/compiler_options）拒绝；同名
  wrapper 不能授权额外指令；整数输入不获得实数分段规则；零点处严格正选择使 JVP/VJP 与
  ``abs`` 不同，整体判 inequivalent。
- 正例附带：有符号零在实数上相等，但必须发出 signed zero 警告；执行结果仍保留 ``-0.0``
  的符号位。

前置条件：CPU 平台与 SymPy 1.14；抽象签名保证结论覆盖全部输入而不是单个样本。
8 种写法不覆盖所有可能的谓词组合。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jax_reorder import (
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
)


def _graphs(left, right, shape=(3,), dtype=jnp.float32):
    """使用抽象签名证明全部动态输入，而不是只比较给定样本。"""
    argument = jax.ShapeDtypeStruct(shape, dtype)
    return jax.make_jaxpr(left)(argument), jax.make_jaxpr(right)(argument)


# 8 种等价的绝对值写法：比较方向与常量位置（x>0 / 0<x）、正负分支的排列，以及 > 与 >=
# 的区别（零点导数不同，见后面的 JVP/VJP 用例）。
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
    """每种写法都要命中 select_n 规则、给出代数结论并保留浮点警告。"""
    left, right = _graphs(jnp.abs, rewritten, dtype=dtype)
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "algebraic"
    assert "select_n" in result.rules
    assert result.warnings
    assert result.backend_result is not None


@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 0, 3), (2, 1, 3)])
def test_absolute_scalar_empty_and_unit_axes(shape):
    """标量、零长度与单位长度轴不改变分段规则，只需按输出坐标逐点成立。"""
    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, VARIANTS[0], shape))
    assert result.status == Status.EQUIVALENT, result


def test_boolean_broadcast_tracks_the_same_element_as_both_branches():
    """谓词按广播后的元素位置生效；两分支都读取同一输入元素。"""

    def left(x):
        return jnp.abs(jnp.broadcast_to(x, (2, 3)))

    # right 的谓词是 x > 0（形状 (3,) 广播），分支分别广播 x 与 -x。
    def right(x):
        return jnp.where(x > 0, jnp.broadcast_to(x, (2, 3)), -x)

    result = compare_algebraic_jaxprs(*_graphs(left, right))
    assert result.status == Status.EQUIVALENT, result


def test_direct_select_and_transparently_named_wrapper_are_supported():
    """直接 select 与一层透明 jit wrapper 都应被看作同一 select 语义。"""

    def select_only(predicate, negative, positive):
        return lax.select(predicate, positive, negative)

    # rewritten 通过 jit 包装同一函数，仍应被接受；wrapper 名不承担额外授权。
    def rewritten(x):
        return jax.jit(select_only)(x > 0, -x, x)

    for right in (lambda x: lax.select(x > 0, x, -x), rewritten):
        result = compare_algebraic_jaxprs(*_graphs(jnp.abs, right))
        assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize(
    "rewritten",
    [
        # 以下写法看似 abs，但阈值、分支符号、分支表达式或谓词至少一处不同。
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
    """错误阈值、分支或谓词必须给出完整输出的负证明，不依赖结构对应。"""
    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, rewritten))
    assert result.status == Status.INEQUIVALENT
    assert result.correspondence is None


def test_comparison_cannot_be_borrowed_from_another_input():
    """谓词必须由同一次 abs 调用的输入产生，不能借用第二个输入 y。"""
    value = jax.ShapeDtypeStruct((3,), jnp.float32)
    left = jax.make_jaxpr(lambda x, y: jnp.abs(x))(value, value)
    right = jax.make_jaxpr(lambda x, y: jnp.where(y > 0, x, -x))(value, value)
    assert compare_algebraic_jaxprs(left, right).status == Status.INEQUIVALENT


def test_condition_broadcast_cannot_swap_equal_length_axes():
    """两个等长轴广播时不能把谓词轴与值轴对调，即使形状都是 (2,)。"""
    value = jax.ShapeDtypeStruct((2,), jnp.float32)
    left = jax.make_jaxpr(lambda x: jnp.abs(jnp.broadcast_to(x, (2, 2))))(value)
    right = jax.make_jaxpr(lambda x: jnp.where(x[:, None] > 0, x[None, :], -x[None, :]))(value)
    assert compare_algebraic_jaxprs(left, right).status == Status.INEQUIVALENT


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_strict_positive_selection_does_not_hide_zero_derivative_difference(mode):
    """``where(x>0,...)`` 与 ``abs`` 在前向值上相同，但零点导数不同。"""
    result = compare_functions(jnp.abs, VARIANTS[0], jnp.zeros(3), modes=("forward", mode))
    assert result.modes[0].status == Status.EQUIVALENT
    assert result.modes[1].status == Status.INEQUIVALENT
    assert result.status == Status.INEQUIVALENT
    # 浮点抽样仅作为反例证据，等价结论仍由抽象图证明决定。
    _, absolute = jax.jvp(jnp.abs, (jnp.float32(0),), (jnp.float32(1),))
    _, rewritten = jax.jvp(VARIANTS[0], (jnp.float32(0),), (jnp.float32(1),))
    assert absolute == 1 and rewritten == -1


def test_non_strict_positive_selection_preserves_jax_first_derivatives():
    """``where(x>=0,...)`` 在 JAX 的 abs 导数约定下与 abs 的一阶导数一致。"""
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
    """+0 与 -0 在实数解释下相等，但执行结果不同，必须发出显式浮点警告。"""
    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, VARIANTS[0], ()))
    assert result.status == Status.EQUIVALENT, result
    assert any("signed zero" in warning for warning in result.warnings)
    # abs(+0)=+0，而 where 写法在 x>0 为假时返回 x，即 -0，符号位不同。
    assert not np.signbit(np.asarray(jnp.abs(jnp.float32(0))))
    assert np.signbit(np.asarray(VARIANTS[0](jnp.float32(0))))


@pytest.mark.parametrize("metadata", ["donation", "inline", "compiler_options"])
def test_select_wrapper_keeps_non_default_call_metadata(metadata):
    """jit 包装的非默认调用元数据参与语义，不能在代数路径上被忽略。"""
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
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        compare_algebraic_jaxprs(left, right)


def test_wrapper_name_does_not_authorize_other_instructions():
    """同名 wrapper 内部多做一步 +1，不能因名字相同就放行。"""

    @jax.jit
    def _where(predicate, negative, positive):
        return lax.select(predicate, positive, negative) + 1

    result = compare_algebraic_jaxprs(*_graphs(jnp.abs, lambda x: _where(x > 0, -x, x)))
    assert result.status == Status.INEQUIVALENT


def test_non_real_input_does_not_acquire_the_real_piecewise_rule():
    """整数输入没有实数除零/符号语义，必须拒绝且不记录 select_absolute 规则。"""
    left, right = _graphs(jnp.abs, VARIANTS[0], dtype=jnp.int32)
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)) as error:
        compare_algebraic_jaxprs(left, right)
    result = error.value
    assert "select_absolute" not in result.diagnostics["rules"]
