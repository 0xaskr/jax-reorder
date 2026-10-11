"""反馈边、xs/ys、元数据和对外输出一起核对，保留数值反例。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core

from examples import feedback_bindings as cases
from jax_reorder import Status, canonicalize, compare_jaxprs
from jax_reorder.functions import compare_functions


def arguments(kind, steps, dtype):
    x, y = jnp.array([0.7, 1.2], dtype), jnp.array([-0.3, 0.4], dtype)
    if kind == "while":
        return x, y, jnp.int32(steps)
    u = jnp.arange(steps * 2, dtype=dtype).reshape(steps, 2) * 0.1
    v = jnp.arange(steps * 2, dtype=dtype).reshape(steps, 2) * -0.2
    return x, y, u, v


@pytest.mark.parametrize("kind", ["scan", "while"])
@pytest.mark.parametrize("suffix", ["reordered", "reordered_again"])
@pytest.mark.parametrize("steps", [0, 1, 3])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_joint_feedback_permutation(kind, suffix, steps, dtype):
    original, candidate = getattr(cases, f"{kind}_original"), getattr(cases, f"{kind}_{suffix}")
    args = arguments(kind, steps, dtype)
    a, b = (jax.make_jaxpr(fn)(*args) for fn in (original, candidate))
    before = str(a), str(b)
    result = compare_jaxprs(a, b)
    assert result.status == Status.EQUIVALENT, result
    assert before == (str(a), str(b))
    assert compare_jaxprs(b, a).status == Status.EQUIVALENT
    for raw, normalized in ((a, result.left), (b, result.right)):
        core.check_jaxpr(normalized.jaxpr)
        assert canonicalize(normalized.jaxpr).graph == normalized.graph
        for expected, actual in zip(
            core.eval_jaxpr(raw, raw.consts, *args),
            core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, *args),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
    for expected, actual in zip(
        jax.tree.leaves(original(*args)), jax.tree.leaves(candidate(*args)), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kind,suffix",
    [
        ("scan", "wrong_initial"),
        ("scan", "wrong_output"),
        ("while", "wrong_initial"),
        ("while", "wrong_feedback"),
    ],
)
def test_incorrect_feedback_or_result_binding_has_a_witness(kind, suffix):
    original, candidate = getattr(cases, f"{kind}_original"), getattr(cases, f"{kind}_{suffix}")
    args = arguments(kind, 3, jnp.float32)
    result = compare_functions(original, candidate, *args)
    assert result.status == Status.DIFFERENT, result
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(
            jax.tree.leaves(original(*args)), jax.tree.leaves(candidate(*args)), strict=True
        )
    )


def test_reordered_scan_under_jit_and_reverse_mode():
    result = compare_functions(
        cases.scan_original,
        cases.scan_reordered_again,
        *arguments("scan", 3, jnp.float32),
        modes=("forward", "jvp", "vjp"),
        transforms=("jit",),
        check_cpu=True,
    )
    assert result.status == Status.EQUIVALENT, result


def test_changed_vjp_accumulation_order_is_not_treated_as_a_port_rename():
    result = compare_functions(
        cases.scan_original,
        cases.scan_reordered,
        *arguments("scan", 3, jnp.float32),
        modes=("forward", "jvp", "vjp"),
        transforms=("jit",),
        check_cpu=True,
    )
    # 两份前向图相同，但 JAX 转置按返回槽位累加余切向量：
    # 左侧 ((carry + total_ct) + difference_ct)，右侧先累加 difference_ct。
    # 这会改变普通 add_any 连接，不能为使标签通过而套用结合律。
    assert [mode.status for mode in result.modes] == [
        Status.EQUIVALENT,
        Status.EQUIVALENT,
        Status.DIFFERENT,
    ]
    assert result.status == Status.DIFFERENT
