"""反馈边、xs/ys、元数据和对外输出一起核对，保留数值反例。

覆盖：
- 正例：scan/while 的 carry、xs/ys、反馈槽位与对外输出同步置换（reordered 与
  reordered_again 两种写法）在 0/1/3 步、float32/bfloat16 下等价；原图不被修改、方向对称、
  规范化幂等且执行结果一致。
- 反例：错误的反馈或结果绑定必须有数值见证；VJP 中因槽位顺序导致的 `add_any` 累加顺序
  变化不能被当成端口重命名（结构模式判 different）。
- 附加：reordered scan 在 jit 内与 reverse 模式下仍保持。
- 前置条件：CPU 平台；固定源码来自 tests/feedback_bindings.py，不涉及多设备。
- 有限步骤与槽位组合不证明任意反馈拓扑。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core

from tests import feedback_bindings as cases
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
    result = compare_functions(original, candidate, *args, equivalence="structural")
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


@pytest.mark.parametrize(
    "equivalence,expected", [("structural", Status.DIFFERENT), ("auto", Status.EQUIVALENT)]
)
def test_changed_vjp_accumulation_order_is_not_treated_as_a_port_rename(equivalence, expected):
    result = compare_functions(
        cases.scan_original,
        cases.scan_reordered,
        *arguments("scan", 3, jnp.float32),
        modes=("forward", "jvp", "vjp"),
        transforms=("jit",),
        check_cpu=True,
        equivalence=equivalence,
    )
    # 两份前向图相同，但 JAX 转置按返回槽位累加余切向量：
    # 左侧 ((carry + total_ct) + difference_ct)，右侧先累加 difference_ct。
    # 结构模式保留 add_any 连接；auto 需要独立的代数证据和浮点警告。
    assert [mode.status for mode in result.modes] == [
        Status.EQUIVALENT,
        Status.EQUIVALENT,
        expected,
    ]
    assert result.status == expected
    assert result.modes[2].comparison.status == Status.DIFFERENT
    if equivalence == "auto":
        assert result.modes[2].equivalence_kind == "algebraic"
        assert result.modes[2].algebra.backend_result and result.modes[2].algebra.warnings
