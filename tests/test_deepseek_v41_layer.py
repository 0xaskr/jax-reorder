"""固定上游 DeepSeek-V4.1-Flash mHC 层、AI 改写和完整来源证据。

覆盖：
- 来源与尺寸：fixture 快照经 hash 校验，源码断言、manifest revision 和配置维度来自真实
  上游文件，而不是手写常量。
- 正例：原层与 post_matmul/post_blocked 两种代数改写候选在 float32/bfloat16、多个 profile
  下 compare_functions 判 equivalent，并核对数值与 sinkhorn 上下文。
- 反例：形状相同但转置错误、缩放错误、舍入方式错误的候选必须被拒绝并给出数值见证。
- 资源：完整层代数证明受预算限制，超限抛 ComparisonBudgetError；符号结论与数值核对分开。
- 前置条件：CPU 平台、业务标记（pytest -m business）与 tests/fixtures/deepseek_v41 快照；
  该快照是上游文件的固定副本，不是运行中依赖。
- 单层验收不能外推到整个模型或权重梯度。
"""

import ast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.deepseek_v41 import candidates
from examples.deepseek_v41.cases import (
    PROFILES,
    cancellation_inputs,
    check_close,
    check_numerics,
    context_inputs,
    in_context,
    inputs,
    signatures,
)
from examples.deepseek_v41.upstream import MANIFEST, ROOT, original, verify_snapshot
from jax_reorder import ComparisonBudgetError, MathematicalUndecidedError, Status, compare_functions

pytestmark = pytest.mark.business


def test_fixed_upstream_source_definitions_and_actual_model_dimensions():
    verify_snapshot()
    assert MANIFEST["revision"] == "d11defb7e5e2fdd8d355b2f6225c74d2a15fa1e1"
    assert MANIFEST["model"] == "DeepSeek-V4.1-Flash"
    source = (ROOT / "deepseek_v41/reference.py.txt").read_text()
    assert "x = hc_post(ffn_out, residual, ffn_post, ffn_comb)" in source
    config = ast.parse((ROOT / "deepseek_v41/config.py.txt").read_text())
    definition = next(
        node
        for node in config.body
        if isinstance(node, ast.ClassDef) and node.name == "DSV41Config"
    )
    values = {
        node.target.id: ast.literal_eval(node.value)
        for node in definition.body
        if isinstance(node, ast.AnnAssign)
    }
    assert (values["hc_mult"], values["dim"], values["hc_sinkhorn_iters"]) == (4, 5120, 20)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("name", ["post_matmul", "post_blocked"])
def test_original_layer_and_algebraic_candidates(profile, dtype, name):
    function = getattr(candidates, name)
    report = compare_functions(
        original().hc_post,
        function,
        *signatures(profile, dtype),
        modes=("forward", "jvp", "vjp"),
        check_cpu=True,
    )
    assert report.status == Status.EQUIVALENT, [(m.mode, m.issues) for m in report.modes]
    assert report.cpu_compiled
    for mode in report.modes:
        assert mode.equivalence_kind == "algebraic"
        assert mode.comparison.status == Status.DIFFERENT
        assert mode.algebra.warnings
        assert mode.algebra.backend_result.backend == "sympy"
    for seed in (17, 53):
        args = inputs(profile, dtype, seed)
        check_numerics(original().hc_post, args, seed)
        check_numerics(function, args, seed)
    check_numerics(function, cancellation_inputs(profile, dtype))


@pytest.mark.parametrize("name", ["wrong_transpose", "wrong_scale", "wrong_rounding"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_shape_preserving_wrong_candidates_are_rejected_with_witness(name, dtype):
    args = cancellation_inputs((1, 1, 4, 7), dtype)
    function = getattr(candidates, name)
    if dtype == jnp.bfloat16 or name == "wrong_rounding":
        with pytest.raises(MathematicalUndecidedError, match="condition_undecided") as error:
            compare_functions(original().hc_post, function, *args, modes=("forward", "jvp", "vjp"))
        assert error.value.path.startswith("forward.algebra")
    else:
        report = compare_functions(
            original().hc_post, function, *args, modes=("forward", "jvp", "vjp")
        )
        assert all(mode.status == Status.INEQUIVALENT for mode in report.modes)
    assert not np.array_equal(np.asarray(original().hc_post(*args)), np.asarray(function(*args)))


@pytest.mark.parametrize("name", ["post_matmul", "post_blocked"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_full_original_sinkhorn_context(name, dtype):
    args = context_inputs((1, 1, 4, 8), dtype, 17)
    left, right = in_context(original().hc_post), in_context(getattr(candidates, name))
    report = compare_functions(left, right, *args, modes=("forward", "jvp", "vjp"), check_cpu=True)
    assert report.status == Status.EQUIVALENT, [(m.mode, m.issues) for m in report.modes]
    assert all("common_jit" in mode.algebra.rules for mode in report.modes)
    for a, b in zip(jax.tree.leaves(left(*args)), jax.tree.leaves(right(*args)), strict=True):
        check_close(a, b)


def test_real_dimensions_in_full_original_context():
    args = context_inputs((1, 1, 4, 5120), jnp.bfloat16, 53)
    left, right = in_context(original().hc_post), in_context(candidates.post_matmul)
    report = compare_functions(left, right, *args, modes=("forward",), check_cpu=True)
    assert report.status == Status.EQUIVALENT, report.modes[0].issues
    for a, b in zip(jax.tree.leaves(left(*args)), jax.tree.leaves(right(*args)), strict=True):
        check_close(a, b)


def test_layer_sympy_result_and_budget_rejection():
    args = signatures((1, 1, 4, 8), jnp.float32)
    report = compare_functions(original().hc_post, candidates.post_matmul, *args)
    mode = report.modes[0]
    assert mode.status == Status.EQUIVALENT
    assert mode.algebra.backend_result.backend == "sympy"
    with pytest.raises(ComparisonBudgetError) as error:
        compare_functions(original().hc_post, candidates.post_matmul, *args, algebra_budget=10)
    limited = error.value
    assert "budget" in limited.reason
    assert limited.path.startswith("forward.algebra")
