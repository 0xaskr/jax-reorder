"""四层复杂 cond 的完整路径回归；预期由手写候选的计算关系决定。

覆盖：
- 正例与标签：tests/nested_cond_four_levels 的场景在全部 profile 与变体下重放 80 个
  样本，比较标签与手写候选一致，不生成候选。
- 实际改写：每层分支数应为 2^level，且规范化结构中第 1 至 4 层都确实发生了换序
  （防止改写被追踪消去后只剩下自比较）。
- 反例：固定非对称输入下遍历全部 16 条路径；wrong_deep_binding 只影响 f3=True 的 8 条，
  其余路径逐位相等，受影响路径出现可复现的数值差异。

前置条件：CPU 平台；样本与预期由固定场景及手写候选提供，测试阶段不生成候选。
四层与 16 条路径不覆盖更深或更多分支的 cond。
"""

import itertools

import jax
import numpy as np
import pytest

from tests import nested_cond_four_levels as case


@pytest.mark.parametrize("profile", case.business.PROFILES)
@pytest.mark.parametrize("variant", case.SCENARIO.variants, ids=lambda v: v.fn.__name__)
def test_four_level_cond_comparison_and_execution(profile, variant):
    result = case.business.verify_variant(
        case.SCENARIO, variant, profile, case.make_samples(profile)
    )
    assert result["actual"] == variant.expected.value
    assert result["samples"] == 80


def test_actual_nesting_and_reordering_at_every_level():
    args = case.make_samples("float32", trials=1)[0]
    graphs = [jax.make_jaxpr(fn)(*args) for fn in (case.original, case.reordered)]
    left, right = [case.structure(graph) for graph in graphs]
    assert left[:2] == right[:2] == (15, 4)
    for level in range(1, 5):
        assert len(left[2][level]) == len(right[2][level]) == 2**level
        assert left[2][level] != right[2][level], f"第 {level} 层没有实际换序"


def test_all_sixteen_paths_execute_and_deep_false_branch_stays_equal():
    # 固定非对称输入，第四层的错误绑定只影响 f3=True 的八条路径。
    samples = case.make_samples("float32", trials=1)[16:32]
    good = jax.jit(case.original)
    wrong = jax.jit(case.wrong_deep_binding)
    tags = []
    affected = 0
    for args, flags in zip(samples, itertools.product((False, True), repeat=4), strict=True):
        original, candidate = good(*args), wrong(*args)
        tag = sum(int(flag) * weight for flag, weight in zip(flags, (8, 4, 2, 1), strict=True))
        assert int(original[2]) == int(candidate[2]) == tag
        tags.append(tag)
        if flags[3]:
            assert not np.allclose(original[0], candidate[0], rtol=2e-5, atol=2e-5)
            affected += 1
        else:
            for a, b in zip(original, candidate, strict=True):
                np.testing.assert_array_equal(a, b)
    assert tags == list(range(16)) and affected == 8
