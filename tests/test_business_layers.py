"""重放 AI hardcode 的真实业务改写，不在测试阶段生成候选或推导预期标签。

覆盖：
- 正例与标签核对：examples/business_layer_cases 中全部场景 × 变体 × profile；基准先与
  独立 NumPy 参照核对，再验证图预判、候选数值、两侧重排执行与规范化幂等。
- 反例：报告数值见证时必须给出足够重放的输入与两侧输出，不能只记录一个布尔值；把负例
  故意预判成 equivalent 时运行器必须报错。
- 附加校验：正例候选确实改变真实方程顺序（防止换序被追踪消去后只剩自比较）；cond 场景的
  错误内部绑定同样被检出，且复现器只改内部绑定。
- 前置条件：CPU 平台；固定种子 20261011/42 只影响数值样本，不决定图标签。
- 这是固定业务用例的重放，不代表所有业务改写；标签仍是待验证预期。
"""

import json

import jax
import pytest

from examples import business_layer_cases as business
from examples import cond_bindings


@pytest.mark.parametrize(
    "scenario,variant,profile",
    [
        pytest.param(scenario, variant, profile, id=f"{variant.fn.__name__}/{profile}")
        for scenario in business.SCENARIOS
        for variant in scenario.variants
        for profile in business.PROFILES
    ],
)
def test_saved_business_variant(scenario, variant, profile):
    # 基准先与独立 NumPy 参照核对；另验证图预判、候选数值、两边重排执行与幂等性。
    samples = business.make_samples(scenario, profile, trials=3, seed=20261011)
    result = business.verify_variant(scenario, variant, profile, samples)
    assert result["actual"] == variant.expected.value
    if variant.numerical == "witness":
        assert result["witness"] is not None
        # 报告必须可序列化，并包含足够重放的输入与两侧输出，不能只记录一个布尔值。
        restored = json.loads(json.dumps(result))
        assert restored["witness"]["inputs"]
        assert restored["witness"]["base_outputs"]
        assert restored["witness"]["candidate_outputs"]


@pytest.mark.parametrize("scenario", business.SCENARIOS, ids=lambda item: item.name)
def test_positive_candidates_change_real_equation_order(scenario):
    # 防止所谓换序被追踪完全消去，以至于只验证同一份原图的自比较。
    args = business.make_samples(scenario, "bfloat16", trials=1, seed=42)[0]
    original = jax.make_jaxpr(scenario.base)(*args)
    for variant in scenario.variants:
        if variant.expected == business.Status.EQUIVALENT:
            candidate = jax.make_jaxpr(variant.fn)(*args)
            # 这里只检查原图确实有差别；等价结论仍由上一组完整结构测试负责。
            assert str(original) != str(candidate), variant.fn.__name__


def test_runner_rejects_incorrect_ai_prediction():
    # 预判不是无条件可信的真值：故意把业务负例写成 equivalent，必须报错。
    scenario = business.SCENARIOS[0]
    wrong_prediction = business.Variant(
        business.norm_wrong_gate, business.Status.EQUIVALENT, "same", "故意注入错误标签"
    )
    samples = business.make_samples(scenario, "float32", trials=1, seed=0)
    with pytest.raises(AssertionError):
        business.verify_variant(scenario, wrong_prediction, "float32", samples)


def test_numeric_oracle_catches_wrong_base():
    # 即使 base 与候选同时算错，图比较仍可能通过；NumPy 参照必须阻止这种假通过。
    original = business.SCENARIOS[0]
    both_wrong = business.Scenario(
        original.name, business.norm_wrong_gate, original.oracle, original.variants
    )
    candidate = business.Variant(
        business.norm_wrong_gate, business.Status.EQUIVALENT, "same", "两边同时错误"
    )
    samples = business.make_samples(both_wrong, "float32", trials=1, seed=0)
    with pytest.raises(AssertionError):
        business.verify_variant(both_wrong, candidate, "float32", samples)


@pytest.mark.parametrize("profile", business.PROFILES)
@pytest.mark.parametrize("variant", cond_bindings.SCENARIO.variants, ids=lambda v: v.fn.__name__)
def test_cond_business_binding(variant, profile):
    # 候选是固定源码；真假分支、NumPy 参照、错误绑定反例和重排执行均需验证。
    business.verify_variant(
        cond_bindings.SCENARIO, variant, profile, cond_bindings.make_samples(profile)
    )


def test_cond_reproducer_changes_internal_bindings_only():
    args = cond_bindings.make_samples("float32", trials=1)[0]
    left, right = [
        jax.make_jaxpr(fn)(*args) for fn in (cond_bindings.original, cond_bindings.gate_first)
    ]
    positions = []
    for graph in (left, right):
        eq = next(eq for eq in graph.eqns if eq.primitive.name == "cond")
        positions.append(tuple(graph.invars.index(v) for v in eq.invars[1:]))
    assert positions == [(0, 2, 1), (1, 0, 2)]
    assert [v.aval for v in left.invars] == [v.aval for v in right.invars]
