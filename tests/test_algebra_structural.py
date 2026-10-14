"""把旧用例已证明结构等价的 IR 成对送入代数层，检验蕴含关系。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from lax_cases import cases, trace_case

from jax_reorder import (
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_jaxprs,
    verify_algebraic_proof,
)
from jax_reorder.algebra import _prepare
from jax_reorder.algebra_calls import CommonCalls
from jax_reorder.algebra_ir import Budget
from jax_reorder.algebra_lower import Lowerer
from jax_reorder.algebra_normalize import PolynomialNormalizer
from jax_reorder.algebra_proof import AlgebraProof, AlgebraSource

# 这些旧用例已有结构证明，但含代数展开器尚未适配的区域、token 或有序操作。
# 明列拒绝范围；不能把任意转换失败当作预期跳过。
EXPANSION_REJECTIONS = {
    **dict.fromkeys(("rng_uniform", "dce_sink", "dce_sink_mlir"), "algebra.effects"),
    **dict.fromkeys(("create_token", "after_all"), "algebra.shape"),
    **dict.fromkeys(
        (
            "custom_linear_solve",
            "scatter-add",
            "scatter-sub",
            "scatter-mul",
            "scatter-min",
            "scatter-max",
            "reduce_window",
            "reduce",
            "select_and_scatter",
            "psum",
            "pmin",
            "pmax",
            "axis_index",
            "all_gather",
            "all_to_all",
            "ppermute",
            "ragged_all_to_all",
            "pbroadcast",
            "reduce_scatter",
            "composite",
            "composite_attributes",
            "psum_invariant",
        ),
        "algebra.metadata",
    ),
}


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_old_primitive_structural_pair_is_also_algebraically_equivalent(case, monkeypatch):
    """涵盖旧原语清单的全部数值域及区域，代数预算不能否定结构证据。"""
    original = trace_case(case)
    normalized = canonicalize(original)
    structural = compare_jaxprs(original, normalized.jaxpr)
    assert structural.status == Status.EQUIVALENT, structural

    def forbidden(*args, **kwargs):
        raise AssertionError("structural proof must not depend on polynomial expansion")

    monkeypatch.setattr(PolynomialNormalizer, "normalize", forbidden)
    result = compare_algebraic_jaxprs(original, normalized.jaxpr, algebra_budget=1)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "structural"
    assert result.work == 0 and not result.warnings
    assert verify_algebraic_proof(original, normalized.jaxpr, result.proof) is None


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_old_structural_pairs_also_agree_in_the_expansion_engine(case):
    """单独走转换、展开和证明复核，避免结构直通掩盖代数引擎的不一致。"""
    original = trace_case(case)
    normalized = canonicalize(original).jaxpr
    assert compare_jaxprs(original, normalized).status == Status.EQUIVALENT
    budget = Budget(10000)
    engine = PolynomialNormalizer(budget)
    sources, signatures, forms = [], [], []
    prepared_pair = tuple(_prepare(graph) for graph in (original, normalized))
    common_calls = (
        None
        if case.name in EXPANSION_REJECTIONS
        else CommonCalls(
            tuple(
                (graph, *prepared)
                for graph, prepared in zip((original, normalized), prepared_pair, strict=True)
            ),
            budget,
        )
    )
    for prepared, correspondence in prepared_pair:
        if case.name in EXPANSION_REJECTIONS:
            with pytest.raises(UnsupportedJaxprError) as caught:
                Lowerer(budget).graph(prepared)
            assert caught.value.path.startswith(EXPANSION_REJECTIONS[case.name])
            continue
        lowered = Lowerer(budget, common_calls).graph(prepared)
        roots = tuple(engine.normalize(expr) for expr in lowered.roots)
        sources.append(AlgebraSource(prepared, correspondence, roots))
        signatures.append(lowered.signature)
        forms.append(tuple(engine.steps[i].result for i in roots))
    if sources:
        assert signatures[0] == signatures[1]
        assert forms[0] == forms[1]
        proof = AlgebraProof(*sources, tuple(engine.steps), budget.limit)
        assert verify_algebraic_proof(original, normalized, proof) is None


def test_structural_pair_with_exponential_expansion_uses_existing_evidence():
    """相同巨大多项式已由图证据确认，不应再受代数展开的资源限制。"""

    def huge(x):
        for _ in range(30):
            x = x * x
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(huge)(x)
    result = compare_algebraic_jaxprs(graph, graph, algebra_budget=1)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "structural" and result.work == 0
    assert verify_algebraic_proof(graph, graph, result.proof) is None


def test_function_structural_pair_passes_all_requested_modes_with_tiny_algebra_budget():
    """函数契约核对后，前向和显式求导模式各自保留结构等价依据。"""

    def left(a, b):
        x = jnp.sin(a)
        y = jnp.cos(b)
        return x * y

    def right(a, b):
        y = jnp.cos(b)
        x = jnp.sin(a)
        return x * y

    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    result = compare_functions(left, right, x, x, modes=("forward", "jvp", "vjp"), algebra_budget=1)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "structural"
    assert all(mode.equivalence_kind == "structural" for mode in result.modes)


@pytest.mark.parametrize("change", ["inputs", "outputs", "source", "mixed", "missing"])
def test_structural_algebra_certificate_is_independently_checked(change):
    """直通证据也必须绑定原图，错误端口和混合证据均不能被放行。"""
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    graph = jax.make_jaxpr(lambda a, b: (a - b, a * b))(x, x)
    result = compare_algebraic_jaxprs(graph, graph)
    assert result.status == Status.EQUIVALENT, result
    proof = result.proof
    other = graph
    if change == "inputs":
        proof = replace(proof, structural=replace(proof.structural, inputs=(1, 0)))
    elif change == "outputs":
        proof = replace(proof, structural=replace(proof.structural, outputs=(1, 0)))
    elif change == "source":
        other = jax.make_jaxpr(lambda a, b: (a + b, a * b))(x, x)
    elif change == "mixed":
        proof = replace(proof, steps=(object(),))
    else:
        proof = replace(proof, structural=None)
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(graph, other, proof)
