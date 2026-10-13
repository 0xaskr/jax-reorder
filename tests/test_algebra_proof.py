"""代数证明的独立复核、篡改拒绝和资源停止路径。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest

from jax_reorder import (
    CorrespondenceError,
    Status,
    canonicalize,
    compare_algebraic_jaxprs,
    verify_algebraic_proof,
)
from jax_reorder.algebra_ir import Budget, Expr, Factor, Index, Term
from jax_reorder.algebra_normalize import PolynomialNormalizer


@pytest.fixture
def contraction():
    """不同原语构成同一收缩，证明包含乘法、求和和索引见证。"""
    args = (jax.ShapeDtypeStruct((2, 3), jnp.float32), jax.ShapeDtypeStruct((3, 4), jnp.float32))
    left = jax.make_jaxpr(lambda a, b: a @ b)(*args)
    right = jax.make_jaxpr(lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1))(*args)
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.EQUIVALENT, result
    return left, right, result.proof


def test_verifier_does_not_call_normalization_or_index_search(contraction, monkeypatch):
    left, right, proof = contraction

    def forbidden(*args, **kwargs):
        raise AssertionError("normalizer must not be called by verifier")

    monkeypatch.setattr(PolynomialNormalizer, "normalize", forbidden)
    monkeypatch.setattr(PolynomialNormalizer, "_term", forbidden)
    import jax_reorder.canonical as module

    monkeypatch.setattr(module, "canonicalize", forbidden)
    assert verify_algebraic_proof(left, right, proof) is None


@pytest.mark.parametrize(
    "change",
    [
        "coefficient",
        "factor",
        "scope",
        "order",
        "missing_term",
        "premise",
        "source",
        "root",
        "version",
    ],
)
def test_tampered_proofs_are_rejected(contraction, change):
    left, right, proof = contraction
    steps = list(proof.steps)
    index = next(i for i, s in enumerate(steps) if s.expression.op == "sum")
    step = steps[index]
    if change == "coefficient":
        steps[index] = replace(step, result=(replace(step.result[0], coefficient=(2, 1)),))
    elif change == "factor":
        term = step.result[0]
        factor = replace(
            term.factors[0], indices=(Index("free", 8),) * len(term.factors[0].indices)
        )
        steps[index] = replace(step, result=(replace(term, factors=(factor, *term.factors[1:])),))
    elif change == "scope":
        uid, extent = step.expression.data
        steps[index] = replace(step, expression=replace(step.expression, data=(uid, extent + 1)))
    elif change == "order":
        steps[index] = replace(step, orders=((100,),))
    elif change == "missing_term":
        steps[index] = replace(step, orders=())
    elif change == "premise":
        steps[index] = replace(step, premises=(index,))
    elif change == "source":
        different = jax.make_jaxpr(lambda a, b: a @ b + 1)(
            jax.ShapeDtypeStruct((2, 3), jnp.float32), jax.ShapeDtypeStruct((3, 4), jnp.float32)
        )
        proof = replace(proof, left=replace(proof.left, graph=different))
    elif change == "root":
        proof = replace(proof, left=replace(proof.left, roots=()))
    elif change == "version":
        proof = replace(proof, version=0)
    proof = replace(proof, steps=tuple(steps))
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(left, right, proof)


def test_constants_and_primitive_parameters_bind_proof_to_original_graph():
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    left = jax.make_jaxpr(lambda x: x + 2)(x)
    right = jax.make_jaxpr(lambda x: 2 + x)(x)
    result = compare_algebraic_jaxprs(left, right)
    changed = jax.make_jaxpr(lambda x: x + 3)(x)
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(changed, right, result.proof)


def test_all_outputs_and_original_graph_are_preserved(contraction):
    left, right, _ = contraction
    before = canonicalize(left).graph, canonicalize(right).graph
    first = compare_algebraic_jaxprs(left, right)
    second = compare_algebraic_jaxprs(right, left)
    assert first.status == second.status == Status.EQUIVALENT
    assert before == (canonicalize(left).graph, canonicalize(right).graph)
    assert first.proof.steps == compare_algebraic_jaxprs(left, right).proof.steps


def test_symmetric_index_search_is_bounded_before_permutations():
    # 同一向量的多个独立索引，颜色完全相同，必须枚举或安全拒绝。
    factors = tuple(Factor(("x",), (Index("bound", i),)) for i in range(9))
    engine = PolynomialNormalizer(Budget(20))
    from jax_reorder import UnsupportedJaxprError

    with pytest.raises(UnsupportedJaxprError, match="budget"):
        engine._term(Term((1, 1), (3,) * 9, factors))


def test_expansion_budget_precedes_large_intermediate_allocation():
    atoms = tuple(Expr("atom", (("input", i), ())) for i in range(40))
    expression = Expr(
        "mul", args=tuple(Expr("add", args=(atoms[i], atoms[i + 1])) for i in range(0, 40, 2))
    )
    from jax_reorder import UnsupportedJaxprError

    engine = PolynomialNormalizer(Budget(1000))
    with pytest.raises(UnsupportedJaxprError, match="budget"):
        engine.normalize(expression)
    assert engine.budget.used <= 1000


def test_repeated_squaring_counts_factor_storage_before_allocation():
    def huge(x):
        for _ in range(30):
            x = x * x
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(huge)(x)
    other = jax.make_jaxpr(lambda x: huge(x) * 1)(x)
    result = compare_algebraic_jaxprs(graph, other, algebra_budget=1000)
    assert result.status == Status.UNKNOWN
    assert "budget" in result.issues[0].reason


def test_deep_chain_ends_with_a_located_budget_failure():
    def chain(x):
        for _ in range(300):
            x = x + 1
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(chain)(x)
    other = jax.make_jaxpr(lambda x: chain(x) + 0)(x)
    result = compare_algebraic_jaxprs(graph, other, algebra_budget=100)
    assert result.status == Status.UNKNOWN
    assert result.issues[0].path.startswith("algebra")
    assert "budget" in result.issues[0].reason
