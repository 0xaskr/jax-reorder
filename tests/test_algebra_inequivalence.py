"""完整输出多项式不恒等：精确系数证据、定义域和独立复核。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    verify_inequivalence_proof,
)
from jax_reorder.algebra import _prepare
from jax_reorder.algebra_calls import CommonCalls
from jax_reorder.algebra_inequivalence import InputElement, try_prove_inequivalent
from jax_reorder.algebra_ir import Budget
from jax_reorder.algebra_lower import Lowerer
from jax_reorder.algebra_proof import AlgebraSource


def graphs(left, right, *shapes, dtype=jnp.float32):
    """只构造抽象 jaxpr，不编译、不运行函数或抽样张量。"""
    args = tuple(jax.ShapeDtypeStruct(shape, dtype) for shape in shapes)
    return jax.make_jaxpr(left)(*args), jax.make_jaxpr(right)(*args)


def candidate(left, right, *, limit=10000):
    """绕过正向规范化，检查负证据生成器自身的接受边界。"""
    budget = Budget(limit)
    prepared = tuple(_prepare(graph) for graph in (left, right))
    calls = CommonCalls(
        tuple(
            (original, *adapted) for original, adapted in zip((left, right), prepared, strict=True)
        ),
        budget,
    )
    lowered = tuple(Lowerer(budget, calls).graph(graph) for graph, _ in prepared)
    sources = tuple(AlgebraSource(graph, correspondence, ()) for graph, correspondence in prepared)
    return try_prove_inequivalent(left, right, sources=sources, lowered=lowered, budget=budget)


@pytest.mark.parametrize(
    "left,right,shapes",
    [
        (lambda x: x + 1, lambda x: x + 2, ((),)),
        (lambda x: 2 * x, lambda x: x, ((3,),)),
        (lambda x, y: x - y, lambda x, y: y - x, ((2,), (2,))),
        (lambda x, y: (x, y), lambda x, y: (y, x), ((2,), (2,))),
        (lambda x: x * x, lambda x: x, ((),)),
        (lambda x: jnp.sum(x, axis=0), lambda x: jnp.sum(x, axis=1), ((2, 2),)),
        (lambda x: x[:2], lambda x: x[1:], ((3,),)),
        (lambda x: x, lambda x: x.T, ((2, 2),)),
        (lambda x, y: x @ y, lambda x, y: y @ x, ((2, 2), (2, 2))),
        (
            lambda x, y: jnp.sum(x) * jnp.sum(y),
            lambda x, y: jnp.sum(x * y),
            ((2,), (2,)),
        ),
        (lambda x: x.reshape((2, 3)), lambda x: x.reshape((3, 2)).T, ((6,),)),
        (lambda x: (x, x + 1), lambda x: (x, x + 2), ((),)),
    ],
    ids=[
        "offset",
        "gain",
        "sign",
        "output_order",
        "degree",
        "reduce_axis",
        "slice",
        "transpose",
        "matmul",
        "cross_terms",
        "reshape",
        "second_output",
    ],
)
def test_real_output_polynomial_difference_has_replayable_evidence(left, right, shapes):
    first, second = graphs(left, right, *shapes)
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == Status.INEQUIVALENT, result
    assert result.proof is None
    assert result.inequivalence_proof.coefficient[0] != 0
    assert result.warnings and result.assumptions
    assert verify_inequivalence_proof(first, second, result.inequivalence_proof) is None
    reverse = compare_algebraic_jaxprs(second, first)
    assert reverse.status == Status.INEQUIVALENT
    assert reverse.inequivalence_proof.monomial == result.inequivalence_proof.monomial
    numerator, denominator = result.inequivalence_proof.coefficient
    assert reverse.inequivalence_proof.coefficient == (-numerator, denominator)


def test_output_slot_and_later_coordinate_are_observable_not_local():
    first, second = graphs(lambda x: (x, x), lambda x: (x, x.T), (2, 2))
    proof = candidate(first, second)
    assert proof.output == 1
    assert proof.coordinate == (0, 1)
    assert proof.monomial == (InputElement(0, (0, 1)),)


@pytest.mark.parametrize(
    "left,right,shapes",
    [
        (lambda x, y: x + y, lambda x, y: y + x, ((2,), (2,))),
        (lambda x: (x + 1) * 0 + x, lambda x: (x + 2) * 0 + x, ((2,),)),
        (lambda x: jnp.sum(x - x), lambda x: jnp.sum(x) - jnp.sum(x), ((3,),)),
        (lambda x: x.T, lambda x: x, ((1, 1),)),
        (lambda x: x + 1, lambda x: x + 2, ((0,),)),
        (lambda x: jnp.sum(x) + 0, lambda x: jnp.sum(x) * 2, ((0,),)),
        (
            lambda x: jnp.sum(jnp.broadcast_to(x, (4,))) ** 2,
            lambda x: 16 * x * x,
            ((),),
        ),
    ],
    ids=[
        "commute",
        "erased_local_difference",
        "cancellation",
        "unit_transpose",
        "empty_output",
        "empty_sum",
        "reused_input_element",
    ],
)
def test_no_false_inequivalence_for_equal_outputs(left, right, shapes):
    assert candidate(*graphs(left, right, *shapes)) is None


@pytest.mark.parametrize(
    "left,right,dtype",
    [
        (lambda x: jnp.sin(x), lambda x: x + 1, jnp.float32),
        (lambda x: jnp.exp(x), lambda x: x, jnp.float32),
        (lambda x: jnp.log(x), lambda x: x, jnp.float32),
        (lambda x: x / x, lambda x: x + 1, jnp.float32),
        (lambda x: x + (-jnp.inf), lambda x: x + 1, jnp.float32),
        (lambda x: (x + 1, jnp.sin(x)), lambda x: (x + 2, jnp.sin(x)), jnp.float32),
        (lambda x: x + 1, lambda x: x + 2, jnp.int32),
        (lambda x: x + 1, lambda x: x + 2, jnp.complex64),
    ],
    ids=[
        "opaque",
        "transcendental",
        "partial",
        "denominator",
        "nonfinite",
        "other_output_opaque",
        "integer",
        "complex",
    ],
)
def test_domain_boundaries_remain_unknown(left, right, dtype):
    first, second = graphs(left, right, (), dtype=dtype)
    assert candidate(first, second) is None
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == Status.UNKNOWN, result
    assert result.inequivalence_proof is None


def test_captured_tensor_is_not_fresh_independent_input():
    value = np.asarray([1.0, 2.0], dtype=np.float32)
    first, second = graphs(lambda x: x + value, lambda x: x + 1, (2,))
    assert candidate(first, second) is None
    assert compare_algebraic_jaxprs(first, second).status == Status.UNKNOWN


@pytest.mark.parametrize("operation", [jnp.log, lambda x: x + (-jnp.inf)])
def test_dead_partial_or_nonfinite_work_does_not_make_domain_assumptions(operation):
    def first(x):
        operation(x)
        return x + 1

    left, right = graphs(first, lambda x: x + 2, ())
    result = compare_algebraic_jaxprs(left, right)
    assert result.status == Status.UNKNOWN, result
    assert "unused" in result.issues[0].reason


def test_dead_region_cannot_be_discarded_to_claim_inequivalence():
    def first(x):
        jax.lax.while_loop(lambda c: c < 3, lambda c: c + 1, x)
        return x + 1

    left, right = graphs(first, lambda x: x + 2, ())
    assert candidate(left, right) is None
    assert compare_algebraic_jaxprs(left, right).status == Status.UNKNOWN


@pytest.fixture
def certificate():
    left, right = graphs(lambda x, y: x * y + x, lambda x, y: x * y, (), ())
    proof = compare_algebraic_jaxprs(left, right).inequivalence_proof
    assert proof is not None
    return left, right, proof


@pytest.mark.parametrize(
    "changes",
    [
        {"coefficient": (2, 1)},
        {"coefficient": (0, 1)},
        {"coefficient": (2, 2)},
        {"coefficient": (1, 0)},
        {"coefficient": (True, 1)},
        {"output": 1},
        {"output": True},
        {"coordinate": (0,)},
        {"monomial": ()},
        {"monomial": (InputElement(1, ()),)},
        {"monomial": (InputElement(2, ()),)},
        {"monomial": (InputElement(0, (1,)),)},
        {"monomial": (InputElement(0, ()), InputElement(0, ()))},
        {"monomial": (InputElement(1, ()), InputElement(0, ()))},
        {"budget": 1},
        {"budget": True},
        {"version": 0},
        {"version": True},
    ],
)
def test_tampered_certificates_are_rejected(certificate, changes):
    left, right, proof = certificate
    with pytest.raises(CorrespondenceError):
        verify_inequivalence_proof(left, right, replace(proof, **changes))


def test_certificate_cannot_be_transferred_to_changed_source_or_empty_output(certificate):
    left, right, proof = certificate
    changed, _ = graphs(lambda x, y: x * y + 2 * x, lambda x, y: x * y, (), ())
    with pytest.raises(CorrespondenceError):
        verify_inequivalence_proof(changed, right, proof)
    prepared, _ = _prepare(changed)
    with pytest.raises(CorrespondenceError):
        verify_inequivalence_proof(
            left, right, replace(proof, left=replace(proof.left, graph=prepared))
        )
    empty_left, empty_right = graphs(lambda x: x + 1, lambda x: x + 2, (0,))
    sources = tuple(_prepare(g) for g in (empty_left, empty_right))
    empty_proof = replace(
        proof,
        left=replace(proof.left, graph=sources[0][0], correspondence=sources[0][1]),
        right=replace(proof.right, graph=sources[1][0], correspondence=sources[1][1]),
        coordinate=(0,),
        monomial=(),
        coefficient=(-1, 1),
    )
    with pytest.raises(CorrespondenceError, match="coordinate"):
        verify_inequivalence_proof(empty_left, empty_right, empty_proof)


def test_independent_checker_does_not_call_search_normalizer_or_cas(certificate, monkeypatch):
    from jax_reorder import algebra_inequivalence, canonical
    from jax_reorder.algebra_normalize import PolynomialNormalizer

    def forbidden(*args, **kwargs):
        raise AssertionError("independent checker must not invoke candidate search or CAS")

    monkeypatch.setattr(algebra_inequivalence, "_polynomial", forbidden)
    monkeypatch.setattr(algebra_inequivalence, "try_prove_inequivalent", forbidden)
    monkeypatch.setattr(PolynomialNormalizer, "normalize", forbidden)
    monkeypatch.setattr(canonical, "canonicalize", forbidden)
    import sympy

    monkeypatch.setattr(sympy, "simplify", forbidden)
    monkeypatch.setattr(sympy, "expand", forbidden)
    assert verify_inequivalence_proof(*certificate) is None


def test_huge_output_shape_does_not_allocate_coordinate_pools():
    first, second = graphs(lambda x: x + 1, lambda x: x + 2, (10**9,))
    result = compare_algebraic_jaxprs(first, second)
    assert result.status == Status.INEQUIVALENT, result
    assert result.inequivalence_proof.coordinate == (0,)


def test_finite_sum_and_expansion_are_budgeted_before_enumeration():
    first, second = graphs(lambda x: jnp.sum(x), lambda x: 2 * jnp.sum(x), (10**9,))
    with pytest.raises(UnsupportedJaxprError, match="budget"):
        candidate(first, second, limit=1000)
    result = compare_algebraic_jaxprs(first, second, algebra_budget=1000)
    assert result.status == Status.UNKNOWN
    assert "budget" in result.issues[0].reason


def test_malicious_certificate_degree_is_budgeted_before_divisor_enumeration(certificate):
    left, right, proof = certificate
    proof = replace(proof, monomial=(InputElement(0, ()),) * 20000)
    with pytest.raises(CorrespondenceError, match="budget"):
        verify_inequivalence_proof(left, right, proof)


def test_different_contract_is_not_a_polynomial_inequivalence():
    first, second = graphs(lambda x: x, lambda x: x[:1], (2,))
    assert candidate(first, second) is None
    assert compare_algebraic_jaxprs(first, second).status == Status.UNKNOWN


def test_high_degree_coefficient_uses_correct_variable_multiplicity():
    left, right = graphs(lambda x, y: (x + y) ** 4, lambda x, y: x**4 + y**4, (), ())
    proof = compare_algebraic_jaxprs(left, right).inequivalence_proof
    assert proof is not None
    assert proof.monomial == (InputElement(0, ()),) * 3 + (InputElement(1, ()),)
    assert proof.coefficient == (4, 1)
    assert verify_inequivalence_proof(left, right, proof) is None


def test_rational_constant_difference_is_exact():
    left, right = graphs(lambda x: x + 0.5, lambda x: x + 0.25, ())
    proof = compare_algebraic_jaxprs(left, right).inequivalence_proof
    assert proof is not None
    assert proof.monomial == ()
    assert proof.coefficient == (1, 4)


def test_canceled_intermediate_difference_cannot_supply_a_forged_final_certificate():
    left, right = graphs(lambda x: (x + 1) * 0, lambda x: (x + 2) * 0, ())
    origin, other = graphs(lambda x: x + 1, lambda x: x + 2, ())
    proof = compare_algebraic_jaxprs(origin, other).inequivalence_proof
    prepared = tuple(_prepare(g) for g in (left, right))
    forged = replace(
        proof,
        left=replace(proof.left, graph=prepared[0][0], correspondence=prepared[0][1]),
        right=replace(proof.right, graph=prepared[1][0], correspondence=prepared[1][1]),
    )
    with pytest.raises(CorrespondenceError, match="coefficient"):
        verify_inequivalence_proof(left, right, forged)
