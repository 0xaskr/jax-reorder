"""真实 jaxpr 上验证受控 CAS、数学定义域和不依赖 CAS 的证据重放。"""

from dataclasses import replace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
import sympy
from jax._src import compiler, dispatch

from jax_reorder import (
    CorrespondenceError,
    Status,
    compare_algebraic_jaxprs,
    verify_algebraic_proof,
)
from jax_reorder.algebra_ir import Budget, Expr, Factor, Index, Term
from jax_reorder.algebra_normalize import PolynomialNormalizer


def forbidden(*args, **kwargs):
    raise AssertionError("numerical execution or CAS search cannot verify this certificate")


def compare(left, right, shape=(5,), **options):
    value = jax.ShapeDtypeStruct(shape, jnp.float32)
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
        patch.object(sympy.Expr, "equals", forbidden),
        patch.object(sympy.Expr, "_random", forbidden),
    ):
        a, b = jax.make_jaxpr(left)(value), jax.make_jaxpr(right)(value)
        result = compare_algebraic_jaxprs(a, b, **options)
    return a, b, result


@pytest.mark.parametrize(
    "left,right",
    [
        (lambda x: jnp.exp(x) * jnp.exp(2 * x), lambda x: jnp.exp(3 * x)),
        (lambda x: jnp.exp(x) * jnp.exp(-x), jnp.ones_like),
        (lambda x: jnp.exp(x * 0), jnp.ones_like),
        (lambda x: jnp.exp(x * 0 + 0.5) ** 2, lambda x: jnp.exp(x * 0 + 1)),
        (lambda x: x / 2, lambda x: x * 0.5),
        (lambda x: (x + 2) / -2, lambda x: -0.5 * x - 1),
        (lambda x: jnp.exp(x / 2), lambda x: jnp.exp(x * 0.5)),
        (lambda x: jnp.exp(jnp.exp(x) / jnp.exp(2 * x)), lambda x: jnp.exp(jnp.exp(-x))),
        (lambda x: jnp.exp(x) / jnp.exp(2 * x), lambda x: jnp.exp(-x)),
        (lambda x: jnp.exp(x) / jnp.exp(x), jnp.ones_like),
        (
            lambda x: jnp.sum(jnp.exp(x)) / jnp.sum(jnp.exp(x)),
            lambda x: jnp.array(1.0, dtype=jnp.float32),
        ),
        (
            lambda x: -jnp.sum(jnp.exp(x)) / -jnp.sum(jnp.exp(x)),
            lambda x: jnp.array(1.0, dtype=jnp.float32),
        ),
        (lambda x: jnp.sum(jnp.exp(x) * jnp.exp(2 * x)), lambda x: jnp.sum(jnp.exp(3 * x))),
        (
            lambda x: jnp.exp(jnp.sum(x)) * jnp.exp(jnp.sum(2 * x)),
            lambda x: jnp.exp(jnp.sum(3 * x)),
        ),
    ],
)
def test_supported_scalar_identities_replay_from_actual_jaxprs(left, right):
    a, b, result = compare(left, right)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "algebraic"
    assert any(step.scalar_rewrites for step in result.proof.steps)
    verify_algebraic_proof(a, b, result.proof)


def test_sympy_generates_candidates_but_checker_does_not_use_it(monkeypatch):
    calls = []
    powsimp = sympy.powsimp

    def observed(*args, **kwargs):
        calls.append(True)
        return powsimp(*args, **kwargs)

    monkeypatch.setattr(sympy, "powsimp", observed)
    a, b, result = compare(lambda x: jnp.exp(x) * jnp.exp(2 * x), lambda x: jnp.exp(3 * x))
    assert result.status == Status.EQUIVALENT
    assert calls
    import jax_reorder.algebra_sympy as backend

    monkeypatch.setattr(sympy, "powsimp", forbidden)
    monkeypatch.setattr(sympy, "cancel", forbidden)
    monkeypatch.setattr(backend, "rewrite", forbidden)
    monkeypatch.setattr(backend.LinearBridge, "__init__", forbidden)
    monkeypatch.setattr(PolynomialNormalizer, "normalize", forbidden)
    verify_algebraic_proof(a, b, result.proof)


@pytest.mark.parametrize(
    "left,right,shape",
    [
        (lambda x: x / x, jnp.ones_like, (5,)),
        (lambda x: jnp.sin(x) / jnp.sin(x), jnp.ones_like, (5,)),
        (lambda x: jnp.maximum(x, 0) / jnp.maximum(x, 0), jnp.ones_like, (5,)),
        (lambda x: x / 0, jnp.zeros_like, (5,)),
        (lambda x: jnp.exp(x / x), lambda x: jnp.exp(jnp.ones_like(x)), (5,)),
        (lambda x: jnp.exp(x / 0), lambda x: jnp.exp(x * 0), (5,)),
        (lambda x: jnp.exp(jnp.log(x) / 2), lambda x: jnp.exp(jnp.log(x) * 0.5), (5,)),
        (lambda x: jnp.exp((x - jnp.inf) / 2), lambda x: jnp.exp((x - jnp.inf) * 0.5), (5,)),
        (
            lambda x: jnp.sum(jnp.exp(x)) / jnp.sum(jnp.exp(x)),
            lambda x: jnp.array(1.0, dtype=jnp.float32),
            (0,),
        ),
        (
            lambda x: jnp.exp(jnp.log(x)) * jnp.exp(jnp.log(x)),
            lambda x: jnp.exp(2 * jnp.log(x)),
            (5,),
        ),
        (
            lambda x: jnp.exp(x - jnp.inf) * jnp.exp(-x + jnp.inf),
            jnp.ones_like,
            (5,),
        ),
    ],
)
def test_partial_nonfinite_and_missing_nonzero_facts_are_not_erased(left, right, shape):
    _, _, result = compare(left, right, shape)
    assert result.status == Status.UNKNOWN, result


def test_inner_and_outer_summation_scopes_are_preserved():
    def left(x):
        return jnp.sum(jnp.exp(jnp.sum(x, axis=0)) * jnp.exp(x), axis=1)

    def right(x):
        return jnp.sum(jnp.exp(jnp.sum(x, axis=0) + x), axis=1)

    a, b, result = compare(left, right, (3, 4))
    assert result.status == Status.EQUIVALENT, result
    verify_algebraic_proof(a, b, result.proof)
    _, _, wrong = compare(
        left, lambda x: jnp.sum(jnp.exp(jnp.sum(x, axis=1)[:, None] + x), axis=1), (3, 4)
    )
    assert wrong.status != Status.EQUIVALENT


def test_defined_exponential_composes_with_reductions_and_replays_independently(monkeypatch):
    a, b, result = compare(
        lambda x: jnp.exp(jnp.sum(x, axis=0) / 2),
        lambda x: jnp.exp(jnp.sum(x * 0.5, axis=0)),
        (3, 4),
    )
    assert result.status == Status.EQUIVALENT, result
    assert any(
        rewrite.kind == "exp_defined"
        for step in result.proof.steps
        for rewrite in step.scalar_rewrites
    )
    monkeypatch.setattr(sympy, "cancel", forbidden)
    monkeypatch.setattr(sympy, "powsimp", forbidden)
    monkeypatch.setattr(PolynomialNormalizer, "normalize", forbidden)
    verify_algebraic_proof(a, b, result.proof)


def test_defined_exponential_certificate_cannot_launder_partial_argument():
    from jax import lax

    from jax_reorder.algebra_ir import ScalarRewrite
    from jax_reorder.algebra_lower import encode
    from jax_reorder.algebra_scalar_proof import replay

    semantic = ("scalar", lax.exp_p, encode({}), encode(jnp.dtype("float32")))
    partial = Term((1, 1), factors=(Factor(("partial",), (), total=False),))
    factor = Factor(semantic, (), ((partial,),), False, True)
    term = Term((1, 1), factors=(factor,))
    forged = replace(term, factors=(replace(factor, total=True),))
    witness = ScalarRewrite("exp_defined", 0, (forged,), (0,))
    with pytest.raises(CorrespondenceError, match="no finite total proof"):
        replay((term,), (witness,), Budget(100))


@pytest.mark.parametrize(
    "change", ["kind", "position", "data", "coefficient", "total", "scope", "missing"]
)
def test_scalar_certificate_tampering_is_rejected(change):
    a, b, result = compare(lambda x: jnp.exp(x) * jnp.exp(2 * x), lambda x: jnp.exp(3 * x))
    proof = result.proof
    index = next(i for i, step in enumerate(proof.steps) if step.scalar_rewrites)
    step = proof.steps[index]
    witness = step.scalar_rewrites[0]
    if change == "kind":
        witness = replace(witness, kind="untrusted_rule")
    elif change == "position":
        witness = replace(witness, position=999)
    elif change == "data":
        witness = replace(witness, data=(0, 0))
    else:
        term = witness.result[0]
        if change == "coefficient":
            term = replace(term, coefficient=(7, 1))
        elif change == "total":
            term = replace(term, factors=(replace(term.factors[0], total=False),))
        elif change == "scope":
            factor = term.factors[0]
            argument = factor.arguments[0][0]
            leaf = replace(argument.factors[0], indices=(Index("bound", 0, scope=9),))
            term = replace(
                term, factors=(replace(factor, arguments=((replace(argument, factors=(leaf,)),),)),)
            )
        witness = replace(witness, result=(term,))
    witnesses = () if change == "missing" else (witness,)
    steps = list(proof.steps)
    steps[index] = replace(step, scalar_rewrites=witnesses)
    with pytest.raises(CorrespondenceError):
        verify_algebraic_proof(a, b, replace(proof, steps=tuple(steps)))


def test_cas_wrong_candidate_is_not_accepted(monkeypatch):
    monkeypatch.setattr(sympy, "powsimp", lambda *args, **kwargs: sympy.S.One)
    _, _, result = compare(lambda x: jnp.exp(x) * jnp.exp(2 * x), jnp.ones_like)
    assert result.status == Status.UNKNOWN
    assert "incorrect scalar rewrite" in result.issues[0].reason


def test_large_reductions_remain_symbolic_and_cas_preflight_obeys_budget():
    reports = []
    for size in (20, 1_000_000):
        _, _, result = compare(
            lambda x: jnp.exp(jnp.sum(x)) * jnp.exp(jnp.sum(2 * x)),
            lambda x: jnp.exp(jnp.sum(3 * x)),
            (size,),
        )
        assert result.status == Status.EQUIVALENT
        reports.append(result)
    assert reports[0].work == reports[1].work
    from jax_reorder.algebra_sympy import LinearBridge
    from jax_reorder.errors import UnsupportedJaxprError

    factors = tuple(Factor(("input", i), ()) for i in range(20))
    polynomial = tuple(Term((1, 1), factors=(factor,)) for factor in factors)
    with patch.object(sympy, "Dummy", forbidden):
        with pytest.raises(UnsupportedJaxprError, match="budget"):
            LinearBridge((polynomial,), Budget(20), lambda x: 0)


def test_nonfinite_exp_factor_never_gets_a_cas_real_symbol():
    from jax import lax

    from jax_reorder.algebra_lower import encode

    semantic = ("scalar", lax.exp_p, encode({"accuracy": None}), encode(jnp.dtype("float32")))
    partial = Expr("atom", (("nonfinite",), ()), total=False)
    expression = Expr("call", semantic, (partial,), total=False)
    root = Expr("mul", args=(expression, expression), total=False)
    engine = PolynomialNormalizer(Budget(100))
    with patch.object(sympy, "Dummy", forbidden):
        engine.normalize(root)
    assert not any(step.scalar_rewrites for step in engine.steps)
