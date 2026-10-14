"""用独立的小域解释器核对符号整数与求和域换元，覆盖捕获与局部绑定。"""

from itertools import product
from math import sin
from random import Random

import pytest

from jax_reorder.algebra_domains import apply_domain_witness, prefix_terms
from jax_reorder.algebra_indices import add, divide, integer, map_index, scale
from jax_reorder.algebra_ir import Budget, Expr, Factor, Index, Term
from jax_reorder.algebra_normalize import PolynomialNormalizer


def evaluate_index(index, frames=(), free=()):
    """不调用被测整数化简器，只按运算定义解释坐标。"""
    if index.kind in ("zero", "const"):
        return index.number
    if index.kind == "bound":
        return frames[index.scope][index.number]
    if index.kind == "free":
        return free[index.number]
    args = [evaluate_index(child, frames, free) for child in index.args]
    if index.kind == "add":
        return sum(args)
    if index.kind == "scale":
        return args[0] * index.number
    if index.kind == "div":
        return args[0] // index.number
    if index.kind == "mod":
        return args[0] % index.number
    raise AssertionError(index)


def evaluate_terms(terms, outer=()):
    """独立枚举测试用的小矩形；实际实现始终保留符号域。"""
    total = 0.0
    for term in terms:
        for point in product(*(range(n) for n in term.bounds)):
            frames = (point, *outer)
            value = float(term.rational)
            for factor in term.factors:
                if factor.scalar:
                    value *= sin(evaluate_terms(factor.arguments[0], frames))
                else:
                    indices = [evaluate_index(index, frames) for index in factor.indices]
                    # 非可分的确定性函数，使错轴、重数和捕获错误可观察。
                    value *= 1 + sum(
                        (i + 2) * (coordinate + 1) ** (i + 1)
                        for i, coordinate in enumerate(indices)
                    )
            total += value
    return total


def test_integer_rewrites_agree_on_signed_inputs_and_nested_divmod():
    random = Random(20261014)
    x, y = Index("free", 0), Index("free", 1)
    for _ in range(100):
        a, b, c = (random.randrange(-5, 6) for _ in range(3))
        d, e = random.randrange(1, 7), random.randrange(1, 7)
        raw = Index("add", args=(Index("scale", a, (x,)), Index("scale", b, (y,)), integer(c)))
        for kind in ("div", "mod"):
            expression = Index(kind, d, (Index("mod", e, (raw,)),))
            simplified = map_index(expression, lambda index: index, {x: (-3, 4), y: (-3, 4)})
            for left, right in product(range(-3, 4), repeat=2):
                assert evaluate_index(expression, free=(left, right)) == evaluate_index(
                    simplified, free=(left, right)
                )
        rebuilt = add(scale(divide(raw, d), d), divide(raw, d, modulo=True))
        for point in ((-3, 2), (0, 0), (3, -2)):
            assert evaluate_index(raw, free=point) == evaluate_index(rebuilt, free=point)


@pytest.mark.parametrize(
    "witness,bounds,coordinates",
    [
        (("split", 0, 3), (12,), (Index("bound", 0),)),
        (("fuse", 0, 1), (3, 4), (Index("bound", 0), Index("bound", 1))),
        (("shift", 0, 2), (4,), (add(Index("bound", 0), integer(2)),)),
        (
            ("shift", 0, 5),
            (7,),
            (
                divide(add(Index("bound", 0), integer(5)), 4),
                divide(add(Index("bound", 0), integer(1)), 4, modulo=True),
            ),
        ),
    ],
)
def test_domain_witness_preserves_sum_by_independent_enumeration(witness, bounds, coordinates):
    original = Term((3, 2), bounds, (Factor(("read",), coordinates),))
    changed = prefix_terms(apply_domain_witness(original, witness), Budget(10000))
    assert evaluate_terms((original,)) == evaluate_terms(changed)


def test_scalar_function_local_sum_does_not_capture_outer_axis():
    # sum_i sin(sum_j x[i,j]) 必须保留两层作用域，不能变成 sin(sum_i,j x[i,j])。
    inner = Term(
        (1, 1), (3,), (Factor(("read",), (Index("bound", 0, scope=1), Index("bound", 0))),)
    )
    original = Term((1, 1), (6,), (Factor(("sin",), (), ((inner,),), scalar=True),))
    changed = apply_domain_witness(original, ("split", 0, 2))
    assert evaluate_terms((original,)) == pytest.approx(evaluate_terms((changed,)), abs=1e-12)
    assert changed.factors[0].arguments[0][0].bounds == (3,)


def test_unsafe_prefix_domains_are_rejected():
    i = Index("bound", 0)
    for term in (
        Term(
            (1, 1), (3,), (Factor(("read",), (add(divide(add(i, integer(2)), 2), integer(-1)),)),)
        ),
        Term((1, 1), (3,), (Factor(("read",), (add(i, integer(2)),), total=False),)),
        Term((1, 1), (3,), (Factor(("read",), (i, add(i, integer(2)))),)),
        Term((1, 1), (3,), (Factor(("read",), (add(i, integer(-2)),)),)),
    ):
        with pytest.raises(ValueError, match="unsafe"):
            apply_domain_witness(term, ("shift", 0, 2))


def test_normalized_scalar_capture_and_nested_reductions_keep_scope():
    i, j = Index("sum", 0), Index("sum", 1)
    read = Expr("atom", (("read",), (i, j)))
    inside = Expr("sum", (1, 3), (read,))
    expression = Expr("sum", (0, 6), (Expr("call", ("sin",), (inside,)),))
    engine = PolynomialNormalizer(Budget(10000))
    result = engine.steps[engine.normalize(expression)].result
    inner = result[0].factors[0].arguments[0][0]
    assert inner.bounds == (3,)
    assert inner.factors[0].indices == (Index("bound", 0, scope=1), Index("bound", 0))
    expected = sum(sin(sum(1 + 2 * (i + 1) + 3 * (j + 1) ** 2 for j in range(3))) for i in range(6))
    assert evaluate_terms(result) == pytest.approx(expected, abs=1e-12)
