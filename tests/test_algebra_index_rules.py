"""用独立整数解释器核对坐标适配，覆盖负坐标及嵌套 div/mod。"""

from itertools import product
from random import Random

from jax_reorder.algebra_indices import Index, add, divide, integer, scale


def evaluate_index(index, free=()):
    """不调用被测整数化简器，只按运算定义解释坐标。"""
    if index.kind in ("zero", "const"):
        return index.number
    if index.kind == "free":
        return free[index.number]
    args = [evaluate_index(child, free) for child in index.args]
    if index.kind == "add":
        return sum(args)
    if index.kind == "scale":
        return args[0] * index.number
    if index.kind == "div":
        return args[0] // index.number
    if index.kind == "mod":
        return args[0] % index.number
    raise AssertionError(index)


def test_integer_rewrites_agree_on_signed_inputs_and_nested_divmod():
    random = Random(20261014)
    x, y = Index("free", 0), Index("free", 1)
    for _ in range(100):
        a, b, c = (random.randrange(-5, 6) for _ in range(3))
        d, e = random.randrange(1, 7), random.randrange(1, 7)
        raw = Index("add", args=(Index("scale", a, (x,)), Index("scale", b, (y,)), integer(c)))
        for kind in ("div", "mod"):
            expression = Index(kind, d, (Index("mod", e, (raw,)),))
            domains = {x: (-3, 4), y: (-3, 4)}
            normalized = add(scale(x, a), scale(y, b), integer(c))
            inner = divide(normalized, e, domains, modulo=True)
            simplified = divide(inner, d, domains, modulo=kind == "mod")
            for left, right in product(range(-3, 4), repeat=2):
                assert evaluate_index(expression, free=(left, right)) == evaluate_index(
                    simplified, free=(left, right)
                )
        rebuilt = add(scale(divide(raw, d), d), divide(raw, d, modulo=True))
        for point in ((-3, 2), (0, 0), (3, -2)):
            assert evaluate_index(raw, free=point) == evaluate_index(rebuilt, free=point)
