"""用独立整数解释器核对坐标适配，覆盖负坐标及嵌套 div/mod。

覆盖：
- 正例：`div`/`mod` 的化简结果与原始表达式在所有采样点上取值相同；`add`/`scale` 重建
  的表达式也与原式一致。
- 反例：以独立解释器为参照，不调用被测化简器的任何规则，避免“两份同样错误”的漏检。
- 边界：自由变量取负值（-3..3），除数取 1..6，嵌套 mod 后再做 div/mod，检查有符号整数
  取模与下取整除法的 Python 语义。

前置条件：纯 Python，无 JAX/设备依赖，也不需要上游快照。
固定种子 20261014 的 100 轮随机样本不构成对全部整数表达式的证明。
"""

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
    """随机采样有符号坐标，逐点比较原式与化简式；div 与 mod 两种路径都覆盖。"""
    # 固定种子让失败可复现；只影响采样点，不改变被测的整数规则。
    random = Random(20261014)
    x, y = Index("free", 0), Index("free", 1)
    for _ in range(100):
        # a/b/c 取负值以覆盖有符号取模；d/e 取正除数，符合下标除法的合法域。
        a, b, c = (random.randrange(-5, 6) for _ in range(3))
        d, e = random.randrange(1, 7), random.randrange(1, 7)
        raw = Index("add", args=(Index("scale", a, (x,)), Index("scale", b, (y,)), integer(c)))
        for kind in ("div", "mod"):
            expression = Index(kind, d, (Index("mod", e, (raw,)),))
            # 自由变量域固定为 [-3, 4)，与解释器采样范围一致。
            domains = {x: (-3, 4), y: (-3, 4)}
            normalized = add(scale(x, a), scale(y, b), integer(c))
            inner = divide(normalized, e, domains, modulo=True)
            simplified = divide(inner, d, domains, modulo=kind == "mod")
            for left, right in product(range(-3, 4), repeat=2):
                assert evaluate_index(expression, free=(left, right)) == evaluate_index(
                    simplified, free=(left, right)
                )
        # 欧几里得分解恒等式：q*d + r == raw，用三个固定点检查重建结果。
        rebuilt = add(scale(divide(raw, d), d), divide(raw, d, modulo=True))
        for point in ((-3, 2), (0, 0), (3, -2)):
            assert evaluate_index(raw, free=point) == evaluate_index(rebuilt, free=point)
