"""Exact, budgeted tail-mask checks for each individual tensor Ref domain."""

from ..errors import UnsupportedJaxprError
from .index_domain import AnalysisBudget, IndexExpr, integer, substitute
from .memory import Expression, Predicate, ValueExpr, truth
from .tensor_domain import TensorDomain, coordinate_ranges


def prove_mask(
    mask: Expression,
    domain: TensorDomain,
    coordinates: dict[str, IndexExpr],
    ranges: dict[str, tuple[int, int]],
    budget: AnalysisBudget,
    path: str,
) -> None:
    valid = Predicate(
        "and",
        tuple(
            Predicate("lt", (substitute(address, coordinates), integer(n)))
            for address, n in zip(domain.address, domain.shape, strict=True)
        ),
    )
    initial = {**ranges, **coordinate_ranges("local", domain.block_shape)}

    def normalized(value: Expression, bounds: dict[str, tuple[int, int]]) -> Expression:
        known = truth(value, bounds)
        if known is not None:
            return known
        if isinstance(value, Predicate) and value.operation == "and":
            terms = []
            pending = list(value.operands)
            while pending:
                item = pending.pop()
                known = truth(item, bounds)
                if known is False:
                    return False
                if known is True:
                    continue
                if isinstance(item, Predicate) and item.operation == "and":
                    pending.extend(item.operands)
                else:
                    terms.append(item)
            return Predicate("and", tuple(sorted(set(terms), key=repr)))
        return value

    pending = [initial]
    while pending:
        local = pending.pop()
        budget.spend(path + ".tail_mask")
        left, right = normalized(mask, local), normalized(valid, local)
        if left == right:
            continue
        if type(left) is bool and type(right) is bool:
            raise UnsupportedJaxprError(
                path,
                "tensor mask leaves a coverage gap in the valid domain"
                if right
                else "tensor access has an unguarded out-of-bounds tail",
            )
        # Shared conjuncts need not drive splitting. If the residual condition
        # differs, the shared condition is still checked on the resulting box.
        expressions: list[Expression] = [left, right]
        if (
            isinstance(left, Predicate)
            and left.operation == "and"
            and isinstance(right, Predicate)
            and right.operation == "and"
        ):
            common = set(left.operands) & set(right.operands)
            expressions = [v for v in (*left.operands, *right.operands) if v not in common]
        axes = set()
        while expressions:
            item = expressions.pop()
            if isinstance(item, IndexExpr):
                if item.kind == "axis" and str(item.value) in local:
                    axes.add(str(item.value))
                expressions.extend(item.args)
            elif isinstance(item, Predicate | ValueExpr):
                expressions.extend(item.operands)
        candidates = [key for key in axes if local[key][0] < local[key][1]]
        if not candidates:
            raise UnsupportedJaxprError(path, "tensor mask has no exact static tail proof")
        key = max(candidates, key=lambda k: local[k][1] - local[k][0])
        low, high = local[key]
        middle = (low + high) // 2
        pending.extend(({**local, key: (low, middle)}, {**local, key: (middle + 1, high)}))
