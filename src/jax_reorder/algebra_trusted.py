"""Trusted SymPy queries and a killable process boundary for the mathematical stage."""

import os
import pickle
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from itertools import islice, product
from math import comb, prod

import sympy as sp

from .algebra_symbolic import SymbolicBuilder, coordinates, rename_bounds
from .errors import (
    ComparisonContractError,
    ComparisonError,
    ComparisonInternalError,
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    UnsupportedJaxprError,
)


@dataclass(frozen=True)
class SympyQuery:
    output: int
    scope: str
    answer: bool | None
    method: str
    left: str
    right: str


@dataclass(frozen=True)
class SympyResult:
    status: str
    queries: tuple[SympyQuery, ...] = ()
    path: str = "algebra.outputs"
    output: int | None = None
    backend: str = "sympy"
    version: str = sp.__version__
    work: int = 0
    elapsed_seconds: float = field(default=0, compare=False)
    rules: tuple[str, ...] = ()
    definitions: tuple[tuple[int, str], ...] = ()

    def __post_init__(self):
        if self.status not in ("equivalent", "inequivalent"):
            raise ValueError(f"invalid mathematical result status: {self.status}")


@dataclass(frozen=True)
class WorkerFailure:
    """失败消息与 SympyResult 分离，不携带第四类比较状态。"""

    error_type: type[ComparisonError]
    path: str
    reason: str
    stage: str
    failure_kind: str
    diagnostics: dict
    traceback: str


class RemoteWorkerError(RuntimeError):
    """作为父进程异常的 cause 保存远端原始异常链。"""


def _finite_expansion(expression, budget):
    """Preflight explicit symbolic sums; numeric shape is not a license for unbounded expansion."""
    costs = {}

    def cost(node):
        if node in costs:
            return costs[node]
        if isinstance(node, (sp.Sum, sp.Product)):
            size = 1
            for variable, start, stop in node.limits:
                extent = stop - start + 1
                if not extent.is_Integer or extent < 0:
                    return 257
                if isinstance(node, sp.Sum) and variable not in node.function.free_symbols:
                    # Sum of an index-independent value is one scalar multiply,
                    # even when the declared axis is large.
                    continue
                size *= int(extent)
                if size > 256:
                    return 257
            count = size * max(1, cost(node.function))
        else:
            count = 0
            for child in node.args:
                count += cost(child)
                if count > 256:
                    break
        costs[node] = count
        return count

    count = cost(expression)
    if count > 256:
        return False
    budget.spend(count, path="algebra.sympy.expand")
    return True


def simplify(expression, budget, *, assumptions=sp.true, expand=True):
    from sympy.core.function import AppliedUndef

    budget.spend(path="algebra.sympy.simplify")
    # Charge a combinatorial upper bound before SymPy expands a power/product.
    terms = {}
    for node in reversed(list(sp.preorder_traversal(expression))):
        count = (
            sum(terms.get(arg, 1) for arg in node.args)
            if node.is_Add
            else prod(terms.get(arg, 1) for arg in node.args)
            if node.is_Mul
            else 1
        )
        if node.is_Pow and node.exp.is_Integer and node.exp > 0 and node.base.is_Add:
            exponent = int(node.exp)
            if exponent > budget.limit:
                budget.spend(budget.limit + 1, path="algebra.sympy.expand")
            count = comb(len(node.base.args) + exponent - 1, exponent)
        if count > budget.limit - budget.used:
            budget.spend(count, path="algebra.sympy.expand")
        terms[node] = count
    expression = rename_bounds(
        expression.expand() if expand and isinstance(expression, sp.Expr) else expression
    )
    if expression.has(AppliedUndef) and expression.has(sp.Max, sp.Min):
        # Keep discrete index conditions; no real inequality model is available.
        return expression
    if expand and expression.has(sp.Sum, sp.Product) and _finite_expansion(expression, budget):
        expression = expression.doit()
    if expression.has(sp.Piecewise, sp.Max, sp.Min, sp.Abs):
        expression = sp.piecewise_exclusive(
            sp.piecewise_fold(expression.rewrite(sp.Piecewise)), deep=True
        )
    if assumptions != sp.true:
        expression = sp.refine(expression, assumptions)
    # Explicit finite expansion was budgeted above. simplify's default doit would
    # bypass that limit and make equivalent sums canonicalize inconsistently.
    return sp.simplify(expression, doit=False)


def _array_compatible(expression, indices):
    """ArrayExpr conversion requires index labels, not arbitrary coordinate expressions.

    convert_indexed_to_array treats x[i+1] and x[i] as the same array with
    different axis labels. Only pure tensor contractions over full axes can
    use that representation; slices and coordinate arithmetic stay indexed.
    """
    bound = {}
    for summation in expression.atoms(sp.Sum):
        for variable, start, stop in summation.limits:
            if not isinstance(variable, sp.Symbol) or start != 0 or not stop.is_Integer:
                return False
            if variable in bound and bound[variable] != stop + 1:
                return False
            bound[variable] = stop + 1
    labels = set(indices) | bound.keys()
    reads = expression.atoms(sp.Indexed)
    # Dropping an absent free axis would also conflate row and column broadcasts.
    if {index for read in reads for index in read.indices} != labels:
        return False

    def supported(node):
        if isinstance(node, sp.Indexed):
            return all(
                isinstance(index, sp.Symbol)
                and index in labels
                and (index not in bound or bound[index] == extent)
                for index, extent in zip(node.indices, node.base.shape, strict=True)
            )
        if isinstance(node, sp.Sum):
            return supported(node.function)
        if node.is_Add or node.is_Mul:
            return all(supported(arg) for arg in node.args)
        return node.is_Number or isinstance(node, sp.Symbol) and node not in labels

    return supported(expression)


def equal(left, right, budget, *, indices=None, assumptions=sp.true, methods=None):
    from sympy.core.function import AppliedUndef

    steps = [] if methods is None else methods
    if left.has(sp.nan, sp.zoo) or right.has(sp.nan, sp.zoo):
        steps.append("nonfinite_expression")
        return None
    if left == right:
        steps.append("identity")
        return True

    def boolean_piecewise(expression):
        # SymPy 的 simplify 可把定义条件变成 Piecewise；它继承 Expr，
        # 但布尔分支不能相减。只把全部分支均为逻辑值的情况交给 ITE。
        if isinstance(expression, sp.Piecewise) and all(
            isinstance(branch, sp.logic.boolalg.Boolean) and not isinstance(branch, sp.Expr)
            for branch, _ in expression.args
        ):
            budget.spend(len(expression.args), path="algebra.sympy.logic")
            steps.append("Piecewise.rewrite(ITE)")
            return expression.rewrite(sp.ITE)
        return expression

    left, right = boolean_piecewise(left), boolean_piecewise(right)
    if (
        indices is not None
        and (left.has(sp.Sum, sp.Indexed) or right.has(sp.Sum, sp.Indexed))
        and _array_compatible(left, indices)
        and _array_compatible(right, indices)
    ):
        from sympy.tensor.array.expressions import convert_indexed_to_array

        budget.spend(path="algebra.sympy.array")
        steps.append("ArrayExpr.doit")
        try:
            a = convert_indexed_to_array(left, first_indices=indices).doit()
            b = convert_indexed_to_array(right, first_indices=indices).doit()
            if a == b:
                return True
        except (TypeError, ValueError, NotImplementedError, IndexError, AttributeError, KeyError):
            pass
    if isinstance(left, sp.logic.boolalg.Boolean) and not isinstance(left, sp.Expr):
        steps.append(
            "simplify_logic(Equivalent)"
            if left.has(AppliedUndef) or right.has(AppliedUndef)
            else "simplify(Equivalent)"
        )
        residual = (
            sp.simplify_logic(sp.Equivalent(left, right), deep=False)
            if left.has(AppliedUndef) or right.has(AppliedUndef)
            else simplify(sp.Equivalent(left, right), budget)
        )
        return True if residual == sp.true else False if residual == sp.false else None
    if (left.has(sp.Sum, sp.Product) or right.has(sp.Sum, sp.Product)) and not (
        left.has(sp.Piecewise, sp.Max, sp.Min, sp.Abs)
        or right.has(sp.Piecewise, sp.Max, sp.Min, sp.Abs)
    ):
        # Simplifying each side first preserves symbolic sum identities that
        # expansion of exp(Sum(...)) would turn into large unevaluated Products.
        # Piecewise and functions rewritten into it need cancellation or bounded
        # expansion first to avoid branch explosion.
        steps.append("simplify(each side, doit=False)")
        left = simplify(left, budget, assumptions=assumptions, expand=False)
        right = simplify(right, budget, assumptions=assumptions, expand=False)
        if left == right:
            return True
    steps.append("simplify(left-right)")
    residual = simplify(left - right, budget, assumptions=assumptions)
    if residual.has(sp.nan, sp.zoo):
        return None
    if residual.has(AppliedUndef):
        steps.append("opaque_residual")
        return None
    if residual.has(sp.Sum, sp.Product) and not _finite_expansion(residual, budget):
        # Expr.equals can call doit internally, outside the explicit expansion
        # budget. Leave the query undecided; the caller raises if no other method decides it.
        steps.append("unevaluated_sum")
        return None
    steps.append("Expr.equals(0)")
    return residual.equals(0)


def _domain(condition, budget):
    if condition in (sp.true, sp.false):
        return condition
    from sympy.core.function import AppliedUndef

    if condition.has(AppliedUndef):
        if sp.count_ops(condition) > 30:
            return rename_bounds(condition)
        return rename_bounds(sp.simplify_logic(condition, deep=False))
    return simplify(condition, budget)


def _explicit_symbols(left, right):
    """Concrete array entries are independent inputs, with one shared identity on both sides."""
    from sympy.core.function import AppliedUndef

    if left.has(AppliedUndef) or right.has(AppliedUndef):
        return None
    if any(
        isinstance(symbol, sp.Symbol) and symbol.is_integer
        for symbol in left.free_symbols | right.free_symbols
    ):
        # Unbounded SymPy integers can witness values outside a source integer dtype.
        # Congruence remains useful, but False is not a source-program counterexample.
        return None
    reads = left.atoms(sp.Indexed) | right.atoms(sp.Indexed)
    replacements = {}
    for position, read in enumerate(sorted(reads, key=sp.default_sort_key)):
        if any(not index.is_Integer for index in read.indices):
            return None
        if any(
            not 0 <= index < extent
            for index, extent in zip(read.indices, read.base.shape, strict=True)
        ):
            return None
        assumptions = {"real": True} if read.is_real else {}
        if read.is_integer:
            assumptions = {"integer": True}
        replacements[read] = sp.Symbol(f"entry_{position}", **assumptions)
    return left.xreplace(replacements), right.xreplace(replacements)


def compare_symbolic(left, right, algebra_budget=10000):
    started = time.monotonic()
    builder = SymbolicBuilder(algebra_budget)
    records = []

    def query(output, scope, x, y, **kwargs):
        methods = []
        answer = equal(x, y, builder.budget, methods=methods, **kwargs)
        records.append(SympyQuery(output, scope, answer, " -> ".join(methods), str(x), str(y)))
        return answer

    def report(*args, **kwargs):
        return SympyResult(
            *args,
            **kwargs,
            definitions=tuple((index, repr(key)) for key, index in builder.definitions.items()),
        )

    try:
        a, b = builder.graph(left), builder.graph(right)
        if len(a) != len(b) or any(
            x.shape != y.shape or x.dtype != y.dtype for x, y in zip(a, b, strict=True)
        ):
            raise ComparisonContractError("algebra.outputs", "output abstract contracts differ")
        undecided, domain_unknown = False, False
        for output, (x, y) in enumerate(zip(a, b, strict=True)):
            if prod(x.shape) == 0:
                continue
            if x.defined == y.defined:
                da, db = x.defined, y.defined
            else:
                da, db = _domain(x.defined, builder.budget), _domain(y.defined, builder.budget)
            same_domain = query(output, "definedness", da, db)
            if same_domain is not True and prod(x.shape) <= 64:
                same_domain = all(
                    query(
                        output,
                        f"definedness_coordinate={point}",
                        _domain(x.read(point)[1], builder.budget),
                        _domain(y.read(point)[1], builder.budget),
                    )
                    is True
                    for point in product(*(range(n) for n in x.shape))
                )
            if same_domain is not True:
                domain_unknown = True
                continue
            if da == sp.false:
                domain_unknown = True
                continue
            bounds = sp.And(*(i < n for i, n in zip(coordinates(x.shape), x.shape, strict=True)))
            answer = query(
                output,
                "symbolic_output",
                x.expression,
                y.expression,
                indices=coordinates(x.shape),
                assumptions=bounds,
            )
            if answer is True:
                continue
            # False with abstract coordinates does not witness a legal source output.
            # Instantiate only indices, keeping every input value symbolic.
            concrete_answers = []
            fully_enumerated = prod(x.shape) <= 64
            points = product(
                *(range(size if fully_enumerated else min(size, 2)) for size in x.shape)
            )
            for point in islice(points, 64 if fully_enumerated else 16):
                indices = tuple(map(sp.Integer, point))
                ex, dx = x.read(indices)
                ey, dy = y.read(indices)
                dx, dy = _domain(dx, builder.budget), _domain(dy, builder.budget)
                if dx == sp.false or equal(dx, dy, builder.budget) is not True:
                    concrete_answers.append(None)
                    continue
                if not (
                    _finite_expansion(ex, builder.budget) and _finite_expansion(ey, builder.budget)
                ):
                    concrete_answers.append(None)
                    continue
                ex, ey = ex.doit(), ey.doit()
                explicit = _explicit_symbols(ex, ey)
                methods = []
                if explicit is None:
                    # Congruence is sound for opaque calls/casts; a generic False is not.
                    decision = (
                        True if equal(ex, ey, builder.budget, methods=methods) is True else None
                    )
                else:
                    decision = equal(*explicit, builder.budget, methods=methods)
                if decision is False and (dx != sp.true or dy != sp.true):
                    decision = None
                concrete_answers.append(decision)
                records.append(
                    SympyQuery(
                        output,
                        f"output_coordinate={point}",
                        decision,
                        " -> ".join(methods),
                        str(ex),
                        str(ey),
                    )
                )
                if decision is False:
                    return report(
                        "inequivalent",
                        tuple(records),
                        output=output,
                        work=builder.budget.used,
                        rules=tuple(sorted(builder.rules)),
                        elapsed_seconds=time.monotonic() - started,
                    )
            undecided |= not (
                fully_enumerated and concrete_answers and all(v is True for v in concrete_answers)
            )
        failure = (
            "condition_undecided" if domain_unknown else "cas_undecided" if undecided else None
        )
        if failure:
            raise MathematicalUndecidedError(
                "algebra.outputs",
                "original conditions are not established"
                if domain_unknown
                else "SymPy did not establish output identity",
                failure_kind=failure,
            )
        return report(
            "equivalent",
            tuple(records),
            work=builder.budget.used,
            rules=tuple(sorted(builder.rules)),
            elapsed_seconds=time.monotonic() - started,
        )
    except ComparisonError as exc:
        exc.diagnostics.update(
            queries=tuple(records),
            work=builder.budget.used,
            elapsed_seconds=time.monotonic() - started,
            rules=tuple(sorted(builder.rules)),
            definitions=tuple((index, repr(key)) for key, index in builder.definitions.items()),
        )
        if isinstance(exc, UnsupportedJaxprError):
            raise UnsupportedJaxprError(
                exc.path,
                exc.reason,
                stage="algebra",
                failure_kind="unsupported_conversion",
                **exc.diagnostics,
            ) from exc
        raise
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.sympy",
            f"{type(exc).__name__}: {exc}",
            stage="algebra",
            queries=tuple(records),
            work=builder.budget.used,
            elapsed_seconds=time.monotonic() - started,
        ) from exc


def isolated_compare(left, right, *, algebra_budget=10000, timeout=10.0):
    if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
        raise ValueError("algebra_timeout must be a positive finite number")
    started = time.monotonic()
    try:
        payload = pickle.dumps((left, right, algebra_budget))
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.transport", f"symbolic transport failed: {exc}", stage="algebra"
        ) from exc
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "jax_reorder.algebra_trusted"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        raise ComparisonInternalError(
            "algebra.sympy.worker", f"worker could not start: {exc}", stage="algebra"
        ) from exc
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise ComparisonTimeoutError(
            "algebra.sympy.timeout",
            f"symbolic worker exceeded {timeout:g}s",
            timeout=timeout,
            elapsed_seconds=time.monotonic() - started,
        ) from exc
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.sympy.worker", f"worker communication failed: {exc}", stage="algebra"
        ) from exc
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if process.returncode:
        raise ComparisonInternalError(
            "algebra.sympy.worker",
            f"worker exited {process.returncode}: {stderr.decode(errors='replace')[-1000:]}",
            stage="algebra",
            elapsed_seconds=time.monotonic() - started,
        )
    try:
        result = pickle.loads(stdout)
        if isinstance(result, WorkerFailure):
            if not isinstance(result.error_type, type) or not issubclass(
                result.error_type, ComparisonError
            ):
                raise TypeError("invalid symbolic worker exception type")
            error = result.error_type(
                result.path,
                result.reason,
                stage=result.stage,
                failure_kind=result.failure_kind,
                **result.diagnostics,
            )
            error.diagnostics["elapsed_seconds"] = time.monotonic() - started
        elif not isinstance(result, SympyResult) or result.status not in (
            "equivalent",
            "inequivalent",
        ):
            raise TypeError("invalid symbolic worker result")
    except Exception as exc:
        raise ComparisonInternalError(
            "algebra.sympy.worker", f"invalid worker response: {exc}", stage="algebra"
        ) from exc
    if isinstance(result, WorkerFailure):
        raise error from RemoteWorkerError(result.traceback)
    return replace(result, elapsed_seconds=time.monotonic() - started)


def _worker_main():
    # 使用规范模块名序列化类型，父进程无需导入 __main__ 中的类。
    from jax_reorder.algebra_trusted import WorkerFailure, compare_symbolic

    try:
        payload = pickle.load(sys.stdin.buffer)
        result = compare_symbolic(*payload)
    except Exception as exc:
        error = (
            exc
            if isinstance(exc, ComparisonError)
            else ComparisonInternalError(
                "algebra.sympy.worker", f"{type(exc).__name__}: {exc}", stage="algebra"
            )
        )
        result = WorkerFailure(
            type(error),
            error.path,
            error.reason,
            error.stage,
            error.failure_kind,
            error.diagnostics,
            traceback.format_exc(),
        )
    pickle.dump(result, sys.stdout.buffer)


if __name__ == "__main__":
    _worker_main()
