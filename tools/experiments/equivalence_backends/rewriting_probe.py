"""Symbolic probes for upstream egglog rules and a minimal indexed adapter.

Run in an isolated environment with egglog==14.0.0 and numpy installed.
Optional --taso-source points to a checkout and additionally requires z3-solver.
No JAX layer is executed. Failure to merge is always unknown, never inequality.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

from egglog import EGraph, Expr, String, StringLike, eq, relation, rewrite, vars_


class Term(Expr):
    """Finite-real indexed terms; indices and symbolic extents remain explicit."""

    @classmethod
    def atom(cls, name: StringLike, indices: StringLike) -> Term: ...

    def __add__(self, other: Term) -> Term: ...

    def __mul__(self, other: Term) -> Term: ...

    def sum(self, axis: StringLike, extent: StringLike) -> Term: ...

    def exp(self) -> Term: ...


def check_pair(egraph, lhs, rhs, *, ruleset=None):
    egraph.register(lhs, rhs)
    if ruleset is None:
        egraph.run(8)
    else:
        egraph.run(8, ruleset=ruleset)
    return "equivalent_under_rules" if egraph.check_bool(eq(lhs).to(rhs)) else "unknown"


def upstream_array_probes():
    import egglog.exp.array_api as array

    x, y, z = [array.NDArray.var(n) for n in ("x", "y", "z")]
    q, kt, scale, p, v = [array.NDArray.var(n) for n in ("Q", "K_transposed", "s", "P", "V")]
    for tensor, shape in (
        (x, (2, 3)),
        (y, (2, 3)),
        (z, (2, 3)),
        (q, (2, 3)),
        (kt, (3, 4)),
        (scale, ()),
        (p, (2, 3)),
        (v, (3, 4)),
    ):
        array.assume_shape(tensor, shape)
        array.assume_dtype(tensor, array.float64)
        array.assume_isfinite(tensor)
    cases = {
        "E2": ((x + y) * z, x * z + y * z),
        "E3": ((q @ kt) * scale, (q * scale) @ kt),
        "E4": (
            p @ v,
            array.sum(array.expand_dims(p, 2) * array.expand_dims(v, 0), 1),
        ),
        "N1": ((q @ kt) * scale, q @ kt),
    }
    results = {
        name: {
            "status": check_pair(EGraph(), lhs, rhs, ruleset=array.array_api_ruleset),
            "rules": "unmodified upstream experimental array_api_ruleset",
            "assumptions": (
                "compatible fixed shapes, float64 dtype, finite elements; no values allocated"
            ),
        }
        for name, (lhs, rhs) in cases.items()
    }
    # The module exports egglog.exp via a wildcard import, not an ndarray exp op.
    results["E5"] = {
        "status": "unknown",
        "reason": "no callable array exp in the evaluated experimental API",
        "callable_array_exp": callable(getattr(array, "exp", None)),
    }
    return results


def indexed_adapter_probes():
    independent = relation("independent_of_axis", Term, String)
    a, b, c = vars_("a b c", Term)
    axis, extent = vars_("axis extent", String)
    q = Term.atom("Q", "i,k")
    k = Term.atom("K", "j,k")
    scale = Term.atom("s", "")
    p = Term.atom("P", "i,k")
    v = Term.atom("V", "k,j")

    def graph():
        result = EGraph()
        result.register(
            rewrite(a * b).to(b * a),
            rewrite((a * b) * c).to(a * (b * c)),
            rewrite(a.sum(axis, extent) * c).to((a * c).sum(axis, extent), independent(c, axis)),
            independent(scale, "k"),
        )
        return result

    # Both E4 frontends use the mathematical definition of contraction. This
    # demonstrates adapter encoding, not an upstream tensor theorem discovery.
    def dot_entry(left, right):
        return (left * right).sum("k", "tokens")

    results = {}
    cases = {
        "E3": ((q * k).sum("k", "features") * scale, ((q * scale) * k).sum("k", "features")),
        "E4": (dot_entry(p, v), (p * v).sum("k", "tokens")),
        "E5": ((q + k).exp(), q.exp() * k.exp()),
        "N1": ((q * k).sum("k", "features") * scale, (q * k).sum("k", "features")),
        "N2": ((p * v).sum("k", "tokens"), (p * v).sum("j", "channels")),
        "guard_dependent_factor": (
            (q * k).sum("k", "features") * q,
            ((q * k) * q).sum("k", "features"),
        ),
    }
    for name, (lhs, rhs) in cases.items():
        results[name] = {
            "status": check_pair(graph(), lhs, rhs),
            "rules": "adapter-authored real multiplication AC and guarded finite-sum linearity",
            "assumptions": (
                "finite real entries; compatible shapes; finite symbolic reduction extents"
            ),
        }
    results["E4"]["qualification"] = (
        "same expression after definition-based lowering, not a new proof"
    )
    results["E5"]["reason"] = "exp is uninterpreted; target identity was not added as an axiom"
    results["U1"] = {
        "status": "unknown",
        "reason": "top_k, scan and mask semantics not implemented by adapter",
    }
    return results


def taso_probes(source: Path):
    import z3

    path = source / "verify" / "verify.py"
    # The upstream CLI ends with a Python 2 print statement. Its declarations
    # and axioms are Python 3 compatible; load them unchanged before __main__.
    definitions = path.read_text().split("if __name__ == '__main__':", 1)[0]
    taso = ModuleType("upstream_taso_verify")
    exec(compile(definitions, str(path), "exec"), taso.__dict__)
    q, k, scale = z3.Consts("probe_Q probe_K probe_s", taso.T)
    lhs = taso.scalar_mul_0(taso.matmul_0(q, taso.transpose_0(k)), scale)
    rhs = taso.matmul_0(taso.scalar_mul_0(q, scale), taso.transpose_0(k))
    results = {"z3_version": z3.get_version_string(), "upstream_axiom_count": len(taso.axioms)}
    for name, left, right in (("E3", lhs, rhs), ("N1", lhs, taso.matmul_0(q, taso.transpose_0(k)))):
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*(axiom for axiom, _ in taso.axioms))
        solver.add(left != right)
        answer = solver.check()
        # A model of uninterpreted tensor axioms need not be a concrete tensor
        # counterexample. Only unsat is usable as relative equality evidence.
        results[name] = {
            "status": "entailed_by_upstream_axioms" if answer == z3.unsat else "unknown",
            "solver_result": str(answer),
            "reason": solver.reason_unknown() if answer == z3.unknown else None,
            "qualification": (
                "conditional on TASO axioms and compatible shapes, not full tensor semantics"
            ),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taso-source", type=Path)
    args = parser.parse_args()
    result = {
        "egglog_version": version("egglog"),
        "model_executed": False,
        "upstream_array": upstream_array_probes(),
        "custom_indexed_adapter": indexed_adapter_probes(),
    }
    if args.taso_source:
        result["taso_axioms"] = taso_probes(args.taso_source)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
