"""Symbolic SMT probes, with no JAX/model execution or random sampling.

Install z3-solver and cvc5 in an isolated environment, then run this file.
Tensor reductions are explicitly encoded at contraction length 3. They are
universal in their real-valued entries, not proofs for arbitrary tensor sizes.
The encodings are written for this experiment; neither solver imports jaxpr.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import time


def run_backend(name: str, timeout_ms: int) -> dict:
    """Ask whether a counterexample exists; unsat proves the encoded equality."""
    api = importlib.import_module("z3" if name == "z3" else "cvc5.pythonic")
    version = importlib.metadata.version("z3-solver" if name == "z3" else "cvc5")
    results = []

    def check(case, left, right, assumptions=(), *, scope, budget=False):
        solver = api.Solver()
        solver.set("timeout" if name == "z3" else "tlimit-per", timeout_ms)
        if budget:
            solver.set("rlimit" if name == "z3" else "rlimit-per", 1)
        solver.add(*assumptions, left != right)
        start = time.monotonic()
        answer = solver.check()
        result = {
            "case": case,
            "solver_result": str(answer),
            "status": "unknown",
            "scope": scope,
            "assumptions": [str(x) for x in assumptions],
            "elapsed_seconds": round(time.monotonic() - start, 6),
        }
        if answer == api.unsat:
            result["status"] = "equivalent"
        elif answer == api.sat:
            result["status"] = "inequivalent"
            result["witness"] = str(solver.model())
        else:
            result["reason"] = str(solver.reason_unknown())
        results.append(result)

    x, y, z, scale = api.Reals("x y z scale")
    check("E2", (x + y) * z, x * z + y * z, scope="all real x,y,z")

    q = api.Reals("q0 q1 q2")
    k = api.Reals("k0 k1 k2")
    dot = sum(a * b for a, b in zip(q, k, strict=True))
    scaled_dot = sum((scale * a) * b for a, b in zip(q, k, strict=True))
    check(
        "E3",
        dot * scale,
        scaled_dot,
        scope="one arbitrary output of QK^T; contraction size 3; real entries",
    )
    # Dot and broadcast-multiply/reduce are independently expanded from their
    # standard finite-sum definitions. Reverse terms to require real algebra.
    p = api.Reals("p0 p1 p2")
    v = api.Reals("v0 v1 v2")
    dot_pv = sum(p[t] * v[t] for t in range(3))
    broadcast_reduce = sum(p[t] * v[t] for t in reversed(range(3)))
    check(
        "E4",
        dot_pv,
        broadcast_reduce,
        scope="one arbitrary output of PV; token size 3; real entries",
    )

    # E6 is deliberately a partial softmax lemma. The adapter must separately
    # justify finite exp(x)>0 and masked exp(-inf)=0. No exp identity is assumed.
    w0, w1, common_factor = api.Reals("w0 w1 common_factor")
    m0, m1 = api.Bools("mask0 mask1")
    a = api.If(m0, w0, 0)
    b = api.If(m1, w1, 0)
    total = a + b
    scaled_total = common_factor * a + common_factor * b
    normalized = api.If(total == 0, 0, a / total)
    normalized_scaled = api.If(scaled_total == 0, 0, common_factor * a / scaled_total)
    conditions = (w0 > 0, w1 > 0, common_factor > 0)
    check(
        "E6_weights_only",
        normalized,
        normalized_scaled,
        conditions,
        scope="two-token guarded normalization including both masks false",
    )
    check(
        "E6_all_mask",
        normalized,
        api.RealVal(0),
        (*conditions, api.Not(m0), api.Not(m1)),
        scope="all-masked row maps to zero by the explicit guard",
    )

    check(
        "N1",
        dot * scale,
        dot,
        (scale > 0, scale != 1),
        scope="omitted positive nonunit scale; contraction size 3",
    )
    a00, a01, a10 = api.Reals("a00 a01 a10")
    check(
        "N2",
        a00 + a01,
        a00 + a10,
        scope="first output of row versus column reduction of a 2x2 tensor",
    )
    check("N3", api.If(x > 0, x, 0), x, scope="removed real-valued ReLU")
    index, length, size = api.Ints("index length size")
    check(
        "N4",
        api.If(index < length, x, 0),
        api.If(index <= length, x, 0),
        (size > 0, index >= 0, index < size, length >= 0, length <= size),
        scope="symbolic valid tensor index and dynamic sequence length",
    )

    if name == "cvc5":
        exp = api.Exponential
        check(
            "E5",
            exp(x + y),
            exp(x) * exp(y),
            scope="native cvc5 real exponential; no added exp axioms",
        )
        check(
            "N5",
            exp(x + y),
            exp(x) + exp(y),
            scope="native cvc5 real exponential; no added exp axioms",
        )
        check(
            "N5_constrained_witness",
            exp(x + y),
            exp(x) + exp(y),
            (x == 0, y == 0),
            scope="exact symbolic witness check using native exp(0), not sampling",
        )
        e0 = api.If(m0, exp(x), 0)
        e1 = api.If(m1, exp(y), 0)
        shifted0 = api.If(m0, exp(x - z), 0)
        shifted1 = api.If(m1, exp(y - z), 0)
        denominator = e0 + e1
        shifted_denominator = shifted0 + shifted1
        check(
            "E6_native_exp_shift",
            api.If(denominator == 0, 0, e0 / denominator),
            api.If(shifted_denominator == 0, 0, shifted0 / shifted_denominator),
            scope=(
                "two real logits, arbitrary real shift and masks, guarded zero row; "
                "mask-to-zero encoding supplied by experiment, not inferred from -inf"
            ),
        )
    else:
        for case in ("E5", "N5"):
            results.append(
                {
                    "case": case,
                    "status": "unknown",
                    "reason": "no real exp semantic encoder in this Z3 probe",
                }
            )
        # A free exp function has more interpretations than the real exp.
        # Its sat result cannot witness mathematical inequivalence of real exp.
        free_exp = api.Function("uninterpreted_exp", api.RealSort(), api.RealSort())
        solver = api.Solver()
        solver.add(free_exp(x + y) != free_exp(x) * free_exp(y))
        answer = solver.check()
        results.append(
            {
                "case": "U1_exp_as_uninterpreted_function",
                "solver_result": str(answer),
                "status": "unknown",
                "reason": "uninterpreted function countermodel is not a real exp witness",
            }
        )

    check(
        "U1_resource_limit",
        x * x,
        api.RealVal(2),
        (x > 0,),
        scope="one-resource-unit solver budget; failure is not inequivalence",
        budget=True,
    )
    return {"backend": name, "version": version, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("z3", "cvc5", "all"), default="all")
    parser.add_argument("--timeout-ms", type=int, default=3000)
    args = parser.parse_args()
    backends = ("z3", "cvc5") if args.backend == "all" else (args.backend,)
    print(json.dumps([run_backend(name, args.timeout_ms) for name in backends], indent=2))


if __name__ == "__main__":
    main()
