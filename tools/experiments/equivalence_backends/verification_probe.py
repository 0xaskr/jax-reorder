"""符号 SMT 探针：不执行 JAX/模型，也不做随机采样。

运行（隔离环境安装 z3-solver 与 cvc5，见
docs/exec-plans/completed/research/equivalence-backends/verification.md）：
  <隔离环境>/python tools/experiments/equivalence_backends/verification_probe.py \
      --backend all --timeout-ms 3000

对实数/整数/布尔表达式查询“是否存在反例”：unsat 证明编码等式，sat 给出
具体模型，unknown 保留原因。张量归约显式按收缩长度 3 编码，只对符号元素
普遍成立，不是任意 shape 的证明；求解器不读取 jaxpr，mask-to-zero 等
编码也由实验提供。输出 JSON；--backend/--timeout-ms 控制后端范围与预算。
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import time


def run_backend(name: str, timeout_ms: int) -> dict:
    """询问是否存在反例；unsat 证明编码等式，返回该后端的完整结果。

    name 为 "z3" 或 "cvc5"，timeout_ms 为求解预算。返回 backend、version
    与逐 case 记录（status/scope/assumptions/耗时，sat 时含 witness）。
    导入缺失依赖直接抛错；不修改任何文件。
    """
    api = importlib.import_module("z3" if name == "z3" else "cvc5.pythonic")
    version = importlib.metadata.version("z3-solver" if name == "z3" else "cvc5")
    results = []

    def check(case, left, right, assumptions=(), *, scope, budget=False):
        """对单个 case 建求解器、加假设与“左右不等”查询并返回记录。

        left/right 为待查表达式，assumptions 为域条件，scope 说明该编码
        覆盖的语义范围，budget=True 时施加 1 资源单位限制以观察 unknown。
        结果写入 results：unsat→equivalent，sat→inequivalent（含模型），
        其余为 unknown 并记录原因。
        """
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
    # dot 与 broadcast-multiply/reduce 按各自有限求和定义独立展开；反转
    # token 顺序以要求实数代数（而非同序巧合）成立。
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

    # E6 故意只编码部分 softmax 引理；适配器须另行证明有限 exp(x)>0 与
    # 遮蔽 exp(-inf)=0，这里不假设任何 exp 恒等式。
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
        # 自由 exp 函数比实数 exp 有更多解释；其 sat 不能作为数学不等价的
        # 见证，因此 U1 保留 unknown。
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
    """解析 backend/timeout 参数，逐个后端运行探针并打印 JSON 列表。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("z3", "cvc5", "all"), default="all")
    parser.add_argument("--timeout-ms", type=int, default=3000)
    args = parser.parse_args()
    backends = ("z3", "cvc5") if args.backend == "all" else (args.backend,)
    print(json.dumps([run_backend(name, args.timeout_ms) for name in backends], indent=2))


if __name__ == "__main__":
    main()
