"""上游 egglog 规则与最小索引适配器的符号探针。

运行（隔离环境，见 docs/development-docs/equivalence-backends/rewriting.md）：
  <隔离环境>/python tools/experiments/equivalence_backends/rewriting_probe.py
  <隔离环境>/python tools/experiments/equivalence_backends/rewriting_probe.py \
      --taso-source /tmp/jax-reorder-equivalence-eval/rewriting/taso

egglog 部分使用未修改的 `egglog.exp.array_api` 规则与自写索引适配器；
--taso-source 额外加载 TASO 的 Z3 公理。合并失败一律 unknown，绝不解释为
不等价；Z3 的 sat 只是公理模型，不是具体张量反例。不执行 JAX 层，输出
JSON 到 stdout。缺依赖时导入失败，退出码非零。
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

from egglog import EGraph, Expr, String, StringLike, eq, relation, rewrite, vars_


class Term(Expr):
    """有限实数索引项；索引与符号长度保持显式，不展开元素。"""

    @classmethod
    def atom(cls, name: StringLike, indices: StringLike) -> Term:
        """声明原子项：名字与显式索引字符串。"""
        ...

    def __add__(self, other: Term) -> Term:
        """声明逐项加法。"""
        ...

    def __mul__(self, other: Term) -> Term:
        """声明逐项乘法。"""
        ...

    def sum(self, axis: StringLike, extent: StringLike) -> Term:
        """声明沿 axis、长度为 extent 的符号求和。"""
        ...

    def exp(self) -> Term:
        """声明未解释的 exp 项，不附加恒等式公理。"""
        ...


def check_pair(egraph, lhs, rhs, *, ruleset=None):
    """注册两侧表达式并运行 8 轮规则，返回 equivalent_under_rules 或 unknown。

    egraph 为待查询图，lhs/rhs 为两侧符号项，ruleset 可选（缺省用全部
    注册规则）。只有 check_bool 成功才判等价；失败仍是 unknown。
    """
    egraph.register(lhs, rhs)
    if ruleset is None:
        egraph.run(8)
    else:
        egraph.run(8, ruleset=ruleset)
    return "equivalent_under_rules" if egraph.check_bool(eq(lhs).to(rhs)) else "unknown"


def upstream_array_probes():
    """用未修改的上游 array_api 规则检查固定/兼容 shape 的恒等式。

    为输入假设固定 shape、float64 与有限元素，但不分配数值；返回每个
    case 的状态、所用规则与假设。E5 因该实验 API 没有可调用的数组 exp
    而保持 unknown。
    """
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
    # 该模块通过 wildcard import 暴露 egglog.exp，不是数组 exp 操作。
    results["E5"] = {
        "status": "unknown",
        "reason": "no callable array exp in the evaluated experimental API",
        "callable_array_exp": callable(getattr(array, "exp", None)),
    }
    return results


def indexed_adapter_probes():
    """用自写索引适配器验证编码方式，而非上游张量定理发现。

    注册实数乘法交换/结合律与受保护的有限求和线性规则，检查 E3/E4/E5、
    负例与 guard 依赖用例。返回每 case 状态、规则来源与假设；不声称
    发现新定理。
    """
    independent = relation("independent_of_axis", Term, String)
    a, b, c = vars_("a b c", Term)
    axis, extent = vars_("axis extent", String)
    q = Term.atom("Q", "i,k")
    k = Term.atom("K", "j,k")
    scale = Term.atom("s", "")
    p = Term.atom("P", "i,k")
    v = Term.atom("V", "k,j")

    def graph():
        """构造只含适配器自写规则的 EGraph，避免把目标恒等式加为公理。"""
        result = EGraph()
        result.register(
            rewrite(a * b).to(b * a),
            rewrite((a * b) * c).to(a * (b * c)),
            rewrite(a.sum(axis, extent) * c).to((a * c).sum(axis, extent), independent(c, axis)),
            independent(scale, "k"),
        )
        return result

    # E4 的两个前端都使用收缩的数学定义；这里只演示适配器编码，不是上游
    # 张量定理发现。
    def dot_entry(left, right):
        """按收缩定义展开一个输出元素：沿 k 求和。"""
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
    """加载 TASO 的 Z3 公理，检查 E3 的蕴含与 N1 的反例查询。

    source 指向 TASO checkout；返回 Z3 版本、公理数量与每个 case 的求解
    结果。只有 unsat 可作为相对等价证据；sat 是未解释公理模型，不算具体
    张量反例。求解超时保留原因。
    """
    import z3

    path = source / "verify" / "verify.py"
    # 上游 CLI 末尾是 Python 2 print 语句；其声明与公理兼容 Python 3，
    # 因此在 __main__ 之前原样加载定义。
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
        # 未解释张量公理的模型未必对应具体张量反例；只有 unsat 可用作相
        # 对等价证据。
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
    """解析参数并输出 egglog/适配器（可选 TASO）JSON 报告；不执行模型。"""
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
