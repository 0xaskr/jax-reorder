"""Probe compiler IR reasoning without compiling or evaluating model inputs."""

import argparse
import json


def mlir_probe():
    import jaxlib
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir, passmanager

    programs = {
        "factored": """module {
          func.func @main(%x: i32, %y: i32, %z: i32) -> i32 {
            %a = arith.addi %x, %y : i32
            %b = arith.muli %a, %z : i32
            return %b : i32
          }
        }""",
        "distributed": """module {
          func.func @main(%x: i32, %y: i32, %z: i32) -> i32 {
            %a = arith.muli %x, %z : i32
            %b = arith.muli %y, %z : i32
            %c = arith.addi %a, %b : i32
            return %c : i32
          }
        }""",
        "add_zero": """module {
          func.func @main(%x: i32) -> i32 {
            %zero = arith.constant 0 : i32
            %a = arith.addi %x, %zero : i32
            return %a : i32
          }
        }""",
        "identity": """module {
          func.func @main(%x: i32) -> i32 {
            return %x : i32
          }
        }""",
        "stablehlo_factored": """module {
          func.func @main(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>,
                          %z: tensor<2x2xi32>) -> tensor<2x2xi32> {
            %a = stablehlo.add %x, %y : tensor<2x2xi32>
            %b = stablehlo.multiply %a, %z : tensor<2x2xi32>
            return %b : tensor<2x2xi32>
          }
        }""",
        "stablehlo_distributed": """module {
          func.func @main(%x: tensor<2x2xi32>, %y: tensor<2x2xi32>,
                          %z: tensor<2x2xi32>) -> tensor<2x2xi32> {
            %a = stablehlo.multiply %x, %z : tensor<2x2xi32>
            %b = stablehlo.multiply %y, %z : tensor<2x2xi32>
            %c = stablehlo.add %a, %b : tensor<2x2xi32>
            return %c : tensor<2x2xi32>
          }
        }""",
    }
    rows = []
    with mlir.make_ir_context():
        modules = {name: ir.Module.parse(source) for name, source in programs.items()}
        for name, left, right in (
            ("E2_arith", "factored", "distributed"),
            ("E2_stablehlo", "stablehlo_factored", "stablehlo_distributed"),
            ("identity_control", "add_zero", "identity"),
        ):
            lhs, rhs = modules[left].operation, modules[right].operation
            flags = ir.OperationEquivalenceFlags.IGNORE_LOCATIONS
            before = lhs.is_structurally_equivalent(rhs, flags)
            pipeline = passmanager.PassManager.parse("builtin.module(canonicalize,cse)")
            pipeline.run(lhs)
            pipeline.run(rhs)
            after = lhs.is_structurally_equivalent(rhs, flags)
            rows.append(
                {
                    "case": name,
                    "syntax_valid": lhs.verify() and rhs.verify(),
                    "same_structure_before": before,
                    "same_structure_after": after,
                    "math_result": "equivalent_by_upstream_rewrite" if after else "unknown",
                }
            )
    return {
        "backend": "jaxlib MLIR",
        "version": jaxlib.__version__,
        "domain": "integer IR; no model execution or handwritten rewrite patterns",
        "results": rows,
    }


def tvm_probe():
    import tvm

    # Current wheels renamed arith/tir to sym/tirx; retain the observed API in the report.
    analyzer_module = tvm.sym if hasattr(tvm, "sym") else tvm.arith
    tir = tvm.tirx if hasattr(tvm, "tirx") else tvm.tir
    analyzer = analyzer_module.Analyzer()
    rows = []

    def equality(name, left, right, domain):
        proved = analyzer.can_prove_equal(left, right)
        rows.append(
            {
                "case": name,
                "domain": domain,
                "result": "equivalent" if proved else "unknown",
                "difference_after_simplify": str(analyzer.simplify(left - right)),
            }
        )

    x, y, z = [tir.Var(name, "int64") for name in ("x", "y", "z")]
    equality("E2_integer", (x + y) * z, x * z + y * z, "int64 arithmetic")
    equality("N1_unconstrained", 2 * x, x, "int64 arithmetic")
    with analyzer.constraint_scope(x > 0):
        rows.append(
            {
                "case": "N1_positive_subdomain",
                "domain": "x > 0; satisfiable subdomain",
                "proved_inequality": analyzer.can_prove(2 * x != x),
            }
        )
    n, i = tir.Var("n", "int64"), tir.Var("i", "int64")
    left = tir.if_then_else(i < n, 1, 0)
    right = tir.if_then_else(i <= n, 1, 0)
    with analyzer.constraint_scope(i == n):
        rows.append(
            {
                "case": "N4_boundary",
                "domain": "i == n; satisfiable subdomain",
                "proved_inequality": analyzer.can_prove(left != right),
            }
        )

    a, b, s = [tir.Var(name, "float32") for name in ("a", "b", "s")]
    equality("E2_float", (a + b) * s, a * s + b * s, "TVM float32 expression semantics")
    equality("E5", tir.exp(a + b), tir.exp(a) * tir.exp(b), "TVM float32 exp semantics")
    p = tvm.te.placeholder((n,), name="p")
    v = tvm.te.placeholder((n,), name="v")
    k = tvm.te.reduce_axis((0, n), name="k")
    equality(
        "E3_symbolic_contraction",
        tvm.te.sum(p[k] * v[k], axis=k) * s,
        tvm.te.sum((p[k] * s) * v[k], axis=k),
        "symbolic-length float32 contraction; no element enumeration",
    )
    return {
        "backend": "Apache TVM analyzer",
        "version": tvm.__version__,
        "api_module": analyzer_module.__name__,
        "results": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("mlir", "tvm"))
    options = parser.parse_args()
    result = mlir_probe() if options.backend == "mlir" else tvm_probe()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
