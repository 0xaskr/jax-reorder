"""编译器与符号分析探针：直接对 IR 或表达式做等价判定实验。

运行（与 docs/development-docs/equivalence-backends/compiler.md 一致）：
  python -m tools.experiments.equivalence_backends.compiler_probe mlir
  <隔离环境>/python tools/experiments/equivalence_backends/compiler_probe.py tvm

mlir 使用 jaxlib 自带的 MLIR 绑定解析手写 IR，运行
`builtin.module(canonicalize,cse)` 后做结构比较；tvm 使用隔离环境中的
TVM analyzer 证明整数/浮点表达式关系。两者都不编译或执行模型，也不读取
jaxpr。结论仅覆盖列出的局部恒等式：规范化后不同不等于数学不等价，整数
或浮点 IR 的推理域也不同于真实 JAX 语义。输出为 JSON；依赖缺失时导入
错误向上传播，退出码非零。
"""

import argparse
import json


def mlir_probe():
    """对三组手写 MLIR/StableHLO 模块做规范化前后的结构比较。

    返回 backend/version/domain/results 字典；每个 case 记录语法校验、
    规范化前/后的结构相等和结论字符串。jaxlib 依赖在函数内导入，缺失时
    直接抛错；不执行模型或生成机器码。
    """
    import jaxlib
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir, passmanager

    # 前两组分别比较 arith 与 StableHLO 的分配律；第三组是消零控制例。
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
            # 只忽略源码 location；类型、属性与参数仍参与结构比较。
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
                    # 只有上游规则确实把两图化为同一结构，才记为等价。
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
    """用 TVM analyzer 证明若干整数/浮点/符号长度表达式关系。

    返回 backend/version/API 模块与逐 case 结果。未证明时状态为 unknown，
    不能解释为不等价；浮点用例只描述该 API 的默认推理域。隔离环境缺少
    tvm 时导入失败，退出码非零。
    """
    import tvm

    # 当前 wheel 把 arith/tir 改名为 sym/tirx；报告中保留实际检测到的 API。
    analyzer_module = tvm.sym if hasattr(tvm, "sym") else tvm.arith
    tir = tvm.tirx if hasattr(tvm, "tirx") else tvm.tir
    analyzer = analyzer_module.Analyzer()
    rows = []

    def equality(name, left, right, domain):
        """查询两表达式是否可证相等；不可证时记为 unknown 并保留化简差。"""
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
        # 可满足子域上的不等关系可证明；这不等于全实数域结论。
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
    """解析 backend 并输出对应 JSON；无文件副作用，退出码由异常决定。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("mlir", "tvm"))
    options = parser.parse_args()
    result = mlir_probe() if options.backend == "mlir" else tvm_probe()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
