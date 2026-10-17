"""在通用公式上审查上游符号组件的复用能力，不执行 JAX 编译。

运行（见 docs/development-docs/equivalence-backends/architecture-reuse.md）：
  uv run --no-sync --offline python \
      tools/experiments/equivalence_backends/architecture_reuse_probe.py \
      --sources-root /tmp/jax-reorder-architecture-eval \
      --deps /tmp/jax-reorder-architecture-eval/deps \
      --json /tmp/jax-reorder-architecture-eval/results.json

源码与可选依赖必须位于仓库之外；每个项目在子进程内运行并受墙钟限制，
父进程校验固定提交。可选 --jax2sympy-fork 的本地改动与固定上游源码分开
报告。这是转换器审查，不替代生产比较器；不编译也不 eager 执行 JAX 数组
运算，结论只覆盖列出的通用公式，不代表真实模型层已通过。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp
from jax._src import compiler, dispatch

REVISIONS = {
    "jax2sympy": "2cb6a6568716b26af0abd542a941f87ae0580e25",
    "nerva-sympy": "981739f5d94f7d4703d75b981e76b30bcda81f22",
    "tensorgrad": "59c85e4b34bc7f217383e9116d445c60e1b69ccf",
    "sympy2jax": "4345d478accc49783b8da5a8ca7a3aad65cd0782",
    "nerva-jax": "97b1c7934e0b4f7e57f69df2e8f6f6c1cd41785c",
}


def forbidden(*args, **kwargs):
    """拦截 JAX 后端编译与 eager 执行，确保审查不运行模型。"""
    raise AssertionError("JAX backend compilation and eager execution are forbidden")


def query(left, right):
    """比较两侧显式符号输出；只有在能证明全等时才返回 True。

    逐元素求残差并用 equals(0) 查询，Piecewise 先显式化分支条件。返回
    same_shape/all_equal/methods；存在 None 时 all_equal 也为 None，不能
    把“未证明”当成“不等价”。
    """
    left, right = np.asarray(left, dtype=object), np.asarray(right, dtype=object)
    if left.shape != right.shape:
        return {"same_shape": False, "all_equal": False}
    answers, methods = [], {"Expr.equals"}
    for a, b in zip(left.flat, right.flat, strict=True):
        a, b = sp.sympify(a), sp.sympify(b)
        residual = a - b
        if residual.has(sp.Piecewise):
            # 在查询残差前把有序分支条件显式化，避免依赖隐式化简。
            residual = sp.simplify(
                sp.piecewise_exclusive(sp.piecewise_fold(residual.rewrite(sp.Piecewise)), deep=True)
            )
            methods.add("rewrite+piecewise_fold+piecewise_exclusive+simplify")
        answer = residual.equals(0)
        answers.append(answer)
    answer = False if False in answers else None if None in answers else True
    return {"same_shape": True, "all_equal": answer, "methods": sorted(methods)}


def measure(outputs):
    """统计输出规模、自由符号、节点数与运算数，供报告横向对比。

    outputs 为符号数组序列；返回可 JSON 序列化的规模字典，并保留首个
    输出的截断字符串。只做结构统计，不做数值求值。
    """
    elements = [sp.sympify(x) for a in outputs for x in np.asarray(a).flat]
    symbols = set().union(*(e.free_symbols for e in elements))
    nodes = [node for e in elements for node in sp.preorder_traversal(e)]
    return {
        "output_shapes": [list(np.shape(a)) for a in outputs],
        "output_elements": len(elements),
        "free_symbols": len(symbols),
        "tree_nodes": len(nodes),
        "unique_nodes": len(set(nodes)),
        "count_ops": sum(int(sp.count_ops(e)) for e in elements),
        "first_output": str(elements[0])[:240] if elements else None,
    }


def jax2sympy_probe(*, convert=None, mappings=None):
    """审查 jax2sympy 在通用公式上的真实 jaxpr 转换输出。

    convert/mappings 可注入（用于本地 fork 审查），缺省导入固定上游版本。
    返回 primitive 映射数量、默认映射重复调用结果、逐 case 记录与 caveats。
    每个 case 记录真实输出契约、规模、NumPy 参照查询结果与耗时/内存；
    失败记为 error，不清除失败信息。
    """
    if convert is None:
        from jax2sympy import jaxpr_to_sympy_expressions as convert
    if mappings is None:
        from jax2sympy.primitive_mapping import primitive_to_sympy_op as mappings

    scalar = jax.ShapeDtypeStruct((), jnp.float32)
    vector = jax.ShapeDtypeStruct((3,), jnp.float32)
    matrix = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    records = []

    def audit(name, function, args, reference):
        """追踪一个通用函数，转换其输出并与独立 SymPy 参照比较。"""
        record = {"case": name, "scope": "actual_jaxpr_to_symbolic_outputs"}
        started = time.perf_counter()
        tracemalloc.start()
        try:
            graph = jax.make_jaxpr(function)(*args)
            record["jaxpr"] = str(graph)
            # 传入全新 var2sym 映射是公开 API 参数，不是修改上游源码。
            outputs, bindings, _, _ = convert(graph, var2sym={})
            inputs = [bindings[v] for v in graph.jaxpr.invars]
            expected = reference(*inputs)
            record.update(measure(outputs))
            record["reference_checks"] = [
                query(a, b) for a, b in zip(outputs, expected, strict=True)
            ]
            record["input_symbols"] = sum(np.size(a) for a in inputs)
            record["status"] = "converted"
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        finally:
            record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
            record["python_peak_bytes"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        records.append(record)

    audit(
        "distribution", lambda x, y, z: (x + y) * z, (scalar,) * 3, lambda x, y, z: [x * z + y * z]
    )
    audit(
        "exponential_addition",
        lambda x, y: jnp.exp(x + y),
        (scalar,) * 2,
        lambda x, y: [sp.exp(x.item()) * sp.exp(y.item())],
    )
    audit(
        "trigonometric_identity",
        lambda x: jnp.sin(x) ** 2 + jnp.cos(x) ** 2,
        (scalar,),
        lambda x: [sp.Integer(1)],
    )
    audit("changed_coefficient", lambda x: x + x, (scalar,), lambda x: [x])
    audit("ordered_outputs", lambda x, y: (y, x), (scalar,) * 2, lambda x, y: [x, y])
    audit("literal_output", lambda x: 2.0, (scalar,), lambda x: [sp.Integer(2)])
    audit("no_inputs", lambda: 2.0, (), lambda: [sp.Integer(2)])
    for coefficient in (1, 2):
        captured = np.full((3,), coefficient, dtype=np.float32)
        audit(
            f"captured_constant_{coefficient}",
            lambda x, c=captured: x + c,
            (vector,),
            lambda x, c=coefficient: [x + c],
        )
    audit(
        "select_branch_order",
        lambda x: jnp.where(x > 0, x, -x),
        (scalar,),
        lambda x: [sp.Piecewise((x.item(), x.item() > 0), (-x.item(), True))],
    )
    audit(
        "matmul",
        lambda a, b: a @ b,
        (matrix, jax.ShapeDtypeStruct((3, 4), jnp.float32)),
        lambda a, b: [a @ b],
    )
    audit(
        "batched_contraction",
        lambda a, b: jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,)))),
        (
            jax.ShapeDtypeStruct((2, 2, 3), jnp.float32),
            jax.ShapeDtypeStruct((2, 3, 2), jnp.float32),
        ),
        lambda a, b: [np.stack([a[i] @ b[i] for i in range(2)])],
    )
    audit(
        "broadcast_reduce",
        lambda x, y: jnp.sum(x * y[None, :], axis=1),
        (matrix, vector),
        lambda x, y: [np.sum(x * y[None, :], axis=1)],
    )
    audit("transpose", lambda x: x.T, (matrix,), lambda x: [x.T])
    audit(
        "nested_jit",
        jax.jit(lambda x: jnp.exp(x) + x),
        (scalar,),
        lambda x: [sp.exp(x.item()) + x.item()],
    )
    audit(
        "conditional_region",
        lambda x: jax.lax.cond(x > 0, lambda y: y + 1, lambda y: y - 1, x),
        (scalar,),
        lambda x: [sp.Piecewise((x.item() + 1, x.item() > 0), (x.item() - 1, True))],
    )
    audit(
        "empty_array",
        lambda x: jnp.exp(x),
        (jax.ShapeDtypeStruct((0,), jnp.float32),),
        lambda x: [x],
    )
    audit("real_log_exp", lambda x: jnp.log(jnp.exp(x)), (scalar,), lambda x: [x])
    audit("division_domain", lambda x: x / x, (scalar,), lambda x: [sp.Integer(1)])
    for length in (8, 32, 128, 512):
        audit(
            f"scale_{length}",
            lambda x: jnp.sum(x * x),
            (jax.ShapeDtypeStruct((length,), jnp.float32),),
            lambda x: [sum(x * x)],
        )

    graph = jax.make_jaxpr(lambda x: x + 1)(scalar)
    try:
        convert(graph)
        convert(graph)
        default_mapping = "two_calls_succeeded"
    except Exception as error:
        default_mapping = f"{type(error).__name__}: {error}"
    return {
        "primitive_mapping_count": len(mappings),
        "primitive_mappings": sorted(mappings),
        "default_mapping_repeat": default_mapping,
        "records": records,
        "caveats": [
            "reference_checks compare value expressions, not source definition domains",
            "division_domain deliberately exposes that the return type has no domain",
            "upstream symbols have no real assumption; no assumptions were injected",
            "python_peak_bytes includes tracing and query; it is not total process RSS",
        ],
    }


def nerva_probe():
    """审查 nerva-sympy 的公式函数，并用新的符号查询检查其恒等式。

    返回 scope、shape、assumptions、checks、规模指标与“未运行上游测试”
    的原因。上游测试工具会无条件导入 TensorFlow/Torch，因此本轮不运行；
    也不声称存在 jaxpr 转换器。
    """
    from nerva_sympy.activation_functions import sigmoid, sigmoid_derivative
    from nerva_sympy.matrix_operations import Diag, hadamard
    from nerva_sympy.softmax_functions import (
        log_softmax,
        log_softmax_jacobian,
        softmax,
        softmax_jacobian,
        stable_softmax,
    )

    x = sp.Matrix(1, 3, sp.symbols("x:3", real=True))
    shift = sp.Symbol("shift", real=True)
    y = softmax(x)
    checks = {
        "normalization": query([sum(y)], [1]),
        "shift_invariance": query(softmax(x + sp.ones(1, 3) * shift), y),
        "stable_softmax": query(stable_softmax(x), y),
        "softmax_jacobian": query(y.jacobian(x), softmax_jacobian(x)),
        "log_softmax_jacobian": query(log_softmax(x).jacobian(x), log_softmax_jacobian(x)),
        "jacobian_missing_outer_product": query(y.jacobian(x), Diag(y)),
        "hadamard_definition": query(hadamard(x, x), sp.Matrix([[v**2 for v in x]])),
        "sigmoid_derivative": query([sp.diff(sigmoid(x[0]), x[0])], [sigmoid_derivative(x[0])]),
    }
    return {
        "scope": "upstream_formula_functions_with_new_symbolic_queries",
        "shape": [1, 3],
        "assumptions": "finite real symbols",
        "checks": checks,
        "metrics": measure([y]),
        "upstream_test_suite_run": False,
        "reason": "upstream test utilities import TensorFlow and Torch unconditionally",
        "jax_converter_provided": False,
    }


def sympy_probe():
    """检查 SymPy 原生符号表示与指标求和转换，不是 JAX 转换器。

    返回矩阵求和转乘积、符号维度张量收缩与节点数；不枚举元素，也不声称
    已接入 jaxpr。
    """
    from sympy.tensor.array.expressions import ArrayContraction, ArraySymbol, ArrayTensorProduct

    m, n, k = sp.symbols("m n k", integer=True, positive=True)
    a, b = sp.MatrixSymbol("A", m, n), sp.MatrixSymbol("B", n, k)
    i, j, q = sp.symbols("i j q", integer=True)
    summed = sp.Sum(a[i, j] * b[j, q], (j, 0, n - 1))
    converted = sp.MatrixExpr.from_index_summation(summed)
    arr_a, arr_b = ArraySymbol("A", (m, n)), ArraySymbol("B", (n, k))
    contract = ArrayContraction(ArrayTensorProduct(arr_a, arr_b), (1, 2))
    return {
        "scope": "native_symbolic_representation_not_a_jax_converter",
        "matrix_sum_to_product": converted == a * b,
        "symbolic_dimensions": str(contract.shape),
        "array_expression": str(contract),
        "array_tree_nodes": sum(1 for _ in sp.preorder_traversal(contract)),
        "sum_tree_nodes": sum(1 for _ in sp.preorder_traversal(summed)),
        "enumerated_elements": False,
    }


def tensorgrad_probe():
    """检查 tensorgrad 手构符号张量的化简与相等 API 语义。

    返回 shape、化简后分配律结论与相等 API 说明；不枚举元素，也不声称
    存在 jaxpr 转换。
    """
    import tensorgrad as tg

    n, m, k = sp.symbols("n m k", integer=True, positive=True)
    a, b, c = tg.Variable("A", i=n, j=m), tg.Variable("B", j=m, k=k), tg.Variable("C", j=m, k=k)
    left, right = a @ (b + c), a @ b + a @ c
    simplified_left, simplified_right = left.full_simplify(), right.full_simplify()
    return {
        "scope": "hand_constructed_symbolic_tensors_not_a_jax_converter",
        "shape": {key: str(value) for key, value in simplified_left.shape.items()},
        "distribution_after_simplify": simplified_left == simplified_right,
        "equality_api": "Tensor.__eq__ delegates to graph isomorphism, not three-state CAS",
        "enumerated_elements": False,
    }


def sympy2jax_probe():
    """检查 sympy2jax 的公式→JAX 测试生成方向，并记录真实 jaxpr。

    返回公式往返比较、jaxpr、输出数量与“任意 jaxpr→SymPy”的支持说明。
    只覆盖本实验列出的公式，不代表完整转换器。
    """
    import sympy2jax

    x, y = sp.symbols("x y")
    formulas = [(x + y) ** 2, sp.sin(x) ** 2 + sp.cos(x) ** 2]
    module = sympy2jax.SymbolicModule(formulas, make_array=False)
    abstract = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(lambda a, b: module(x=a, y=b))(abstract, abstract)
    return {
        "scope": "SymPy_formulas_to_JAX_test_generation",
        "formula_roundtrip": query(formulas, module.sympy()),
        "jaxpr": str(graph),
        "output_count": len(graph.jaxpr.outvars),
        "arbitrary_jaxpr_to_sympy": False,
        "imported_source": sympy2jax.__file__,
    }


def nerva_jax_probe():
    """把 nerva-jax 的抽象追踪图转换为 SymPy，并与 nerva-sympy 函数比较。

    对固定 2×3 输入逐个函数对执行 make_jaxpr + 转换 + 查询；返回 scope、
    shape、逐 case 记录与“未运行整模型/非程序对比较器”的边界说明。
    """
    from jax2sympy import jaxpr_to_sympy_expressions as convert
    from nerva_jax import activation_functions as ja
    from nerva_jax import matrix_operations as jm
    from nerva_jax import softmax_functions as js
    from nerva_sympy import activation_functions as sa
    from nerva_sympy import matrix_operations as sm
    from nerva_sympy import softmax_functions as ss

    matrix = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    pairs = [
        ("softmax", js.softmax, ss.softmax, (matrix,)),
        ("stable_softmax", js.stable_softmax, ss.stable_softmax, (matrix,)),
        ("log_softmax", js.log_softmax, ss.log_softmax, (matrix,)),
        ("stable_log_softmax", js.stable_log_softmax, ss.stable_log_softmax, (matrix,)),
        ("sigmoid", ja.Sigmoid, sa.Sigmoid, (matrix,)),
        ("sigmoid_gradient", ja.Sigmoid_gradient, sa.Sigmoid_gradient, (matrix,)),
        ("tanh", ja.Hyperbolic_tangent, sa.Hyperbolic_tangent, (matrix,)),
        ("relu", ja.Relu, sa.Relu, (matrix,)),
        ("hadamard", jm.hadamard, sm.hadamard, (matrix, matrix)),
        (
            "matrix_product",
            jm.product,
            sm.product,
            (matrix, jax.ShapeDtypeStruct((3, 2), jnp.float32)),
        ),
    ]
    results = []
    for name, jax_fn, sympy_fn, args in pairs:
        record = {"case": name}
        started = time.perf_counter()
        try:
            graph = jax.make_jaxpr(jax_fn)(*args)
            record["primitives"] = sorted({e.primitive.name for e in graph.jaxpr.eqns})
            outputs, bindings, _, _ = convert(graph, var2sym={})
            symbolic_inputs = [sp.Matrix(bindings[v]) for v in graph.jaxpr.invars]
            record.update(query(outputs[0], sympy_fn(*symbolic_inputs)))
            record["status"] = "converted"
        except Exception as error:
            record.update(status="error", error=f"{type(error).__name__}: {error}")
        record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        results.append(record)
    return {
        "scope": "nerva_jax_abstract_trace_to_SymPy_compared_with_nerva_sympy",
        "shape": [2, 3],
        "records": results,
        "whole_model_run": False,
        "program_pair_comparator": False,
    }


def main():
    """运行父进程或单个 worker，输出汇总 JSON。

    --worker 模式只在子进程内：设置隔离搜索路径、阻断 JAX 编译/eager 执行，
    运行一个项目探针并打印其 JSON。父进程模式校验各项目固定提交，为每个
    项目启动带 45 秒墙钟限制的子进程，汇总退出码/超时/结果，并按需把
    报告写入 --json。参数非法时由 argparse 以退出码 2 结束。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-root", type=Path, required=True)
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--jax2sympy-fork", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--worker",
        choices=("jax2sympy", "nerva-sympy", "sympy", "tensorgrad", "sympy2jax", "nerva-jax"),
    )
    args = parser.parse_args()
    sources = args.sources_root.resolve()
    if args.worker:
        # 隔离环境路径优先，避免误用仓库内同名包或全局安装版本。
        search_paths = [
            str(args.jax2sympy_fork.resolve() if args.jax2sympy_fork else sources / "jax2sympy"),
            str(sources / "nerva-sympy" / "src"),
            str(sources / "nerva-jax" / "src"),
            str(sources / "tensorgrad"),
            str(sources / "sympy2jax"),
        ]
        if args.deps:
            search_paths.append(str(args.deps.resolve()))
        sys.path[:0] = search_paths
        probes = {
            "jax2sympy": jax2sympy_probe,
            "nerva-sympy": nerva_probe,
            "sympy": sympy_probe,
            "tensorgrad": tensorgrad_probe,
            "sympy2jax": sympy2jax_probe,
            "nerva-jax": nerva_jax_probe,
        }
        # 初始化 JAX 后端；worker 只 tracing，不编译也不执行生成的计算。
        jax.devices()
        # 阻断后端编译与 eager 执行，保证审查只读取表达式。
        with (
            patch.object(compiler, "backend_compile_and_load", forbidden),
            patch.object(dispatch, "apply_primitive", forbidden),
        ):
            result = probes[args.worker]()
        print(json.dumps(result, default=str))
        return

    report = {
        "jax_version": jax.__version__,
        "sympy_version": sp.__version__,
        "numpy_version": np.__version__,
        "jax_compiled": False,
        "jax_executed": False,
        "projects": {},
    }
    if args.jax2sympy_fork:
        # 本地 fork 单独记录文件摘要，不与固定上游结果混合。
        report["jax2sympy_fork"] = {
            "path": str(args.jax2sympy_fork.resolve()),
            "files_sha256": {
                name: hashlib.sha256(
                    (args.jax2sympy_fork / "jax2sympy" / name).read_bytes()
                ).hexdigest()
                for name in ("translate.py", "primitive_mapping.py")
            },
        }
    for project in ("jax2sympy", "nerva-sympy", "sympy", "tensorgrad", "sympy2jax", "nerva-jax"):
        if project in REVISIONS:
            # 父进程独立校验固定提交，不信任 sources 目录里的自报版本。
            revision = subprocess.check_output(
                ["git", "-C", str(sources / project), "rev-parse", "HEAD"], text=True
            ).strip()
            if revision != REVISIONS[project]:
                raise ValueError(f"{project}: expected {REVISIONS[project]}, got {revision}")
        else:
            revision = sp.__version__
        command = [sys.executable, __file__, "--sources-root", str(sources), "--worker", project]
        if args.deps:
            command.extend(["--deps", str(args.deps.resolve())])
        if args.jax2sympy_fork:
            command.extend(["--jax2sympy-fork", str(args.jax2sympy_fork.resolve())])
        started = time.perf_counter()
        record = {"revision": revision}
        try:
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=45,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            record["exit_code"] = completed.returncode
            record["stderr"] = completed.stderr[-3000:]
            if completed.returncode == 0:
                record["result"] = json.loads(completed.stdout)
            else:
                record["stdout"] = completed.stdout[-1000:]
        except subprocess.TimeoutExpired:
            record["status"] = "timeout"
        record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report["projects"][project] = record
    if args.deps:
        report["isolated_dependencies"] = {
            d.metadata["Name"]: d.version
            for d in importlib.metadata.distributions(path=[str(args.deps)])
        }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json:
        args.json.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
