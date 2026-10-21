"""包分层与导入图约束。

覆盖：
- 正例：包内全部模块可导入；公共 API 名称与 __all__ 保持稳定。
- 约束：实现按 errors → core → algebra → functions 单向依赖；core 不导入
  algebra/functions，algebra 不导入 functions；内部导入图无环。
- 这些是静态结构与导入检查，不验证算法语义；算法正确性由其余测试和开发记录覆盖。
"""

import ast
import importlib
from pathlib import Path

import jax_reorder

PACKAGE = Path(jax_reorder.__file__).parent
EXPECTED_PUBLIC = {
    "AlgebraComparison",
    "Canonicalized",
    "Comparison",
    "ComparisonBudgetError",
    "ComparisonContractError",
    "ComparisonError",
    "ComparisonStage",
    "ComparisonFailureKind",
    "ComparisonInternalError",
    "ComparisonTimeoutError",
    "CompilationError",
    "CorrespondenceError",
    "CoreExecutionBinding",
    "EquationCorrespondence",
    "FunctionComparison",
    "FunctionSignature",
    "FunctionTrace",
    "GraphCorrespondence",
    "InvalidJaxprError",
    "JaxprError",
    "LocalKernelComparison",
    "LocalKernelEvidence",
    "MathematicalUndecidedError",
    "ModeComparison",
    "Status",
    "TraceError",
    "UnsupportedJaxprError",
    "canonicalize",
    "compare_algebraic_jaxprs",
    "compare_function_traces",
    "compare_functions",
    "compare_jaxprs",
    "compare_pallas_kernels",
    "trace_function",
    "verify_correspondence",
    "verify_pallas_kernel_comparison",
}


def module_paths():
    """返回包内全部模块文件（不含 __pycache__）。"""
    return sorted(
        path
        for path in PACKAGE.rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "py.typed"
    )


def module_name(path):
    """把文件路径转成包的绝对模块名；__init__.py 归到所在包。"""
    parts = list(path.relative_to(PACKAGE).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return "jax_reorder" + ("." + ".".join(parts) if parts else "")


def imported_modules(path, *, include_type_checking=True):
    """解析包内导入；循环检查只看运行时依赖，分层检查也包含类型依赖。"""
    module = module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result = set()
    pending = [tree]
    while pending:
        node = pending.pop()
        if (
            not include_type_checking
            and isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            pending.extend(node.orelse)
            continue
        pending.extend(ast.iter_child_nodes(node))
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
                base = base[: len(base) - (node.level - 1)]
                if node.module:
                    result.add(".".join(base + [node.module]))
                else:
                    result.update(f"{'.'.join(base)}.{alias.name}" for alias in node.names)
            elif node.module and node.module.startswith("jax_reorder"):
                result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(
                alias.name for alias in node.names if alias.name.startswith("jax_reorder")
            )
    return result


def internal_graph(*, include_type_checking=True):
    """返回仅包含包内模块的导入图。"""
    paths = {module_name(path): path for path in module_paths()}
    graph = {}
    for module, path in paths.items():
        deps = {
            dep
            for dep in imported_modules(path, include_type_checking=include_type_checking)
            if dep in paths and dep != module
        }
        graph[module] = deps
    return graph


def test_all_modules_importable():
    names = [module_name(path) for path in module_paths()]
    for name in names:
        importlib.import_module(name)
    assert len(names) == len(set(names))


def test_import_graph_has_no_cycles():
    graph = internal_graph(include_type_checking=False)
    state = {}

    def visit(node):
        """0/1/2 分别表示未访问、在栈上、已完成；回到栈上节点即为环。"""
        if state.get(node) == 1:
            return False
        if state.get(node) == 2:
            return True
        state[node] = 1
        for dep in graph[node]:
            if not visit(dep):
                return False
        state[node] = 2
        return True

    for module in graph:
        assert visit(module), f"import cycle involving {module}"


def test_layering_direction():
    graph = internal_graph()
    for module, deps in graph.items():
        for dep in deps:
            top = dep.split(".")[1]
            if module.startswith("jax_reorder.core"):
                assert top not in {"algebra", "functions"}, f"core imports {dep}"
            if module.startswith("jax_reorder.algebra"):
                assert top != "functions", f"algebra imports {dep}"
            if module == "jax_reorder.errors":
                assert not dep.startswith("jax_reorder."), f"errors imports {dep}"


def test_public_api_surface_is_stable():
    assert set(jax_reorder.__all__) == EXPECTED_PUBLIC
    for name in EXPECTED_PUBLIC:
        assert hasattr(jax_reorder, name)
