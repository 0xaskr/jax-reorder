"""由 basedpyright 检查的调用方契约；不执行 tracing 或数学查询。"""

from typing import assert_type

import jax
from jax._src import core

from jax_reorder import (
    AlgebraComparison,
    Canonicalized,
    Comparison,
    ComparisonFailureKind,
    ComparisonStage,
    CoreExecutionBinding,
    FunctionComparison,
    FunctionTrace,
    LocalKernelComparison,
    UnsupportedJaxprError,
    canonicalize,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_jaxprs,
    compare_pallas_kernels,
    trace_function,
)
from jax_reorder.core.canonical import Graph


def exception_types() -> None:
    error = UnsupportedJaxprError("eqns[0]", "unsupported", query_count=2)
    assert_type(error.default_stage, ComparisonStage)
    assert_type(error.default_kind, ComparisonFailureKind)
    assert_type(error.stage, ComparisonStage)
    assert_type(error.failure_kind, ComparisonFailureKind)
    assert_type(error.path, str)
    assert_type(error.diagnostics, dict[str, object])
    assert_type(error.at("left"), UnsupportedJaxprError)


def graph_types(graph: core.Jaxpr) -> None:
    normalized = canonicalize(graph)
    assert_type(normalized, Canonicalized)
    assert_type(normalized.graph, Graph)
    assert_type(normalized.equation_order, tuple[int, ...])
    assert_type(normalized.lineage[graph], tuple[core.Jaxpr, tuple[int, ...]])
    assert_type(normalized.regions[graph], Canonicalized)
    assert_type(compare_jaxprs(graph, graph), Comparison)
    assert_type(compare_algebraic_jaxprs(graph, graph), AlgebraComparison)
    assert_type(compare_pallas_kernels(graph, graph), LocalKernelComparison)
    local = compare_pallas_kernels(graph, graph, check_protocol=True, execution_cores=1)
    assert_type(local.execution_cores, int | None)
    binding = CoreExecutionBinding((((0,),), ((1,),)))
    bound = compare_pallas_kernels(graph, graph, check_protocol=True, execution=binding)
    assert_type(bound.execution, CoreExecutionBinding | None)


def function_types() -> None:
    argument = jax.ShapeDtypeStruct((), "float32")
    trace = trace_function(lambda x: x, argument)
    assert_type(trace, FunctionTrace)
    assert_type(trace.abstract_inputs, tuple[jax.ShapeDtypeStruct, ...])
    assert_type(compare_functions(lambda x: x, lambda x: x, argument), FunctionComparison)
