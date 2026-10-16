"""把旧用例已证明结构等价的 IR 成对送入代数层，检验蕴含关系。"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from lax_cases import cases, trace_case

from jax_reorder import (
    CorrespondenceError,
    Status,
    canonicalize,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_jaxprs,
    verify_correspondence,
)


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_primitive_structural_pair_bypasses_mathematical_queries(case, monkeypatch):
    """涵盖旧原语清单的全部数值域及区域，代数预算不能否定结构证据。"""
    original = trace_case(case)
    normalized = canonicalize(original)
    structural = compare_jaxprs(original, normalized.jaxpr)
    assert structural.status == Status.EQUIVALENT, structural

    def forbidden(*args, **kwargs):
        raise AssertionError("structural proof must not invoke the mathematical worker")

    from jax_reorder import algebra_trusted

    monkeypatch.setattr(algebra_trusted, "isolated_compare", forbidden)
    result = compare_algebraic_jaxprs(original, normalized.jaxpr, algebra_budget=1)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "structural"
    assert result.work == 0 and not result.warnings
    assert (
        verify_correspondence(
            original, normalized.jaxpr, result.correspondence, derivative_scope="primal"
        )
        is None
    )


def test_structural_pair_with_exponential_expansion_uses_existing_evidence():
    """相同巨大多项式已由图证据确认，不应再受代数展开的资源限制。"""

    def huge(x):
        for _ in range(30):
            x = x * x
        return x

    x = jax.ShapeDtypeStruct((), jnp.float32)
    graph = jax.make_jaxpr(huge)(x)
    result = compare_algebraic_jaxprs(graph, graph, algebra_budget=1)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "structural" and result.work == 0
    assert (
        verify_correspondence(graph, graph, result.correspondence, derivative_scope="primal")
        is None
    )


def test_function_structural_pair_passes_all_requested_modes_with_tiny_algebra_budget():
    """函数契约核对后，前向和显式求导模式各自保留结构等价依据。"""

    def left(a, b):
        x = jnp.sin(a)
        y = jnp.cos(b)
        return x * y

    def right(a, b):
        y = jnp.cos(b)
        x = jnp.sin(a)
        return x * y

    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    result = compare_functions(left, right, x, x, modes=("forward", "jvp", "vjp"), algebra_budget=1)
    assert result.status == Status.EQUIVALENT, result
    assert result.equivalence_kind == "structural"
    assert all(mode.equivalence_kind == "structural" for mode in result.modes)


@pytest.mark.parametrize("change", ["inputs", "outputs", "source", "missing"])
def test_structural_correspondence_is_independently_checked(change):
    """直通证据也必须绑定原图，错误端口和缺失证据均不能被放行。"""
    x = jax.ShapeDtypeStruct((3,), jnp.float32)
    graph = jax.make_jaxpr(lambda a, b: (a - b, a * b))(x, x)
    result = compare_algebraic_jaxprs(graph, graph)
    assert result.status == Status.EQUIVALENT, result
    proof = result.correspondence
    other = graph
    if change == "inputs":
        proof = replace(proof, inputs=(1, 0))
    elif change == "outputs":
        proof = replace(proof, outputs=(1, 0))
    elif change == "source":
        other = jax.make_jaxpr(lambda a, b: (a + b, a * b))(x, x)
    else:
        proof = None
    with pytest.raises(CorrespondenceError):
        verify_correspondence(graph, other, proof, derivative_scope="primal")
