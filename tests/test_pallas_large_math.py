"""Public mathematical comparisons at pinned upstream attention/RoPE sizes."""

import pytest

from jax_reorder import Status, compare_algebraic_jaxprs
from tests.pallas_upstream_state_cases import trace


@pytest.mark.parametrize(
    "family,mode",
    [
        ("attention", "identity_mul"),
        ("attention", "wrong_scale"),
        ("quantized_rope", "identity_mul"),
        ("quantized_rope", "wrong_sign"),
        ("quantized_rope", "wrong_scale"),
    ],
)
def test_real_kernel_math_completes_under_default_budget(family, mode):
    result = compare_algebraic_jaxprs(trace(family), trace(family, mode), scope="device_local")
    expected = Status.EQUIVALENT if mode == "identity_mul" else Status.INEQUIVALENT
    assert result.status is expected


def test_complete_paged_attention_value_scale_has_exact_counterexample():
    from tests.pallas_paged_attention_cases import trace as paged_trace

    result = compare_algebraic_jaxprs(
        paged_trace(), paged_trace(inline=True, wrong=True), scope="device_local"
    )
    assert result.status is Status.INEQUIVALENT


@pytest.mark.parametrize("change", ["inline", "value", "page"])
def test_bounded_runtime_paged_attention_math_keeps_page_and_value_dependencies(change):
    from tests.pallas_paged_attention_cases import trace as paged_trace

    candidate = paged_trace(
        inline=True,
        bounded_pages=True,
        wrong=change == "value",
        wrong_page=change == "page",
    )
    result = compare_algebraic_jaxprs(
        paged_trace(bounded_pages=True), candidate, scope="device_local"
    )
    assert result.status is (Status.EQUIVALENT if change == "inline" else Status.INEQUIVALENT)
