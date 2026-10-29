"""Real complete KDA kernels: pinned history, fixed AI cases and honest blockers."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
    compare_pallas_kernels,
)
from tests.pallas_oracle import run_worker
from tests.upstream_tpu import kda_cases as cases
from tests.upstream_tpu.kda_oracle import samples
from tests.upstream_tpu.sources import read_manifest, read_source

pytestmark = pytest.mark.business
TRACE_CASES = [
    (revision, variant, config)
    for config in cases.CONFIGURATIONS
    for revision, variant in cases.CANDIDATES
    if cases.supports_configuration(revision, config)
]


def test_source_provenance_and_per_revision_helpers():
    manifest = read_manifest(cases.REPO)
    assert manifest["revision"] == cases.REVISION
    assert manifest["revisions"]["latest"]["mega_last_change"].startswith("c500d1bb")
    assert set(manifest["revisions"]) == {"latest", *cases.HISTORY}
    for key, entry in manifest["files"].items():
        assert entry["revision"] == manifest["revisions"][key.split("_")[0]]["revision"]
        read_source(cases.REPO, key)
    assert "Apache License" in read_source(cases.REPO, "latest_license")
    for revision, variant in cases.CANDIDATES:
        namespace = cases.source_namespace(revision, variant)
        globals_ = namespace["_bwd_mega_kernel"].__globals__
        assert globals_["compute_m1_recompute"] is namespace["compute_m1_recompute"]
        if variant in cases.AI_EXPECTATIONS and variant != "cumsum_forward":
            assert globals_["compute_dav"] is getattr(cases, variant)
        if variant == "cumsum_forward":
            assert globals_["compute_reverse_cumsum_dg_scan"] is cases.cumsum_forward
    assert (
        cases.source_namespace("latest")["compute_m1_recompute"]
        is not (cases.source_namespace("jul30")["compute_m1_recompute"])
    )


@pytest.mark.parametrize(
    "revision,variant,config",
    TRACE_CASES,
    ids=[f"{r}-{v}-{c.name}" for r, v, c in TRACE_CASES],
)
def test_complete_native_launcher_trace(revision, variant, config):
    graph = cases.trace(revision, variant, config)
    (call,) = tuple(cases.pallas_calls(graph))
    assert call.params["interpret"] is False
    assert tuple(call.params["grid_mapping"].grid) == ()
    assert len(graph.jaxpr.outvars) == len(cases.OUTPUTS) == 6
    expected = cases.signatures(config)
    assert tuple(v.aval.shape for v in graph.jaxpr.outvars) == (
        expected[0].shape,
        expected[1].shape,
        expected[2].shape,
        expected[3].shape,
        expected[6].shape,
        expected[10].shape,
    )
    counts = cases.primitive_counts(graph)
    assert counts["pallas_call"] == 1
    for primitive in ("run_scoped", "scan", "dma_start", "dma_wait", "dot_general", "swap"):
        assert counts[primitive] > 0


@pytest.mark.parametrize("revision", ["jul30"])
def test_older_revision_option_is_not_silently_dropped(revision):
    with pytest.raises(ValueError, match="no cast_dg_to_q_dtype"):
        cases.function(revision, config=cases.Configuration(dtype="bfloat16", cast_gate=True))


@pytest.mark.parametrize("revision,variant", cases.CANDIDATES)
def test_complete_structure_compares_history_and_fixed_ai_cases(revision, variant):
    result = compare_jaxprs(
        cases.trace(), cases.trace(revision, variant), include_derivative_rules=False
    )
    expected = (
        Status.EQUIVALENT
        if (revision, variant)
        in (("latest", "original"), ("oct09", "original"), ("latest", "dav_reordered"))
        else Status.DIFFERENT
    )
    assert result.status is expected
    if expected is Status.EQUIVALENT:
        assert result.correspondence is not None


@pytest.mark.parametrize("variant", ["original", "dav_wrong_scale"])
@pytest.mark.parametrize("stage", ["semantic", "mathematics"])
def test_original_kernel_unobserved_outputs_require_explicit_parent_boundary(variant, stage):
    # Dead get values now have no initialization obligation, but the default
    # local boundary still includes all nine outputs, including placeholders.
    base, candidate = cases.trace(), cases.trace(variant=variant)
    with pytest.raises(UnsupportedJaxprError, match="uninitialized tensor output/scratch"):
        if stage == "semantic":
            compare_pallas_kernels(base, candidate, equivalence="semantic", analysis_budget=100000)
        else:
            compare_algebraic_jaxprs(base, candidate, scope="device_local", algebra_budget=100000)


@pytest.mark.parametrize("variant", ["original", "dav_reordered"])
def test_original_kernel_parent_outputs_pass_complete_resource_checks(variant):
    result = compare_pallas_kernels(
        cases.trace(),
        cases.trace(variant=variant),
        equivalence="semantic",
        output_boundary="parent",
        analysis_budget=100000,
    )
    assert result.status is Status.EQUIVALENT
    assert len(result.left_protocol.projection.outputs) == len(cases.OUTPUTS) == 6
    assert result.left_protocol.projection.communication
    assert result.left_protocol.output_boundary == "parent"


@pytest.mark.parametrize(
    "variant,kind", [("dav_reordered", "structural"), ("dav_scale_inside", "algebraic")]
)
def test_extracted_dav_helper_positive_is_separate_from_complete_kernel_proof(variant, kind):
    """Diagnose the actual replaced helper on small tiles; retain whole-kernel checks above."""
    signatures = (
        jax.ShapeDtypeStruct((1, 4, 8), jnp.float32),
        jax.ShapeDtypeStruct((1, 4, 8), jnp.float32),
        jax.ShapeDtypeStruct((1, 4, 4), jnp.float32),
    )
    original = cases.source_namespace()["compute_dav"]
    candidate = getattr(cases, variant)
    graphs = [
        jax.make_jaxpr(functools.partial(function, scale=0.25))(*signatures)
        for function in (original, candidate)
    ]
    result = compare_algebraic_jaxprs(*graphs)
    assert result.status is Status.EQUIVALENT and result.equivalence_kind == kind
    if kind == "algebraic":
        assert result.warnings


def test_numpy_recurrence_adjoint_against_independent_autodiff():
    """Check all six manual-adjoint outputs with a separate JAX recurrence."""
    args, expected = samples(7)
    q, key, value, beta, _, _, cumulative, _, do, dht, initial = (
        jnp.asarray(x, jnp.float32) for x in args
    )
    raw_gate = jnp.diff(cumulative[0, 0], axis=0, prepend=jnp.zeros((1, 128))) * jnp.log(2.0)

    def objective(q, key, value, beta, gate, state):
        def step(state, xs):
            qt, kt, vt, bt, gt, dot = xs
            decayed = state * jnp.exp(gt)[:, None]
            residual = vt - kt @ decayed
            state = decayed + bt * jnp.outer(kt, residual)
            output = (qt @ state) * 128**-0.5
            return state, jnp.sum(output * dot)

        final, loss = jax.lax.scan(step, state, (q, key, value, beta, gate, do[0, 0]))
        return jnp.sum(loss) + jnp.sum(final * dht[0, 0, 0])

    actual = jax.grad(objective, argnums=tuple(range(6)))(
        q[0, 0], key[0, 0], value[0, 0], beta[0, 0], raw_gate, initial[0, 0, 0]
    )
    for i, (a, b) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(
            np.asarray(a),
            b.reshape(a.shape),
            rtol=3e-4,
            atol=3e-6,
            err_msg=cases.OUTPUTS[i],
        )


@pytest.mark.pallas_communication
@pytest.mark.parametrize("config", cases.CONFIGURATIONS, ids=lambda c: c.name)
def test_complete_outputs_in_official_tpu_cpu_interpreter(config):
    run_worker(
        "tests.upstream_tpu.kda_oracle",
        entry="regression_worker",
        kwargs={"config": dataclasses.asdict(config)},
        expected={"case": "kda_regression", "status": "ok"},
        devices=1,
        startup_timeout=180,
        execution_timeout=120,
    )
