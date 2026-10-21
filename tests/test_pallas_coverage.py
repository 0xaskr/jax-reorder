"""Source coverage is measured on real traces, separately from execution."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax.experimental import pallas as pl

from jax_reorder import (
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_jaxprs,
    verify_correspondence,
)
from jax_reorder.core.adapters.pallas.structural import PARAMETER_SCHEMAS
from tests.pallas_coverage_cases import CASES, context, equations, finish, function, trace_case
from tests.pallas_inventory import inventory


def test_public_api_inventory_has_no_unclassified_exports():
    assert inventory()
    # Every native schema must name a real tracing case below, so extending the
    # admission registry alone cannot silently count as new source coverage.
    covered = {name for case in CASES for name in case.primitives} | {"num_programs"}
    assert {primitive.name for primitive in PARAMETER_SCHEMAS} <= covered


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_official_surface_with_reordered_and_wrong_candidates(case):
    with context(case):
        a, b, wrong = (trace_case(case, v) for v in ("base", "reorder", "wrong"))
        actual = list(equations(a))
        assert set(case.primitives) <= {e.primitive.name for e in actual}
        for eq in actual:
            if eq.primitive in PARAMETER_SCHEMAS:
                assert eq.params.keys() == PARAMETER_SCHEMAS[eq.primitive]
        # All candidates contain an actual sin/cos order change in a live region.
        assert [e.primitive for e in actual] != [e.primitive for e in equations(b)]
        result = compare_jaxprs(a, b, search_budget=100)
        assert result.status is Status.EQUIVALENT
        assert result.correspondence is not None
        verify_correspondence(a, b, result.correspondence)
        core.check_jaxpr(result.left.jaxpr)
        assert compare_jaxprs(a, wrong, search_budget=100).status is Status.DIFFERENT


@pytest.mark.parametrize("case", [c for c in CASES if c.execute], ids=lambda case: case.name)
def test_cpu_interpreter_checks_both_candidates(case):
    outputs = []
    for variant in ("base", "reorder", "wrong"):
        fn, (sig,) = function(case, variant, interpret=True)
        x = jnp.linspace(0.1, 0.9, np.prod(sig.shape), dtype=jnp.float32).reshape(sig.shape)
        with context(case):
            outputs.append(np.asarray(fn(x)))
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=2e-6, atol=2e-6)
    assert np.isfinite(outputs[0]).all()
    assert not np.allclose(outputs[0], outputs[2], rtol=1e-3, atol=1e-3)


def test_dynamic_num_programs_is_not_only_static_frontend_folding():
    sig = jax.ShapeDtypeStruct((16, 128), jnp.float32)

    def fn(variant):
        def outer(n, x):
            spec = pl.BlockSpec((8, 128), lambda i: (i, 0))

            def kernel(a, b):
                b[...] = finish(a[...] + pl.num_programs(0), variant)

            return pl.pallas_call(
                kernel, out_shape=sig, grid=(n,), in_specs=(spec,), out_specs=spec
            )(x)

        return jax.make_jaxpr(outer)(jax.ShapeDtypeStruct((), jnp.int32), sig)

    a, b = fn("base"), fn("reorder")
    assert "num_programs" in {e.primitive.name for e in equations(a)}
    assert compare_jaxprs(a, b).status is Status.EQUIVALENT
    assert compare_jaxprs(a, fn("wrong")).status is Status.DIFFERENT


@pytest.mark.parametrize("mode", ["blocked", "element", "squeezed", "bounded"])
def test_block_dimension_descriptors(mode):
    dim, index, shape = {
        "blocked": (pl.Blocked(8), lambda i: (i, 0), (16, 128)),
        "element": (pl.Element(8), lambda i: (8 * i, 0), (16, 128)),
        "squeezed": (pl.Squeezed(), lambda i: (i, 0), (2, 128)),
        "bounded": (pl.BoundedSlice(8), lambda i: (pl.ds(8 * i, 8), 0), (16, 128)),
    }[mode]
    sig = jax.ShapeDtypeStruct(shape, jnp.float32)
    spec = pl.BlockSpec((dim, 128), index, pipeline_mode=pl.Buffered(2))

    def trace(variant):
        def kernel(x, y):
            y[...] = finish(x[...], variant)

        return jax.make_jaxpr(
            pl.pallas_call(
                kernel,
                out_shape=sig,
                grid=(2,),
                in_specs=(spec,),
                out_specs=spec,
                cost_estimate=pl.CostEstimate(
                    flops=1024, transcendentals=2048, bytes_accessed=8192
                ),
            )
        )(sig)

    a, b = trace("base"), trace("reorder")
    assert compare_jaxprs(a, b).status is Status.EQUIVALENT
    assert compare_jaxprs(a, trace("wrong")).status is Status.DIFFERENT


def test_indirect_array_index_is_rejected_by_locked_pallas_frontend():
    sig = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    spec = pl.BlockSpec((pl.Indirect(8), 128), lambda i: (jnp.arange(8) + 8 * i, 0))
    fn = pl.pallas_call(
        lambda x, y: y.__setitem__(Ellipsis, x[...]),
        out_shape=sig,
        grid=(2,),
        in_specs=(spec,),
        out_specs=spec,
    )
    with pytest.raises(ValueError, match="must return integer scalars"):
        jax.make_jaxpr(fn)(sig)


def test_ref_transpose_cpu_discharge_is_an_upstream_execution_limit():
    case = next(c for c in CASES if c.name == "ref_transpose")
    fn, (sig,) = function(case, interpret=True)
    with pytest.raises(NotImplementedError, match="TransposeTransform"):
        fn(jnp.ones(sig.shape, sig.dtype))


@pytest.mark.parametrize(
    "name,variant",
    [
        ("pipeline_captures", "capture_wrong"),
        ("pipeline_allocations", "allocation_wrong"),
    ],
)
def test_pipeline_capture_and_allocation_index_bindings_are_semantic(name, variant):
    case = next(c for c in CASES if c.name == name)
    with context(case):
        assert (
            compare_jaxprs(trace_case(case), trace_case(case, variant)).status is Status.DIFFERENT
        )


def test_unreviewed_high_level_primitive_subclass_is_not_accepted_as_einshape():
    from jax._src import hijax
    from jax._src.pallas.einshape import Einshape

    class DifferentEinshape(Einshape):
        pass

    case = next(c for c in CASES if c.name == "einshape")
    graph = trace_case(case)
    outer = graph.eqns[0]
    kernel = outer.params["jaxpr"]
    changed_eqns = []
    for eq in kernel.eqns:
        if eq.primitive is hijax.call_hi_primitive_p:
            prim = copy.copy(eq.params["_prim"])
            prim.__class__ = DifferentEinshape
            eq = eq.replace(params={"_prim": prim})
        changed_eqns.append(eq)
    changed = graph.replace(
        eqns=[outer.replace(params={**outer.params, "jaxpr": kernel.replace(eqns=changed_eqns)})]
    )
    with pytest.raises(UnsupportedJaxprError, match="high-level primitive"):
        canonicalize(changed)
