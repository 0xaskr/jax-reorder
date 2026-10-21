"""Pallas view/retile acceptance, unsafe near-misses, and independent oracles.

Native source graphs are abstract-traced on CPU. CPU interpret runs are separate
numerical evidence, and do not establish TPU compilation or concurrency support.
"""

from dataclasses import fields, is_dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core
from jax._src.pallas import primitives as pp
from jax._src.state import types as st
from jax.experimental import pallas as pl

from jax_reorder import (
    ComparisonBudgetError,
    CorrespondenceError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import (
    AnalysisBudget,
    axis,
    bounds,
    prove_block_domain,
    substitute,
)
from jax_reorder.core.memory import functionalize_call
from jax_reorder.core.model import ProgramView, RefType
from jax_reorder.core.normalization import (
    NormalizedEvidence,
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.view_graph import verify_view_correspondence
from tests.pallas_domain_oracle import enumerate_blocked
from tests.pallas_projection_oracle import at_point


def build(
    block,
    reverse=False,
    *,
    size=1024,
    mode="add",
    tail=False,
    gap=0,
    interpret=False,
    mapping=None,
    in_mapping=None,
    aliases=None,
    compiler_params=None,
):
    programs = (size + block - 1) // block

    def default_mapping(p):
        return (programs - 1 - p if reverse else p,)

    mapping = mapping or default_mapping
    spec = pl.BlockSpec((block,), mapping)
    inspec = pl.BlockSpec((block,), in_mapping) if in_mapping else spec
    dtype = jnp.int32 if mode == "program_id" else jnp.float32

    def kernel(x, y):
        if mode == "program_id":
            y[:] = jnp.broadcast_to(pl.program_id(0), (block,))
            return
        if tail:
            lane = jnp.arange(block)
            global_index = mapping(pl.program_id(0))[0] * block + lane
            value = pp.load(x, (lane,), mask=global_index < size, other=0.0)
            pp.store(y, (lane,), value + 1.0, mask=global_index < size - gap)
        elif mode == "uninitialized":
            y[:] = y[:] + x[:]
        elif mode == "partial":
            y[: block // 2] = x[: block // 2]
        elif mode == "duplicate":
            y[:] = x[:]
            y[:] = x[:] + 1
        elif mode == "input_write":
            x[:] = x[:] + 1
            y[:] = x[:]
        elif mode == "unused_read":
            unused = y[:]
            del unused
            y[:] = x[:]
        elif mode == "nonlinear":
            y[:] = jnp.sin(x[:]) * jnp.exp(x[:]) + 1
        else:
            y[:] = x[:] + (2.0 if mode == "add_two" else 1.0)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((size,), dtype),
        grid=(programs,),
        in_specs=(inspec,),
        out_specs=spec,
        interpret=interpret,
        input_output_aliases=aliases or {},
        compiler_params=compiler_params,
    )


def trace(block=128, reverse=False, **kwargs):
    size = kwargs.get("size", 1024)
    return jax.make_jaxpr(build(block, reverse, **kwargs))(
        jax.ShapeDtypeStruct((size,), jnp.float32)
    )


def view(block=128, reverse=False, **kwargs):
    return adapt(trace(block, reverse, **kwargs)).view


def call(v):
    return v.root.operations[0].call


def test_self_comparison_has_structural_witness():
    a, b = view(), view()
    result = compare_program_views(a, b)
    assert result.status == Status.EQUIVALENT
    assert result.equivalence_kind == "structural"
    assert len(result.evidence.regions) == 4  # root, kernel, input/output maps
    verify_view_correspondence(a, b, result.evidence)
    assert isinstance(call(a).kernel.inputs[0].type, RefType)
    assert {e.kind for e in call(a).kernel.effects} == {"read", "write"}
    assert result.warnings == ()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mode", ["add", "nonlinear"])
def test_retile_reverse_grid_symbolically(reverse, mode):
    a, b = view(128, mode=mode), view(256, reverse, mode=mode)
    result = compare_program_views(a, b)
    assert result.status == Status.EQUIVALENT
    assert result.equivalence_kind == "normalized"
    assert result.evidence.left.projection.outputs == result.evidence.right.projection.outputs
    assert result.evidence.left.projection.domains != result.evidence.right.projection.domains
    assert "every valid output coordinate has exactly one writer" in (
        result.evidence.left.projection.conditions
    )
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())


def test_large_shape_does_not_enumerate(monkeypatch):
    size = 2**24 + 13
    a, b = view(128, size=size, tail=True), view(256, True, size=size, tail=True)

    def forbidden(*args, **kwargs):
        pytest.fail("comparison must not execute/enumerate source jaxprs")

    monkeypatch.setattr(core, "eval_jaxpr", forbidden)
    result = compare_program_views(a, b, analysis_budget=256)
    assert result.equivalence_kind == "normalized"
    assert result.work < 256
    assert len(result.evidence.left.projection.outputs) == 1


def test_tail_mask_gap_is_rejected():
    bad = view(128, size=1037, tail=True, gap=1)
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(bad, budget=AnalysisBudget())


@pytest.mark.parametrize("block,reverse", [(128, False), (256, True)])
def test_staged_tail_padding_does_not_escape_valid_output(block, reverse):
    source = view(block, reverse, size=1037)
    proof = normalize_view(source, budget=AnalysisBudget())
    x = np.arange(1037, dtype=np.float32)
    actual = np.asarray([at_point(proof.projection.outputs[0], (x,), (i,)) for i in range(x.size)])
    np.testing.assert_array_equal(actual, x + 1)
    verify_normalized_evidence(
        source, source, NormalizedEvidence(proof, proof, ((0, 0),)), budget=AnalysisBudget()
    )


def test_program_id_data_is_preserved():
    a, b = view(128, mode="program_id"), view(256, mode="program_id")
    result = compare_program_views(a, b)
    assert result.status == Status.INEQUIVALENT
    assert result.evidence.coordinate == 128
    assert (result.evidence.left, result.evidence.right) == (1, 0)


def test_store_then_get_uses_current_version():
    def program(block):
        def kernel(x, y, z):
            y[:] = x[:] + 1
            z[:] = y[:] * 2

        spec = pl.BlockSpec((block,), lambda p: (p,))
        f = pl.pallas_call(
            kernel,
            out_shape=(jax.ShapeDtypeStruct((1024,), jnp.float32),) * 2,
            grid=(1024 // block,),
            in_specs=(spec,),
            out_specs=(spec, spec),
        )
        return adapt(jax.make_jaxpr(f)(jax.ShapeDtypeStruct((1024,), jnp.float32))).view

    a, b = program(128), program(256)
    projection = functionalize_call(call(a), budget=AnalysisBudget())
    assert projection.outputs[1].operands[0] == projection.outputs[0]
    assert len(projection.versions) == 2
    assert compare_program_views(a, b).equivalence_kind == "normalized"


@pytest.mark.parametrize(
    "mode,reason",
    [
        ("uninitialized", "uninitialized"),
        ("duplicate", "multiple writes"),
        ("partial", "uninitialized"),
        ("input_write", "read-only"),
    ],
)
def test_unsafe_memory_is_rejected_before_projection(mode, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        normalize_view(view(mode=mode), budget=AnalysisBudget())


def test_unused_uninitialized_read_is_still_checked():
    # JAX's tracing DCE removes unused get operations. Insert the retained source
    # equation explicitly to exercise the complete-graph admission requirement.
    graph = trace()
    outer = graph.eqns[0]
    kernel = outer.params["jaxpr"]
    unused = kernel.eqns[0].replace(
        invars=[kernel.invars[1]],
        outvars=[core.DropVar(kernel.eqns[0].outvars[0].aval)],
        effects={st.ReadEffect(kernel.invars[1])},
    )
    kernel = kernel.replace(eqns=[unused, *kernel.eqns], effects=kernel.effects | unused.effects)
    changed = graph.jaxpr.replace(eqns=[outer.replace(params={**outer.params, "jaxpr": kernel})])
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        normalize_view(adapt(changed).view, budget=AnalysisBudget())


def test_formula_mismatch_is_not_inequivalence():
    with pytest.raises(MathematicalUndecidedError, match="mathematical backend"):
        compare_program_views(view(), view(256, mode="add_two"))


def test_equal_read_sets_do_not_prove_corresponding_reads():
    a = view()
    b = view(256, in_mapping=lambda p: (3 - p,))
    with pytest.raises(MathematicalUndecidedError):
        compare_program_views(a, b)


@pytest.mark.parametrize("mapping", [lambda p: (0,), lambda p: (p * 2,), lambda p: (p * p,)])
def test_nonbijective_and_nonlinear_maps_are_rejected(mapping):
    with pytest.raises(UnsupportedJaxprError, match="bijection|integer index operation"):
        normalize_view(view(mapping=mapping), budget=AnalysisBudget())


def test_index_overflow_is_rejected_even_if_cancelled_later():
    with pytest.raises(UnsupportedJaxprError, match="overflow"):
        normalize_view(
            view(mapping=lambda p: ((p + jnp.int32(2147483647)) - jnp.int32(2147483647),)),
            budget=AnalysisBudget(),
        )


def test_nonnegative_div_rem_index_map_is_proved_symbolically():
    mapped = view(256, mapping=lambda p: (lax.div(p, 2) * 2 + lax.rem(p, 2),))
    assert compare_program_views(view(128), mapped).equivalence_kind == "normalized"


def test_negative_truncating_division_is_not_mistaken_for_floor_division():
    mapped = view(256, mapping=lambda p: (lax.div(-p, 2),))
    with pytest.raises(UnsupportedJaxprError, match="integer index operation div"):
        normalize_view(mapped, budget=AnalysisBudget())


def test_analysis_budget_is_classified():
    with pytest.raises(ComparisonBudgetError):
        compare_program_views(view(), view(256), analysis_budget=1)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"interpret": True}, "interpret=False"),
    ],
)
def test_unsupported_call_contract_is_rejected_even_in_self_comparison(kwargs, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        view(**kwargs)


def test_views_contain_no_jax_or_sympy_objects():
    v = view(128, size=1037, tail=True)
    pending = [v]
    while pending:
        item = pending.pop()
        if is_dataclass(item):
            assert type(item).__module__.startswith("jax_reorder.core.model")
            pending.extend(getattr(item, f.name) for f in fields(item))
        elif isinstance(item, tuple):
            pending.extend(item)
        else:
            assert item is None or type(item) in (bool, int, str, bytes)
    assert isinstance(v, ProgramView)


def test_structural_and_normalized_witness_tampering_is_detected():
    a = view()
    result = compare_program_views(a, a)
    bad = replace(result.evidence, regions=result.evidence.regions[:-1])
    with pytest.raises(CorrespondenceError):
        verify_view_correspondence(a, a, bad)
    b = view(256, True)
    result = compare_program_views(a, b)
    bad = replace(result.evidence, terminal_outputs=())
    with pytest.raises(CorrespondenceError):
        verify_normalized_evidence(a, b, bad, budget=AnalysisBudget())


def test_repeated_squaring_preserves_a_dag_without_exponential_expansion():
    def program(block):
        def kernel(x, y):
            value = x[:]
            for _ in range(80):
                value = value * value
            y[:] = value

        spec = pl.BlockSpec((block,), lambda p: (p,))
        f = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
            grid=(1024 // block,),
            in_specs=(spec,),
            out_specs=spec,
        )
        return adapt(jax.make_jaxpr(f)(jax.ShapeDtypeStruct((1024,), jnp.float32))).view

    result = compare_program_views(program(128), program(256), analysis_budget=512)
    assert result.equivalence_kind == "normalized"
    expr = result.evidence.left.projection.outputs[0]
    for _ in range(80):
        assert expr.operands[0] is expr.operands[1]
        expr = expr.operands[0]


def test_fake_same_name_primitive_does_not_get_admitted():
    graph = trace()
    outer = graph.eqns[0]
    kernel = outer.params["jaxpr"]
    real_add = kernel.eqns[1]
    fake = core.Primitive("add")
    fake.def_abstract_eval(lambda a, b: a)
    changed_eq = real_add.replace(primitive=fake)
    changed_kernel = kernel.replace(eqns=[kernel.eqns[0], changed_eq, *kernel.eqns[2:]])
    changed = graph.jaxpr.replace(
        eqns=[outer.replace(params={**outer.params, "jaxpr": changed_kernel})]
    )
    with pytest.raises(UnsupportedJaxprError, match="unsupported primitive add"):
        adapt(changed)


def test_unknown_pallas_parameter_fails_closed():
    graph = trace()
    outer = graph.eqns[0]
    changed = graph.jaxpr.replace(
        eqns=[outer.replace(params={**outer.params, "new_semantics": True})]
    )
    # JAX may catch this before the adapter's schema guard; both are classified.
    from jax_reorder import InvalidJaxprError

    with pytest.raises((UnsupportedJaxprError, InvalidJaxprError)):
        adapt(changed)


def test_effect_binding_changes_are_not_hidden_by_same_primitive_names():
    a = view()
    op = call(a).kernel.operations[0]
    assert op.effects[0].ref == op.access.ref
    bad_effect = replace(op.effects[0], ref=call(a).kernel.operations[-1].access.ref)
    bad_op = replace(op, effects=(bad_effect,))
    bad_kernel = replace(call(a).kernel, operations=(bad_op, *call(a).kernel.operations[1:]))
    bad_call = replace(call(a), kernel=bad_kernel)
    bad_root = replace(a.root, operations=(replace(a.root.operations[0], call=bad_call),))
    evidence = compare_program_views(a, a).evidence
    with pytest.raises(CorrespondenceError):
        verify_view_correspondence(a, replace(a, root=bad_root), evidence)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.int32])
def test_retiling_preserves_dtype_and_literal_bits(dtype):
    def program(block):
        def kernel(x, y):
            y[:] = x[:] + jnp.array(1, dtype=dtype)

        spec = pl.BlockSpec((block,), lambda p: (p,))
        f = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((1024,), dtype),
            grid=(1024 // block,),
            in_specs=(spec,),
            out_specs=spec,
        )
        return adapt(jax.make_jaxpr(f)(jax.ShapeDtypeStruct((1024,), dtype))).view

    a, b = program(128), program(256)
    assert compare_program_views(a, b).equivalence_kind == "normalized"
    assert call(a).blocks[0].array.dtype == str(np.dtype(dtype))


def test_native_compiler_parameters_are_not_silently_ignored():
    from jax.experimental.pallas import tpu as pltpu

    native = view(compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)))
    assert call(native).compiler.explicit
    assert call(native).grid[0].semantics == "parallel"
    assert compare_program_views(native, view(256, True)).equivalence_kind == "normalized"


def test_alias_is_encoded_and_requires_matching_independent_blocks():
    native = view(aliases={0: 0})
    assert call(native).aliases == ((0, 0),)
    assert compare_program_views(native, native).equivalence_kind == "structural"
    proof = normalize_view(native, budget=AnalysisBudget())
    assert proof.rule == "local-control-memory-to-values"
    assert compare_program_views(native, view()).equivalence_kind == "normalized"


def test_default_jit_wrapper_is_preserved_in_structural_and_normalized_evidence():
    def program(block, reverse=False):
        return adapt(
            jax.make_jaxpr(jax.jit(build(block, reverse)))(
                jax.ShapeDtypeStruct((1024,), jnp.float32)
            )
        ).view

    a, b = program(128), program(256, True)
    assert a.root.operations[0].primitive == "jit"
    assert a.root.operations[0].regions[0].arguments == a.root.operations[0].operands
    structural = compare_program_views(a, a)
    assert structural.equivalence_kind == "structural"
    assert len(structural.evidence.regions) == 5
    normalized = compare_program_views(a, b)
    assert normalized.equivalence_kind == "normalized"
    assert tuple(boundary.primitive for boundary in normalized.evidence.left.boundaries) == (
        "jit",
        "pallas_call",
    )


@pytest.mark.parametrize(
    "options", [{"donate_argnums": (0,)}, {"keep_unused": True}, {"inline": True}]
)
def test_nondefault_jit_contract_is_not_erased(options):
    graph = jax.make_jaxpr(jax.jit(build(128), **options))(
        jax.ShapeDtypeStruct((1024,), jnp.float32)
    )
    with pytest.raises(UnsupportedJaxprError, match="JIT execution contract"):
        adapt(graph)


@pytest.mark.parametrize("size,block", [(13, 8), (16, 8), (17, 8), (35, 16), (3, 1)])
@pytest.mark.parametrize("reverse", [False, True])
def test_symbolic_domain_matches_independent_small_oracle(size, block, reverse):
    v = view(block, reverse, size=size, tail=True)
    domain = prove_block_domain(call(v).blocks[0], call(v).grid[0].size, AnalysisBudget())
    count = (size + block - 1) // block
    good, writes, outside = enumerate_blocked(
        size, block, count, lambda p: count - 1 - p if reverse else p, lambda i: i < size
    )
    assert good and not outside and len(writes) == size
    assert substitute(
        domain.address, dict(zip(("program", "lane"), domain.inverse, strict=True))
    ) == axis("global")
    for i in range(size):
        p, r = (bounds(expr, {"global": (i, i)})[0] for expr in domain.inverse)
        assert (count - 1 - p if reverse else p) * block + r == i


@pytest.mark.parametrize("size,tail", [(1024, False), (1037, True)])
def test_cpu_interpret_is_a_separate_numerical_oracle(size, tail):
    x = np.arange(size, dtype=np.float32)
    for block, reverse in ((128, False), (256, True)):
        actual = build(block, reverse, size=size, tail=tail, interpret=True)(x)
        np.testing.assert_array_equal(actual, x + 1)
    assert (
        compare_program_views(
            view(128, size=size, tail=tail), view(256, True, size=size, tail=tail)
        ).equivalence_kind
        == "normalized"
    )
