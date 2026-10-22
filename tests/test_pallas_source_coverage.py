"""Source admissions preserve local coordinates, memory validity and branch types."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_reorder import (
    CorrespondenceError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_pallas_kernels,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.call_bindings import extract_call
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    NormalizedEvidence,
    normalize_view,
    verify_normalized_evidence,
)
from jax_reorder.core.tensor_domain import prove_tensor_domain
from tests.pallas_projection_oracle import at_point
from tests.pallas_source_cases import (
    concatenate_call,
    prefetch_call,
    squeezed_call,
    stateful_jit_call,
    value_call,
    weak_cond_call,
)


@pytest.mark.parametrize("axis", [0, 1])
def test_static_concatenate_matches_piecewise_ref_stores_and_preserves_input_order(axis):
    signature = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    source = jax.make_jaxpr(concatenate_call(axis))(signature, signature)
    for mode in ("pieces", "swap"):
        candidate = jax.make_jaxpr(concatenate_call(axis, mode))(signature, signature)
        semantic = compare_pallas_kernels(source, candidate, equivalence="semantic")
        assert semantic.status is (Status.DIFFERENT if mode == "swap" else Status.EQUIVALENT)
        assert compare_algebraic_jaxprs(source, candidate).status is (
            Status.INEQUIVALENT if mode == "swap" else Status.EQUIVALENT
        )
    args = tuple(
        np.random.default_rng(seed).normal(size=signature.shape).astype(np.float32)
        for seed in (7, 11)
    )
    expected = np.concatenate(args, axis=axis)
    np.testing.assert_array_equal(concatenate_call(axis, interpret=True)(*args), expected)
    np.testing.assert_array_equal(concatenate_call(axis, "pieces", interpret=True)(*args), expected)


def concatenate_operation():
    signature = jax.ShapeDtypeStruct((8, 128), jnp.float32)
    source = jax.make_jaxpr(concatenate_call(0))(signature, signature)
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    op = next(op for op in call.kernel.operations if op.primitive == "concatenate")
    return op, tuple(block.ref.inner for block in call.blocks[:2])


def test_concatenate_definition_obligations_follow_the_selected_operand():
    from jax_reorder.core.index_domain import integer
    from jax_reorder.core.masked_access import require_defined
    from jax_reorder.core.memory import ValueExpr, scalar_type
    from jax_reorder.core.tensor_values import project_coordinates, tensor_value

    op, types = concatenate_operation()
    known = ValueExpr("literal", scalar_type(types[0]), data=np.float32(1).tobytes())
    missing = ValueExpr("undefined", scalar_type(types[1]))
    expression = tensor_value(op, (known, missing), types, {}, {}, (), AnalysisBudget())
    for row in (0, 8):
        selected = project_coordinates(
            expression, {"local:0": integer(row), "local:1": integer(0)}, {}, {}, AnalysisBudget()
        )
        if row == 0:
            require_defined(selected, {}, AnalysisBudget(), "concatenate.selected")
        else:
            with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
                require_defined(selected, {}, AnalysisBudget(), "concatenate.selected")


@pytest.mark.parametrize("change", ["dimension", "shape", "dtype"])
def test_concatenate_rejects_unproved_coordinate_interfaces(change):
    from jax_reorder.core.index_domain import integer
    from jax_reorder.core.model import ParameterView
    from jax_reorder.core.tensor_values import tensor_value

    op, types = concatenate_operation()
    if change == "dimension":
        op = replace(op, parameters=(ParameterView("dimension", 2),))
    else:
        types = (
            types[0],
            replace(
                types[1], **({"shape": (7, 128)} if change == "shape" else {"dtype": "float16"})
            ),
        )
    with pytest.raises(UnsupportedJaxprError, match="concatenate interface"):
        tensor_value(op, (integer(1), integer(2)), types, {}, {}, (), AnalysisBudget())


@pytest.mark.parametrize(
    "shape,squeezed", [((2, 8, 128), (0,)), ((4, 2, 8, 128), (1,)), ((2, 3, 8, 128), (0, 1))]
)
def test_squeezed_block_axes_match_retained_singleton_views(shape, squeezed):
    signature = jax.ShapeDtypeStruct(shape, jnp.float32)
    left = jax.make_jaxpr(squeezed_call(shape, squeezed))(signature)
    for wrong in (False, True):
        right = jax.make_jaxpr(squeezed_call(shape, squeezed, retain_axes=True, wrong=wrong))(
            signature
        )
        assert compare_algebraic_jaxprs(left, right).status is (
            Status.INEQUIVALENT if wrong else Status.EQUIVALENT
        )
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    proof = normalize_view(adapt(left).view, budget=AnalysisBudget())
    for point in [tuple(0 for _ in shape), tuple(n - 1 for n in shape)]:
        assert at_point(proof.projection.outputs[0], (x,), point) == x[point] + 1
    np.testing.assert_array_equal(squeezed_call(shape, squeezed, interpret=True)(x), x + 1)


@pytest.mark.parametrize("axes", [(1,), (0, 0), (3,)])
def test_squeezed_block_metadata_cannot_forge_coordinate_rank(axes):
    source = jax.make_jaxpr(squeezed_call((2, 8, 128), (0,)))(
        jax.ShapeDtypeStruct((2, 8, 128), jnp.float32)
    )
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    with pytest.raises(UnsupportedJaxprError, match="squeezed"):
        prove_tensor_domain(
            replace(call.blocks[0], squeezed_axes=axes), (2,), AnalysisBudget(), writer=False
        )


@pytest.mark.parametrize(
    "operation", ["stop_gradient", "slice", "slice_stride", "reduce_min", "reduce_and", "reduce_or"]
)
def test_reviewed_value_operations_reach_public_mathematical_comparison(operation):
    signature = jax.ShapeDtypeStruct((8, 256), jnp.float32)
    left = jax.make_jaxpr(value_call(operation))(signature)
    for wrong in (False, True):
        right = jax.make_jaxpr(value_call(operation, reference=True, wrong=wrong))(signature)
        assert compare_algebraic_jaxprs(left, right).status is (
            Status.INEQUIVALENT if wrong else Status.EQUIVALENT
        )
    x = np.random.default_rng(7).normal(size=signature.shape).astype(np.float32)
    original = value_call(operation, interpret=True)(x)
    candidate = value_call(operation, reference=True, interpret=True)(x)
    expected = {
        "stop_gradient": lambda: x[:, :128] * 2,
        "slice": lambda: x[:, 128:256],
        "slice_stride": lambda: x[:, ::2],
        "reduce_min": lambda: x[:, :128].min(axis=0),
        "reduce_and": lambda: (x[:, :128] > 0).all(axis=0),
        "reduce_or": lambda: (x[:, :128] > 0).any(axis=0),
    }[operation]()
    np.testing.assert_array_equal(original, expected)
    np.testing.assert_array_equal(candidate, expected)


def test_mutable_scalar_prefetch_keeps_serial_state_without_mutating_hbm_input():
    signature = jax.ShapeDtypeStruct((384,), jnp.float32)
    original = jax.make_jaxpr(prefetch_call())(signature)
    for mode, expected in [("reference", Status.EQUIVALENT), ("reset_each", Status.INEQUIVALENT)]:
        candidate = jax.make_jaxpr(prefetch_call(mode))(signature)
        assert compare_algebraic_jaxprs(original, candidate).status is expected
    x = np.arange(384, dtype=np.float32)
    proof = normalize_view(adapt(original).view, budget=AnalysisBudget())
    for point in [(0,), (127,), (128,), (383,)]:
        assert at_point(proof.projection.outputs[0], (x,), point) == x[point] + point[0] // 128 + 1


def test_mutable_prefetch_does_not_invent_parallel_ownership():
    source = jax.make_jaxpr(prefetch_call(parallel=True))(jax.ShapeDtypeStruct((384,), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="serial ownership"):
        normalize_view(adapt(source).view, budget=AnalysisBudget())


def assert_ordered_index_map_equivalent(source, reference):
    results = (
        compare_pallas_kernels(source, reference, equivalence="semantic"),
        compare_algebraic_jaxprs(source, reference, scope="device_local"),
        compare_algebraic_jaxprs(source, reference),
    )
    for result in results:
        assert result.status is Status.EQUIVALENT
        assert any("index maps" in text and "kernel" in text for text in result.assumptions)


@pytest.mark.parametrize(
    "initial,step,blocks,literal",
    [(0, 1, 3, False), (1, 1, 4, False), (1, 2, 6, False), (0, 1, 3, True), (2, -1, 3, True)],
)
def test_mutable_prefetch_index_map_reads_state_after_the_previous_kernel(
    initial, step, blocks, literal
):
    signature = jax.ShapeDtypeStruct((128 * blocks,), jnp.float32)
    options = dict(initial=initial, step=step, dynamic_initial=literal)

    def trace(call):
        function = (lambda x: call(jnp.array([initial], jnp.int32), x)) if literal else call
        return jax.make_jaxpr(function)(signature)

    source = trace(prefetch_call(indexed_prefetch=True, **options))
    reference = jax.make_jaxpr(prefetch_call(indexed_reference=True, initial=initial, step=step))(
        signature
    )
    assert_ordered_index_map_equivalent(source, reference)

    # Independent sequential model: bind the input window, run the kernel,
    # then use its updated counter when binding the next program's window.
    # Ordinary Pallas interpret discards mutable scalar-prefetch updates.
    x = np.arange(signature.shape[0], dtype=np.float32)
    expected = np.empty((384,), np.float32)
    counter = initial
    for program in range(3):
        block = x[counter * 128 : (counter + 1) * 128]
        counter += step
        expected[program * 128 : (program + 1) * 128] = block + counter
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget())
    actual = np.asarray([at_point(proof.projection.outputs[0], (x,), (i,)) for i in range(384)])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"initial": 2, "step": -1},
        {"indexed_prefetch": True},
        {"initial": 2, "step": -1, "indexed_prefetch": True},
        {"step": 0, "output_blocks": 1, "accumulate_output": True},
        {"programs": 5, "output_blocks": 2, "output_wrap": True},
        {
            "programs": 5,
            "output_blocks": 2,
            "output_wrap": True,
            "accumulate_output": True,
        },
        {
            "programs": 5,
            "output_blocks": 2,
            "output_wrap": True,
            "accumulate_output": True,
            "multiple_outputs": True,
        },
    ],
    ids=[
        "forward",
        "reverse",
        "both",
        "both-reverse",
        "consecutive",
        "revisit",
        "restore",
        "mixed-lifetimes",
    ],
)
def test_mutable_prefetch_output_map_keeps_the_current_window_until_kernel_completion(options):
    programs = options.get("programs", 3)
    output_blocks = options.get("output_blocks", 3)
    signature = jax.ShapeDtypeStruct((128 * programs,), jnp.float32)
    source = jax.make_jaxpr(prefetch_call(indexed_output=True, **options))(signature)
    reference_options = {
        **options,
        "indexed_prefetch": False,
        "indexed_reference": options.get("indexed_prefetch", False),
        "indexed_output_reference": True,
    }
    reference = jax.make_jaxpr(prefetch_call(**reference_options))(signature)
    assert_ordered_index_map_equivalent(source, reference)

    # Bind both windows from the old state, then execute and commit the kernel.
    # A store must not move to the next window when the kernel increments state.
    x = np.arange(signature.shape[0], dtype=np.float32)
    expected = np.full((128 * output_blocks,), np.nan, np.float32)
    total = np.zeros((128,), np.float32)
    counter = options.get("initial", 0)
    for program in range(programs):
        input_block = counter if options.get("indexed_prefetch") else program
        output_block = counter % output_blocks if options.get("output_wrap") else counter
        counter += options.get("step", 1)
        value = x[input_block * 128 : (input_block + 1) * 128] + counter
        window = slice(output_block * 128, (output_block + 1) * 128)
        if options.get("accumulate_output"):
            if program < output_blocks:
                expected[window] = 0
            expected[window] += value
        else:
            expected[window] = value
        total += value
    expected_outputs = (expected, total) if options.get("multiple_outputs") else (expected,)
    proof = normalize_view(adapt(source).view, budget=AnalysisBudget())
    assert len(proof.projection.outputs) == len(expected_outputs)
    for output, expected in zip(proof.projection.outputs, expected_outputs, strict=True):
        assert not np.isnan(expected).any()
        actual = np.asarray([at_point(output, (x,), (i,)) for i in range(len(expected))])
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("change", ["reset_each", "step"])
def test_mutable_prefetch_output_map_preserves_changed_state_and_final_revisited_values(change):
    signature = jax.ShapeDtypeStruct((384,), jnp.float32)
    options = dict(indexed_output=True, output_blocks=2, output_wrap=True)
    source = jax.make_jaxpr(prefetch_call(**options))(signature)
    candidate = jax.make_jaxpr(
        prefetch_call(
            "reset_each" if change == "reset_each" else "counter",
            step=3 if change == "step" else 1,
            **options,
        )
    )(signature)
    assert (
        compare_pallas_kernels(source, candidate, equivalence="semantic").status is Status.DIFFERENT
    )
    for scope in ("device_local", "full_function"):
        result = (
            compare_algebraic_jaxprs(source, candidate, scope=scope)
            if scope == "device_local"
            else compare_algebraic_jaxprs(source, candidate)
        )
        assert result.status is Status.INEQUIVALENT


def test_mutable_prefetch_output_map_rejects_unwritten_output_blocks():
    source = jax.make_jaxpr(prefetch_call(indexed_output=True, step=0))(
        jax.ShapeDtypeStruct((384,), jnp.float32)
    )
    with pytest.raises(UnsupportedJaxprError, match="cover|defined|uninitialized"):
        normalize_view(adapt(source).view, budget=AnalysisBudget())


def test_mutable_prefetch_output_map_does_not_invent_automatic_revisit_restoration():
    source = jax.make_jaxpr(
        prefetch_call(indexed_output=True, output_blocks=2, output_wrap=True, output_revisit=None)
    )(jax.ShapeDtypeStruct((384,), jnp.float32))
    with pytest.raises(UnsupportedJaxprError, match="revisit a block nonconsecutively"):
        normalize_view(adapt(source).view, budget=AnalysisBudget())


@pytest.mark.parametrize("change", ["exact", "shape", "coordinate", "data", "float", "unknown"])
def test_prefetch_constant_integer_keeps_original_shape_and_exact_dtype(change):
    from jax_reorder.core.index_domain import axis, integer
    from jax_reorder.core.memory import ValueExpr, scalar_type
    from jax_reorder.core.model import ArrayType, ParameterView
    from jax_reorder.core.tensor_memory import _prefetch_constant_integer

    dtype = "float32" if change == "float" else "int32"
    shape = (4,) if change == "shape" else (2, 2)
    coordinates = (
        (integer(0), integer(3))
        if change == "coordinate"
        else (axis("unbound"), integer(0))
        if change == "unknown"
        else (integer(1), integer(0))
    )
    value = ValueExpr(
        "constant_array",
        scalar_type(ArrayType((), dtype, False, ())),
        coordinates,
        (ParameterView("shape", shape),),
        np.arange(3 if change == "data" else 4, dtype=dtype).tobytes(),
    )
    if change == "exact":
        assert _prefetch_constant_integer(value, "test.prefetch") == integer(2)
    elif change in ("float", "unknown"):
        assert _prefetch_constant_integer(value, "test.prefetch") is value
    else:
        with pytest.raises(UnsupportedJaxprError, match="prefetch constant"):
            _prefetch_constant_integer(value, "test.prefetch")


@pytest.mark.parametrize("assumptions", [(), ("prefetch index maps use the initial state",)])
@pytest.mark.parametrize("mapped", ["input", "output"])
def test_mutable_prefetch_index_map_execution_assumption_replays(assumptions, mapped):
    source = jax.make_jaxpr(
        prefetch_call(**{"indexed_prefetch" if mapped == "input" else "indexed_output": True})
    )(jax.ShapeDtypeStruct((384,), jnp.float32))
    view = adapt(source).view
    proof = normalize_view(view, budget=AnalysisBudget())
    evidence = NormalizedEvidence(proof, proof, ((0, 0),))
    verify_normalized_evidence(view, view, evidence, budget=AnalysisBudget())
    forged = replace(proof, projection=replace(proof.projection, assumptions=assumptions))
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(
            view, view, replace(evidence, left=forged), budget=AnalysisBudget()
        )


@pytest.mark.parametrize("change", ["reset_each", "step"])
def test_mutable_prefetch_index_map_preserves_state_reset_and_stride_differences(change):
    signature = jax.ShapeDtypeStruct((768,), jnp.float32)
    source = jax.make_jaxpr(prefetch_call(indexed_prefetch=True))(signature)
    candidate = jax.make_jaxpr(
        prefetch_call(
            "reset_each" if change == "reset_each" else "counter",
            indexed_prefetch=True,
            step=2 if change == "step" else 1,
        )
    )(signature)
    assert (
        compare_pallas_kernels(source, candidate, equivalence="semantic").status is Status.DIFFERENT
    )
    assert (
        compare_algebraic_jaxprs(source, candidate, scope="device_local").status
        is Status.INEQUIVALENT
    )


@pytest.mark.parametrize("entry", ["semantic", "mathematical"])
@pytest.mark.parametrize("mapped", ["input", "output"])
@pytest.mark.parametrize(
    "change,reason",
    [
        ("parallel", "complete serial ownership"),
        ("out_of_bounds", "out-of-bounds block"),
        ("unknown_initial", "static integer is required"),
    ],
)
def test_mutable_prefetch_index_map_rejects_unproved_execution(change, reason, mapped, entry):
    signature = jax.ShapeDtypeStruct((384,), jnp.float32)
    options = {"indexed_prefetch" if mapped == "input" else "indexed_output": True}
    if change == "parallel":
        options["parallel"] = True
    elif change == "out_of_bounds":
        options["initial"] = 1
    elif change == "unknown_initial":
        options["dynamic_initial"] = True
    arguments = (
        (jax.ShapeDtypeStruct((1,), jnp.int32), signature)
        if change == "unknown_initial"
        else (signature,)
    )
    source = jax.make_jaxpr(prefetch_call(**options))(*arguments)
    # Comparing identical sources must still validate the address/state proof.
    with pytest.raises(UnsupportedJaxprError, match=reason):
        if entry == "semantic":
            compare_pallas_kernels(source, source, equivalence="semantic")
        else:
            compare_algebraic_jaxprs(source, source, scope="device_local")


def test_default_kernel_jit_preserves_all_ref_effect_bindings():
    signature = jax.ShapeDtypeStruct((128,), jnp.float32)
    original = jax.make_jaxpr(stateful_jit_call())(signature)
    for mode, expected in [("inline", Status.EQUIVALENT), ("wrong", Status.INEQUIVALENT)]:
        right = jax.make_jaxpr(stateful_jit_call(mode))(signature)
        assert compare_algebraic_jaxprs(original, right).status is expected


def test_contiguous_ref_slot_footprint_is_proved_without_enumerating_values():
    from jax_reorder.core.model import RefTransform, SliceIndex
    from jax_reorder.core.ref_views import view_coordinates, view_footprint
    from jax_reorder.core.storage import Span
    from jax_reorder.core.tensor_storage import Box

    shape = (2, 8, 16, 128)
    views = (
        RefTransform(
            "index", (), (1, SliceIndex(0, 8, 1), SliceIndex(0, 16, 1), SliceIndex(0, 128, 1))
        ),
        RefTransform("transpose", (1, 0, 2)),
    )
    budget = AnalysisBudget(32)
    result, backward, _ = view_coordinates(shape, views, "test.view", budget)
    assert view_footprint(shape, result, backward, views, "test.view", budget) == (
        Box((Span(1, 2), Span(0, 8), Span(0, 16), Span(0, 128))),
    )


@pytest.mark.parametrize("mode", ["stride", "diagonal"])
def test_nonrectangular_ref_footprints_do_not_expand_into_their_bounding_box(mode):
    from jax_reorder.core.index_domain import integer, scale
    from jax_reorder.core.model import RefTransform, SliceIndex
    from jax_reorder.core.ref_views import view_footprint
    from jax_reorder.core.tensor_domain import coordinate

    shape = (4, 4)
    backward = (
        {"local:0": integer(0), "local:1": scale(coordinate("local", 0), 2)}
        if mode == "stride"
        else {"local:0": coordinate("local", 0), "local:1": coordinate("local", 0)}
    )
    boxes = view_footprint(
        shape,
        (2,),
        backward,
        (RefTransform("index", (), (SliceIndex(0, 2, 1),)),),
        "test.view",
        AnalysisBudget(),
    )
    assert sum(box.size for box in boxes) == 2
    assert len(boxes) == 2


def paged_projection(**options):
    from jax_reorder.core.call_bindings import binding_value
    from jax_reorder.core.tensor_memory import functionalize_tensor_call
    from tests.pallas_paged_attention_cases import trace

    budget = AnalysisBudget()
    site = extract_call(adapt(trace(**options)).view, budget=budget)
    assert site.operation.call is not None
    proof = functionalize_tensor_call(
        site.operation.call,
        budget=budget,
        inputs=tuple(binding_value(argument, budget=budget) for argument in site.arguments),
    )
    return proof


@pytest.mark.parametrize("bounded_pages", [False, True])
def test_complete_upstream_paged_attention_reaches_values_under_declared_routing(bounded_pages):
    from jax_reorder.core.memory import expressions_equal
    from tests.pallas_paged_attention_cases import inputs, page_tables, reference
    from tests.pallas_upstream_state_cases import verify_sources

    verify_sources()
    direct = paged_projection(bounded_pages=bounded_pages)
    loop = paged_projection(inline=True, bounded_pages=bounded_pages)
    wrong = paged_projection(inline=True, wrong=True, bounded_pages=bounded_pages)
    assert len(direct.outputs) == len(loop.outputs) == len(wrong.outputs) == 3
    assert expressions_equal(direct.outputs, loop.outputs)
    assert not expressions_equal(direct.outputs, wrong.outputs)
    assert expressions_equal(direct.outputs[1:], wrong.outputs[1:])
    if bounded_pages:
        wrong_source = paged_projection(inline=True, bounded_pages=True, wrong_page=True)
        assert not expressions_equal(direct.outputs, wrong_source.outputs)
        values = inputs()
        for pages in page_tables():
            arguments = values + (pages.reshape(1, 8),)
            for proof, changes in (
                (direct, {}),
                (loop, {}),
                (wrong, {"wrong": True}),
                (wrong_source, {"wrong_page": True}),
            ):
                for value, expected in zip(
                    proof.outputs, reference(values, pages, **changes), strict=True
                ):
                    actual = at_point(value, arguments, np.indices(expected.shape))
                    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("bounded_pages", [False, True])
def test_complete_paged_attention_public_semantics_fit_default_analysis_budget(bounded_pages):
    from tests.pallas_paged_attention_cases import trace

    source = trace(bounded_pages=bounded_pages)
    for options in ({}, {"wrong": True}, {"wrong_page": True}):
        candidate = trace(inline=True, bounded_pages=bounded_pages, **options)
        result = compare_pallas_kernels(source, candidate, equivalence="semantic")
        assert result.status is (Status.DIFFERENT if options else Status.EQUIVALENT)


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("unknown_lengths", [False, True])
def test_paged_attention_does_not_guess_unbounded_lengths_or_page_indices(inline, unknown_lengths):
    from tests.pallas_paged_attention_cases import function, signatures, trace

    if unknown_lengths:
        source = trace(inline=inline, dynamic_routing=True)
    else:
        original = function(inline=inline, dynamic_routing=True)
        source = jax.make_jaxpr(
            lambda q, k, v, pages: original(q, k, v, jnp.full((1,), 128, jnp.int32), pages)
        )(*signatures(bounded_pages=True))
    with pytest.raises(UnsupportedJaxprError, match="finite|static|bounds"):
        compare_pallas_kernels(source, source, equivalence="semantic")


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "inline,wrong,bounded_pages",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (False, False, True),
        (True, False, True),
    ],
)
def test_complete_paged_attention_actual_execution_has_no_races(inline, wrong, bounded_pages):
    from tests.pallas_oracle import run_worker

    run_worker(
        "tests.pallas_paged_attention_cases",
        entry="execute",
        kwargs=dict(inline=inline, wrong=wrong, bounded_pages=bounded_pages),
        expected=dict(
            inline=inline, wrong=wrong, bounded_pages=bounded_pages, status="ok", races=False
        ),
        devices=1,
    )


def test_cond_join_preserves_source_weak_types_but_accepts_the_legal_boundary():
    signature = jax.ShapeDtypeStruct((128,), jnp.float32)
    source = jax.make_jaxpr(weak_cond_call())(signature)
    cond = next(eq for eq in source.eqns[0].params["jaxpr"].eqns if eq.primitive.name == "cond")
    assert cond.outvars[0].aval.weak_type
    assert not cond.params["branches"][1].jaxpr.outvars[0].aval.weak_type
    reference = jax.make_jaxpr(weak_cond_call(reference=True))(signature)
    assert compare_algebraic_jaxprs(source, reference).status is Status.EQUIVALENT


@pytest.mark.parametrize("change", ["dtype", "shape"])
def test_cond_join_does_not_relax_value_dtype_or_shape(change):
    from jax._src import core

    source = jax.make_jaxpr(weak_cond_call())(jax.ShapeDtypeStruct((128,), jnp.float32))
    outer = source.eqns[0]
    kernel = outer.params["jaxpr"]
    equations = list(kernel.eqns)
    index = next(i for i, eq in enumerate(equations) if eq.primitive.name == "cond")
    cond = equations[index]
    branches = list(cond.params["branches"])
    value = np.float32(1) if change == "dtype" else np.ones((1,), np.int32)
    aval = core.ShapedArray(np.shape(value), np.asarray(value).dtype)
    branches[1] = branches[1].replace(
        jaxpr=branches[1].jaxpr.replace(outvars=[core.Literal(value, aval)])
    )
    equations[index] = cond.replace(params={"branches": tuple(branches)})
    changed = source.replace(
        eqns=[outer.replace(params={**outer.params, "jaxpr": kernel.replace(eqns=equations)})]
    )
    with pytest.raises(UnsupportedJaxprError, match="conditional branch port types"):
        adapt(changed)


@pytest.mark.parametrize("operation", ["reduce_min", "reduce_and", "reduce_or"])
@pytest.mark.parametrize("obligation", ["defined", "memory"])
def test_new_reductions_retain_every_operand_definedness_and_memory_obligation(
    operation, obligation
):
    import sympy as sp

    from jax_reorder.algebra.expression import SymbolicTensor, coordinates
    from jax_reorder.algebra.pallas import _operation
    from jax_reorder.algebra.symbolic import SymbolicBuilder

    dtype = np.dtype("float32" if operation == "reduce_min" else "bool")
    function = {"reduce_min": jnp.min, "reduce_and": jnp.all, "reduce_or": jnp.any}[operation]
    equation = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((4,), dtype)).jaxpr.eqns[0]
    (index,) = coordinates((4,))
    cells = sp.IndexedBase("cells", shape=(4,), real=True)
    original = SymbolicTensor(
        (4,),
        dtype,
        sp.Integer(1) if operation == "reduce_min" else sp.true,
        cells[index] > 0 if obligation == "defined" else sp.true,
        memory_validity=index < 3 if obligation == "memory" else sp.true,
    )
    result = _operation(SymbolicBuilder(1000), equation, [original])
    assert result.defined == (
        sp.And(*(cells[i] > 0 for i in range(4))) if obligation == "defined" else sp.true
    )
    assert result.memory_validity == (sp.false if obligation == "memory" else sp.true)


@pytest.mark.parametrize(
    "mode", ["read", "input", "parameters", "capture", "obligations", "budget"]
)
def test_tensor_computation_sharing_preserves_each_output_context(mode, monkeypatch):
    from types import SimpleNamespace

    import sympy as sp

    from jax_reorder import ComparisonBudgetError
    from jax_reorder.algebra import pallas
    from jax_reorder.algebra.expression import coordinates
    from jax_reorder.algebra.symbolic import SymbolicBuilder
    from jax_reorder.core.index_domain import add, axis, integer, scale
    from jax_reorder.core.memory import MemoryProjection, ScalarType, ValueExpr
    from jax_reorder.core.model import ParameterView

    builder = SymbolicBuilder(1000)
    aval = SimpleNamespace(shape=(2, 2), dtype=np.dtype("float32"))
    sources = [builder.input(aval, i) for i in range(2)]
    if mode == "obligations":
        sources[0] = replace(
            sources[0],
            defined=coordinates((2, 2))[0] < 1,
            memory_validity=coordinates((2, 2))[0] < 1,
        )
    typ = ScalarType("float32", False, ())
    row = axis("output:0") if mode == "capture" else axis("tensor:r")

    def root(port=0, reduce_axis=1, reverse=False):
        body = ValueExpr("input", typ, (row, axis("tensor:c")), input=port)
        at = add(integer(1), scale(axis("output:0"), -1)) if reverse else axis("output:0")
        return ValueExpr(
            "tensor.reduce_sum",
            typ,
            (body, at),
            tuple(
                ParameterView(name, value)
                for name, value in (
                    ("__input_shapes", ((2, 2),)),
                    ("__input_dtypes", ("float32",)),
                    ("__input_axes", (("tensor:r", "tensor:c"),)),
                    ("__output_shape", (2,)),
                    ("axes", (reduce_axis,)),
                )
            ),
        )

    first = root()
    second = root(
        port=int(mode == "input"),
        reduce_axis=0 if mode == "parameters" else 1,
        reverse=mode in ("read", "obligations", "budget"),
    )
    projection = MemoryProjection((first, second), (), (), (), ())
    equation = SimpleNamespace(
        params={"projection": projection, "rule": "tensor-sharing-test"},
        outvars=tuple(
            SimpleNamespace(aval=SimpleNamespace(shape=(2,), dtype=aval.dtype)) for _ in range(2)
        ),
    )
    operations = []
    original = pallas._operation

    def operation(builder, equation, values):
        operations.append(equation.primitive.name)
        return original(builder, equation, values)

    monkeypatch.setattr(pallas, "_operation", operation)
    results = pallas.projected_outputs(builder, equation, sources)
    assert operations == ["reduce_sum"] * (2 if mode in ("input", "parameters", "capture") else 1)
    replacements = {
        source.expression.base[r, c]: sp.Integer(100 * port + 2 * r + c)
        for port, source in enumerate(sources)
        for r in range(2)
        for c in range(2)
    }
    actual = [
        [result.read((i,))[0].doit().xreplace(replacements) for i in range(2)] for result in results
    ]
    expected = {
        "input": [[1, 5], [201, 205]],
        "parameters": [[1, 5], [2, 4]],
        "capture": [[1, 5], [1, 5]],
    }.get(mode, [[1, 5], [5, 1]])
    assert actual == expected
    if mode == "obligations":
        assert [[result.read((i,))[1] for i in range(2)] for result in results] == [
            [sp.true, sp.false],
            [sp.false, sp.true],
        ]
        assert [[pallas._memory_read(result, (i,)) for i in range(2)] for result in results] == [
            [sp.true, sp.false],
            [sp.false, sp.true],
        ]
    if mode == "budget":
        with pytest.raises(ComparisonBudgetError):
            pallas.projected_outputs(SymbolicBuilder(builder.budget.used - 1), equation, sources)
