"""Multi-axis address correspondence, rectangular state and independent CPU oracles."""

import json
from dataclasses import replace
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import (
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_functions,
    compare_pallas_kernels,
    verify_pallas_kernel_comparison,
)
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget, IndexExpr
from jax_reorder.core.memory import Predicate, ValueExpr
from jax_reorder.core.normalization import compare_program_views, normalize_view
from tests.pallas_oracle import run_worker

EMPTY_RESHAPES = (
    ((8, 0), (8, 1, 0), None),
    ((0, 8), (0, 1, 8), (1, 0)),
    ((2, 0, 4), (8, 0), None),
    ((0, 2, 0, 4), (0, 0, 8), (3, 1, 2, 0)),
)


def empty_reshape_call(shape, dimensions, *, reference=None, partial=False, interpret=False):
    def kernel(x, y):
        y[...] = x[...]

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        interpret=interpret,
    )

    def function(x, *args):
        if reference is None:
            view = lax.reshape(x, shape, dimensions=dimensions)
            value = (
                jnp.max(view, initial=False).astype(jnp.float32)
                if x.dtype == jnp.bool_
                else jnp.sum(view)
            )
        else:
            value = jnp.float32(reference)
        if partial:
            value += jnp.log(args[0]) * 0
        return call(jnp.broadcast_to(value, (8, 128)))

    return function


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bool_])
@pytest.mark.parametrize("input_shape,shape,dimensions", EMPTY_RESHAPES)
def test_empty_reshape_preserves_empty_reduction_and_nonempty_outputs(
    input_shape, shape, dimensions, dtype
):
    signature = jax.ShapeDtypeStruct(input_shape, dtype)
    source = empty_reshape_call(shape, dimensions)
    for value, expected in ((0, Status.EQUIVALENT), (1, Status.INEQUIVALENT)):
        reference = empty_reshape_call(shape, dimensions, reference=value)
        assert compare_functions(source, reference, signature).status is expected
    actual = empty_reshape_call(shape, dimensions, interpret=True)(
        np.empty(input_shape, dtype=dtype)
    )
    np.testing.assert_array_equal(np.asarray(actual), np.zeros((8, 128), dtype=np.float32))


def test_empty_reshape_does_not_erase_a_nonempty_partial_computation():
    source = empty_reshape_call((8, 1, 0), None, partial=True)
    reference = empty_reshape_call((8, 1, 0), None, reference=0)
    with pytest.raises(MathematicalUndecidedError) as error:
        compare_functions(
            source,
            reference,
            jax.ShapeDtypeStruct((8, 0), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.float32),
        )
    assert error.value.failure_kind == "condition_undecided"


@pytest.mark.parametrize("invalid", ["dimensions", "size"])
def test_empty_reshape_still_checks_the_view_contract(invalid):
    from jax_reorder.core.model import ParameterView
    from jax_reorder.core.tensor_values import tensor_value

    view = adapt(
        jax.make_jaxpr(lambda x: lax.reshape(x, (8, 1, 0)))(
            jax.ShapeDtypeStruct((8, 0), jnp.float32)
        )
    ).view
    (op,) = view.root.operations
    if invalid == "dimensions":
        op = replace(
            op,
            parameters=tuple(
                ParameterView(p.name, (0, 0)) if p.name == "dimensions" else p
                for p in op.parameters
            ),
        )
    else:
        op = replace(
            op, results=(replace(op.results[0], type=replace(op.results[0].type, shape=(8,))),)
        )
    with pytest.raises(UnsupportedJaxprError, match="permutation|number of elements"):
        tensor_value(
            op,
            (IndexExpr("axis", value="local:0"),),
            (view.root.inputs[0].type,),
            {},
            {},
            (),
            AnalysisBudget(),
        )


def rank_one_reduction(kind, block=128, *, guarded=False, reverse=False, interpret=False):
    size = 256
    spec = pl.BlockSpec((block,), lambda p: (size // block - 1 - p if reverse else p,))

    def kernel(x, y):
        def body():
            data = x[:]
            value = {"sum": jnp.sum, "max": jnp.max, "dot": lambda v: jnp.dot(v, v)}[kind](data)
            y[:] = jnp.broadcast_to(value, (block,))

        if guarded:
            lax.cond(pl.program_id(0) >= 0, body, body)
        else:
            body()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((size,), jnp.float32),
        in_specs=(spec,),
        out_specs=spec,
        grid=(size // block,),
        interpret=interpret,
    )


def rank_one_broadcast(
    width=1, *, placement="direct", source="iota", explicit=False, interpret=False
):
    size, block = 256, 128
    spec = pl.BlockSpec((block,), lambda p: (p,))

    def bias(value, extent):
        value = (
            jnp.arange(width, dtype=jnp.int32).astype(jnp.float32) + 3
            if source == "iota"
            else value
            if source == "input"
            else jnp.full((1,), 3, dtype=jnp.float32)
        )
        return jnp.broadcast_to(value, (extent,)) if explicit else value

    def kernel(x, *refs):
        def body():
            refs[-1][:] = (
                x[:]
                if placement == "outer"
                else x[:] + bias(refs[0][:] if source == "input" else None, block)
            )

        if placement == "cond":
            lax.cond(pl.program_id(0) >= 0, body, body)
        else:
            body()

    inputs = (
        (spec, pl.BlockSpec((1,), lambda p: (0,)))
        if source == "input" and placement != "outer"
        else (spec,)
    )
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((size,), jnp.float32),
        in_specs=inputs,
        out_specs=spec,
        grid=(size // block,),
        interpret=interpret,
    )
    if placement == "outer":
        return lambda x, *args: call(x + bias(args[0] if args else None, size))
    return call


@pytest.mark.parametrize("source", ["iota", "input", "constant"])
@pytest.mark.parametrize("placement", ["direct", "cond"])
def test_rank_one_broadcast_projects_the_singleton_coordinate(source, placement):
    inputs = (np.arange(256, dtype=np.float32),)
    if source == "input":
        inputs += (np.asarray([7], dtype=np.float32),)
    expected = inputs[0] + (7 if source == "input" else 3)
    function = rank_one_broadcast(placement=placement, source=source)
    source_jaxpr = jax.make_jaxpr(function)(*inputs)
    projection = normalize_view(adapt(source_jaxpr).view, budget=AnalysisBudget()).projection
    np.testing.assert_array_equal(evaluate(projection.outputs[0], inputs, expected.shape), expected)
    actual = rank_one_broadcast(placement=placement, source=source, interpret=True)(*inputs)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("placement", ["direct", "cond"])
def test_rank_one_broadcast_keeps_distinct_coordinate_projections(placement):
    signature = jax.ShapeDtypeStruct((256,), jnp.float32)
    views = tuple(
        adapt(jax.make_jaxpr(rank_one_broadcast(width, placement=placement))(signature)).view
        for width in (1, 128)
    )
    outputs = tuple(
        normalize_view(view, budget=AnalysisBudget()).projection.outputs for view in views
    )
    assert outputs[0] != outputs[1]
    with pytest.raises(MathematicalUndecidedError) as error:
        compare_program_views(*views)
    assert error.value.failure_kind == "cas_undecided"


@pytest.mark.parametrize("kind", ["sum", "max", "dot"])
@pytest.mark.parametrize("guarded", [False, True])
def test_rank_one_reduction_retains_its_input_domain(kind, guarded):
    inputs = np.arange(256, dtype=np.float32)
    views, outputs = [], []
    for block in (128, 256):
        source = jax.make_jaxpr(rank_one_reduction(kind, block, guarded=guarded))(inputs)
        view = adapt(source).view
        views.append(view)
        projection = normalize_view(view, budget=AnalysisBudget()).projection
        expression = projection.outputs[0]
        assert isinstance(expression, ValueExpr) and expression.operation.startswith("tensor.")
        parameters = dict((p.name, p.value) for p in expression.parameters)
        assert parameters["__input_shapes"] == ((block,),) * (2 if kind == "dot" else 1)
        chunks = inputs.reshape(-1, block)
        reduced = (
            np.sum(chunks, axis=1)
            if kind == "sum"
            else np.max(chunks, axis=1)
            if kind == "max"
            else np.einsum("ij,ij->i", chunks, chunks)
        )
        actual = np.asarray(
            rank_one_reduction(kind, block, guarded=guarded, interpret=True)(inputs)
        )
        np.testing.assert_array_equal(actual, np.repeat(reduced, block))
        outputs.append(actual)
    assert not np.array_equal(*outputs)
    with pytest.raises(MathematicalUndecidedError, match="mathematical backend"):
        compare_program_views(*views)


@pytest.mark.parametrize("kind", ["sum", "max", "dot"])
@pytest.mark.parametrize("guarded", [False, True])
def test_rank_one_reduction_preserves_equivalent_program_traversals(kind, guarded):
    signature = jax.ShapeDtypeStruct((256,), jnp.float32)
    left, right = (
        adapt(
            jax.make_jaxpr(rank_one_reduction(kind, guarded=guarded, reverse=reverse))(signature)
        ).view
        for reverse in (False, True)
    )
    result = compare_program_views(left, right)
    assert result.status is Status.EQUIVALENT and result.equivalence_kind == "normalized"


@pytest.mark.parametrize("projector", ["straight", "control", "tensor"])
def test_unknown_value_semantics_cannot_enter_a_pointwise_projection(projector):
    if projector == "tensor":
        source = trace()
        primitive = "add"
    else:
        source = jax.make_jaxpr(rank_one_reduction("sum", guarded=projector == "control"))(
            jax.ShapeDtypeStruct((256,), jnp.float32)
        )
        primitive = "reduce_sum"

    def replace_value_rule(region):
        return replace(
            region,
            operations=tuple(
                replace(
                    op,
                    primitive="unreviewed_value_rule"
                    if op.primitive == primitive
                    else op.primitive,
                    regions=tuple(
                        replace(child, region=replace_value_rule(child.region))
                        for child in op.regions
                    ),
                )
                for op in region.operations
            ),
        )

    view = adapt(source).view
    (outer,) = view.root.operations
    call = replace(outer.call, kernel=replace_value_rule(outer.call.kernel))
    changed = replace(view, root=replace(view.root, operations=(replace(outer, call=call),)))
    with pytest.raises(UnsupportedJaxprError, match="tensor coordinate rule"):
        normalize_view(changed, budget=AnalysisBudget())


def function(
    kind="add",
    *,
    shape=(16, 256),
    block=(8, 128),
    permutation=None,
    reverse=(),
    mapping_error=None,
    parallel=False,
    interpret=False,
):
    rank = len(shape)
    permutation = tuple(range(rank)) if permutation is None else permutation
    tiles = tuple((n + b - 1) // b for n, b in zip(shape, block, strict=True))
    grid = tuple(tiles[i] for i in permutation)

    def mapping(*p):
        result = tuple(
            tiles[i] - 1 - p[permutation.index(i)] if i in reverse else p[permutation.index(i)]
            for i in range(rank)
        )
        if mapping_error == "duplicate":
            result = (jnp.int32(0), *result[1:])
        if mapping_error == "oob":
            result = (result[0] + 1, *result[1:])
        return result

    spec = pl.BlockSpec(block, mapping)
    scratch = ()
    if kind in ("scratch", "uninitialized_scratch"):
        scratch = (tpu.VMEM(block, jnp.float32),)
    elif kind.startswith("dma"):
        scratch = (tpu.SemaphoreType.DMA,)
    elif kind in ("counter", "loop"):
        scratch = (tpu.SemaphoreType.REGULAR,)

    def kernel(x, y, *resources):
        full = (slice(None),) * rank
        if kind == "scratch":
            resources[0][full] = x[full] + 1
            y[full] = resources[0][full]
        elif kind == "uninitialized_scratch":
            y[full] = resources[0][full]
        elif kind in ("dma", "dma_early", "dma_pending"):
            copy = tpu.make_async_copy(x, y, resources[0])
            copy.start()
            if kind == "dma_early":
                y[full] = y[full] + 1
            if kind != "dma_pending":
                copy.wait()
        elif kind == "dma_rows":
            first = tpu.async_copy(x.at[:4, :], y.at[4:, :], resources[0])
            first.wait()
            tpu.async_copy(x.at[4:, :], y.at[:4, :], resources[0]).wait()
        elif kind == "alias":
            y[full] = x[full]
            y[:, :64] = y[:, :64] + 1
            y[:, 64:] = x[:, 64:] + 2
        elif kind == "counter":
            pl.semaphore_signal(resources[0], 1)
            pl.semaphore_wait(resources[0], 1)
            y[full] = x[full] + 1
        elif kind == "loop":
            y[full] = x[full]

            def body(i, _):
                pl.semaphore_signal(resources[0], 1)
                y[full] = y[full] + i
                pl.semaphore_wait(resources[0], 1)

            lax.fori_loop(0, 3, body, None)
        elif kind in ("columns", "gap", "scalar"):
            half = block[1] // 2
            y[:, :half] = x[:, half:]
            if kind != "gap":
                y[:, half:] = x[:, :half]
            if kind == "scalar":
                y[full] = y[full] + x[2, 3]
        elif kind == "rows":
            y[:4, :] = x[4:, :]
            y[4:, :] = x[:4, :]
        elif kind == "branch":
            lax.cond(
                x[0, 0] > 0,
                lambda: y.__setitem__(full, x[full] + 1),
                lambda: y.__setitem__(full, x[full] - 1),
            )
        elif kind in ("tail", "tail_gap", "tail_unsafe"):
            rows = jnp.arange(block[0])[:, None]
            cols = jnp.arange(block[1])[None, :]
            p = mapping(*(pl.program_id(i) for i in range(len(grid))))
            mask = (p[0] * block[0] + rows < shape[0]) & (p[1] * block[1] + cols < shape[1])
            data = pp.load(x, (rows, cols), mask=None if kind == "tail_unsafe" else mask, other=0)
            write_mask = (
                mask if kind != "tail_gap" else mask & (p[1] * block[1] + cols < shape[1] - 1)
            )
            pp.store(y, (rows, cols), data + 1, mask=write_mask)
        elif kind == "coordinate":
            y[full] = jnp.broadcast_to(jnp.float32(pl.program_id(0)), block)
        else:
            y[full] = x[full] + (2 if kind == "add_two" else 1)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=grid,
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=scratch,
        input_output_aliases={0: 0} if kind == "alias" else {},
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("parallel",) * len(grid) if parallel else None
        ),
        interpret=interpret,
    )


def trace(**kwargs):
    shape = kwargs.get("shape", (16, 256))
    return jax.make_jaxpr(function(**kwargs))(jax.ShapeDtypeStruct(shape, jnp.float32))


def proof(**kwargs):
    return normalize_view(adapt(trace(**kwargs)).view, budget=AnalysisBudget())


def evaluate(expression, inputs, shape):
    def point(value, coordinates, cache):
        if id(value) in cache:
            return cache[id(value)]
        if type(value) is bool:
            result = value
        elif isinstance(value, IndexExpr):
            args = [point(v, coordinates, cache) for v in value.args]
            if value.kind == "constant":
                result = int(value.value)
            elif value.kind == "axis":
                assert str(value.value).startswith("output:")
                result = coordinates[int(str(value.value).split(":")[1])]
            elif value.kind == "sum":
                result = sum(args)
            elif value.kind == "scale":
                result = args[0] * int(value.value)
            elif value.kind == "div":
                result = args[0] // int(value.value)
            else:
                assert value.kind == "mod"
                result = args[0] % int(value.value)
        elif isinstance(value, Predicate):
            args = [point(v, coordinates, cache) for v in value.operands]
            result = {
                "lt": lambda a, b: a < b,
                "le": lambda a, b: a <= b,
                "ge": lambda a, b: a >= b,
                "gt": lambda a, b: a > b,
                "eq": lambda a, b: a == b,
                "ne": lambda a, b: a != b,
                "and": lambda *v: all(v),
                "or": lambda *v: any(v),
                "not": lambda a: not a,
            }[value.operation](*args)
        else:
            assert isinstance(value, ValueExpr)
            if value.operation == "select_n":
                result = point(
                    value.operands[int(point(value.operands[0], coordinates, cache)) + 1],
                    coordinates,
                    cache,
                )
            else:
                args = [point(v, coordinates, cache) for v in value.operands]
                if value.operation == "input":
                    array = inputs[value.input]
                    assert len(args) == array.ndim and all(
                        0 <= i < n for i, n in zip(args, array.shape, strict=True)
                    )
                    result = array[tuple(args)]
                elif value.operation == "constant_array":
                    array = np.frombuffer(value.data, dtype=value.type.dtype).reshape(
                        dict((p.name, p.value) for p in value.parameters)["shape"]
                    )
                    result = array[tuple(args)]
                elif value.operation == "literal":
                    result = np.frombuffer(value.data, dtype=value.type.dtype)[0]
                elif value.operation == "convert_element_type":
                    result = np.asarray(args[0]).astype(value.type.dtype)
                else:
                    result = {
                        "add": np.add,
                        "sub": np.subtract,
                        "mul": np.multiply,
                        "gt": np.greater,
                        "eq": np.equal,
                        "clamp": lambda lo, a, hi: np.clip(a, lo, hi),
                    }[value.operation](*args)
        cache[id(value)] = result
        return result

    return np.array([point(expression, c, {}) for c in np.ndindex(shape)]).reshape(shape)


@pytest.mark.parametrize(
    "shape,block,permutation,reverse",
    [
        ((16, 256), (8, 128), (0, 1), ()),
        ((24, 256), (8, 128), (1, 0), (0, 1)),
        ((4, 16, 256), (2, 8, 128), (2, 0, 1), (2,)),
        ((1, 16, 256), (1, 8, 128), (1, 2, 0), ()),
    ],
)
def test_tensor_grid_maps_preserve_all_axes(shape, block, permutation, reverse):
    source = trace(shape=shape, block=block, permutation=permutation, reverse=reverse)
    result = compare_pallas_kernels(source, source, check_protocol=True)
    verify_pallas_kernel_comparison(source, source, result)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x,), shape), x + 1
    )


@pytest.mark.parametrize(
    "kind", ["columns", "rows", "scalar", "scratch", "dma", "dma_rows", "counter"]
)
def test_rectangles_and_resources_keep_element_correspondence(kind):
    shape = (16, 256)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    actual = evaluate(proof(kind=kind).projection.outputs[0], (x,), shape)
    expected = np.empty_like(x)
    for i in range(0, 16, 8):
        for j in range(0, 256, 128):
            tile = x[i : i + 8, j : j + 128]
            if kind in ("columns", "scalar"):
                expected[i : i + 8, j : j + 128] = np.roll(tile, 64, axis=1) + (
                    tile[2, 3] if kind == "scalar" else 0
                )
            elif kind in ("rows", "dma_rows"):
                expected[i : i + 8, j : j + 128] = np.roll(tile, 4, axis=0)
            else:
                expected[i : i + 8, j : j + 128] = tile + (0 if kind == "dma" else 1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("gap", "uninitialized"),
        ("uninitialized_scratch", "uninitialized"),
        ("dma_early", "before completion"),
        ("dma_pending", "DMA destination accessed before completion wait"),
    ],
)
def test_tensor_unsafe_resources_are_rejected(kind, reason):
    with pytest.raises(UnsupportedJaxprError, match=reason):
        proof(kind=kind)


@pytest.mark.parametrize("mapping_error", ["duplicate", "oob"])
def test_tensor_invalid_writer_map_is_preserved_structurally_but_not_proved(mapping_error):
    source = trace(mapping_error=mapping_error)
    assert compare_pallas_kernels(source, source).status is Status.EQUIVALENT
    with pytest.raises(UnsupportedJaxprError, match="bijection|out-of-bounds|redundant"):
        compare_pallas_kernels(source, source, check_protocol=True)


def test_tensor_retile_and_permute_match_in_core_without_cas():
    left = adapt(trace()).view
    right = adapt(trace(block=(4, 64), permutation=(1, 0), reverse=(0,))).view
    result = compare_program_views(left, right)
    assert result.status is Status.EQUIVALENT and result.equivalence_kind == "normalized"


def test_tensor_projection_reaches_public_algebra():
    source = trace()
    reference = jax.make_jaxpr(lambda x: x + 1)(jax.ShapeDtypeStruct((16, 256), jnp.float32))
    assert compare_algebraic_jaxprs(source, reference).status is Status.EQUIVALENT


def test_large_tensor_grid_is_symbolic():
    source = trace(shape=(2**10, 2**20), block=(8, 128))
    budget = AnalysisBudget(256)
    result = normalize_view(adapt(source).view, budget=budget)
    assert result.projection.domains[0].size == 2**30 and budget.used < 256


@pytest.mark.parametrize("kind", ["tail", "tail_gap", "tail_unsafe"])
def test_tensor_tail_masks_cover_both_axes(kind):
    if kind == "tail_gap":
        with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
            proof(kind=kind, shape=(13, 239))
        return
    source = trace(kind=kind, shape=(13, 239))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(13 * 239, dtype=np.float32).reshape(13, 239)
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x,), x.shape), x + 1
    )


@pytest.mark.parametrize("kind", ["loop", "branch"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_tensor_control_retains_resource_and_value_bindings(kind, sign):
    x = sign * np.arange(1, 16 * 256 + 1, dtype=np.float32).reshape(16, 256)
    actual = evaluate(proof(kind=kind).projection.outputs[0], (x,), x.shape)
    np.testing.assert_array_equal(actual, x + (3 if kind == "loop" else sign))


@pytest.mark.parametrize("mode", ["split", "merge", "singleton"])
def test_grid_rank_and_array_rank_need_not_match(mode):
    if mode == "split":
        shape, block, grid, mapping = (16, 256), (8, 128), (4,), lambda p: (p // 2, p % 2)
    elif mode == "merge":
        shape, block, grid, mapping = (512,), (128,), (2, 2), lambda i, j: (i * 2 + j,)
    else:
        shape, block, grid, mapping = (16, 256), (8, 128), (1, 2, 2), lambda unused, i, j: (i, j)
    spec = pl.BlockSpec(block, mapping)
    full = (slice(None),) * len(shape)

    def kernel(x, y):
        y[full] = x[full] + 1

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=grid,
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct(shape, jnp.float32))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x,), shape), x + 1
    )
    if mode != "merge":
        assert (
            compare_program_views(adapt(source).view, adapt(trace()).view).status
            is Status.EQUIVALENT
        )


def test_transpose_preserves_array_axes_and_block_maps():
    source_spec = pl.BlockSpec((8, 128), lambda i, j: (i, j))
    target_spec = pl.BlockSpec((128, 8), lambda i, j: (j, i))

    def kernel(x, y):
        y[:, :] = x[:, :].T

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((256, 16), jnp.float32),
        grid=(2, 2),
        in_specs=(source_spec,),
        out_specs=target_spec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((16, 256), jnp.float32))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x,), (256, 16)), x.T
    )


def test_input_broadcast_has_no_unique_writer_requirement():
    row_spec = pl.BlockSpec((128,), lambda i, j: (j,))
    matrix_spec = pl.BlockSpec((8, 128), lambda i, j: (i, j))

    def kernel(x, row, y):
        y[:, :] = x[:, :] + row[:][None, :]

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
        grid=(2, 2),
        in_specs=(matrix_spec, row_spec),
        out_specs=matrix_spec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(
        jax.ShapeDtypeStruct((16, 256), jnp.float32), jax.ShapeDtypeStruct((256,), jnp.float32)
    )
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
    row = np.arange(256, dtype=np.float32)
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x, row), x.shape), x + row[None, :]
    )


def test_tensor_write_versions_and_alias_preserve_source_values():
    spec = pl.BlockSpec((8, 128), lambda i, j: (i, j))

    def kernel(x, y):
        y[:, :] = x[:, :]
        y[:, :64] = y[:, :64] + 1
        y[:, 64:] = x[:, 64:] + 2

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
        grid=(2, 2),
        in_specs=(spec,),
        out_specs=spec,
        input_output_aliases={0: 0},
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((16, 256), jnp.float32))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
    expected = x + np.where(np.arange(256) % 128 < 64, 1, 2)[None, :]
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x,), x.shape), expected
    )
    projection = result.left_protocol.projection
    forged = replace(
        result,
        left_protocol=replace(
            result.left_protocol,
            projection=replace(
                projection, versions=tuple(replace(v, rectangle=()) for v in projection.versions)
            ),
        ),
    )
    from jax_reorder import CorrespondenceError

    with pytest.raises(CorrespondenceError):
        verify_pallas_kernel_comparison(source, source, forged)


def test_outer_matrix_and_row_preprocessing_keep_input_coordinates():
    f = function()
    source = jax.make_jaxpr(lambda x, row: f(x + row[None, :]))(
        jax.ShapeDtypeStruct((16, 256), jnp.float32), jax.ShapeDtypeStruct((256,), jnp.float32)
    )
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
    row = np.arange(256, dtype=np.float32)
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x, row), x.shape),
        x + row[None, :] + 1,
    )


def test_changed_input_map_cannot_be_replaced_with_the_output_map():
    shape = (16, 256)
    inspec = pl.BlockSpec((8, 128), lambda i, j: (1 - i, j))
    outspec = pl.BlockSpec((8, 128), lambda i, j: (i, j))

    def kernel(x, y):
        y[:, :] = x[:, :] + 1

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(2, 2),
        in_specs=(inspec,),
        out_specs=outspec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct(shape, jnp.float32))
    assert compare_pallas_kernels(trace(), source).status is Status.DIFFERENT
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x,), shape), np.roll(x, 8, axis=0) + 1
    )


@pytest.mark.parametrize("mode", ["branch_gap", "data_index", "noncontiguous_dma", "stride"])
def test_additional_tensor_rejection_boundaries(mode):
    spec = pl.BlockSpec((8, 128), lambda i, j: (i, j))

    def kernel(x, y, *sem):
        if mode == "branch_gap":
            lax.cond(
                x[0, 0] > 0,
                lambda: y.__setitem__((slice(None), slice(None)), x[:, :]),
                lambda: None,
            )
        elif mode == "data_index":
            y[:, :] = x[:, :] + x[x[0, 0].astype(jnp.int32), 0]
        elif mode == "stride":
            y[:, :64] = x[:, ::2]
            y[:, 64:] = x[:, 1::2]
        else:
            tpu.async_copy(x.at[:, :64], y.at[:, :64], sem[0]).wait()
            y[:, 64:] = x[:, 64:]

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
        grid=(2, 2),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(tpu.SemaphoreType.DMA,) if mode == "noncontiguous_dma" else (),
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((16, 256), jnp.float32))
    assert compare_pallas_kernels(source, source).status is Status.EQUIVALENT
    if mode == "stride":
        result = compare_pallas_kernels(source, source, check_protocol=True)
        x = np.arange(16 * 256, dtype=np.float32).reshape((16, 256))
        blocks = x.reshape((16, 2, 128))
        expected = np.concatenate((blocks[:, :, ::2], blocks[:, :, 1::2]), axis=-1)
        np.testing.assert_array_equal(
            evaluate(result.left_protocol.projection.outputs[0], (x,), x.shape),
            expected.reshape(x.shape),
        )
        return
    with pytest.raises(
        UnsupportedJaxprError, match="uninitialized|static integer|noncontiguous|unit stride|bounds"
    ):
        compare_pallas_kernels(source, source, check_protocol=True)


def test_parallel_tensor_counter_does_not_guess_a_core_target():
    with pytest.raises(UnsupportedJaxprError, match="core target"):
        proof(kind="counter", parallel=True)


def test_three_dimensional_flat_grid_is_a_symbolic_mixed_radix_map():
    shape, block = (4, 16, 256), (2, 8, 128)
    spec = pl.BlockSpec(block, lambda p: (p // 4, (p // 2) % 2, p % 2))

    def kernel(x, y):
        y[:, :, :] = x[:, :, :] + 1

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(8,),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct(shape, jnp.float32))
    assert (
        compare_program_views(
            adapt(source).view, adapt(trace(shape=shape, block=block)).view
        ).status
        is Status.EQUIVALENT
    )


def run_oracle(kind):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    params = tpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=2 if kind.startswith("parallel_") else 1,
    )
    shape = (13, 239) if kind == "tail" else (16, 256)
    options = {
        "kind": kind.removeprefix("parallel_"),
        "shape": shape,
        "parallel": kind.startswith("parallel_"),
    }
    x = np.arange(prod(shape), dtype=np.float32).reshape(shape)
    compiled = (
        jax.jit(function(**options, interpret=params))
        .lower(jax.ShapeDtypeStruct(shape, jnp.float32))
        .compile()
    )
    print("READY", flush=True)
    actual = jax.block_until_ready(compiled(x))
    jax.effects_barrier()
    expected = evaluate(proof(**options).projection.outputs[0], (x,), shape)
    np.testing.assert_array_equal(actual, expected)
    assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps({"case": kind, "status": "ok", "races": False}), flush=True)


@pytest.mark.pallas_communication
@pytest.mark.parametrize(
    "kind",
    [
        "add",
        "columns",
        "scratch",
        "dma",
        "dma_rows",
        "counter",
        "tail",
        "alias",
        "loop",
        "branch",
        "parallel_scratch",
        "parallel_dma",
    ],
)
def test_tensor_official_cpu_oracle(kind):
    expected = {"case": kind, "status": "ok", "races": False}

    def run():
        return run_worker(
            "tests.test_pallas_tensor_memory", kind, expected=expected, entry="run_oracle"
        )

    if kind == "tail":
        with pytest.raises(AssertionError, match="NotImplementedError: masked load_p"):
            run()
    else:
        assert run() == expected


def test_tensor_tail_against_generic_cpu_interpret():
    x = np.arange(13 * 239, dtype=np.float32).reshape(13, 239)
    actual = function(kind="tail", shape=x.shape, interpret=True)(x)
    np.testing.assert_array_equal(actual, x + 1)


def test_float_iota_is_not_an_exact_integer_coordinate():
    def kernel(y):
        y[:, :] = lax.broadcasted_iota(jnp.float32, (8, 128), 1)

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        grid=(1, 1),
        out_specs=pl.BlockSpec((8, 128), lambda i, j: (i, j)),
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)()
    assert compare_pallas_kernels(source, source).status is Status.EQUIVALENT
    with pytest.raises(UnsupportedJaxprError, match="integer coordinate dtype"):
        compare_pallas_kernels(source, source, check_protocol=True)


def test_program_axis_values_are_not_replaced_by_array_axes():
    x = np.zeros((16, 256), dtype=np.float32)
    a = proof(kind="coordinate").projection.outputs[0]
    b = proof(kind="coordinate", permutation=(1, 0)).projection.outputs[0]
    np.testing.assert_array_equal(
        evaluate(a, (x,), x.shape), np.broadcast_to((np.arange(16) // 8)[:, None], x.shape)
    )
    np.testing.assert_array_equal(
        evaluate(b, (x,), x.shape), np.broadcast_to((np.arange(256) // 128)[None, :], x.shape)
    )


@pytest.mark.parametrize("mode", ["finite", "data", "infinite"])
def test_tensor_while_termination_is_proved_or_rejected(mode):
    spec = pl.BlockSpec((8, 128), lambda i, j: (i, j))

    def kernel(x, y):
        y[:, :] = x[:, :]

        def condition(i):
            return (
                i < 3 if mode == "finite" else (x[0, 0] > i if mode == "data" else jnp.bool_(True))
            )

        def body(i):
            y[:, :] = y[:, :] + 1
            return i + 1

        lax.while_loop(condition, body, jnp.int32(0))

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
        grid=(2, 2),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct((16, 256), jnp.float32))
    from jax_reorder import ComparisonBudgetError

    if mode != "finite":
        with pytest.raises(
            ComparisonBudgetError if mode == "infinite" else UnsupportedJaxprError,
            match="budget|finite carry",
        ):
            normalize_view(adapt(source).view, budget=AnalysisBudget(100))
    else:
        result = normalize_view(adapt(source).view, budget=AnalysisBudget())
        x = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
        np.testing.assert_array_equal(evaluate(result.projection.outputs[0], (x,), x.shape), x + 3)


def test_outer_tensor_constant_preserves_every_element():
    f = function()
    constant = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
    source = jax.make_jaxpr(lambda x: f(x + constant))(
        jax.ShapeDtypeStruct(constant.shape, jnp.float32)
    )
    result = compare_pallas_kernels(source, source, check_protocol=True)
    x = constant + 7
    np.testing.assert_array_equal(
        evaluate(result.left_protocol.projection.outputs[0], (x,), x.shape), x + constant + 1
    )


def test_single_block_axes_and_large_mixed_radix_domains_stay_symbolic():
    a = adapt(trace(shape=(8, 128), block=(8, 128))).view
    b = adapt(trace(shape=(8, 128), block=(4, 64))).view
    assert compare_program_views(a, b).status is Status.EQUIVALENT
    shape = (2**10, 2**20)
    spec = pl.BlockSpec((8, 128), lambda p: (p // 8192, p % 8192))

    def kernel(x, y):
        y[:, :] = x[:, :] + 1

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
        grid=(2**20,),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(),
    )
    source = jax.make_jaxpr(f)(jax.ShapeDtypeStruct(shape, jnp.float32))
    budget = AnalysisBudget(1000)
    result = normalize_view(adapt(source).view, budget=budget)
    assert result.projection.domains[0].size == 2**30 and budget.used < 1000


def test_tensor_local_device_coordinate_keeps_its_mesh_range():
    from jax.sharding import PartitionSpec as P

    from jax_reorder.core.index_domain import integer
    from jax_reorder.core.memory import _substitute
    from tests.test_pallas_sharding import device_mesh

    spec = pl.BlockSpec((8, 128), lambda i, j: (i, j))

    def kernel(x, y):
        y[:, :] = x[:, :] + jnp.float32(lax.axis_index("d"))

    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
        grid=(2, 2),
        in_specs=(spec,),
        out_specs=spec,
        compiler_params=tpu.CompilerParams(),
    )
    mapped = jax.shard_map(
        f, mesh=device_mesh(), in_specs=P("d", None), out_specs=P("d", None), check_vma=False
    )
    source = jax.make_jaxpr(mapped)(jax.ShapeDtypeStruct((32, 256), jnp.float32))
    result = compare_pallas_kernels(source, source, check_protocol=True)
    assert result.evidence.axes[0][:2] == ("d", 2)
    x = np.arange(16 * 256, dtype=np.float32).reshape(16, 256)
    expression = _substitute(result.left_protocol.projection.outputs[0], {"device:d": integer(1)})
    np.testing.assert_array_equal(evaluate(expression, (x,), x.shape), x + 1)
