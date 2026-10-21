"""Finite index maps, independent owner suffixes and grouped padded windows."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas import primitives as pp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from jax_reorder import ComparisonBudgetError, Status, UnsupportedJaxprError, compare_pallas_kernels
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.call_bindings import extract_call
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.tensor_memory import functionalize_tensor_call
from tests.pallas_oracle import run_worker
from tests.pallas_projection_oracle import at_point


def grouped_tail(mode="base", *, interpret=False, steps=4, width=200):
    def body(x, y, total):
        p, j = pl.program_id(0), pl.program_id(1)
        rows, cols = jnp.arange(8)[:, None], jnp.arange(128)[None, :]
        input_mask = (p * 8 + rows < 13) & (j * 128 + cols < steps * 128)
        total_mask = (p * 8 + rows < 13) & (j // 2 * 128 + cols < width)
        if mode == "gap":
            total_mask &= j // 2 * 128 + cols < width - 1
        elif mode == "unsafe":
            total_mask = p * 8 + rows < 13

        @pl.when(j % 2 == 0)
        def initialize():
            pp.store(total, (rows, cols), jnp.zeros((8, 128), jnp.float32), mask=total_mask)

        value = pp.load(x, (rows, cols), mask=input_mask, other=0)
        pp.store(y, (rows, cols), value * 2 + j, mask=input_mask)
        old = pp.load(total, (rows, cols), mask=total_mask, other=0)
        pp.store(total, (rows, cols), old + value, mask=total_mask)

    full = pl.BlockSpec((8, 128), lambda p, j: (p, j))
    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((13, steps * 128), jnp.float32),
            jax.ShapeDtypeStruct((13, width), jnp.float32),
        ),
        grid=(2, steps),
        interpret=interpret,
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, 128), lambda p, j: (p, j // 2))),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )


def native_tail(*, steps=4, width=200, interpret=False):
    """TPU-native slice loads and masked stores, with independent tail outputs."""

    def body(x, y, total):
        p, j = pl.program_id(0), pl.program_id(1)
        rows, cols = jnp.arange(8)[:, None], jnp.arange(128)[None, :]
        y_mask = jnp.broadcast_to(p * 8 + rows < 13, (8, 128))
        total_mask = (p * 8 + rows < 13) & (j // 2 * 128 + cols < width)
        value = x[...]
        pp.store(y, (slice(None), slice(None)), value * 2 + j, mask=y_mask)
        pp.store(total, (slice(None), slice(None)), value, mask=total_mask)

    full = pl.BlockSpec((8, 128), lambda p, j: (p, j))
    return pl.pallas_call(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((13, steps * 128), jnp.float32),
            jax.ShapeDtypeStruct((13, width), jnp.float32),
        ),
        grid=(2, steps),
        in_specs=[full],
        out_specs=(full, pl.BlockSpec((8, 128), lambda p, j: (p, j // 2))),
        interpret=interpret,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )


def masked_reduction(*, interpret=False):
    def body(x, y):
        p, j = pl.program_id(0), pl.program_id(1)
        rows, cols = jnp.arange(8)[:, None], jnp.arange(128)[None, :]
        mask = (p * 8 + rows < 13) & (j * 128 + cols < 239)
        values = pp.load(x, (rows, cols), mask=mask, other=7)
        pp.store(
            y,
            (slice(None), slice(None)),
            jnp.sum(values, axis=1, keepdims=True),
            mask=p * 8 + rows < 13,
        )

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((13, 2), jnp.float32),
        grid=(2, 2),
        in_specs=[pl.BlockSpec((8, 128), lambda p, j: (p, j))],
        out_specs=pl.BlockSpec((8, 1), lambda p, j: (p, j)),
        interpret=interpret,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )


def permutation_map(kind="select", *, collision=False):
    def mapping(p, j):
        if kind == "coupled":
            return (p + j) % 2, j
        if kind == "loop":
            j = jax.lax.fori_loop(0, j, lambda _, n: n + 1, 0)
        if kind == "cond":
            k = jax.lax.cond(j < 2, lambda: 2 - 2 * j, lambda: 7 - 2 * j)
        else:
            k = jnp.where(j < 2, 2 - 2 * j, 7 - 2 * j)
        return p, jnp.where(j == 3, 0, k) if collision else k

    return mapping


def permutation(kind="select", *, collision=False, groups=2):
    def body(x, y):
        y[...] = x[...] + pl.program_id(1) + 10 * pl.program_id(0)

    spec = pl.BlockSpec((8, 128), permutation_map(kind, collision=collision))
    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8 * groups, 512), jnp.float32),
        grid=(groups, 4),
        in_specs=[spec],
        out_specs=spec,
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )


def grouped_permutation(*, groups=2, kind="select"):
    def body(x, y):
        j = pl.program_id(1)

        @pl.when(j % 2 == 0)
        def initialize():
            y[...] = jnp.zeros((8, 128), jnp.float32)

        y[...] += x[...]

    def mapping(p, j):
        return permutation_map(kind)(p, j // 2)

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8 * groups, 512), jnp.float32),
        grid=(groups, 8),
        in_specs=[pl.BlockSpec((8, 128), lambda p, j: (p, j))],
        out_specs=pl.BlockSpec((8, 128), mapping),
        compiler_params=tpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
    )


def interleaved(*, bad=False):
    def body(x, y, scratch, sem):
        k = pl.program_id(2)

        @pl.when(k == 0)
        def initialize():
            if not bad:
                scratch[...] = jnp.zeros((8, 128), jnp.float32)

        pl.semaphore_signal(sem, 1)
        pl.semaphore_wait(sem, 1)
        scratch[...] += x[...]
        y[...] = scratch[...]

    return pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((16, 256), jnp.float32),
        grid=(2, 2, 3),
        in_specs=[pl.BlockSpec((8, 128), lambda i, p, k: (p, i * 3 + k))],
        out_specs=pl.BlockSpec((8, 128), lambda i, p, k: (p, i)),
        scratch_shapes=[tpu.VMEM((8, 128), jnp.float32), tpu.SemaphoreType.REGULAR],
        compiler_params=tpu.CompilerParams(
            dimension_semantics=("arbitrary", "parallel", "arbitrary")
        ),
    )


def trace(fn, shape):
    return jax.make_jaxpr(fn)(jax.ShapeDtypeStruct(shape, jnp.float32))


def project(source, *, limit=10000, cores=None):
    call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
    return functionalize_tensor_call(call, budget=AnalysisBudget(limit), execution_cores=cores)


def array(shape, seed=3):
    return np.random.default_rng(seed).integers(-4, 5, size=shape).astype(np.float32)


def check(proof, args, expected):
    for value, output in zip(proof.outputs, expected, strict=True):
        output = np.asarray(output)
        for row in (0, output.shape[0] - 1):
            for col in range(output.shape[1]):
                np.testing.assert_equal(at_point(value, args, (row, col)), output[row, col])


@pytest.mark.parametrize("steps,width", [(4, 200), (5, 300)])
def test_grouped_padded_outputs_keep_distinct_domains(steps, width):
    source = trace(grouped_tail(steps=steps, width=width), (13, steps * 128))
    proof = project(source)
    x = array((13, steps * 128))
    tiles = x.reshape(13, steps, 128)
    total = np.concatenate([tiles[:, i : i + 2].sum(axis=1) for i in range(0, steps, 2)], axis=1)[
        :, :width
    ]
    check(proof, (x,), ((tiles * 2 + np.arange(steps)[None, :, None]).reshape(x.shape), total))


@pytest.mark.parametrize("steps,width", [(4, 200), (5, 300)])
def test_native_padded_output_windows_preserve_the_last_write(steps, width):
    x = array((16, steps * 128))
    proof = project(trace(native_tail(steps=steps, width=width), x.shape))
    tiles = x[:13].reshape(13, steps, 128)
    expected = (
        (tiles * 2 + np.arange(steps)[None, :, None]).reshape(13, steps * 128),
        np.concatenate([tiles[:, min(i + 1, steps - 1)] for i in range(0, steps, 2)], axis=1)[
            :, :width
        ],
    )
    check(proof, (x,), expected)
    actual = native_tail(steps=steps, width=width, interpret=True)(x)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_output, expected_output)


def test_padded_reduction_uses_the_masked_load_other_value():
    x = array((13, 239))
    proof = project(trace(masked_reduction(), x.shape))
    expected = np.stack((x[:, :128].sum(axis=1), x[:, 128:].sum(axis=1) + 17 * 7), axis=1)
    # The numerical projection oracle vectorizes both select arms. Supply a
    # sentinel padding buffer, then verify the mask excludes every sentinel.
    padded = np.pad(x, ((0, 3), (0, 17)), constant_values=12345)
    check(proof, (padded,), (expected,))
    np.testing.assert_array_equal(np.asarray(masked_reduction(interpret=True)(x)), expected)


def test_grouped_padded_mask_cannot_leave_a_valid_output_gap():
    with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
        project(trace(grouped_tail("gap"), (13, 512)))


def test_initialized_vmem_padding_can_be_read_without_leaking_to_hbm():
    proof = project(trace(grouped_tail("unsafe"), (13, 512)))
    x = array((13, 512))
    tiles = x.reshape(13, 4, 128)
    expected = (
        (tiles * 2 + np.arange(4)[None, :, None]).reshape(x.shape),
        np.concatenate([tiles[:, i : i + 2].sum(axis=1) for i in (0, 2)], axis=1)[:, :200],
    )
    for value, target in zip(proof.outputs, expected, strict=True):
        for point in np.ndindex(target.shape):
            assert at_point(value, (x,), point) == target[point]


@pytest.mark.parametrize("kind", ["select", "cond", "loop", "coupled"])
def test_finite_irregular_writer_permutation_has_exact_inverse(kind):
    x = array((16, 512))
    source = trace(permutation(kind), x.shape)
    proof = project(source)
    inverse = np.arange(4) if kind == "coupled" else np.argsort([2, 0, 3, 1])
    owner = np.arange(16)[:, None] // 8
    owner = (owner + np.arange(4)[None, :]) % 2 if kind == "coupled" else owner
    expected = (x.reshape(16, 4, 128) + inverse[None, :, None] + 10 * owner[:, :, None]).reshape(
        x.shape
    )
    check(proof, (x,), (expected,))
    assert compare_pallas_kernels(source, source, check_protocol=True).status is Status.EQUIVALENT


@pytest.mark.parametrize("kind", ["select", "cond", "loop"])
def test_grouped_static_permutation_preserves_symbolic_parallel_axis(kind):
    costs = []
    for groups in (4, 2**20):
        source = trace(grouped_permutation(groups=groups, kind=kind), (8 * groups, 1024))
        call = extract_call(adapt(source).view, budget=AnalysisBudget()).operation.call
        budget = AnalysisBudget(3000)
        proof = functionalize_tensor_call(call, budget=budget)
        costs.append(budget.used)
        if groups == 4:
            x = array((32, 1024))
            values = x.reshape(32, 4, 2, 128).sum(axis=2)[:, np.argsort([2, 0, 3, 1])]
            check(proof, (x,), (values.reshape(32, 512),))
    assert costs[0] == costs[1]


def test_static_permutation_collision_is_refused():
    with pytest.raises(UnsupportedJaxprError, match="redundant|bijection"):
        project(trace(permutation(collision=True), (16, 512)))


def test_finite_map_inverse_preserves_large_independent_parallel_axes():
    costs = []
    for groups in (4, 2**20):
        call = extract_call(
            adapt(trace(permutation(groups=groups), (8 * groups, 512))).view,
            budget=AnalysisBudget(),
        ).operation.call
        budget = AnalysisBudget(1000)
        functionalize_tensor_call(call, budget=budget)
        costs.append(budget.used)
    assert costs[0] == costs[1]


def test_finite_map_budget_checked_before_enumeration():
    def mapping(p, j):
        return jnp.where(p % 2 == 0, p + 1, p - 1), j

    def body(x, y):
        y[...] = x[...]

    spec = pl.BlockSpec((8, 128), mapping)
    fn = pl.pallas_call(
        body,
        out_shape=jax.ShapeDtypeStruct((8 * 2**20, 512), jnp.float32),
        grid=(2**20, 4),
        in_specs=[spec],
        out_specs=spec,
    )
    with pytest.raises(ComparisonBudgetError, match="finite_index_map"):
        project(trace(fn, (8 * 2**20, 512)), limit=1000)


def test_interleaved_grid_can_reinitialize_scratch_for_each_contiguous_serial_suffix():
    proof = project(trace(interleaved(), (16, 768)), cores=1)
    x = np.arange(16 * 768, dtype=np.float32).reshape(16, 768)
    for row in (0, 7, 8, 15):
        for group in (0, 1):
            for lane in (0, 127):
                expected = sum(x[row, (group * 3 + k) * 128 + lane] for k in range(3))
                assert at_point(proof.outputs[0], (x,), (row, group * 128 + lane)) == expected
    with pytest.raises(UnsupportedJaxprError):
        project(trace(interleaved(bad=True), (16, 768)), cores=1)


def execute(case):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    if case.startswith("tail"):
        steps, width = (5, 300) if case == "tail5" else (4, 200)
        fn, shape = lambda: grouped_tail(steps=steps, width=width), (13, steps * 128)
    else:
        fn, shape = lambda: permutation(case), (16, 512)
    params = tpu.InterpretParams(
        detect_races=True,
        dma_execution_mode="on_wait",
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
    )
    if case.startswith("tail"):
        # TPU's race interpreter does not implement masked load_p. The official
        # generic interpreter supplies the separate CPU numerical oracle.
        compiled = (
            jax.jit(grouped_tail(interpret=True, steps=steps, width=width))
            .lower(array(shape))
            .compile()
        )
    else:
        with tpu.force_tpu_interpret_mode(params):
            compiled = jax.jit(fn()).lower(array(shape)).compile()
    print("READY", flush=True)
    proof = project(trace(fn(), shape), cores=1 if case == "interleaved" else None)
    for seed in (3, 19):
        x = array(shape, seed)
        actual = jax.block_until_ready(compiled(x))
        jax.effects_barrier()
        actual = actual if isinstance(actual, tuple) else (actual,)
        check(proof, (x,), actual)
        if not case.startswith("tail"):
            assert interpreter.races is not None and not interpreter.races.races_found
    print("RESULT " + json.dumps(dict(case=case, status="ok")), flush=True)


@pytest.mark.pallas_communication
@pytest.mark.parametrize("case", ["tail", "tail5", "select", "cond", "loop", "coupled"])
def test_extended_grid_matches_official_cpu(case):
    run_worker(
        "tests.test_pallas_grid_extended",
        case,
        expected=dict(case=case, status="ok"),
        entry="execute",
        devices=1,
        execution_timeout=60,
    )


@pytest.mark.parametrize(
    "case", ["tail4", "tail5", "select", "cond", "loop", "coupled", "grouped", "interleaved"]
)
def test_extended_grid_offline_tpu_lowering_and_ir_verification(case):
    from tests.pallas_pipeline_cases import environment

    if case.startswith("tail"):
        steps, width = (4, 200) if case == "tail4" else (5, 300)
        fn, shape = lambda: native_tail(steps=steps, width=width), (16, steps * 128)
    elif case == "grouped":
        fn, shape = grouped_permutation, (16, 1024)
    elif case == "interleaved":
        fn, shape = interleaved, (16, 768)
    else:
        fn, shape = lambda: permutation(case), (16, 512)
    with environment():
        lowered = (
            jax.jit(fn())
            .trace(jax.ShapeDtypeStruct(shape, jnp.float32))
            .lower(lowering_platforms=("tpu",))
        )
    ir = lowered.compiler_ir()
    assert ir.operation.verify()
    assert "custom_call" in str(ir)
    # Lowering proves backend IR admissibility, not semantic equivalence or a
    # runtime scheduling contract. In particular interleaved state is refused
    # by this core and by the official TPU interpreter's prefix restriction.
