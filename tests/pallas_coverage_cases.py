"""Fixed source cases for the locked TPU Pallas surface, including combinations.

Abstract TPU metadata enables legal source tracing on CPU. It is not a device,
compiler run, or hardware execution result. Candidates are handwritten below.
"""

from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import config, core
from jax._src.mesh import AbstractDevice
from jax._src.pallas.mosaic import pipeline
from jax._src.pallas.mosaic import primitives as mp
from jax._src.state import primitives as state
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


@dataclass(frozen=True)
class Case:
    name: str
    primitives: tuple[str, ...]
    apis: tuple[str, ...]
    execute: bool = False
    abstract_tpu: bool = False


CASES = (
    Case(
        "index",
        ("program_id", "multiple_of"),
        ("pl.program_id", "pl.num_programs", "pl.multiple_of", "pl.align_to", "pl.cdiv"),
        True,
    ),
    Case("broadcast", ("broadcast_to",), ("pl.broadcast_to",), True),
    Case("ref_views", ("get", "swap"), ("pl.ds", "pl.dslice", "pl.Slice"), True),
    Case("ref_reshape", ("get",), (), True),
    Case("ref_transpose", ("get",), ()),
    Case("ref_bitcast", ("get",), (), True),
    Case("loop", ("scan", "cond"), ("pl.loop", "pl.when"), True),
    Case("scoped", ("run_scoped",), ("pl.run_scoped", "pl.with_scoped"), True),
    Case(
        "new_ref", ("new_ref", "freeze"), ("pl.empty", "pl.empty_like", "pl.empty_ref_like"), True
    ),
    Case("empty_ref", ("empty_ref", "free_ref"), (), True),
    Case("linear", ("pin", "unpin"), ()),
    Case("create_linear", ("create_linear", "unpin"), ()),
    Case("run_state", ("run_state",), ("pl.run_state",), True),
    Case("global", ("get_global",), ("pl.get_global",)),
    Case("addupdate", ("addupdate",), (), True),
    Case("masked", ("masked_load", "masked_swap"), ("tpu.load", "tpu.store"), True),
    Case(
        "dependency",
        ("add_dependency", "delay", "trace_value"),
        ("tpu.touch", "pl.delay", "tpu.trace_value"),
    ),
    Case(
        "semaphore",
        ("semaphore_signal", "semaphore_wait", "semaphore_read"),
        ("pl.semaphore_signal", "pl.semaphore_wait", "pl.semaphore_read"),
    ),
    Case("barrier", ("get_barrier_semaphore",), ("tpu.get_barrier_semaphore",)),
    Case(
        "core_helpers",
        ("semaphore_signal", "semaphore_wait", "cond"),
        ("tpu.core_barrier", "tpu.run_on_first_core"),
    ),
    Case(
        "dma", ("dma_start", "dma_wait"), ("tpu.async_copy", "tpu.make_async_copy", "tpu.sync_copy")
    ),
    Case("select_ref", ("dma_start", "dma_wait"), ("pl.select_ref",)),
    Case("bitcast", ("bitcast",), ("tpu.bitcast",), True),
    Case("roll", ("roll",), ("tpu.roll",), True),
    Case("reciprocal", ("reciprocal",), ("tpu.reciprocal",), True),
    Case(
        "packing",
        ("pack_elementwise", "unpack_elementwise"),
        ("tpu.pack_elementwise", "tpu.unpack_elementwise"),
    ),
    Case("rounding", ("stochastic_round",), ("tpu.stochastic_round",)),
    Case("annotate", ("masked_load",), ("tpu.annotate",), True),
    Case("memory_space", ("with_memory_space_constraint",), ("tpu.with_memory_space_constraint",)),
    Case("einshape", ("call_hi_primitive",), ("tpu.einshape",), True),
    Case(
        "stateful_random",
        ("prng_seed", "prng_random_bits"),
        (
            "tpu.prng_seed",
            "tpu.prng_random_bits",
            "tpu.stateful_bits",
            "tpu.stateful_normal",
            "tpu.stateful_bernoulli",
            "tpu.stateful_uniform",
        ),
    ),
    Case("keyed_random", ("prng_split", "prng_join"), ("tpu.to_pallas_key", "tpu.sample_block")),
    Case(
        "mxu_fifo",
        ("matmul_push_rhs", "matmul_lhs_fifo", "matmul_pop_fifo"),
        ("tpu.matmul_push_rhs", "tpu.matmul_lhs_fifo", "tpu.matmul_pop_fifo"),
        abstract_tpu=True,
    ),
    Case(
        "mxu_acc",
        ("matmul_push_rhs", "matmul_acc_lhs", "matmul_pop"),
        ("tpu.ACC", "tpu.matmul_acc_lhs", "tpu.matmul_pop"),
        abstract_tpu=True,
    ),
    Case("conv", ("conv",), ("tpu.conv",), abstract_tpu=True),
    Case("pipeline", ("emit_pipeline",), ("tpu.emit_pipeline",), abstract_tpu=True),
    Case("pipeline_captures", ("emit_pipeline",), (), abstract_tpu=True),
    Case("pipeline_legacy", ("scan", "dma_start", "dma_wait"), (), abstract_tpu=True),
    Case("pipeline_lowering", ("pipeline_body",), (), abstract_tpu=True),
    Case(
        "pipeline_allocations",
        ("emit_pipeline",),
        ("tpu.emit_pipeline_with_allocations",),
        abstract_tpu=True,
    ),
)


def finish(value, variant):
    if variant == "reorder":
        b, a = jnp.cos(value), jnp.sin(value)
    else:
        a, b = jnp.sin(value), jnp.cos(value)
    return a + b if variant == "wrong" else a - b


def function(case, variant="base", *, interpret=False):
    name = case.name
    shape = (8, 256) if name.startswith("mxu") else (8, 128)
    sig = jax.ShapeDtypeStruct(shape, jnp.float32)
    if name == "pipeline_lowering":
        original = trace_case(next(c for c in CASES if c.name == "pipeline"), variant)
        outer = original.eqns[0].params["grid_mapping"]
        eq = next(e for e in equations(original) if e.primitive is pipeline.emit_pipeline_p)
        lowered = pipeline.emit_pipeline_to_jaxpr(
            [v.aval for v in eq.invars],
            grid_names=outer.grid_names,
            grid_sizes=outer.grid,
            **eq.params,
        )

        def lowered_kernel(x, y):
            # Evaluate the official lowering on tracers, producing a real native
            # pipeline_body graph. This is source tracing, not device execution.
            core.eval_jaxpr(lowered, lowered.consts, x, y)

        return pl.pallas_call(lowered_kernel, out_shape=sig, grid=(1,)), (sig,)
    scratch = ()
    if name in {"dma", "select_ref"}:
        scratch = (tpu.SemaphoreType.DMA,)
    elif name in {"semaphore", "core_helpers"}:
        scratch = (tpu.SemaphoreType.REGULAR((2,)),)
    elif name == "mxu_acc":
        scratch = (tpu.ACC(0)(shape, jnp.float32),)

    def kernel(x, y, *resources):
        value = x[...]
        if name == "index":
            n = pl.num_programs(0)
            index = pl.multiple_of(pl.program_id(0) * 8, 8)
            value += pl.cdiv(n, 2) + pl.align_to(index, 8)
        elif name == "broadcast":
            value = pl.broadcast_to(value[:1], shape)
        elif name == "ref_views":
            idx = jnp.int32(0)
            value = x.at[pl.ds(idx, 8), pl.dslice(0, 128)][...]
        elif name == "ref_reshape":
            value = x.reshape((2, 4, 128))[...].reshape(shape)
        elif name == "ref_transpose":
            value = x.transpose((1, 0))[...].T
        elif name == "ref_bitcast":
            value = lax.bitcast_convert_type(x.bitcast(jnp.int32)[...], jnp.float32)
        elif name == "loop":
            y[...] = value

            @pl.loop(0, 2)
            def body(i):
                @pl.when(i == 1)
                def update():
                    y[...] += 1

            value = y[...]
        elif name == "scoped":

            @pl.with_scoped(tpu.VMEM(shape, jnp.float32), other=tpu.VMEM(shape, jnp.float32))
            def body(first, *, other):
                first[...] = value
                other[...] = first[...] + 1
                return other[...]

            value = body()
        elif name == "new_ref":
            temp = pl.empty_ref_like(tpu.VMEM(shape, jnp.float32))
            temp[...] = value
            value = jax.ref.freeze(temp)
        elif name == "empty_ref":
            temp = core.empty_ref(sig, memory_space=tpu.VMEM)
            temp[...] = value
            value = temp[...]
            core.free_ref(temp)
        elif name == "linear":
            value = state.unpin(state.pin(value, to="vmem"))
        elif name == "create_linear":
            value = state.unpin(state.create_linear(core.ShapedArray(shape, jnp.float32), "vmem"))
        elif name == "run_state":
            value = pl.run_state(lambda ref: ref.__setitem__(Ellipsis, ref[...] + 1))(value)
        elif name == "global":
            temp = pl.get_global(tpu.VMEM(shape, jnp.float32))
            temp[...] = value
            value = temp[...]
        elif name == "addupdate":
            y[...] = value
            jax.ref.addupdate(y, (), value)
            value = y[...]
        elif name == "masked":
            mask = jnp.arange(128)[None, :] < 64
            value = tpu.load(x, mask=jnp.broadcast_to(mask, shape))
            value = jnp.where(mask, value, 0)
            tpu.store(y, value, mask=jnp.ones(shape, jnp.bool_))
        elif name == "dependency":
            tpu.touch(x)
            pl.delay(1)
            tpu.trace_value("total", jnp.sum(value))
        elif name == "semaphore":
            (sem,) = resources
            pl.semaphore_signal(sem.at[1], 2)
            pl.semaphore_wait(sem.at[1], 1, decrement=False)
            value += pl.semaphore_read(sem.at[1])
            pl.semaphore_wait(sem.at[1], 2)
        elif name == "barrier":
            sem = tpu.get_barrier_semaphore(tag="coverage")
            pl.semaphore_signal(sem, 1)
            pl.semaphore_wait(sem, 1)
        elif name == "core_helpers":
            tpu.run_on_first_core("core")(lambda: y.__setitem__(Ellipsis, value))
            tpu.core_barrier(resources[0].at[0], core_axis_name="core")
        elif name in {"dma", "select_ref"}:
            (sem,) = resources
            if name == "select_ref":
                source = pl.select_ref(jnp.int32(0), x, x.at[:, :])
                tpu.async_copy(source, y, sem).wait()
            else:
                tpu.make_async_copy(x, y, sem).start()
                tpu.make_async_copy(x, y, sem).wait()
                tpu.async_copy(x, y, sem, priority=1, add=True).wait()
                tpu.sync_copy(x, y)
            value = y[...]
        elif name == "bitcast":
            value = tpu.bitcast(tpu.bitcast(value, jnp.int32), jnp.float32)
        elif name == "roll":
            value = tpu.roll(value, 1, axis=1, stride=1, stride_axis=0)
        elif name == "reciprocal":
            value = tpu.reciprocal(value + 2, approx=True, full_range=False)
        elif name == "packing":
            packed = tpu.pack_elementwise((value, value + 1), packed_dtype=jnp.bfloat16)
            value = tpu.unpack_elementwise(
                packed, index=1, packed_dtype=jnp.bfloat16, unpacked_dtype=jnp.float32
            )
        elif name == "rounding":
            value = tpu.stochastic_round(
                value, jnp.zeros(shape, jnp.uint32), target_dtype=jnp.bfloat16
            ).astype(jnp.float32)
        elif name == "annotate":
            value = tpu.load(tpu.annotate(x, no_hazard=True, no_bank_conflict=True))
        elif name == "einshape":
            value = tpu.einshape("(ab)c->abc", value, a=2).reshape(shape)
        elif name == "stateful_random":
            tpu.prng_seed(4)
            value += tpu.prng_random_bits(shape).astype(jnp.float32)
            value += tpu.stateful_uniform(shape=shape)
            value += tpu.stateful_normal(shape=shape)
            value += tpu.stateful_bernoulli(shape=shape).astype(jnp.float32)
            value += tpu.stateful_bits(shape=shape).astype(jnp.float32)
        elif name == "keyed_random":
            key = tpu.to_pallas_key(jax.random.key(4))
            key = mp.wrap_pallas_seed(*mp.unwrap_pallas_seed(key), impl=key.dtype._impl)
            key = jax.random.fold_in(key, 1)
            value += tpu.sample_block(jax.random.uniform, key, shape, shape, shape, (0, 0))
        elif name in {"mxu_fifo", "mxu_acc"}:
            tpu.matmul_push_rhs(jnp.eye(256, dtype=jnp.float32), 0, 0)
            if name == "mxu_fifo":
                tpu.matmul_lhs_fifo(value, 0, load_staged_rhs=0)
                value = tpu.matmul_pop_fifo(shape, jnp.float32, 0)
            else:
                (acc,) = resources
                tpu.matmul_acc_lhs(acc, value, load_staged_rhs=0)
                value = tpu.matmul_pop(acc)
        elif name == "conv":
            value = tpu.conv(
                value.reshape((1, 8, 1, 128)),
                jnp.eye(128).reshape((1, 1, 128, 128)),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                padding="SAME",
            ).reshape(shape)
        elif name.startswith("pipeline"):
            offset = value[0, int(variant == "capture_wrong")] if name == "pipeline_captures" else 1
            block_shape = (4, 128) if name == "pipeline_allocations" else shape
            pipeline_grid = (2,) if name == "pipeline_allocations" else (1,)
            spec = pl.BlockSpec(block_shape, lambda i: (i, 0))

            def body(a, b):
                b[...] = finish(a[...] + offset, variant)

            if name == "pipeline_allocations":
                run, allocate = tpu.emit_pipeline_with_allocations(
                    body, grid=pipeline_grid, in_specs=(spec,), out_specs=(spec,)
                )
                allocations = allocate(x, y)
                if variant == "allocation_wrong":
                    first, *rest = allocations
                    allocations = (
                        replace(first, _spec=pl.BlockSpec(block_shape, lambda i: (1 - i, 0))),
                        *rest,
                    )
                pl.run_scoped(lambda allocations: run(x, y, allocations=allocations), allocations)
            else:
                tpu.emit_pipeline(
                    body, grid=(1,), in_specs=(spec,), out_specs=(spec,), tiling=tpu.Tiling.COMPACT
                )(x, y)
            return
        y[...] = finish(value, variant)

    fn = pl.pallas_call(
        kernel,
        out_shape=sig,
        grid=(("core", 2),) if name == "core_helpers" else (1,),
        scratch_shapes=scratch,
        interpret=interpret,
        in_specs=(pl.BlockSpec(memory_space=tpu.VMEM) if name == "annotate" else pl.BlockSpec(),),
    )
    if name == "memory_space":
        return lambda x: fn(tpu.with_memory_space_constraint(x, tpu.VMEM)), (sig,)
    return fn, (sig,)


def context(case):
    if case.abstract_tpu:
        return jax.sharding.use_abstract_mesh(
            jax.sharding.AbstractMesh((), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
        )
    return nullcontext()


def trace_case(case, variant="base"):
    with context(case), config.use_emit_pipeline_primitive(case.name != "pipeline_legacy"):
        fn, args = function(case, variant)
        return jax.make_jaxpr(fn)(*args)


def equations(graph):
    """Visit actual nested graphs, including index maps inside descriptors."""
    pending, seen = [graph], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, core.Jaxpr):
            for eq in value.eqns:
                yield eq
                pending.extend(eq.params.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, f.name) for f in fields(value))
