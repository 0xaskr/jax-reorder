"""Compile Pallas source cases on an offline TPU topology, without execution.

Run ``python -m tests.pallas_aot --suite remote_grid --output report.json``.
The default suite, ``all``, writes a mapping from suite names to their records.
Individual suites retain their existing record fields and acceptance rules.
``composed_dma`` is a source-support inventory: compiler rejections are recorded
without a failing exit status. Every other suite fails on unexpected errors.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.experimental.topologies import get_topology_desc
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import tests.pallas_buffered_revisit_cases as buffered_revisit
import tests.pallas_composed_dma_cases as composed_dma
import tests.pallas_dynamic_range_cases as dynamic_range
import tests.pallas_remote_grid_cases as remote_grid
import tests.pallas_remote_revisit_cases as remote_revisit
import tests.pallas_scheduling_cases as scheduling


@dataclass(frozen=True)
class Case:
    record: dict[str, object]
    build: Callable[[Mesh], Callable]
    shape: tuple[int, ...]
    dtype: object = np.float32
    partition: P = P("d")
    inputs: int = 1
    outputs: int = 1
    replicate: bool = False
    expected_error: Callable[[Exception], bool] | None = None


def run_cases(
    cases,
    *,
    devices=2,
    axis="d",
    record_trace=True,
    record_backend_compile=False,
    require_custom_call=True,
    failure_status="failed",
):
    jax.config.update("jax_platforms", "cpu")
    topology = get_topology_desc(
        platform="tpu",
        topology_name="v5:2x2x1",
        chip_config_name="megacore",
        chips_per_host_bounds=(2, 2, 1),
        num_slices=1,
        wrap=(False, False, False),
    )
    mesh = Mesh(np.asarray(topology.devices[:devices]), (axis,))
    records = []
    for case in cases:
        item = dict(case.record, kernel_execution=False)
        records.append(item)
        try:
            with jax.set_mesh(mesh):
                sharding = NamedSharding(mesh, case.partition)
                function = case.build(mesh)
                if case.replicate:
                    function = jax.shard_map(
                        function, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False
                    )
                signature = jax.ShapeDtypeStruct(case.shape, case.dtype, sharding=sharding)
                staged = jax.jit(
                    function,
                    in_shardings=(sharding,) * case.inputs if case.inputs > 1 else sharding,
                    out_shardings=(sharding,) * case.outputs if case.outputs > 1 else sharding,
                ).trace(*([signature] * case.inputs))
                if record_trace:
                    item["trace"] = "passed"
                lowered = staged.lower(lowering_platforms=("tpu",))
                item["lower"] = "passed"
                ir = lowered.compiler_ir("stablehlo")
                assert ir.operation.verify() and (
                    not require_custom_call or "tpu_custom_call" in str(ir)
                )
                item["ir_verify"] = "passed"
                item["executable"] = type(lowered.compile().runtime_executable()).__name__
                if record_backend_compile:
                    item["backend_compile"] = "passed"
                item["status"] = "passed"
        except Exception as error:
            expected = case.expected_error is not None and case.expected_error(error)
            item.update(
                status="expected_rejection" if expected else failure_status,
                error=f"{type(error).__name__}: {error}",
            )
        print(json.dumps(item), flush=True)
    return records


def _run_buffered_revisit():
    return run_cases(
        [
            Case(
                dict(options={key: str(value) for key, value in options.items()}),
                lambda mesh, options=options: buffered_revisit.function(**options, mesh=mesh),
                (2 * options.get("programs", 3) * buffered_revisit.WIDTH,),
                outputs=2 if options.get("multiple") else 1,
                expected_error=lambda error, options=options: (
                    "pipeline_mode" in options
                    and "not supported with double buffering" in str(error)
                ),
            )
            for options in (
                dict(mode="normal"),
                dict(mode="partial"),
                dict(mode="accumulate"),
                dict(mode="normal", programs=6, consecutive=2),
                dict(mode="accumulate", multiple=True),
                dict(mode="wrong_block"),
                dict(mode="normal", pipeline_mode=pl.Buffered(2, revisit=pl.RevisitMode.ANY)),
            )
        ]
    )


def _composed_dma_cases():
    for name in (
        "local",
        "local_holes",
        "padding",
        "indexed",
        "reshape",
        "transpose",
        "mixed",
        "holes",
    ):
        replicated = name.startswith("local") or name == "padding"

        def build(mesh, name=name):
            if name.startswith("local"):
                return composed_dma.local_holes(destination_holes=name == "local_holes")
            if name == "padding":
                return composed_dma.native_padded_view()
            return composed_dma.remote_case_function(mode=name, mesh=mesh)

        shape = (
            (32, 128)
            if name.startswith("local")
            else (16, 384)
            if name == "padding"
            else (1024 if name == "holes" else 512,)
        )
        yield Case(
            dict(case=name),
            build,
            shape,
            partition=P() if replicated else P("d"),
            replicate=replicated,
        )


def _run_composed_dma():
    return run_cases(
        _composed_dma_cases(), require_custom_call=False, failure_status="source_rejected"
    )


def _run_dynamic_range():
    return run_cases(
        [
            Case(
                dict(mode=mode),
                lambda mesh, mode=mode: (
                    dynamic_range.layout_function(hint=mode == "layout_hint")
                    if mode.startswith("layout")
                    else dynamic_range.function(mode)
                ),
                (128, 128) if mode.startswith("layout") else dynamic_range.SHAPE,
                dtype=np.float32 if mode.startswith("layout") else np.int32,
                partition=P(),
                inputs=2 if mode.startswith("layout") else 1,
            )
            for mode in (
                "and",
                "mod",
                "div",
                "lax_div",
                "negative_div",
                "carry",
                "alias_carry",
                "wrong_guard",
                "layout_hint",
                "layout_default",
            )
        ],
        devices=1,
        record_trace=False,
    )


def _run_remote_grid():
    return run_cases(
        [
            Case(
                dict(name=name, options=options),
                lambda mesh, options=options: remote_grid.function(**options, mesh=mesh),
                (2 * remote_grid.LOCAL_SIZE,),
            )
            for name, options in (
                ("serial", {}),
                ("flat", dict(flat=True)),
                ("parallel", dict(parallel=True)),
                ("interleaved", dict(parallel=True, transpose_grid=True)),
                ("all_parallel", dict(all_parallel=True)),
                ("rank_three", dict(three_dimensional=True, parallel=True)),
            )
        ],
        record_backend_compile=True,
    )


def _run_remote_revisit():
    return run_cases(
        [
            Case(
                dict(explicit=explicit, mode=mode, multiple=multiple),
                lambda mesh, explicit=explicit, mode=mode, multiple=multiple: (
                    remote_revisit.function(
                        explicit=explicit, mode=mode, multiple=multiple, mesh=mesh
                    )
                ),
                (6 * remote_revisit.WIDTH,),
                outputs=2 if multiple else 1,
            )
            for explicit, mode, multiple in (
                (False, "normal", False),
                (False, "wrong_block", False),
                (True, "normal", False),
                (True, "wrong_block", False),
                (True, "partial_revisit", False),
                (True, "normal", True),
            )
        ],
        record_backend_compile=True,
    )


def source_bound_core_kernel():
    mesh = tpu.TensorCoreMesh(axis_name="core", num_cores=2)

    def body(x, y, scratch, sem, dma):
        core = jax.lax.axis_index("core")

        @pl.when(core == 0)
        def signal():
            pl.semaphore_signal(sem, 1, core_index=1)

        @pl.when(core == 1)
        def wait():
            pl.semaphore_wait(sem, 1)

        chunk = pl.ds(core * 128, 128)
        tpu.make_async_copy(x.at[:, chunk], scratch, dma).start()
        tpu.make_async_copy(x.at[:, chunk], scratch, dma).wait()
        scratch[...] *= 2
        tpu.make_async_copy(scratch, y.at[:, chunk], dma).start()
        tpu.make_async_copy(scratch, y.at[:, chunk], dma).wait()

    return pl.kernel(
        body,
        out_type=jax.ShapeDtypeStruct((8, 256), jnp.float32),
        mesh=mesh,
        scratch_types=[
            tpu.VMEM((8, 128), jnp.float32),
            tpu.SemaphoreType.REGULAR,
            tpu.SemaphoreType.DMA,
        ],
    )


def _run_scheduling():
    return run_cases(
        [
            Case(
                dict(name=name),
                lambda mesh, build=build: build(),
                shape,
                partition=P(),
                replicate=True,
            )
            for name, build, shape in (
                ("interleaved_independent", scheduling.interleaved, (16, 384)),
                ("interleaved_contiguous_suffix", scheduling.contiguous_suffix, (16, 768)),
                (
                    "interleaved_independent_dma",
                    lambda: scheduling.interleaved(dma=True),
                    (16, 384),
                ),
                ("interleaved_banked", scheduling.banked_grid, (16, 128)),
                ("interleaved_banked_dma", lambda: scheduling.banked_grid(mode="dma"), (16, 128)),
                ("explicit_forward_candidate", scheduling.routed, (8, 256)),
                ("explicit_self_candidate", lambda: scheduling.routed(mode="self"), (8, 256)),
                ("source_bound_tensor_core_mesh", source_bound_core_kernel, (8, 256)),
            )
        ],
        devices=None,
        axis="aot",
        record_backend_compile=True,
    )


SUITES = {
    "buffered_revisit": _run_buffered_revisit,
    "composed_dma": _run_composed_dma,
    "dynamic_range": _run_dynamic_range,
    "remote_grid": _run_remote_grid,
    "remote_revisit": _run_remote_revisit,
    "scheduling": _run_scheduling,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", *SUITES), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        if args.suite != "scheduling":
            parser.error("--output is required")
        args.output = Path("/tmp/pallas-scheduling-aot.json")
    suites = SUITES if args.suite == "all" else {args.suite: SUITES[args.suite]}
    reports = {suite: run() for suite, run in suites.items()}
    result = reports if args.suite == "all" else reports[args.suite]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    failed = False
    for suite, records in reports.items():
        if suite == "composed_dma":
            continue
        accepted = ("passed", "expected_rejection") if suite == "buffered_revisit" else ("passed",)
        failed |= any(item["status"] not in accepted for item in records)
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
