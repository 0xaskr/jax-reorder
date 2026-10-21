"""Trace, lower and verify Pallas TPU IR without executing kernels.

Run: uv run python -m tools.pallas_aot_check --output /tmp/pallas-aot.json
--compile uses libtpu and a compile-only TPU v5p 2x2x1 topology. No TPU is
connected and no kernel is executed. Install jax[tpu] matching the project
version in a separate environment to enable this optional compiler check.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import jax
import numpy as np

from tests import pallas_grid_expansion_cases as grid
from tests import pallas_pipeline_cases as pipeline


def run(compile_executable=False):
    report = {
        "jax_version": jax.__version__,
        "target": "tpu",
        "abstract_device": "TPU v5",
        "kernel_execution": False,
        "cases": [],
    }
    sharding = None
    mesh = None
    if compile_executable:
        from jax.experimental.topologies import get_topology_desc
        from jax.sharding import Mesh, NamedSharding, PartitionSpec

        # Disable runtime TPU discovery; the topology owns a compile-only client.
        jax.config.update("jax_platforms", "cpu")
        topology = get_topology_desc(
            platform="tpu",
            topology_name="v5:2x2x1",
            chip_config_name="megacore",
            chips_per_host_bounds=(2, 2, 1),
            num_slices=1,
            wrap=(False, False, False),
        )
        mesh = Mesh(np.asarray(topology.devices), ("aot",))
        sharding = NamedSharding(mesh, PartitionSpec())
        report.update(
            libtpu_version=importlib.metadata.version("libtpu"),
            topology="v5:2x2x1",
            compile_only_devices=len(topology.devices),
        )
    pipeline.verify_sources()
    cases = [
        ("group2", grid.grouped, ((16, 512),), np.float32),
        ("group3", lambda: grid.grouped(group_size=3), ((16, 768),), np.float32),
        ("grid_dma", grid.grouped_dma, ((8, 512),), np.float32),
        ("upstream_double_buffer_copy", lambda: pipeline.function("copy"), ((512, 512),), np.int32),
    ]
    for name, build, shapes, dtype in cases:
        item = {"name": name}
        report["cases"].append(item)
        try:
            environment = jax.set_mesh(mesh) if mesh is not None else pipeline.environment()
            with environment:
                function = build()
                if mesh is not None:
                    # Pallas has no automatic SPMD partitioner. Compile the
                    # unchanged local kernel on every topology participant.
                    function = jax.shard_map(
                        function,
                        mesh=mesh,
                        in_specs=jax.sharding.PartitionSpec(),
                        out_specs=jax.sharding.PartitionSpec(),
                        check_vma=False,
                    )
                staged = (
                    jax.jit(function, in_shardings=sharding, out_shardings=sharding)
                    if sharding is not None
                    else jax.jit(function)
                )
                stage = staged.trace(
                    *(jax.ShapeDtypeStruct(s, dtype, sharding=sharding) for s in shapes)
                )
                item["trace"] = "passed"
                lowered = stage.lower(lowering_platforms=("tpu",))
                item["tpu_lowering"] = "passed"
                ir = lowered.compiler_ir("stablehlo")
                assert ir is not None and ir.operation.verify()
                assert "tpu_custom_call" in str(ir)
                item["ir_verification"] = "passed"
                if compile_executable:
                    executable = lowered.compile()
                    item["executable_type"] = type(executable.runtime_executable()).__name__
                    item["backend_compile"] = "passed"
                else:
                    item["backend_compile"] = "not_requested"
            item["status"] = "passed"
        except Exception as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
    failed = any(case["status"] != "passed" for case in report["cases"])
    return {**report, "status": "failed" if failed else "passed"}, int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", action="store_true", dest="compile_executable")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, code = run(args.compile_executable)
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
