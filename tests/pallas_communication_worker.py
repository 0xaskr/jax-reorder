"""Compile, announce READY, then execute one isolated official TPU oracle case."""

import json
import sys

import jax
import numpy as np
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
from jax.experimental.pallas import tpu as pltpu

from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.remote import analyze_remote_call
from tests.pallas_dma_cases import sample_case
from tests.pallas_remote_cases import (
    exchange,
    remote_function,
    remote_graph,
    request_reply,
    three_device_route,
)
from tests.test_pallas_dma import chained, function, reused, scheduled


def main():
    case, mode = sys.argv[1:]
    params = pltpu.InterpretParams(
        dma_execution_mode=mode,
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
        allow_hbm_allocation_in_run_scoped=False,
    )
    if case in ("route3_direct", "route3_relay"):
        kernel = three_device_route(relay=case == "route3_relay")
        view = adapt(remote_graph(kernel, semaphores=4, devices=3)).view
        shard = view.root.operations[0]
        call = shard.regions[0].region.operations[0].call
        proof = analyze_remote_call(call, shard.shard_map.mesh, budget=AnalysisBudget())
        assert len(proof.programs) == 3 and len(proof.protocol.counters) == 4
        f = remote_function(kernel, semaphores=4, devices=3, interpret=params)
        x = np.arange(384, dtype=np.float32)
        expected = np.tile(x[:128], 3)
    elif case in ("exchange", "request_reply", "delayed_request_reply"):
        kernel = (
            exchange
            if case == "exchange"
            else request_reply(delayed=case == "delayed_request_reply")
        )
        f = remote_function(kernel, semaphores=2 if case == "exchange" else 4, interpret=params)
        x = np.arange(256, dtype=np.float32)
        expected = np.roll(x, 128) if case == "exchange" else np.tile(x[:128], 2)
    elif case in ("chain", "schedule", "reuse"):
        kernel = {"chain": chained, "schedule": scheduled(True), "reuse": reused}[case]
        f = function(kernel, outputs=2, arrays=int(case == "reuse"), interpret=params)
        x = np.arange(128, dtype=np.float32)
        expected = (x, x if case == "chain" else x + 1)
    else:
        f = sample_case(case, params)
        x = np.arange(256, dtype=np.float32)
        expected = np.roll(x, 128) if case == "swapped" else x
    compiled = jax.jit(f).lower(jax.ShapeDtypeStruct(x.shape, x.dtype)).compile()
    print("READY", flush=True)
    actual = jax.block_until_ready(compiled(x))
    jax.effects_barrier()
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)
    assert interpreter.races is not None and not interpreter.races.races_found
    print(
        "RESULT " + json.dumps({"case": case, "mode": mode, "status": "ok", "races": False}),
        flush=True,
    )


if __name__ == "__main__":
    main()
