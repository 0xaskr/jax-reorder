"""Compile, announce READY, then execute one isolated official TPU oracle case."""

import json
import sys

import jax
import numpy as np
from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter
from jax.experimental.pallas import tpu as pltpu

from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import compare_program_views, normalize_view
from jax_reorder.core.remote import analyze_remote_call
from tests.pallas_dma_cases import sample_case
from tests.pallas_protocol_cases import (
    acknowledged_exchange,
    local_regular_function,
    mixed_dma,
    regular_control,
    regular_remote,
    regular_remote_control,
    shared_local_remote_counter,
    signaled_completion,
)
from tests.pallas_remote_cases import (
    exchange,
    partial_exchange,
    projection_array,
    remote_function,
    remote_graph,
    request_reply,
    reuse_source,
    sliced_exchange,
    three_device_route,
)
from tests.test_pallas_dma import chained, function, reused, scheduled


def main():
    case, mode = sys.argv[1:]
    projection = None
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
    if case in ("protocol_regular_local", "protocol_observed_local"):
        x = np.arange(128, dtype=np.float32)
        protocol_function = (
            local_regular_function(regular_control)
            if case == "protocol_regular_local"
            else local_regular_function()
        )
        source = jax.make_jaxpr(protocol_function)(jax.ShapeDtypeStruct(x.shape, x.dtype))
        projection = normalize_view(adapt(source).view, budget=AnalysisBudget()).projection
        f = (
            local_regular_function(regular_control, interpret=params)
            if case == "protocol_regular_local"
            else local_regular_function(interpret=params)
        )
        expected = x + 3
    elif case.startswith("protocol_"):
        kernels = {
            "protocol_ack": (acknowledged_exchange(), dict(size=256, arrays=1, regular=1)),
            "protocol_mixed": (mixed_dma(), dict(arrays=1, semaphores=3)),
            "protocol_shared": (shared_local_remote_counter, dict(size=256, arrays=1)),
            "protocol_partial": (sliced_exchange(split_wait=True), dict(size=256)),
            "protocol_signal": (signaled_completion(), dict(arrays=1, regular=1)),
            "protocol_regular_remote": (regular_remote_control, dict(semaphores=0, regular=1)),
            "protocol_observed_remote": (regular_remote, dict(semaphores=0, regular=1)),
        }
        kernel, options = kernels[case]
        projection = normalize_view(
            adapt(remote_graph(kernel, **options)).view, budget=AnalysisBudget()
        ).projection
        f = remote_function(kernel, **options, interpret=params)
        size = options.get("size", 128)
        x = np.arange(size * 2, dtype=np.float32)
        if case == "protocol_shared":
            expected = np.concatenate((x[:128], x[384:], x[256:384], x[128:256]))
        elif case == "protocol_partial":
            expected = np.concatenate((x[384:], x[256:384], x[128:256], x[:128]))
        elif case in ("protocol_regular_remote", "protocol_observed_remote"):
            expected = x
        else:
            expected = np.roll(x, size)
    elif case in ("route3_direct", "route3_relay"):
        kernel = three_device_route(relay=case == "route3_relay")
        view = adapt(remote_graph(kernel, semaphores=4, devices=3)).view
        shard = view.root.operations[0]
        call = shard.regions[0].region.operations[0].call
        proof = analyze_remote_call(call, shard.shard_map.mesh, budget=AnalysisBudget())
        assert len(proof.programs) == 3 and len(proof.protocol.counters) == 4
        other = adapt(
            remote_graph(three_device_route(relay=case != "route3_relay"), semaphores=4, devices=3)
        ).view
        comparison = compare_program_views(view, other)
        assert comparison.equivalence_kind == "normalized"
        assert comparison.evidence.left.remote.discharged_obligations
        projection = comparison.evidence.left.projection
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
        if case != "delayed_request_reply":
            proof = normalize_view(
                adapt(remote_graph(kernel, semaphores=2 if case == "exchange" else 4)).view,
                budget=AnalysisBudget(),
            )
            assert len(proof.remote.transfers) == 2
            projection = proof.projection
    elif case in ("remote_slices", "remote_reuse", "remote_partial"):
        reuse = case == "remote_reuse"
        partial = case == "remote_partial"
        kernel = partial_exchange if partial else reuse_source() if reuse else sliced_exchange()
        options = dict(size=256, arrays=int(reuse or partial), semaphores=4 if reuse else 2)
        proof = normalize_view(adapt(remote_graph(kernel, **options)).view, budget=AnalysisBudget())
        assert len(proof.remote.transfers) == (2 if partial else 4)
        projection = proof.projection
        f = remote_function(kernel, **options, interpret=params)
        x = np.arange(512, dtype=np.float32)
        expected = (
            np.concatenate(
                (x[384:] + np.arange(128), x[256:384], x[128:256] + np.arange(128), x[:128] + 1)
            )
            if partial
            else np.concatenate((x[256:384], x[384:] + 1, x[:128], x[128:256] + 1))
            if reuse
            else np.concatenate((x[384:], x[256:384], x[128:256], x[:128]))
        )
    elif case in ("chain", "schedule", "reuse"):
        kernel = {"chain": chained, "schedule": scheduled(True), "reuse": reused}[case]
        f = function(kernel, outputs=2, arrays=int(case == "reuse"), interpret=params)
        x = np.arange(128, dtype=np.float32)
        expected = (x, x if case == "chain" else x + 1)
    else:
        f = sample_case(case, params)
        x = np.arange(256, dtype=np.float32)
        expected = np.roll(x, 128) if case == "swapped" else x
    if projection is not None:
        np.testing.assert_array_equal(
            projection_array(projection.outputs[0], (x,), expected.size), expected
        )
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
