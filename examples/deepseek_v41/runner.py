"""复现真实 mHC 层的证明、数值检查与 CPU 编译后性能数据。

运行：uv run python -m examples.deepseek_v41.runner --benchmark --json /tmp/deepseek-v41.json
"""

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_reorder import Status, compare_functions

from . import candidates
from .cases import (
    cancellation_inputs,
    check_close,
    check_numerics,
    context_inputs,
    in_context,
    inputs,
    signatures,
)
from .upstream import MANIFEST, REVISION, original, verify_snapshot

CORRECT = (candidates.post_matmul, candidates.post_blocked)
INCORRECT = (candidates.wrong_transpose, candidates.wrong_scale, candidates.wrong_rounding)


def prove(left, right, args, expected=True):
    start = time.perf_counter()
    report = compare_functions(left, right, *args, modes=("forward", "jvp", "vjp"), check_cpu=True)
    for mode in report.modes:
        if (mode.status == Status.EQUIVALENT) != expected:
            raise AssertionError((right.__name__, mode.mode, mode.status, mode.issues))
    return {
        "seconds": time.perf_counter() - start,
        "status": report.status.value,
        "cpu_compiled": report.cpu_compiled,
        "modes": [
            {
                "mode": m.mode,
                "structural": m.comparison.status.value,
                "status": m.status.value,
                "kind": m.equivalence_kind,
                "work": m.algebra.work if m.algebra else None,
                "warnings": m.algebra.warnings if m.algebra else (),
                "issues": [{"path": issue.path, "reason": issue.reason} for issue in m.issues],
            }
            for m in report.modes
        ],
    }


def benchmark(profile, dtype, repeats):
    """warmup 后交错运行，计时包含同步，编译与编译器报告单独记录。"""
    args = inputs(profile, dtype, 101)
    functions = (original().hc_post, *CORRECT)
    compiled, records = {}, {}
    for function in functions:
        start = time.perf_counter()
        executable = function.lower(*args).compile()
        compile_seconds = time.perf_counter() - start
        memory = executable.memory_analysis()
        cost = executable.cost_analysis()
        name = function.__name__
        compiled[name] = executable
        records[name] = {
            "compile_seconds": compile_seconds,
            "temp_bytes": memory.temp_size_in_bytes,
            "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
            "cost": {key: float(cost[key]) for key in ("flops", "bytes accessed") if key in cost},
            "samples_us": [],
        }
        for _ in range(3):
            jax.block_until_ready(executable(*args))
    names = list(compiled)
    for iteration in range(repeats):
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            start = time.perf_counter_ns()
            jax.block_until_ready(compiled[name](*args))
            records[name]["samples_us"].append((time.perf_counter_ns() - start) / 1000)
    for record in records.values():
        record["median_us"] = float(np.median(record["samples_us"]))
        record["p10_us"], record["p90_us"] = map(
            float, np.percentile(record["samples_us"], [10, 90])
        )
    baseline = records["hc_post"]["median_us"]
    for record in records.values():
        record["speedup"] = baseline / record["median_us"]
    return {
        "profile": profile,
        "dtype": np.dtype(dtype).name,
        "repeats": repeats,
        "functions": records,
    }


def run(trials, include_benchmark, repeats):
    verify_snapshot()
    result = {
        "source": {
            "revision": REVISION,
            "pull_request": MANIFEST["pull_request"],
            "state_at_download": MANIFEST["pull_request_state_at_download"],
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "devices": [device.device_kind for device in jax.devices()],
            "x64": jax.config.x64_enabled,
        },
        "comparisons": [],
        "counterexamples": [],
        "contexts": [],
        "benchmarks": [],
    }
    for dtype in (jnp.float32, jnp.bfloat16):
        profile = (1, 1, 4, 5120)
        for function in CORRECT:
            record = {
                "candidate": function.__name__,
                "profile": profile,
                "dtype": np.dtype(dtype).name,
                "proof": prove(original().hc_post, function, signatures(profile, dtype)),
                "samples": [],
            }
            for seed in range(trials):
                args = inputs(profile, dtype, 41 + seed)
                record["samples"].append(
                    {
                        "seed": 41 + seed,
                        "base": check_numerics(original().hc_post, args, seed),
                        "candidate": check_numerics(function, args, seed),
                    }
                )
            record["cancellation"] = check_numerics(function, cancellation_inputs(profile, dtype))
            result["comparisons"].append(record)
            print(
                np.dtype(dtype).name,
                function.__name__,
                "D=5120 forward/JVP/VJP proven; numerical checks passed",
                flush=True,
            )
        for function in INCORRECT:
            args = cancellation_inputs((1, 1, 4, 7), dtype)
            proof = prove(original().hc_post, function, args, expected=False)
            a, b = original().hc_post(*args), function(*args)
            difference = np.asarray(a, np.float32) - np.asarray(b, np.float32)
            if not np.any(difference):
                raise AssertionError("missing counterexample")
            result["counterexamples"].append(
                {
                    "candidate": function.__name__,
                    "dtype": np.dtype(dtype).name,
                    "proof": proof,
                    "base_output": np.asarray(a, np.float32).tolist(),
                    "candidate_output": np.asarray(b, np.float32).tolist(),
                    "max_abs_difference": float(np.max(np.abs(difference))),
                }
            )
        for function in CORRECT:
            args = context_inputs(profile, dtype, 53)
            left, right = in_context(original().hc_post), in_context(function)
            proof = prove(left, right, args)
            errors = [
                check_close(a, b)
                for a, b in zip(
                    jax.tree.leaves(left(*args)), jax.tree.leaves(right(*args)), strict=True
                )
            ]
            result["contexts"].append(
                {
                    "candidate": function.__name__,
                    "dtype": np.dtype(dtype).name,
                    "profile": profile,
                    "sinkhorn_iters": 20,
                    "proof": proof,
                    "max_abs_errors": errors,
                }
            )
            print(
                np.dtype(dtype).name,
                function.__name__,
                "full 20-iteration mHC context proven",
                flush=True,
            )
        if include_benchmark:
            for profile in ((1, 1, 4, 5120), (8, 1, 4, 5120), (1, 128, 4, 5120)):
                measured = benchmark(profile, dtype, repeats)
                result["benchmarks"].append(measured)
                print(
                    "benchmark",
                    np.dtype(dtype).name,
                    profile,
                    {
                        name: round(item["speedup"], 3)
                        for name, item in measured["functions"].items()
                    },
                    flush=True,
                )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.trials < 1 or args.repeats < 3:
        parser.error("trials 至少为 1，repeats 至少为 3")
    result = run(args.trials, args.benchmark, args.repeats)
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("All requested comparisons and execution checks passed.")


if __name__ == "__main__":
    main()
