"""复现真实 mHC 层的证明、数值检查与 CPU 编译后性能数据。

默认对 FP32/BF16 下的两个正例执行 forward/JVP/VJP 三模式比较，再用独立
NumPy 参照核对前向、JVP、VJP；三个错误候选必须被拒绝并保存数值反例；
完整上下文使用上游 20 次 Sinkhorn 生成的混合系数。--benchmark 另测 CPU
编译后延迟、编译内存与 XLA cost analysis。

运行：
  uv run python -m examples.deepseek_v41.runner --benchmark --json /tmp/deepseek-v41.json

报告写入 --json 指定路径，未指定则不落盘；全部通过时打印结束行并返回 0，
断言失败、参数非法或快照校验失败时抛出异常/parser.error，退出码非零。
数值与性能结论仅限本机 CPU、锁定版本和声明签名，不是误差上界或 GPU/TPU 结论。
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
    """对 left/right 执行 forward、JVP、VJP 三模式比较并断言预期结论。

    args 是四个输入口的抽象签名；expected=False 表示期望三种模式都判为
    different。返回 JSON 可序列化的记录：耗时、总状态、CPU 编译标志，以及
    各模式的结构结论、代数状态、工作量、警告和问题列表。结论不符时抛出
    AssertionError，不写文件。
    """
    start = time.perf_counter()
    report = compare_functions(left, right, *args, modes=("forward", "jvp", "vjp"), check_cpu=True)
    # 三种模式分别核对；前向等价不能代替求导图的结论。
    for mode in report.modes:
        if (mode.status == Status.EQUIVALENT) != expected:
            raise AssertionError((right.__name__, mode.mode, mode.status, mode.issues))
    return {
        "seconds": time.perf_counter() - start,
        "status": report.status.value,
        "cpu_compiled": report.cpu_compiled,
        # structural 是原图结构比较，work/warnings 是代数证明信息，两者分开保留。
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
    """warmup 后交错运行，计时包含同步，编译与编译器报告单独记录。

    profile/dtype 决定输入形状与精度，repeats 是每个函数的计时样本数。
    返回包含编译秒数、XLA 内存分析、cost analysis、逐次样本、中位数、
    p10/p90 及相对 base 加速比的字典；不修改输入、不写文件。计时前对
    每个函数 warmup 3 次，编译时间不计入中位延迟。
    """
    args = inputs(profile, dtype, 101)
    functions = (original().hc_post, *CORRECT)
    compiled, records = {}, {}
    for function in functions:
        # 编译时间单独记录，便于区分一次性成本与稳态延迟。
        start = time.perf_counter()
        executable = function.lower(*args).compile()
        compile_seconds = time.perf_counter() - start
        # memory_analysis/cost_analysis 是 XLA 的静态估计，不是硬件计数器。
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
        # 先 warmup，避免把首次执行的初始化成本混入稳态样本。
        for _ in range(3):
            jax.block_until_ready(executable(*args))
    names = list(compiled)
    for iteration in range(repeats):
        # 交错轮换执行顺序，抵消设备状态随时间的漂移。
        order = names[iteration % len(names) :] + names[: iteration % len(names)]
        for name in order:
            start = time.perf_counter_ns()
            # block_until_ready 把异步调度时间计入本次测量，反映实际完成延迟。
            jax.block_until_ready(compiled[name](*args))
            records[name]["samples_us"].append((time.perf_counter_ns() - start) / 1000)
    for record in records.values():
        record["median_us"] = float(np.median(record["samples_us"]))
        record["p10_us"], record["p90_us"] = map(
            float, np.percentile(record["samples_us"], [10, 90])
        )
    baseline = records["hc_post"]["median_us"]
    # 加速比 = base 中位延迟 / 当前候选中位延迟，大于 1 表示候选更快。
    for record in records.values():
        record["speedup"] = baseline / record["median_us"]
    return {
        "profile": profile,
        "dtype": np.dtype(dtype).name,
        "repeats": repeats,
        "functions": records,
    }


def run(trials, include_benchmark, repeats):
    """执行一次完整验证流程并返回 JSON 可序列化的报告字典。

    trials 为每种 dtype 的随机样本数，include_benchmark 控制是否测量 CPU
    性能，repeats 原样传给 benchmark。副作用是打印进度与基准摘要行；任一
    阶段失败时抛出断言异常而不吞掉错误。报告包含来源、环境、比较、反例、
    完整上下文和可选基准数据。
    """
    verify_snapshot()
    result = {
        # 来源与环境固定记录在报告里，避免脱离上下文解读结论。
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
    # 两种精度都跑：BF16 暴露低精度舍入和 dtype 转换差异。
    for dtype in (jnp.float32, jnp.bfloat16):
        # 真实模型配置 H=4、D=5120；单 token 形状用于控制 CPU 计算量。
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
            # 抵消样本单独记录：它比随机样本更容易暴露连接和舍入差异。
            record["cancellation"] = check_numerics(function, cancellation_inputs(profile, dtype))
            result["comparisons"].append(record)
            print(
                np.dtype(dtype).name,
                function.__name__,
                "D=5120 forward/JVP/VJP proven; numerical checks passed",
                flush=True,
            )
        for function in INCORRECT:
            # 错误候选只要求被拒绝并提供数值见证；不要求结构状态为某固定值。
            args = cancellation_inputs((1, 1, 4, 7), dtype)
            proof = prove(original().hc_post, function, args, expected=False)
            a, b = original().hc_post(*args), function(*args)
            difference = np.asarray(a, np.float32) - np.asarray(b, np.float32)
            # 图不同必须同时有实际输出差异，避免只凭结构下结论。
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
            # 完整上下文验证真实的 20 次 Sinkhorn 系数生成与 hc_pre 组合。
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
            # 覆盖单 token、批量与长序列，避免只报告有利形状的加速比。
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
    """解析命令行参数，执行 run 并按需写入 JSON 报告。

    默认 trials=3、repeats=30，--benchmark 才测量性能，--json 指定报告
    路径。参数非法时 parser.error 以退出码 2 结束；验证失败时异常向上
    传播，退出码非零；全部通过返回 0。
    """
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
