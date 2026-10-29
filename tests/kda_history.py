"""Reproduce the pinned KDA history/AI campaign, without TPU hardware.

uv run python -m tests.kda_history --output /tmp/kda-history.json

Graph checks retain interpret=False. Numerical checks use the official TPU CPU
interpreter in isolated workers. Failures and unsupported configurations are
reported, never relabelled as mathematical inequivalence or successful proofs.
"""

import argparse
import collections
import dataclasses
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import jax
import sympy

from jax_reorder import (
    ComparisonError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
    compare_pallas_kernels,
)
from tests.pallas_oracle import run_worker
from tests.upstream_tpu import kda_cases as cases
from tests.upstream_tpu.sources import read_manifest, read_source


def checker_source_hash():
    root = Path(__file__).resolve().parents[1]
    return hashlib.sha256(
        b"".join(
            str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0"
            for path in sorted((root / "src/jax_reorder").rglob("*.py"))
        )
    ).hexdigest()


def comparison_worker(
    config, output, candidate=None, *, budget=10000, output_boundary="kernel", comparison="all"
):
    settings = cases.Configuration(**config)
    base = cases.trace(config=settings)
    rows = []
    print("READY", flush=True)
    candidates = (candidate,) if candidate else cases.CANDIDATES
    stages = {
        "whole_structure": lambda other: compare_jaxprs(
            base, other, include_derivative_rules=False
        ),
        "local_semantic": lambda other: compare_pallas_kernels(
            base,
            other,
            equivalence="semantic",
            analysis_budget=budget,
            output_boundary=output_boundary,
        ),
        "local_mathematics": lambda other: compare_algebraic_jaxprs(
            base,
            other,
            scope="device_local",
            algebra_timeout=60,
            algebra_budget=budget,
            output_boundary=output_boundary,
        ),
    }
    if comparison != "all":
        stages = {f"local_{comparison}": stages[f"local_{comparison}"]}

    def checkpoint():
        # Retain completed stages even if a subsequent comparison exceeds the
        # worker wall clock or crashes. An incomplete row is never a proof.
        Path(output).write_text(
            json.dumps({"configuration": config, "comparisons": rows}, separators=(",", ":")) + "\n"
        )

    for revision, variant in candidates:
        row = {
            "revision": revision,
            "variant": variant,
            "checker_source_sha256": checker_source_hash(),
            "output_boundary": output_boundary,
            "expectation": cases.AI_EXPECTATIONS.get(
                variant, "self_control" if revision == "latest" else "historical_measurement"
            ),
        }
        rows.append(row)
        if not cases.supports_configuration(revision, settings):
            row["trace"] = {
                "status": "configuration_unavailable",
                "reason": "This revision has no cast_dg_to_q_dtype argument.",
            }
            checkpoint()
            continue
        other = cases.trace(revision, variant, settings)
        (call,) = tuple(cases.pallas_calls(other))
        mapping = call.params["grid_mapping"]
        row["trace"] = {
            "status": "ok",
            "public_outputs": len(other.jaxpr.outvars),
            "kernel_outputs": mapping.num_outputs,
            "scalar_prefetch": mapping.num_index_operands,
            "grid": list(mapping.grid),
            "interpret": call.params["interpret"],
            "primitives": cases.primitive_counts(other),
        }
        checkpoint()
        for name, compare in stages.items():
            started = time.monotonic()
            row[name] = {"status": "running"}
            checkpoint()
            try:
                result = compare(other)
                row[name] = {
                    "status": result.status.value,
                    "equivalence_kind": getattr(result, "equivalence_kind", None),
                }
                if getattr(result, "difference", None):
                    row[name]["difference"] = result.difference
                if getattr(result, "assumptions", None):
                    row[name]["assumptions"] = result.assumptions
                for field in ("warnings", "work", "rules"):
                    value = getattr(result, field, None)
                    if value is not None:
                        row[name][field] = value
                proofs = getattr(result, "normalizations", ()) or tuple(
                    proof
                    for proof in (
                        getattr(result, "left_protocol", None),
                        getattr(result, "right_protocol", None),
                    )
                    if proof is not None
                )
                if proofs:
                    row[name]["resources"] = [
                        {
                            "outputs": len(proof.projection.outputs),
                            "transfers": len(proof.projection.communication),
                            "waits": len(proof.projection.waits),
                            "output_boundary": proof.output_boundary,
                        }
                        for proof in proofs
                    ]
            except ComparisonError as error:
                row[name] = {
                    "status": "incomplete",
                    "exception": type(error).__name__,
                    "reason": str(error),
                    "failure_kind": error.failure_kind.value,
                    "stage": error.stage.value,
                }
            row[name]["seconds"] = round(time.monotonic() - started, 4)
            checkpoint()
    print("RESULT " + json.dumps({"case": "kda_graphs", "status": "ok"}), flush=True)


def run(
    output,
    selected,
    *,
    graphs=True,
    numerics=True,
    budget=10000,
    worker_timeout=180,
    output_boundary="kernel",
    comparison="all",
    candidates=cases.CANDIDATES,
):
    manifest = read_manifest(cases.REPO)
    for key in manifest["files"]:
        read_source(cases.REPO, key)
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 3,
        "repository": manifest["repository"],
        "baseline_revision": cases.REVISION,
        "revisions": manifest["revisions"],
        "jax": jax.__version__,
        "sympy": sympy.__version__,
        "checker_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "checker_source_sha256": checker_source_hash(),
        "harness_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in (
                "tests/upstream_tpu/kda_cases.py",
                "tests/upstream_tpu/kda_oracle.py",
                "tests/kda_history.py",
            )
        },
        "scope": "Dense device-local complete KDA mega kernel; CP inactive.",
        "output_boundary": output_boundary,
        "evidence": {
            "graphs": "Native interpret=False Pallas tracing and public comparison APIs.",
            "numerics": "Official TPU CPU interpreter, 2 seeds, all 6 public outputs.",
            "oracle": "Independent NumPy token recurrence and its manual reverse derivative.",
            "tpu_hardware": False,
            "tpu_compilation": False,
            "numeric_close_rtol": 3e-4,
            "numeric_close_atol": 3e-6,
            "search_budget": 10000,
            "analysis_budget": budget,
            "worker_timeout_seconds": worker_timeout,
            "algebra_timeout_seconds": 60,
        },
        "ai_expectations": cases.AI_EXPECTATIONS,
        "configurations": [],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kda-history-") as temp:
        for config in selected:
            print(f"Running {config.name}", flush=True)
            data = {"name": config.name, "settings": dataclasses.asdict(config)}
            for phase, enabled, module, entry, expected in (
                (
                    "graphs",
                    graphs,
                    "tests.kda_history",
                    "comparison_worker",
                    {"case": "kda_graphs", "status": "ok"},
                ),
                (
                    "numerics",
                    numerics,
                    "tests.upstream_tpu.kda_oracle",
                    "worker",
                    {"case": "kda", "status": "ok"},
                ),
            ):
                if not enabled:
                    continue
                started = time.monotonic()
                # Bound each candidate independently so a single difficult
                # kernel does not hide the remaining history/AI measurements.
                phase_candidates = candidates if phase == "graphs" else (None,)
                data[phase] = {"configuration": dataclasses.asdict(config)}
                if phase == "graphs":
                    data[phase]["comparisons"] = []
                for i, candidate in enumerate(phase_candidates):
                    path = Path(temp) / f"{config.name}-{phase}-{i}.json"
                    kwargs = {"config": dataclasses.asdict(config), "output": str(path)}
                    if candidate:
                        kwargs["candidate"] = candidate
                        kwargs.update(
                            budget=budget, output_boundary=output_boundary, comparison=comparison
                        )
                    failure = None
                    try:
                        run_worker(
                            module,
                            entry=entry,
                            kwargs=kwargs,
                            expected=expected,
                            devices=1,
                            startup_timeout=180,
                            execution_timeout=worker_timeout,
                        )
                    except (AssertionError, TimeoutError) as error:
                        failure = str(error)
                    partial = json.loads(path.read_text()) if path.exists() else {}
                    if phase == "graphs":
                        rows = partial.get("comparisons", [])
                        if failure:
                            if not rows:
                                rows = [{"revision": candidate[0], "variant": candidate[1]}]
                            rows[-1]["worker_failure"] = failure
                            for value in rows[-1].values():
                                if isinstance(value, dict) and value.get("status") == "running":
                                    value.update(status="incomplete", reason=failure)
                        data[phase]["comparisons"].extend(rows)
                    else:
                        data[phase].update(partial)
                    if failure:
                        data[phase]["status"] = "worker_failed"
                        data[phase].setdefault("worker_failures", []).append(failure)
                    if phase == "graphs":
                        print(
                            f"  {candidate}: "
                            + ", ".join(
                                f"{key}={value['status']}"
                                for row in rows
                                for key, value in row.items()
                                if isinstance(value, dict) and key.startswith("local_")
                            ),
                            flush=True,
                        )
                    report["summary"] = summarize(
                        {**report, "configurations": [*report["configurations"], data]}
                    )
                    output.write_text(
                        json.dumps(
                            {**report, "configurations": [*report["configurations"], data]},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                data[phase]["seconds"] = round(time.monotonic() - started, 3)
                print(f"  {phase}: {data[phase].get('status', 'recorded')}", flush=True)
            report["configurations"].append(data)
            report["summary"] = summarize(report)
            output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    return report


def summarize(report):
    counts = collections.Counter()
    for config in report["configurations"]:
        for phase in ("graphs", "numerics"):
            counts["worker_failures"] += config.get(phase, {}).get("status") == "worker_failed"
        for row in config.get("graphs", {}).get("comparisons", ()):
            counts[f"trace_{row.get('trace', {}).get('status', 'incomplete')}"] += 1
            for stage in ("whole_structure", "local_semantic", "local_mathematics"):
                if stage in row:
                    counts[f"{stage}_{row[stage]['status']}"] += 1
        for row in config.get("numerics", {}).get("results", ()):
            counts["numerical_executions"] += 1
            counts["numerical_race_reports"] += row["races"]
            counts["numerical_nonfinite_outputs"] += sum(
                not metric["finite"] for metric in row["vs_base"].values()
            )
            if row["variant"] in cases.AI_EXPECTATIONS:
                agrees = all(m["close"] for m in row["vs_base"].values())
                expected_agreement = cases.AI_EXPECTATIONS[row["variant"]] != "inequivalent"
                counts["ai_numerical_expectations_met"] += agrees == expected_agreement
                counts["ai_numerical_expectations_missed"] += agrees != expected_agreement
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", choices=[c.name for c in cases.CONFIGURATIONS])
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--worker-timeout", type=float, default=180)
    parser.add_argument("--output-boundary", choices=("kernel", "parent"), default="kernel")
    parser.add_argument("--comparison", choices=("all", "semantic", "mathematics"), default="all")
    parser.add_argument("--revision", action="append", choices=("latest", *cases.HISTORY))
    parser.add_argument("--variant", action="append", choices=("original", *cases.AI_EXPECTATIONS))
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--graphs-only", action="store_true")
    group.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    selected = [c for c in cases.CONFIGURATIONS if not args.config or c.name in args.config]
    report = run(
        args.output,
        selected,
        graphs=not args.numerics_only,
        numerics=not args.graphs_only,
        budget=args.budget,
        worker_timeout=args.worker_timeout,
        output_boundary=args.output_boundary,
        comparison=args.comparison,
        candidates=tuple(
            (revision, variant)
            for revision, variant in cases.CANDIDATES
            if (not args.revision or revision in args.revision)
            and (not args.variant or variant in args.variant)
        ),
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    if report["summary"].get("worker_failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
