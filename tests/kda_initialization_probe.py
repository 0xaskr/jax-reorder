"""Explicit AI diagnostic candidate; never replace the pinned KDA baseline.

Rewrite only ``ref[index] = jnp.zeros_like(ref[index])`` to zeros constructed
from that Ref view's shape and dtype. Original snapshots, graph checks and
resource admission remain unchanged. This is not a proof of the rewrite.

The optional ``--initialize-placeholders`` variant additionally initializes
disabled ABI outputs through automatic VMEM output staging. It changes that
kernel's private output values and memory specs; it is a separate AI candidate.
``--initialize-tail-padding`` also zeroes the unused db rows before the full
tail DMA, whose active source otherwise includes uninitialized scratch.
"""

import argparse
import ast
import copy
import dataclasses
import functools
import hashlib
import json
import time
from pathlib import Path

import jax
from jax.experimental.pallas import tpu

from jax_reorder import (
    ComparisonError,
    compare_algebraic_jaxprs,
    compare_jaxprs,
    compare_pallas_kernels,
)
from tests.pallas_oracle import run_worker
from tests.upstream_tpu import kda_cases as cases
from tests.upstream_tpu.kda_oracle import differences, samples
from tests.upstream_tpu.sources import _bound_names, _execute_definitions, read_source


class ShapeInitialization(ast.NodeTransformer):
    def __init__(self):
        self.edits = []

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Subscript):
            return node
        target, value = node.targets[0], node.value
        if not (
            isinstance(target.value, ast.Name)
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and (value.func.value.id, value.func.attr) == ("jnp", "zeros_like")
            and len(value.args) == 1
            and not value.keywords
        ):
            return node
        read = copy.deepcopy(target)
        read.ctx = ast.Load()
        if ast.dump(read) != ast.dump(value.args[0]):
            return node
        view = ast.Subscript(
            ast.Attribute(copy.deepcopy(target.value), "at", ast.Load()),
            copy.deepcopy(target.slice),
            ast.Load(),
        )
        replacement = ast.Call(
            ast.Attribute(ast.Name("jnp", ast.Load()), "zeros", ast.Load()),
            [ast.Attribute(view, "shape", ast.Load())],
            [ast.keyword("dtype", ast.Attribute(copy.deepcopy(target.value), "dtype", ast.Load()))],
        )
        self.edits.append(
            {
                "line": node.lineno,
                "original": ast.unparse(value),
                "replacement": ast.unparse(replacement),
            }
        )
        node.value = ast.copy_location(replacement, value)
        return node


class PlaceholderInitialization(ast.NodeTransformer):
    def __init__(self):
        self.edits = []

    def visit_FunctionDef(self, node):
        if node.name not in ("_bwd_mega_kernel", "chunk_kda_bwd_fusion"):
            return node
        self.generic_visit(node)
        if node.name == "_bwd_mega_kernel":
            added = ast.parse(
                """
if DISABLE_RECOMPUTE:
    h_all_ref[...] = jnp.zeros(h_all_ref.shape, dtype=h_all_ref.dtype)
if not FUSE_GATE_BWD:
    gate_dA_ref[...] = jnp.zeros(gate_dA_ref.shape, dtype=gate_dA_ref.dtype)
    gate_dbias_ref[...] = jnp.zeros(gate_dbias_ref.shape, dtype=gate_dbias_ref.dtype)
"""
            ).body
            node.body.extend(added)
            self.edits.append(
                {"line": node.lineno, "addition": "\n".join(ast.unparse(n) for n in added)}
            )
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) == ("pltpu", "PrefetchScalarGridSpec")
        ):
            return node
        for keyword in node.keywords:
            if keyword.arg != "out_specs":
                continue
            assert ast.dump(keyword.value) == ast.dump(
                ast.parse("[any_spec] * 9", mode="eval").body
            )
            replacement = ast.parse(
                "[any_spec] * 6 + [pl.BlockSpec() if disable_recompute else any_spec]"
                " + [pl.BlockSpec() if not fuse_gate_bwd else any_spec] * 2",
                mode="eval",
            ).body
            self.edits.append(
                {
                    "line": keyword.value.lineno,
                    "original": ast.unparse(keyword.value),
                    "replacement": ast.unparse(replacement),
                }
            )
            keyword.value = replacement
        return node


class TailPaddingInitialization(ast.NodeTransformer):
    def __init__(self):
        self.edits = []

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == "_launch_tail_output":
            added = ast.parse(
                """
if REM > 0:
    _ai_db_tail_padding = db_buf.at[
        output_buf, pl.ds(REM * MB_PAD, (CHUNK_BATCH - REM) * MB_PAD), :
    ]
    _ai_db_tail_padding[...] = jnp.zeros(_ai_db_tail_padding.shape, dtype=db_buf.dtype)
"""
            ).body
            node.body[:0] = added
            self.edits.append(
                {"line": node.lineno, "addition": "\n".join(ast.unparse(n) for n in added)}
            )
        return node


@functools.cache
def candidate(
    config=cases.DEFAULT_CONFIG,
    initialize_placeholders=False,
    variant="original",
    *,
    initialize_tail_padding=False,
):
    source = read_source(cases.REPO, "latest_mega")
    tree = ast.parse(source)
    rewrite = ShapeInitialization()
    tree = ast.fix_missing_locations(rewrite.visit(tree))
    assert any(edit["line"] == 1575 for edit in rewrite.edits)
    edits = tuple(rewrite.edits)
    if initialize_placeholders:
        placeholders = PlaceholderInitialization()
        tree = ast.fix_missing_locations(placeholders.visit(tree))
        assert len(placeholders.edits) == 2
        edits += tuple(placeholders.edits)
    if initialize_tail_padding:
        tail_padding = TailPaddingInitialization()
        tree = ast.fix_missing_locations(tail_padding.visit(tree))
        assert len(tail_padding.edits) == 1
        edits += tuple(tail_padding.edits)
    original = cases.source_namespace(variant=variant)["_bwd_mega_kernel"].__globals__
    selected = [
        node
        for node in tree.body
        if (_bound_names(node) or isinstance(node, ast.ImportFrom) and node.module == "__future__")
        and not (
            variant in cases.AI_EXPECTATIONS
            and variant != "cumsum_forward"
            and _bound_names(node) == {"compute_dav"}
        )
    ]
    namespace = _execute_definitions(
        Path(str(original["__file__"]) + ":ai_shape_initialization"), tree, selected, original
    )
    options = dict(cases.function(config=config).keywords)
    return (
        functools.partial(namespace["chunk_kda_bwd_fusion"], **options),
        edits,
    )


def graph_worker(
    config,
    output,
    budget=10000,
    initialize_placeholders=False,
    variant="original",
    comparison=None,
    initialize_tail_padding=False,
):
    settings = cases.Configuration(**config)
    function, edits = candidate(
        settings, initialize_placeholders, variant, initialize_tail_padding=initialize_tail_padding
    )
    reference, _ = candidate(
        settings, initialize_placeholders, initialize_tail_padding=initialize_tail_padding
    )
    original = cases.trace(config=settings)
    changed = jax.make_jaxpr(function)(*cases.signatures(settings))
    diagnostic_base = jax.make_jaxpr(reference)(*cases.signatures(settings))
    data = {
        "source": "AI shape-initialization candidate; not the original baseline",
        "configuration": config,
        "analysis_budget": budget,
        "initialize_placeholders": initialize_placeholders,
        "initialize_tail_padding": initialize_tail_padding,
        "variant": variant,
        "requested_comparison": comparison,
        "comparison_reference": "Shape-initialized AI candidate with identical placeholder policy",
        "edits": edits,
        "original_primitives": cases.primitive_counts(original),
        "candidate_primitives": cases.primitive_counts(changed),
        "comparisons": {},
    }
    print("READY", flush=True)
    for stage, name, compare in (
        (
            "structure",
            "original_structure",
            lambda: compare_jaxprs(original, changed, include_derivative_rules=False),
        ),
        (
            "semantic",
            "candidate_semantic_self" if variant == "original" else "diagnostic_semantic_pair",
            lambda: compare_pallas_kernels(
                diagnostic_base, changed, equivalence="semantic", analysis_budget=budget
            ),
        ),
        (
            "mathematics",
            "candidate_mathematical_self"
            if variant == "original"
            else "diagnostic_mathematical_pair",
            lambda: compare_algebraic_jaxprs(
                diagnostic_base, changed, scope="device_local", algebra_budget=budget
            ),
        ),
    ):
        if comparison is not None and comparison != stage:
            continue
        started = time.monotonic()
        data["comparisons"][name] = {"status": "running"}
        Path(output).write_text(json.dumps(data, separators=(",", ":")) + "\n")
        try:
            result = compare()
            data["comparisons"][name] = {
                "status": result.status.value,
                "equivalence_kind": getattr(result, "equivalence_kind", None),
                "protocol_checked": getattr(result, "protocol_checked", None),
                "warnings": getattr(result, "warnings", ()),
            }
        except ComparisonError as error:
            data["comparisons"][name] = {
                "status": "incomplete",
                "exception": type(error).__name__,
                "reason": str(error),
            }
        data["comparisons"][name]["seconds"] = round(time.monotonic() - started, 4)
        Path(output).write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print('RESULT {"case":"kda_initialization_graphs","status":"ok"}', flush=True)


def numerical_worker(
    config, output, initialize_placeholders=False, variant="original", initialize_tail_padding=False
):
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    settings = cases.Configuration(**config)
    function, _ = candidate(
        settings, initialize_placeholders, variant, initialize_tail_padding=initialize_tail_padding
    )
    params = tpu.InterpretParams(
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    with tpu.force_tpu_interpret_mode(params):
        compiled = [
            jax.jit(f).lower(*cases.signatures(settings)).compile()
            for f in (cases.function(config=settings), function)
        ]
    print("READY", flush=True)
    rows = []
    for seed in (7, 23):
        args, _ = samples(seed, settings)
        actual = []
        for f in compiled:
            actual.append(jax.block_until_ready(f(*args)))
            jax.effects_barrier()
            assert interpreter.races is not None and not interpreter.races.races_found
        metrics = differences(actual[1], actual[0])
        assert all(m["finite"] for m in metrics.values()), metrics
        if variant in ("original", "dav_reordered"):
            assert all(m["exact"] for m in metrics.values()), metrics
        elif cases.AI_EXPECTATIONS[variant] != "inequivalent":
            assert all(m["close"] for m in metrics.values()), metrics
        else:
            assert any(not m["close"] for m in metrics.values()), metrics
        rows.append({"seed": seed, "vs_original": metrics, "races": False})
    Path(output).write_text(json.dumps({"results": rows}, separators=(",", ":")) + "\n")
    print('RESULT {"case":"kda_initialization_numerics","status":"ok"}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        choices=[c.name for c in cases.CONFIGURATIONS],
        default=cases.DEFAULT_CONFIG.name,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--worker-timeout", type=int, default=180)
    parser.add_argument("--comparison", choices=["structure", "semantic", "mathematics"])
    parser.add_argument("--initialize-placeholders", action="store_true")
    parser.add_argument("--initialize-tail-padding", action="store_true")
    parser.add_argument(
        "--variant", choices=["original", *cases.AI_EXPECTATIONS], default="original"
    )
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--graphs-only", action="store_true")
    phases.add_argument("--numerics-only", action="store_true")
    args = parser.parse_args()
    config = next(c for c in cases.CONFIGURATIONS if c.name == args.config)
    root = Path(__file__).resolve().parents[1]
    report = {
        "baseline_revision": cases.REVISION,
        "configuration": dataclasses.asdict(config),
        "initialize_placeholders": args.initialize_placeholders,
        "initialize_tail_padding": args.initialize_tail_padding,
        "variant": args.variant,
        "worker_timeout_seconds": args.worker_timeout,
        "requested_comparison": args.comparison,
        "scope": "Independent AI diagnostic candidate; no original-base semantic proof.",
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checker_source_sha256": hashlib.sha256(
            b"".join(
                str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0"
                for p in sorted((root / "src/jax_reorder").rglob("*.py"))
            )
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report["worker_failures"] = {}
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
    for phase, entry in (("graphs", "graph_worker"), ("numerics", "numerical_worker")):
        if (phase == "numerics" and args.graphs_only) or (phase == "graphs" and args.numerics_only):
            continue
        path = args.output.with_suffix(f".{phase}.json")
        kwargs = {
            "config": dataclasses.asdict(config),
            "output": str(path),
            "initialize_placeholders": args.initialize_placeholders,
            "initialize_tail_padding": args.initialize_tail_padding,
            "variant": args.variant,
        }
        if phase == "graphs":
            kwargs["budget"] = args.budget
            kwargs["comparison"] = args.comparison
        path.write_text("{}\n")
        try:
            run_worker(
                "tests.kda_initialization_probe",
                entry=entry,
                kwargs=kwargs,
                expected={"case": f"kda_initialization_{phase}", "status": "ok"},
                devices=1,
                startup_timeout=180,
                execution_timeout=args.worker_timeout,
            )
        except (AssertionError, TimeoutError) as error:
            report["worker_failures"][phase] = {
                "exception": type(error).__name__,
                "reason": str(error),
            }
        report[phase] = json.loads(path.read_text())
        if phase in report["worker_failures"]:
            for row in report[phase].get("comparisons", {}).values():
                if row["status"] == "running":
                    row.update(
                        status="incomplete",
                        checkpoint_status="running",
                        **report["worker_failures"][phase],
                    )
        args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")
        print(phase, "recorded", flush=True)
    if report["worker_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
