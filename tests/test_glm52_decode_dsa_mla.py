"""只追踪 GLM-5.2 核心的 jaxpr，检查支持结果与能力缺口，不运行 layer。"""

import ast
import hashlib
import json
from collections import Counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
from jax._src import compiler, dispatch

from examples import glm52_decode_dsa_mla as layer
from examples.full_layers.runner import raw_schedule
from jax_reorder import Status, trace_function

pytestmark = pytest.mark.business


def nested_graphs(value):
    if hasattr(value, "eqns"):
        yield value
        for eq in value.eqns:
            for child in eq.params.values():
                yield from nested_graphs(child)
    elif hasattr(value, "jaxpr"):
        yield from nested_graphs(value.jaxpr)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from nested_graphs(child)
    elif isinstance(value, dict):
        for child in value.values():
            yield from nested_graphs(child)


def primitives(graph):
    return {eq.primitive.name for child in nested_graphs(graph) for eq in child.eqns}


def forbidden(*args, **kwargs):
    raise AssertionError("GLM layer comparison must not compile or execute array operations")


@pytest.fixture(
    scope="module", params=[("small", jnp.float32), ("glm", jnp.float32), ("glm", jnp.bfloat16)]
)
def comparisons(request):
    profile, dtype = request.param
    # 允许比较器记录设备环境；以下所有层追踪都禁止进入后端编译或 eager 运算。
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        args = layer.signatures(profile, dtype)
        baseline = trace_function(layer.function(profile), *args)
        records = [layer.compare(case, baseline, args, profile) for case in layer.CASES]
    return profile, dtype, args, baseline, records


def test_fixed_source_and_five_plus_five_manual_core_cases():
    data = (layer.ROOT / (layer.SOURCE + ".txt")).read_bytes()
    tree = json.loads((layer.ROOT / "upstream-attention-tree.json").read_text())
    entry = next(item for item in tree["files"] if item["path"] == layer.SOURCE)
    blob = b"blob " + str(len(data)).encode() + b"\0" + data
    assert hashlib.sha1(blob).hexdigest() == entry["sha"]
    imports = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(ast.parse(data))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports == {"__future__", "functools", "jax", "jax.numpy"}
    assert Counter(case.expected for case in layer.CASES) == {"equivalent": 5, "inequivalent": 5}
    assert len({case.function for case in layer.CASES}) == 10
    assert {case.target for case in layer.CASES} == {"mla", "dsa"}


def test_signatures_and_core_output_contract_without_arrays(comparisons):
    profile, dtype, args, baseline, _ = comparisons
    assert all(isinstance(x, jax.ShapeDtypeStruct) for x in jax.tree.leaves(args))
    p = layer.PROFILES[profile]
    assert args[0].shape == (p.batch, p.index_heads, p.index_dim)
    assert args[3].shape == (p.batch, p.heads, p.latent + p.rope)
    assert args[3].dtype == dtype
    assert [v.aval.shape for v in baseline.jaxpr.outvars] == [
        (p.batch, p.heads, p.latent),
        (p.batch, p.top_k),
    ]
    if profile == "glm":
        assert (p.heads, p.latent, p.rope, p.index_heads, p.index_dim, p.top_k) == (
            64,
            512,
            64,
            32,
            128,
            2048,
        )


def test_complete_graphs_contain_dsa_mla_and_no_pallas(comparisons):
    _, _, _, baseline, records = comparisons
    for graph in (baseline.jaxpr, *(candidate.jaxpr for _, candidate in records)):
        names = primitives(graph)
        assert not any("pallas" in name for name in names)
        assert {"approx_top_k", "top_k", "gather", "scan", "exp"} <= names


@pytest.mark.parametrize("case_index", [0, 1])
def test_schedule_changes_are_inside_the_core_loop_and_structurally_equal(comparisons, case_index):
    _, _, _, baseline, records = comparisons
    record, candidate = records[case_index]
    assert record["outcome"] == "proven_equivalent", record
    assert record["kind"] == "structural"
    assert record["structural"] == "equivalent"

    def schedules(graph):
        return [
            raw_schedule(eq.params["jaxpr"])
            for child in nested_graphs(graph)
            for eq in child.eqns
            if eq.primitive.name == "scan"
        ]

    original, changed = schedules(baseline.jaxpr), schedules(candidate.jaxpr)
    assert len(original) == len(changed) == 2
    target = 1 if layer.CASES[case_index].target == "mla" else 0
    assert original[target] != changed[target]


@pytest.mark.parametrize("case_index", range(2, 10), ids=[c.name for c in layer.CASES[2:]])
def test_core_rewrites_prove_finite_changes_and_report_remaining_gaps(comparisons, case_index):
    _, _, _, _, records = comparisons
    record, _ = records[case_index]
    assert record["structural"] == "different", record
    if case_index in (2, 3, 4):
        assert record["status"] == "equivalent", record
        assert record["outcome"] == "proven_equivalent"
        assert record["kind"] == "algebraic"
        assert record["issues"] == []
        assert record["warnings"]
        if case_index == 3:
            assert "guarded_softmax_finite" in record["rules"]
        return
    assert record["status"] == "unknown", record
    assert record["outcome"] == "coverage_gap"
    assert record["failure_kind"] == "algebra_incomplete"
    # 这是明确的能力基线；其他 unknown 原因会失败，不能掩盖新退化。
    assert record["issues"] == [
        {"path": "algebra.outputs", "reason": "normal forms differ; no algebraic proof found"}
    ]
    assert record["kind"] is None
    assert record["warnings"]


@pytest.mark.parametrize("expected", ["equivalent", "inequivalent"])
def test_unknown_is_a_reported_gap_and_nonzero_exit(expected):
    outcome = layer.classify(expected, Status.UNKNOWN)
    assert outcome == "coverage_gap"
    assert layer.exit_code({"comparisons": [{"outcome": outcome}]}) == 2


def test_false_acceptance_takes_priority_over_unknown():
    outcome = layer.classify("inequivalent", Status.EQUIVALENT)
    assert outcome == "false_acceptance"
    assert (
        layer.exit_code({"comparisons": [{"outcome": "coverage_gap"}, {"outcome": outcome}]}) == 1
    )
    assert layer.classify("inequivalent", Status.DIFFERENT) == "structural_mismatch_only"


def test_negative_case_requires_a_mathematical_inequivalence_result():
    outcome = layer.classify("inequivalent", Status.INEQUIVALENT)
    assert outcome == "proven_inequivalent"
    assert layer.exit_code({"comparisons": [{"outcome": outcome}]}) == 0
    assert layer.exit_code({"comparisons": [{"outcome": "structural_mismatch_only"}]}) == 2
    assert layer.classify("equivalent", Status.INEQUIVALENT) == "unexpected_mismatch"


def test_cli_writes_jaxprs_and_reports_gaps_without_compiling(tmp_path, monkeypatch, capsys):
    report = tmp_path / "report.json"
    graphs = tmp_path / "jaxprs"
    monkeypatch.setattr(
        "sys.argv",
        ["glm52", "--profile", "small", "--json", str(report), "--jaxpr-dir", str(graphs)],
    )
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        assert layer.main() == 2
    result = json.loads(report.read_text())
    assert not result["executed"] and not result["compiled"]
    assert len(list(graphs.glob("*.jaxpr"))) == 11
    assert result["summary"] == {
        "proven_equivalent": 5,
        "proven_inequivalent": 0,
        "structural_mismatch_only": 0,
        "coverage_gap": 5,
        "false_acceptance": 0,
        "unexpected_mismatch": 0,
    }
    assert "PASS" not in capsys.readouterr().out
