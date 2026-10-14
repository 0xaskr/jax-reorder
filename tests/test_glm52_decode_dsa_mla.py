"""纯 JAX DSA/MLA：来源、5+5 手写候选、稀疏边界与数值反例。"""

import ast
import hashlib
import json

import jax
import numpy as np
import pytest

from examples import glm52_decode_dsa_mla as layer
from examples.full_layers.runner import raw_schedule

pytestmark = pytest.mark.business


def primitives(graph):
    if hasattr(graph, "eqns"):
        for eq in graph.eqns:
            yield eq.primitive.name
            for value in eq.params.values():
                yield from primitives(value)
    elif hasattr(graph, "jaxpr"):
        yield from primitives(graph.jaxpr)
    elif isinstance(graph, (tuple, list)):
        for value in graph:
            yield from primitives(value)


def test_original_reference_is_pure_jax_and_matches_upstream_git_blob():
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
    names = set(primitives(jax.make_jaxpr(layer.base)(*layer.inputs("small"))))
    assert not any("pallas" in name for name in names)
    assert {"approx_top_k", "top_k", "gather", "scan"} <= names


def test_exactly_five_manual_positive_and_negative_versions():
    assert len(layer.EQUIVALENT) == len(layer.INEQUIVALENT) == 5
    assert len(set((*layer.EQUIVALENT, *layer.INEQUIVALENT))) == 10
    args = layer.inputs("small")
    assert raw_schedule(jax.make_jaxpr(layer.base)(*args)) != raw_schedule(
        jax.make_jaxpr(layer.query_first)(*args)
    )


@pytest.mark.parametrize("function", layer.EQUIVALENT, ids=lambda f: f.__name__)
def test_complete_forward_equivalence(function):
    layer.compare(function, layer.inputs("small"))


@pytest.mark.parametrize("function", layer.INEQUIVALENT, ids=lambda f: f.__name__)
def test_bad_versions_not_accepted_and_have_output_witnesses(function):
    layer.compare(function, layer.inputs("small"))
    for seed in (17, 18):
        witness = layer.counterexample(function, layer.inputs("small", seed))
        assert witness["max_abs_difference"] > 2e-5


@pytest.mark.parametrize("profile", layer.PROFILES)
@pytest.mark.parametrize("function", (layer.base, *layer.EQUIVALENT), ids=lambda f: f.__name__)
def test_cpu_outputs_and_topk_match_independent_numpy(profile, function):
    for seed in (17, 18):
        layer.check_numeric(function, layer.inputs(profile, seed))


@pytest.mark.parametrize("lengths", [(0, 2), (1, 1), (4, 12)])
def test_empty_short_and_page_boundary_sequences(lengths):
    args = layer.inputs("small", lengths=lengths)
    for function in (layer.base, *layer.EQUIVALENT):
        layer.check_numeric(function, args)
    output, ids = layer.base(*args)
    for t, length in enumerate(lengths):
        valid = np.asarray(ids[t])
        assert np.count_nonzero(valid >= 0) == min(length, layer.TOP_K)
        assert np.all(valid[valid >= 0] < length)
        if length == 0:
            np.testing.assert_array_equal(output[t], 0)


def test_indexer_really_changes_the_attention_output():
    args = list(layer.inputs("small"))
    before, old_ids = layer.base(*args)
    args[3] = -args[3]
    after, new_ids = layer.base(*args)
    assert not np.array_equal(old_ids[0], new_ids[0])
    assert not np.allclose(before[0], after[0], atol=2e-5, rtol=2e-4)


def test_invalid_trial_count_fails():
    with pytest.raises(ValueError, match="trials must be positive"):
        layer.run("small", 0)
