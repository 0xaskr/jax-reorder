"""Replay pre-labelled transformations and independently specified CPU witnesses."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from examples.bfs_reordering import ai_cases, llama_mlp, model_mesh, verify_case
from jax_reorder import Status, canonicalize, compare_jaxprs


@pytest.mark.parametrize("index", range(18))
def test_saved_ai_case(index):
    mesh = model_mesh()
    with jax.set_mesh(mesh):
        case = list(ai_cases(mesh))[index]
        comparison = verify_case(case)
        for graph, fn in ((comparison.left, case.base), (comparison.right, case.candidate)):
            actual = jax.jit(
                lambda *xs, graph=graph: core.eval_jaxpr(graph.jaxpr, graph.jaxpr.consts, *xs)
            )(*jax.tree.leaves(case.args))
            for a, b in zip(actual, jax.tree.leaves(fn(*case.args)), strict=True):
                np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)
            assert canonicalize(graph.jaxpr).graph == graph.graph


def test_llama_layer_with_nontrivial_shapes():
    mesh = model_mesh()
    rng = np.random.default_rng(17)
    with jax.set_mesh(mesh):
        args = tuple(
            jax.device_put(
                jnp.asarray(rng.normal(size=shape), dtype=jnp.float32), NamedSharding(mesh, spec)
            )
            for shape, spec in (
                ((2, 3), P("data", None)),
                ((3, 4), P(None, "tensor")),
                ((3, 4), P(None, "tensor")),
                ((4, 3), P("tensor", None)),
            )
        )
        left = jax.make_jaxpr(lambda *xs: llama_mlp(*xs, mesh=mesh))(*args)
        right = jax.make_jaxpr(lambda *xs: llama_mlp(*xs, mesh=mesh, reverse=True))(*args)
        result = compare_jaxprs(left, right)
        assert result.status == Status.EQUIVALENT, result
        a = llama_mlp(*args, mesh=mesh)
        b = llama_mlp(*args, mesh=mesh, reverse=True)
        np.testing.assert_array_equal(a, b)
