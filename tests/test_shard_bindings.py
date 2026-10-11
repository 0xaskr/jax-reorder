"""真实多 CPU 通信边界的输入输出同步置换与错误绑定。"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import Status, compare_functions, compare_jaxprs


def original(mesh):
    @partial(jax.shard_map, mesh=mesh, in_specs=(P("devices"), P("devices")), out_specs=(P(), P()))
    def body(x, y):
        a = lax.psum(lax.sin(x), "devices")
        b = lax.pmax(lax.cos(y), "devices")
        return a, b

    return body


def reordered(mesh):
    @partial(jax.shard_map, mesh=mesh, in_specs=(P("devices"), P("devices")), out_specs=(P(), P()))
    def body(y, x):
        b = lax.pmax(lax.cos(y), "devices")
        a = lax.psum(lax.sin(x), "devices")
        return b, a

    def run(x, y):
        b, a = body(y, x)
        return a, b

    return run


def reordered_again(mesh):
    @partial(jax.shard_map, mesh=mesh, in_specs=(P("devices"), P("devices")), out_specs=(P(), P()))
    def body(y, x):
        b = lax.pmax(lax.cos(y), "devices")
        a = lax.psum(lax.sin(x), "devices")
        return a, b

    return lambda x, y: body(y, x)


def wrong_input(mesh):
    candidate = reordered(mesh)
    return lambda x, y: candidate(y, x)


def wrong_output(mesh):
    candidate = reordered(mesh)

    def run(x, y):
        a, b = candidate(x, y)
        return b, a

    return run


@pytest.fixture
def setup():
    mesh = Mesh(np.asarray(jax.devices()), ("devices",))
    args = jnp.linspace(-0.8, 1.1, 2 * mesh.size), jnp.linspace(0.2, 1.6, 2 * mesh.size)
    return mesh, args


@pytest.mark.parametrize("candidate", [reordered, reordered_again])
def test_shard_map_internal_interface_permutation(candidate, setup):
    mesh, args = setup
    a, b = original(mesh), candidate(mesh)
    result = compare_functions(a, b, *args, check_cpu=True)
    assert result.status == Status.EQUIVALENT, result
    for expected, actual in zip(a(*args), b(*args), strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert result.modes[0].comparison.correspondence is not None


@pytest.mark.parametrize("candidate", [wrong_input, wrong_output])
def test_wrong_shard_map_binding_has_a_witness(candidate, setup):
    mesh, args = setup
    a, b = original(mesh), candidate(mesh)
    result = compare_functions(a, b, *args)
    assert result.status == Status.DIFFERENT, result
    assert any(not np.array_equal(x, y) for x, y in zip(a(*args), b(*args), strict=True))


def test_unknown_shard_map_interface_metadata_fails_explicitly(setup):
    mesh, args = setup
    graph = jax.make_jaxpr(original(mesh))(*args)
    eq = graph.eqns[0]
    bad = graph.replace(eqns=[eq.replace(params={**eq.params, "new_interface": True})])
    result = compare_jaxprs(bad, bad)
    assert result.status == Status.UNKNOWN
    assert "schema changed" in result.issues[0].reason
