"""Lossless semantic fields, wide CPU signatures, and refusal regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core, pjit

from jax_reorder import InvalidJaxprError, Status, canonicalize, compare_jaxprs

DTYPES = [
    jnp.bool_,
    jnp.int8,
    jnp.int16,
    jnp.int32,
    jnp.int64,
    jnp.uint8,
    jnp.uint16,
    jnp.uint32,
    jnp.uint64,
    jnp.float16,
    jnp.bfloat16,
    jnp.float32,
    jnp.float64,
    jnp.complex64,
    jnp.complex128,
    jnp.float8_e4m3fn,
    jnp.float8_e5m2,
    jnp.int4,
    jnp.uint4,
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (3,), (2, 3), (1, 2, 1)])
def test_cpu_dtype_shape_domain_is_not_a_sample_whitelist(dtype, shape):
    with jax.enable_x64():
        x = jnp.ones(shape, dtype=dtype)

        def before(x):
            a = lax.reshape(x, (x.size,))
            b = lax.reshape(x, (1, x.size))
            return a, b

        def after(x):
            b = lax.reshape(x, (1, x.size))
            a = lax.reshape(x, (x.size,))
            return a, b

        a, b = jax.make_jaxpr(before)(x), jax.make_jaxpr(after)(x)
        result = compare_jaxprs(a, b)
        assert result.status == Status.EQUIVALENT, result
        actual = jax.jit(lambda x: core.eval_jaxpr(result.right.jaxpr, [], x))(x)
        for expected, value in zip(before(x), actual, strict=True):
            np.testing.assert_array_equal(value, expected)


def test_prng_key_arrays_in_lax_shape_operations_and_captures():
    for implementation in ("threefry2x32", "rbg"):
        keys = jax.random.split(jax.random.key(0, impl=implementation), 4)
        graph = jax.make_jaxpr(lambda x: lax.reshape(x, (2, 2)))(keys)
        normalized = canonicalize(graph)
        actual = jax.jit(
            lambda x, normalized=normalized: core.eval_jaxpr(normalized.jaxpr, [], x)[0]
        )(keys)
        np.testing.assert_array_equal(
            jax.random.key_data(actual), jax.random.key_data(keys).reshape(2, 2, -1)
        )
        left = jax.make_jaxpr(lambda keys=keys: lax.reshape(keys, (2, 2)))()
        changed = jax.random.split(jax.random.key(1, impl=implementation), 4)
        right = jax.make_jaxpr(lambda changed=changed: lax.reshape(changed, (2, 2)))()
        assert compare_jaxprs(left, right).status == Status.DIFFERENT


def test_nan_payloads_and_float8_formats_do_not_collapse():
    nan1 = np.array([0x7FC00001], np.uint32).view(np.float32)
    nan2 = np.array([0x7FC00002], np.uint32).view(np.float32)
    a, b = [jax.make_jaxpr(lambda x=x: x)() for x in (nan1, nan2)]
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    a, b = [
        jax.make_jaxpr(lambda x: x)(jnp.zeros(1, dtype=t))
        for t in (jnp.float8_e4m3fn, jnp.float8_e5m2)
    ]
    assert compare_jaxprs(a, b).status == Status.DIFFERENT


def test_jit_diagnostic_names_do_not_hide_other_parameters():
    one = jax.jit(lambda x: lax.neg(x))
    two = jax.jit(lambda x: lax.neg(x), donate_argnums=(0,))
    x = np.float32(1)
    a, b = jax.make_jaxpr(one)(x), jax.make_jaxpr(two)(x)
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    params = dict(a.eqns[0].params, compiler_options_kvs=(("opaque", object()),))
    bad = a.replace(eqns=[a.eqns[0].replace(params=params)])
    result = compare_jaxprs(bad, bad)
    assert result.status == Status.UNKNOWN
    assert "opaque parameter" in result.issues[0].reason


def test_unknown_primitive_cannot_impersonate_lax_by_name():
    graph = jax.make_jaxpr(lax.sin)(np.float32(1))
    impostor = core.Primitive("sin")
    graph = graph.replace(eqns=[graph.eqns[0].replace(primitive=impostor)])
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.UNKNOWN
    assert "unsupported primitive" in result.issues[0].reason


def test_implicit_rng_order_crosses_nested_call_boundaries():
    @jax.jit
    def random(low, high):
        return lax.rng_uniform(low, high, (2,))

    def fn(low, high):
        a = random(low, high)
        b = random(low, high)
        return b, a

    graph = jax.make_jaxpr(fn)(np.float32(0), np.float32(1))
    assert len(graph.eqns) == 2
    result = canonicalize(graph)
    assert result.equation_order == (0, 1)
    swapped = graph.replace(eqns=list(reversed(graph.eqns)))
    assert compare_jaxprs(graph, swapped).status == Status.DIFFERENT


def test_unrelated_program_order_api_is_explicitly_unsupported():
    fn = pjit.program_order(lax.sin, enforce=False)
    graph = jax.make_jaxpr(fn)(np.float32(1))
    assert compare_jaxprs(graph, graph).status == Status.UNKNOWN


def test_cyclic_nested_graph_fails_before_jax_recurses():
    graph = jax.make_jaxpr(jax.jit(lax.sin))(np.float32(1))
    cyclic = graph.replace(eqns=list(graph.eqns))
    cyclic.eqns[0] = cyclic.eqns[0].replace(params=dict(cyclic.eqns[0].params, jaxpr=cyclic))
    with pytest.raises(InvalidJaxprError, match="cyclic nested jaxpr"):
        canonicalize(cyclic)


def test_cyclic_parameter_container_reports_unknown():
    graph = jax.make_jaxpr(jax.jit(lax.sin))(np.float32(1))
    container = []
    container.append(container)
    eq = graph.eqns[0]
    graph = graph.replace(eqns=[eq.replace(params=dict(eq.params, compiler_options_kvs=container))])
    result = compare_jaxprs(graph, graph)
    assert result.status == Status.UNKNOWN
    assert "cyclic parameter" in result.issues[0].reason


def test_reducer_callable_is_represented_by_its_traced_body():
    x = np.array([1, 2, 3], np.float32)

    def trace(sign):
        def reduce(a, b):
            return a[0] + b[0], a[1] * b[1] if sign else a[1] + b[1]

        return jax.make_jaxpr(
            lambda x: lax.reduce((x, x), (np.float32(0), np.float32(1)), reduce, (0,))
        )(x)

    assert compare_jaxprs(trace(True), trace(True)).status == Status.EQUIVALENT
    assert compare_jaxprs(trace(True), trace(False)).status == Status.DIFFERENT


def test_partially_unused_multi_result_keeps_all_result_types():
    graph = jax.make_jaxpr(lambda x: lax.top_k(x, 2)[0])(np.array([1, 3, 2], np.float32))
    normalized = canonicalize(graph)
    assert normalized.complete
    assert len(normalized.graph.nodes[0].outputs) == 2
    assert normalized.graph.outputs == (("result", 0, 0),)


def test_opaque_pytree_auxiliary_data_is_not_silently_compared():
    from jax_reorder import UnsupportedJaxprError
    from jax_reorder.semantics import Encoder

    @jax.tree_util.register_pytree_node_class
    class Box:
        def tree_flatten(self):
            return (), object()

        @classmethod
        def tree_unflatten(cls, auxiliary, children):
            return cls()

    tree = jax.tree.structure(Box())
    with pytest.raises(UnsupportedJaxprError, match="opaque parameter"):
        Encoder(None, None).encode(tree, "tree")
