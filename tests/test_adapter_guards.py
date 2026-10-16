"""版本、未知字段、描述符继承及语义编码深度的拒绝与保留检查。"""

import jax
import jax.numpy as jnp
import jaxlib
import pytest
from jax import lax
from jax._src import core
from jax._src.lax.lax import NoDCEEffect

from jax_reorder import UnsupportedJaxprError, compare_jaxprs
from jax_reorder.semantics import Encoder, encoded_equal


@pytest.mark.parametrize("module", [jax, jaxlib])
def test_unreviewed_version_requires_explicit_adapter_upgrade(module, monkeypatch):
    graph = jax.make_jaxpr(lax.sin)(jnp.float32(1))
    monkeypatch.setattr(module, "__version__", "0.11.3")
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "version upgrade validation" in result.reason


def test_unknown_abstract_value_subclass_cannot_hide_new_fields():
    class Extended(core.ShapedArray):
        __slots__ = ("extra_semantic_field",)

    base = core.ShapedArray((), jnp.float32)
    value = object.__new__(Extended)
    for name in core.ShapedArray.__slots__:
        if name != "__weakref__" and hasattr(base, name):
            object.__setattr__(value, name, getattr(base, name))
    object.__setattr__(value, "extra_semantic_field", 3)
    with pytest.raises(UnsupportedJaxprError, match="descriptor subclass"):
        Encoder(lambda *args: None, lambda *args: None).encode(value, "aval")


def test_unknown_effect_subclass_is_not_accepted_as_plain_no_dce():
    class ExtraEffect(NoDCEEffect):
        pass

    graph = jax.make_jaxpr(lambda x: x)(jnp.float32(1))
    graph = graph.replace(effects=frozenset((ExtraEffect(),)))
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "unsupported effect" in result.reason


def test_context_subclass_cannot_hide_inherited_compute_fields():
    class Extended(core.JaxprEqnContext):
        __slots__ = ("annotation",)

    graph = jax.make_jaxpr(lax.neg)(jnp.float32(1))
    context = object.__new__(Extended)
    original = graph.eqns[0].ctx
    for name in core.JaxprEqnContext.__slots__:
        if name != "__weakref__" and hasattr(original, name):
            object.__setattr__(context, name, getattr(original, name))
    object.__setattr__(context, "annotation", 1)
    altered = graph.replace(eqns=[graph.eqns[0].replace(ctx=context)])
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(altered, altered)
    result = error.value
    assert "descriptor subclass" in result.reason


def test_deep_encoded_values_are_compared_without_recursion_failure():
    encoder = Encoder(lambda *args: None, lambda *args: None)
    left, same, other = 1, 1, 2
    for _ in range(2000):
        left, same, other = (left,), (same,), (other,)
    a, b, c = (encoder.encode(value, "value") for value in (left, same, other))
    assert encoded_equal(a, b)
    assert not encoded_equal(a, c)
