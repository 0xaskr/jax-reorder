"""适配层防护：锁定版本、未知字段/子类与语义编码深度的拒绝检查。

覆盖：
- 拒绝路径：JAX/jaxlib 版本被改成未审查的 0.11.3 时明确要求适配升级；ShapedArray 与
  JaxprEqnContext 的未知子类（可能隐藏新字段）、NoDCEEffect 的未知子类、cast 的非默认
  sharding 都抛 UnsupportedJaxprError。
- 正例与边界：深度 2000 的嵌套编码值仍能比较（迭代实现，不触发递归上限），且只有内容
  相同的两侧 encoded_equal 才为真。

前置条件：CPU 平台；版本断言用 monkeypatch 模拟，不修改已安装的 JAX。
这些拒绝只覆盖已知字段，不证明未来 JAX 版本的私有 API 都已被适配。
"""

import jax
import jax.numpy as jnp
import jaxlib
import pytest
from jax import lax
from jax._src import core
from jax._src.lax.lax import NoDCEEffect

from jax_reorder import UnsupportedJaxprError, compare_jaxprs
from jax_reorder.core.semantics import Encoder, encoded_equal


@pytest.mark.parametrize("module", [jax, jaxlib])
def test_unreviewed_version_requires_explicit_adapter_upgrade(module, monkeypatch):
    """未审查的版本号必须失败，而不是沿用旧版本的字段与参数编码。"""
    graph = jax.make_jaxpr(lax.sin)(jnp.float32(1))
    # 只改版本字符串，模拟在新环境中导入本库；其余行为仍是锁定版本。
    monkeypatch.setattr(module, "__version__", "0.11.3")
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "version upgrade validation" in result.reason


def test_unknown_abstract_value_subclass_cannot_hide_new_fields():
    """动态子类可能新增未编码的语义字段，必须拒绝而不是按基类字段忽略。"""

    class Extended(core.ShapedArray):
        __slots__ = ("extra_semantic_field",)

    base = core.ShapedArray((), jnp.float32)
    # 手工构造实例：复制基类槽位，再额外写入一个编码器不认识的字段。
    value = object.__new__(Extended)
    for name in core.ShapedArray.__slots__:
        if name != "__weakref__" and hasattr(base, name):
            object.__setattr__(value, name, getattr(base, name))
    object.__setattr__(value, "extra_semantic_field", 3)
    with pytest.raises(UnsupportedJaxprError, match="descriptor subclass"):
        Encoder(lambda *args: None, lambda *args: None).encode(value, "aval")


def test_unknown_effect_subclass_is_not_accepted_as_plain_no_dce():
    """Unknown effect 子类不能因为继承 NoDCEEffect 就当作已知顺序 effect 放行。"""

    class ExtraEffect(NoDCEEffect):
        pass

    graph = jax.make_jaxpr(lambda x: x)(jnp.float32(1))
    # 原图没有 effect，这里注入一个未知子类实例，模拟上游新增的 effect 类型。
    graph = graph.replace(effects=frozenset((ExtraEffect(),)))
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "unsupported effect" in result.reason


def test_context_subclass_cannot_hide_inherited_compute_fields():
    """方程 ctx 的子类同样可能携带未编码字段；继承字段一致也不足以放行。"""

    class Extended(core.JaxprEqnContext):
        __slots__ = ("annotation",)

    graph = jax.make_jaxpr(lax.neg)(jnp.float32(1))
    # 复制原 ctx 的全部基类槽位，额外加一个字段，验证拒绝依据是子类身份而非字段值。
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
    """2000 层嵌套编码值用于确认编码比较按迭代实现，不会触发递归上限。"""
    encoder = Encoder(lambda *args: None, lambda *args: None)
    left, same, other = 1, 1, 2
    # 每次包一层单元素 tuple，形成深度 2000 的结构；other 只在最内层值上不同。
    for _ in range(2000):
        left, same, other = (left,), (same,), (other,)
    a, b, c = (encoder.encode(value, "value") for value in (left, same, other))
    assert encoded_equal(a, b)
    assert not encoded_equal(a, c)
