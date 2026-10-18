"""语义字段无损编码：宽 dtype/shape 域、边界值与拒绝回归。

覆盖：
- 正例：19 种 dtype × 5 种 shape（含标量、零长度、单元素与全广播）在 X64 下换序后仍
  equivalent，且规范化图执行结果与原始 Python 函数一致；PRNG key 数组参与 reshape 和
  capture；多结果只使用部分输出时仍保留全部结果类型。
- 反例：NaN payload 位模式、float8 两种格式、jit 的 donation 诊断参数、不同的 folded
  reducer 体、不同的 PRNG key 都判 different。
- 拒绝路径：未知/冒名 primitive、不透明参数、program_order API、循环嵌套 jaxpr 与循环
  参数容器、opaque pytree aux 数据都抛 InvalidJaxprError / UnsupportedJaxprError。

前置条件：需要宽 dtype 的用例在内部临时开启 X64；CPU 平台即可，无需多设备或上游快照。
有限 dtype/shape 样本不证明任意抽象值域。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core, pjit

from jax_reorder import (
    InvalidJaxprError,
    Status,
    UnsupportedJaxprError,
    canonicalize,
    compare_jaxprs,
)

# 19 种 dtype：涵盖布尔、有符号/无符号整数、半精度、bfloat16、float8 与 4 位整数。
# int64/uint64/float64 只在 X64 开启时实际存在，因此对应用例内部开启 X64。
# 每种 dtype 在形状 (0,) 零长度时也有合法 aval，用于确认零元素域不被当作缺失。
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
    """dtype 与 shape 属于结构语义：只换两次 reshape 的定义顺序应判 equivalent。"""
    with jax.enable_x64():
        x = jnp.ones(shape, dtype=dtype)

        def before(x):
            a = lax.reshape(x, (x.size,))
            b = lax.reshape(x, (1, x.size))
            return a, b

        # after 只交换两个互不依赖的 reshape 的定义顺序。
        def after(x):
            b = lax.reshape(x, (1, x.size))
            a = lax.reshape(x, (x.size,))
            return a, b

        a, b = jax.make_jaxpr(before)(x), jax.make_jaxpr(after)(x)
        result = compare_jaxprs(a, b)
        assert result.status == Status.EQUIVALENT, result
        # 重排后的图必须真的可执行：jit + eval_jaxpr 后与原始数值逐元素对比。
        actual = jax.jit(lambda x: core.eval_jaxpr(result.right.jaxpr, [], x))(x)
        for expected, value in zip(before(x), actual, strict=True):
            np.testing.assert_array_equal(value, expected)


def test_prng_key_arrays_in_lax_shape_operations_and_captures():
    """key dtype 数组可以进入普通 shape 运算；不同 key 数组必须判 different。

    分别使用 threefry2x32 与 rbg 两种 PRNG 实现，确认 key 的 aval 编码不依赖实现名。
    """
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
        # 换成 seed=1 的另一组 key 后位内容不同，结构比较必须判 different。
        left = jax.make_jaxpr(lambda keys=keys: lax.reshape(keys, (2, 2)))()
        changed = jax.random.split(jax.random.key(1, impl=implementation), 4)
        right = jax.make_jaxpr(lambda changed=changed: lax.reshape(changed, (2, 2)))()
        assert compare_jaxprs(left, right).status == Status.DIFFERENT


def test_nan_payloads_and_float8_formats_do_not_collapse():
    """NaN 的位级 payload 和 float8 的格式差异都不能被“都是特殊值”抹平。"""
    nan1 = np.array([0x7FC00001], np.uint32).view(np.float32)
    nan2 = np.array([0x7FC00002], np.uint32).view(np.float32)
    # 两个 NaN 的 payload 位不同：若只比较“是否 NaN”会漏判，因此要求 different。
    a, b = [jax.make_jaxpr(lambda x=x: x)() for x in (nan1, nan2)]
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    # e4m3fn 与 e5m2 的指数/尾数位宽不同，dtype 属于语义。
    a, b = [
        jax.make_jaxpr(lambda x: x)(jnp.zeros(1, dtype=t))
        for t in (jnp.float8_e4m3fn, jnp.float8_e5m2)
    ]
    assert compare_jaxprs(a, b).status == Status.DIFFERENT


def test_jit_diagnostic_names_do_not_hide_other_parameters():
    """jit 的调用参数（如 donation）属于语义；诊断名相同也不能掩盖差异。"""
    one = jax.jit(lambda x: lax.neg(x))
    two = jax.jit(lambda x: lax.neg(x), donate_argnums=(0,))
    x = np.float32(1)
    a, b = jax.make_jaxpr(one)(x), jax.make_jaxpr(two)(x)
    assert compare_jaxprs(a, b).status == Status.DIFFERENT
    # 注入无法解释的参数对象，确认拒绝理由落在 opaque parameter 而不是静默忽略。
    params = dict(a.eqns[0].params, compiler_options_kvs=(("opaque", object()),))
    bad = a.replace(eqns=[a.eqns[0].replace(params=params)])
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(bad, bad)
    result = error.value
    assert "opaque parameter" in result.reason


def test_unknown_primitive_cannot_impersonate_lax_by_name():
    """比较按 Primitive 对象身份而非名称；同名冒名对象必须拒绝。"""
    graph = jax.make_jaxpr(lax.sin)(np.float32(1))
    impostor = core.Primitive("sin")
    graph = graph.replace(eqns=[graph.eqns[0].replace(primitive=impostor)])
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "unsupported primitive" in result.reason


def test_implicit_rng_order_crosses_nested_call_boundaries():
    """rng_uniform 的顺序约束藏在 jit 子图内，外层 effects 为空也不能交换两次调用。"""

    @jax.jit
    def random(low, high):
        return lax.rng_uniform(low, high, (2,))

    def fn(low, high):
        a = random(low, high)
        b = random(low, high)
        # 返回顺序与调用顺序相反，便于检查结果端口没有被顺序约束混淆。
        return b, a

    graph = jax.make_jaxpr(fn)(np.float32(0), np.float32(1))
    # 两次调用各留下一个 jit 方程；即使它们没有显式数据依赖也不能交换。
    assert len(graph.eqns) == 2
    result = canonicalize(graph)
    assert result.equation_order == (0, 1)
    swapped = graph.replace(eqns=list(reversed(graph.eqns)))
    assert compare_jaxprs(graph, swapped).status == Status.DIFFERENT


def test_unrelated_program_order_api_is_explicitly_unsupported():
    """根图带有无法解释的 program_order 语义时必须拒绝，而不是按普通纯节点比较。"""
    fn = pjit.program_order(lax.sin, enforce=False)
    graph = jax.make_jaxpr(fn)(np.float32(1))
    with pytest.raises(UnsupportedJaxprError):
        compare_jaxprs(graph, graph)


def test_cyclic_nested_graph_fails_before_jax_recurses():
    """畸形参数环必须在 JAX 自身递归前被识别，避免栈溢出而不是给出拒绝原因。"""
    graph = jax.make_jaxpr(jax.jit(lax.sin))(np.float32(1))
    # 把 jit 的 jaxpr 参数指回整张图，构造一个嵌套环。
    cyclic = graph.replace(eqns=list(graph.eqns))
    cyclic.eqns[0] = cyclic.eqns[0].replace(params=dict(cyclic.eqns[0].params, jaxpr=cyclic))
    with pytest.raises(InvalidJaxprError, match="cyclic nested jaxpr"):
        canonicalize(cyclic)


def test_cyclic_parameter_container_reports_unknown():
    """参数容器自身成环时报告“循环参数”，不能对未解释结构做猜测性编码。"""
    graph = jax.make_jaxpr(jax.jit(lax.sin))(np.float32(1))
    # container 的第一个元素就是它自己，构成 Python 层参数环。
    container = []
    container.append(container)
    eq = graph.eqns[0]
    graph = graph.replace(eqns=[eq.replace(params=dict(eq.params, compiler_options_kvs=container))])
    with pytest.raises(UnsupportedJaxprError) as error:
        compare_jaxprs(graph, graph)
    result = error.value
    assert "cyclic parameter" in result.reason


def test_reducer_callable_is_represented_by_its_traced_body():
    """reduce 的组合函数不是 opaque callable：比较的是它实际追踪出的子图。"""
    x = np.array([1, 2, 3], np.float32)

    def trace(sign):
        # sign 只改变追踪出的二元运算，因此两份图必须判 different。
        def reduce(a, b):
            return a[0] + b[0], a[1] * b[1] if sign else a[1] + b[1]

        return jax.make_jaxpr(
            lambda x: lax.reduce((x, x), (np.float32(0), np.float32(1)), reduce, (0,))
        )(x)

    assert compare_jaxprs(trace(True), trace(True)).status == Status.EQUIVALENT
    assert compare_jaxprs(trace(True), trace(False)).status == Status.DIFFERENT


def test_partially_unused_multi_result_keeps_all_result_types():
    """只使用多结果中的一个输出时，另一个结果的 aval 仍要保留。"""
    graph = jax.make_jaxpr(lambda x: lax.top_k(x, 2)[0])(np.array([1, 3, 2], np.float32))
    normalized = canonicalize(graph)
    assert normalized.complete
    # top_k 的两个结果（值与下标）都出现在节点输出中，但图只引用第 0 个。
    assert len(normalized.graph.nodes[0].outputs) == 2
    assert normalized.graph.outputs == (("result", 0, 0),)


def test_opaque_pytree_auxiliary_data_is_not_silently_compared():
    """自定义 pytree 的 aux 数据不透明时必须拒绝，不能按对象地址或忽略处理。"""
    from jax_reorder import UnsupportedJaxprError
    from jax_reorder.core.semantics import Encoder

    # tree_flatten 返回 object() 作为 aux，比较层无法解释它的内容。
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
