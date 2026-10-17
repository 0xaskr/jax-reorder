"""索引代数的数学正例、不可误接受的连接与数值边界。

覆盖：
- 正例：15 组逐元素多项式恒等式在 float32/bfloat16/float16 下证明等价，并带假设与浮点
  警告；空域、单位轴与广播维；矩阵乘法与“广播乘积再求和”是同一索引表达式；矩阵链结合、
  分配律、批量广播、转置与 einsum；独立求和顺序与线性；未使用求和索引贡献其长度；大轴
  保持符号表示。
- 反例：错误下标连接（a-b/b-a、a@b/b@a、归约轴、广播方向、输出槽位交换）得到完整输出
  的负证明；独立求和乘积漏掉交叉项判 inequivalent。
- 拒绝与未决：max 差值的未完成查询抛 MathematicalUndecidedError/ComparisonTimeoutError；
  整数/复数域不获得交换律；精度 cast、窄化 cast、部分函数（log/sqrt/1÷x）的 0*x 或
  x-x 不能消项；effect 与非有限常量拒绝进入代数；非法/耗尽预算分别抛 ValueError 与
  ComparisonBudgetError。
- 附加正例：默认 auto 保留结构报告，显式 structural 模式不看代数；实数等价保留浮点见证
  与 bitwise 警告。

前置条件：CPU 平台与 SymPy 1.14；数值见证（如 33554432）只用于展示浮点差异，不作为
等价证明。有限恒等式样本不覆盖全部数学改写。
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax._src import core

from jax_reorder import (
    ComparisonBudgetError,
    ComparisonContractError,
    ComparisonTimeoutError,
    MathematicalUndecidedError,
    Status,
    UnsupportedJaxprError,
    compare_algebraic_jaxprs,
    compare_function_traces,
    compare_functions,
    compare_jaxprs,
    trace_function,
)


def signature(shape=(), dtype=jnp.float32):
    """只提供签名，避免测试依赖某个样本数值。"""
    return jax.ShapeDtypeStruct(shape, dtype)


def algebra(left, right, *args, **options):
    """从实际 JAX 追踪图调用底层代数入口，统一构造比较所需的两张图。"""
    a, b = jax.make_jaxpr(left)(*args), jax.make_jaxpr(right)(*args)
    result = compare_algebraic_jaxprs(a, b, **options)
    return result


# 15 组逐元素恒等式：交换律、结合律、分配律、平方差、完全平方、重复相加、双重取负、
# 加减零/乘一、a-a、a**0 以及 sin 的交换律。每组都是实数域上成立的改写。
POLYNOMIALS = [
    (lambda a, b, c: a + b, lambda a, b, c: b + a),
    (lambda a, b, c: a * b, lambda a, b, c: b * a),
    (lambda a, b, c: (a + b) + c, lambda a, b, c: a + (c + b)),
    (lambda a, b, c: (a * b) * c, lambda a, b, c: a * (c * b)),
    (lambda a, b, c: a * (b + c), lambda a, b, c: a * b + a * c),
    (lambda a, b, c: (a + b) * (a - b), lambda a, b, c: a * a - b * b),
    (lambda a, b, c: (a + b) ** 2, lambda a, b, c: a * a + 2 * a * b + b * b),
    (lambda a, b, c: a + a + a, lambda a, b, c: 3 * a),
    (lambda a, b, c: lax.neg(lax.neg(a)), lambda a, b, c: a),
    (lambda a, b, c: a + 0, lambda a, b, c: a),
    (lambda a, b, c: a * 1, lambda a, b, c: a),
    (lambda a, b, c: a - a, lambda a, b, c: jnp.zeros_like(a)),
    (lambda a, b, c: a**0, lambda a, b, c: jnp.ones_like(a)),
    (lambda a, b, c: jnp.sin(a + b), lambda a, b, c: jnp.sin(b + a)),
    (lambda a, b, c: (a * b) + (a * b), lambda a, b, c: 2 * (a * b)),
]


@pytest.mark.parametrize("pair", POLYNOMIALS)
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16, jnp.float16])
def test_elementwise_polynomial_rules(pair, dtype):
    """每个恒等式在三种浮点精度下都要有代数结论、前提条件和浮点警告。"""
    result = algebra(*pair, *(signature((2, 3), dtype) for _ in range(3)))
    assert result.status == Status.EQUIVALENT, result
    # backend_result 记录实际查询，assumptions/warnings 说明结论的适用前提。
    assert result.backend_result and result.assumptions and result.warnings


@pytest.mark.parametrize("shape", [(0,), (1,), (3,), (2, 1, 3)])
def test_polynomial_empty_unit_and_broadcast_dimensions(shape):
    """零长度域、单位长度轴与长度 1 广播都应该能建立同一索引表达式。"""
    result = algebra(
        lambda a, b: a * (b + 1), lambda a, b: a * b + a, signature(shape), signature(())
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("dims", [(2, 3, 4), (3, 3, 3), (2, 0, 4), (2, 1, 4), (0, 3, 4)])
def test_matmul_and_broadcast_reduce_are_the_same_index_expression(dims):
    """矩阵乘法与“广播乘积沿收缩轴求和”必须得到同一索引公式，含零长度维。"""
    m, k, n = dims
    result = algebra(
        lambda a, b: a @ b,
        lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1),
        signature((m, k)),
        signature((k, n)),
    )
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("batch", [(), (2,), (1, 2)])
def test_matrix_chain_and_distributivity(dtype, batch):
    """矩阵链结合律与左右分配律，带批量维和半精度时仍成立。"""
    a, b, c = [signature((*batch, *shape), dtype) for shape in ((2, 3), (3, 4), (4, 5))]
    result = algebra(lambda a, b, c: (a @ b) @ c, lambda a, b, c: a @ (b @ c), a, b, c)
    assert result.status == Status.EQUIVALENT, result
    result = algebra(lambda a, b, c: a @ (b + c), lambda a, b, c: a @ b + a @ c, a, b, b)
    assert result.status == Status.EQUIVALENT, result
    result = algebra(lambda a, b, c: (a + b) @ c, lambda a, b, c: a @ c + b @ c, a, a, b)
    assert result.status == Status.EQUIVALENT, result


def test_batch_broadcast_transpose_and_einsum():
    """批量广播、双重转置和 einsum 都应与普通矩阵乘法等价。"""
    a, b = signature((1, 2, 3)), signature((5, 3, 4))
    result = algebra(
        lambda a, b: a @ b,
        lambda a, b: jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2),
        a,
        b,
    )
    assert result.status == Status.EQUIVALENT, result
    a, b = signature((2, 3)), signature((3, 4))
    for other in (lambda a, b: (b.T @ a.T).T, lambda a, b: jnp.einsum("ik,kj->ij", a, b)):
        assert algebra(lambda a, b: a @ b, other, a, b).status == Status.EQUIVALENT


@pytest.mark.parametrize("shape", [(2, 3), (3, 3), (0, 3), (3, 0), (1, 1)])
def test_independent_sum_order_and_linearity(shape):
    """独立归约轴可以互换顺序，求和也对加法线性；形状允许退化与零长度。"""
    a = signature(shape)
    result = algebra(
        lambda x: jnp.sum(jnp.sum(x, axis=0), axis=0),
        lambda x: jnp.sum(jnp.sum(x, axis=1), axis=0),
        a,
    )
    assert result.status == Status.EQUIVALENT, result
    result = algebra(lambda x, y: jnp.sum(x + y), lambda x, y: jnp.sum(x) + jnp.sum(y), a, a)
    assert result.status == Status.EQUIVALENT, result


@pytest.mark.parametrize("length", [0, 1, 7, 4096])
def test_unused_sum_index_contributes_its_extent(length):
    """对广播结果的求和等于乘以长度：未被元素引用的索引仍贡献其范围。"""
    result = algebra(
        lambda x: jnp.sum(jnp.broadcast_to(x, (length,))), lambda x: x * length, signature()
    )
    assert result.status == Status.EQUIVALENT, result


def test_product_of_independent_sums_retains_all_cross_terms():
    """两个独立求和的乘积展开后包含交叉项；只保留对角项是错误改写。"""
    x = signature((4,))
    result = algebra(
        lambda a, b: jnp.sum(a) * jnp.sum(b), lambda a, b: jnp.sum(a[:, None] * b[None, :]), x, x
    )
    assert result.status == Status.EQUIVALENT, result
    wrong = algebra(lambda a, b: jnp.sum(a) * jnp.sum(b), lambda a, b: jnp.sum(a * b), x, x)
    assert wrong.status == Status.INEQUIVALENT
    # 负结论必须绑定到具体输出槽位。
    assert wrong.backend_result.output is not None


@pytest.mark.parametrize(
    "pair,shapes",
    [
        # 以下每组都是“看起来相似但被比较的索引连接不同”：操作数交换、非对称乘法、
        # 归约轴、广播方向、输出槽位、转置未参与索引。
        ((lambda a, b: a - b, lambda a, b: b - a), ((3,), (3,))),
        ((lambda a, b: a @ b, lambda a, b: b @ a), ((3, 3), (3, 3))),
        ((lambda a, b: jnp.sum(a, axis=0), lambda a, b: jnp.sum(a, axis=1)), ((3, 3), ())),
        ((lambda a, b: a + b[:, None], lambda a, b: a + b[None, :]), ((3, 3), (3,))),
        ((lambda a, b: (a, b), lambda a, b: (b, a)), ((3,), (3,))),
        ((lambda a, b: a @ b, lambda a, b: a @ b.T), ((3, 3), (3, 3))),
    ],
)
def test_wrong_index_connections_have_complete_output_negative_proofs(pair, shapes):
    """错误连接的负结论需要覆盖完整真实输出，结构路径不参与。"""
    result = algebra(*pair, *(signature(s) for s in shapes))
    assert result.status == Status.INEQUIVALENT, result
    assert result.correspondence is None
    assert result.backend_result.output is not None


def test_unfinished_maximum_difference_raises_instead_of_claiming_inequality():
    """未完成的查询必须抛未决异常，不能把“没算出来”当成不等价。"""
    with pytest.raises((MathematicalUndecidedError, ComparisonTimeoutError)) as error:
        algebra(
            lambda a, b: jnp.max(a + b),
            lambda a, b: jnp.max(a) + jnp.max(b),
            signature((3,)),
            signature((3,)),
        )
    result = error.value
    assert result.failure_kind in ("cas_undecided", "timeout")


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.complex64])
def test_non_real_domains_do_not_gain_commutativity(dtype):
    """整数与复数域不套用实数代数规则，交换律必须拒绝或未决。"""
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        algebra(
            lambda a, b: a + b, lambda a, b: b + a, signature((3,), dtype), signature((3,), dtype)
        )


def test_precision_cast_and_partial_function_boundaries():
    """显式 precision、窄化 cast 与部分函数都不能被实数代数规则绕过。"""
    x = signature((3, 3))
    # 显式 HIGH precision 的 dot 与默认 dot 不承诺同一位级语义，因此拒绝。
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        algebra(
            lambda a, b: lax.dot_general(
                a, b, (((1,), (0,)), ((), ())), precision=lax.Precision.HIGH
            ),
            lambda a, b: a @ b,
            x,
            x,
        )
    # bfloat16 往返 cast 有信息损失，不能当作恒等。
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        algebra(lambda a: a.astype(jnp.bfloat16).astype(jnp.float32), lambda a: a, x)
    # log/sqrt/1/x 在部分点无定义，0*partial(x) 与 partial(x)-partial(x) 不能消项。
    for partial in (jnp.log, jnp.sqrt, lambda x: 1 / x):
        for left in (
            lambda x, partial=partial: 0 * partial(x),
            lambda x, partial=partial: partial(x) - partial(x),
        ):
            with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
                algebra(left, lambda x: jnp.zeros_like(x), signature((3,)))


def test_dead_pure_work_is_ignored_only_after_complete_semantic_check():
    """纯而无用的计算可被忽略，但 effect 与非有限常量仍要进入完整语义检查。"""

    def unused(x):
        _ = x * x + jnp.sin(x)
        return x + 1

    result = algebra(unused, lambda x: 1 + x, signature((3,)))
    assert result.status == Status.EQUIVALENT, result

    # 未使用分支里的 debug.print 带 effect，不能被删除。
    def unsafe(x):
        jax.debug.print("value {}", x)
        return x + 1

    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        algebra(unsafe, lambda x: x + 1, signature())
    # 含 inf 常量的交换律不进入实数代数（NaN/Inf 不在有限实数解释内）。
    with pytest.raises((MathematicalUndecidedError, UnsupportedJaxprError)):
        algebra(lambda x: x + jnp.inf, lambda x: jnp.inf + x, signature())


def test_default_auto_retains_structural_report_and_explicit_mode():
    """auto 先结构后代数并保留两份报告；structural 模式只看结构。"""
    a, b, c = (signature((3,)) for _ in range(3))
    left, right = POLYNOMIALS[4]
    auto = compare_functions(left, right, a, b, c, modes=("forward", "jvp", "vjp"), check_cpu=True)
    assert auto.status == Status.EQUIVALENT, auto
    assert auto.equivalence_kind == "algebraic" and auto.cpu_compiled
    # 每个模式：结构报告仍为 DIFFERENT，代数路径给出等价。
    assert all(
        m.algebra and m.algebra.backend_result and m.comparison.status == Status.DIFFERENT
        for m in auto.modes
    )
    structural = compare_functions(left, right, a, b, c, equivalence="structural")
    assert structural.status == Status.DIFFERENT and structural.modes[0].algebra is None
    same = compare_functions(left, left, a, b, c)
    # 两侧其实是同一个函数：结构等价直接通过，不触发代数。
    assert same.equivalence_kind == "structural" and same.modes[0].algebra is None
    graph = jax.make_jaxpr(left)(a, b, c)
    assert compare_jaxprs(graph, jax.make_jaxpr(right)(a, b, c)).status == Status.DIFFERENT


def test_signature_and_environment_mismatch_do_not_enter_algebra():
    """接口或追踪环境不同属于契约差异，不进入代数比较。"""
    x = signature((2,))
    with pytest.raises(ComparisonContractError):
        compare_functions(lambda x: x, lambda x: x, x, right_args=(signature((3,)),))
    left = trace_function(lambda x: x, x)
    # 直接替换追踪上下文，模拟两侧在不同的 JAX 配置下追踪。
    right = replace(left, context=("changed",))
    with pytest.raises(ComparisonContractError, match="context_mismatch"):
        compare_function_traces(left, right)


def test_input_abstract_memory_and_manual_axis_are_not_lost():
    """输入抽象属性（memory_space 与 manual_axis_type）在签名转换后必须逐字段保留。"""
    from jax_reorder.functions import _abstract, _encoder

    # 单设备 AbstractMesh 足以构造 Manual 轴类型，不需要真实多设备。
    mesh = jax.sharding.AbstractMesh((1,), ("x",), axis_types=(jax.sharding.AxisType.Manual,))
    for value in (
        jax.ShapeDtypeStruct((2,), jnp.float32, _memory_space=core.MemorySpace.Host),
        jax.ShapeDtypeStruct(
            (2,),
            jnp.float32,
            sharding=jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
            manual_axis_type=core.ManualAxisType(varying=frozenset({"x"})),
        ),
    ):
        original = core.typeof(value)
        reproduced = core.typeof(_abstract(value))
        # 用同一编码器比较，确保转换前后的完整语义字段一致。
        assert _encoder().encode(original, "input") == _encoder().encode(reproduced, "input")


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_is_rejected(budget):
    """0、负数、布尔与非整数预算都在入口处被拒绝。"""
    with pytest.raises(ValueError, match="budget"):
        compare_functions(lambda x: x, lambda x: x, signature(), algebra_budget=budget)


def test_budget_is_deterministic_and_large_axes_are_symbolic():
    """同样的输入得到同样的预算失败路径；大轴保持符号，不展开成元素。"""
    f, g = POLYNOMIALS[6]
    x = signature((4,))
    errors = []
    # 交替左右两侧，确认预算耗尽的原因与路径不依赖参数顺序。
    for first, second in ((f, g), (g, f), (f, g)):
        with pytest.raises(ComparisonBudgetError) as error:
            algebra(first, second, x, x, x, algebra_budget=1)
        errors.append(error.value)
    assert "budget" in errors[0].reason
    assert errors[0].reason == errors[2].reason
    assert errors[0].path == errors[2].path
    reports = []
    # k=3 与 k=4096 的工作量应相同：索引公式按轴长度符号表示，而不是逐元素展开。
    for k in (3, 4096):
        reports.append(
            algebra(
                lambda a, b: a @ b,
                lambda a, b: jnp.sum(a[:, :, None] * b[None, :, :], axis=1),
                signature((2, k)),
                signature((k, 4)),
            )
        )
    assert all(r.status == Status.EQUIVALENT for r in reports)
    assert reports[0].work == reports[1].work


def test_real_equivalence_keeps_floating_point_witnesses_and_warning():
    """实数等价不等于浮点逐位等价：必须保留浮点见证和 bitwise 警告。"""

    def f(a, b, c):
        return (a + b) + c

    def g(a, b, c):
        return a + (b + c)

    # 33554432 附近的浮点吸收让两种加法顺序结果不同，但实数代数上仍然等价。
    args = tuple(jnp.float32(x) for x in (33554432, -33554432, 1))
    report = compare_functions(f, g, *args)
    assert report.status == Status.EQUIVALENT
    assert f(*args) != g(*args)
    assert "bitwise" in report.modes[0].algebra.warnings[0]
    # 溢出、x*x-x*x 与 +0/-0 三类案例：实数等价但字节表示不同。
    cases = [
        (lambda x: (x + x) - x, lambda x: x, jnp.finfo(jnp.float32).max),
        (lambda x: x * x - x * x, lambda x: jnp.zeros_like(x), jnp.finfo(jnp.float32).max),
        (lambda x: x + 0, lambda x: x, -0.0),
    ]
    for left, right, value in cases:
        x = jnp.float32(value)
        result = compare_functions(left, right, x)
        assert result.status == Status.EQUIVALENT, result
        assert np.asarray(left(x)).tobytes() != np.asarray(right(x)).tobytes()
