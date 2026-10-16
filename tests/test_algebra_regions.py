"""固定区域、捕获与反馈的代数证明，以及独立的一阶模式。"""

import jax
import jax.numpy as jnp
import pytest
from jax import lax

from jax_reorder import Status, canonicalize, compare_functions, compare_jaxprs, trace_function


def pair(kind, length=3):
    """生成同一控制结构中的两种乘加次序，捕获顺序由实际 JAX 追踪决定。"""

    def build(swapped):
        def expression(a, b):
            return b * a + b if swapped else a * b + b

        if kind == "jit":
            return jax.jit(expression)
        if kind == "cond":

            def conditional(a, b):
                return lax.cond(a[0] > 0, lambda: expression(a, b), lambda: expression(a, b) + 1)

            return conditional
        if kind == "scan":

            def scanned(a, b):
                return lax.scan(
                    lambda c, x: (c + expression(a, b), c * x), a, jnp.ones((length, 2))
                )

            return scanned

        def loop(a, b):
            return lax.while_loop(
                lambda c: c[0] < length, lambda c: (c[0] + 1, c[1] + expression(a, b)), (0, a)
            )[1]

        return loop

    return build(False), build(True)


@pytest.mark.parametrize("kind", ["jit", "cond", "scan", "while"])
@pytest.mark.parametrize("length", [0, 1, 3])
@pytest.mark.parametrize("mode", ["forward", "jvp", "vjp"])
def test_algebra_inside_fixed_regions(kind, length, mode):
    a, b = jnp.array([0.5, 0.8]), jnp.array([0.2, -0.3])
    result = compare_functions(*pair(kind, length), a, b, modes=(mode,), check_cpu=True)
    if kind == "while" and mode == "vjp":
        assert result.status == Status.UNKNOWN
        assert result.modes[0].failure_kind == "trace_rejected"
    else:
        assert result.status == Status.EQUIVALENT, result
        assert result.cpu_compiled


def test_nested_cond_scan_and_jit_regions():
    def make(swapped):
        @jax.jit
        def run(a, b):
            def body(c, _):
                return lax.cond(
                    a[0] > 0, lambda: c + (b + a) if swapped else c + (a + b), lambda: c
                ), c

            return lax.scan(body, a, None, length=2)

        return run

    result = compare_functions(
        make(False), make(True), jnp.ones(2), jnp.ones(2), modes=("forward", "jvp", "vjp")
    )
    assert result.status == Status.EQUIVALENT, result
    # 重复返回同一 Var 时，结构证据须与父节点的结果槽位保持一致。
    original = trace_function(make(False), jnp.ones(2), jnp.ones(2), mode="vjp").jaxpr
    normalized = canonicalize(original, derivative_scope="primal")
    assert (
        compare_jaxprs(original, normalized.jaxpr, derivative_scope="primal").status
        == Status.EQUIVALENT
    )


@pytest.mark.parametrize("kind", ["jit", "cond", "scan", "while"])
def test_region_wrong_binding_and_control_parameters_never_match(kind):
    left, right = pair(kind)
    result = compare_functions(left, lambda a, b: right(b, a), jnp.ones(2), jnp.ones(2))
    assert result.status != Status.EQUIVALENT, result
    if kind in ("scan", "while"):
        wrong, _ = pair(kind, length=4)
        result = compare_functions(left, wrong, jnp.ones(2), jnp.ones(2))
        assert result.status != Status.EQUIVALENT, result


def test_custom_derivatives_are_checked_separately_from_forward():
    def make(scale):
        @jax.custom_jvp
        def function(x):
            return x * x

        @function.defjvp
        def rule(primals, tangents):
            (x,), (t,) = primals, tangents
            return function(x), scale * x * t

        return function

    result = compare_functions(make(2), make(3), jnp.float32(1), modes=("forward", "jvp", "vjp"))
    assert result.modes[0].status == Status.EQUIVALENT
    assert result.modes[1].status == result.modes[2].status == Status.INEQUIVALENT
    assert result.status == Status.INEQUIVALENT


def test_transform_order_and_unrequested_derivatives_remain_explicit():
    left, right = pair("cond")
    result = compare_functions(
        left, right, jnp.ones((3, 2)), jnp.ones((3, 2)), transforms=("vmap", "jit")
    )
    assert result.status == Status.EQUIVALENT, result
    assert result.unverified_modes == ("jvp", "vjp")
    assert result.modes[0].left.signature.transforms == ("vmap", "jit")
