"""混合区域的完整接口、全部分支及规范化执行验收。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core

from examples import mixed_regions as cases
from jax_reorder import Status, compare_functions


def schedule(graph):
    """读取原始 Primitive 顺序，避免把只改 Python 名字当作改写覆盖。"""
    rows, pending = [], [graph]
    while pending:
        value = pending.pop()
        if isinstance(value, core.Jaxpr):
            rows.append(tuple(eq.primitive.name for eq in value.eqns))
            for eq in value.eqns:
                pending.extend(eq.params.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif type(value) is dict:
            pending.extend(value.values())
    return tuple(rows)


@pytest.mark.parametrize("name", ["cond_scan_while_cond", "scan_cond_scan", "jit_cond_scan"])
@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_mixed_interfaces_and_actual_branches(name, flag, dtype):
    original, candidate = getattr(cases, name), getattr(cases, name + "_reordered")
    args = (
        jnp.bool_(flag),
        dtype(0.7),
        dtype(-0.3),
        jnp.array([0.1, -0.2], dtype),
        jnp.array([-0.3, 0.4], dtype),
    )
    result = compare_functions(original, candidate, *args, check_cpu=True)
    assert result.status == Status.EQUIVALENT, result
    mode = result.modes[0]
    assert schedule(mode.left.jaxpr) != schedule(mode.right.jaxpr)
    for expected, actual in zip(
        jax.tree.leaves(original(*args)), jax.tree.leaves(candidate(*args)), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    for raw, normalized in (
        (mode.left.jaxpr, mode.comparison.left.jaxpr),
        (mode.right.jaxpr, mode.comparison.right.jaxpr),
    ):
        core.check_jaxpr(normalized)
        for expected, actual in zip(
            core.eval_jaxpr(raw, raw.consts, *args),
            core.eval_jaxpr(normalized, normalized.consts, *args),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
