"""The complete MJX-JAX quaternion product and independent AI rewrites.

MJX-JAX's documented TPU support is distinct from the repository's NVIDIA Warp
backend. These cases exercise the original JAX function, not a Pallas kernel.
"""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from tests.upstream_tpu.sources import load_definitions

REVISION = "a72c9b84d003a1980a651a271a04edafde93d7e5"
CASES = (("reordered", True, True), ("regrouped", False, True), ("wrong_cross", False, False))


@cache
def original():
    return load_definitions("mujoco", "math", ("quat_mul",), dict(jax=jax, jp=jnp))["quat_mul"]


def reordered(u, v):
    """AI structural positive: compute independent z, y, w, x outputs in that order."""
    z = u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0]
    y = u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1]
    w = u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3]
    x = u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2]
    return jnp.array([w, x, y, z])


def regrouped(u, v):
    """AI algebraic positive: group the scalar product and antisymmetric terms."""
    w = u[0] * v[0] - (u[1] * v[1] + u[2] * v[2] + u[3] * v[3])
    x = (u[0] * v[1] + u[1] * v[0]) + (u[2] * v[3] - u[3] * v[2])
    y = (u[0] * v[2] + u[2] * v[0]) + (u[3] * v[1] - u[1] * v[3])
    z = (u[0] * v[3] + u[3] * v[0]) + (u[1] * v[2] - u[2] * v[1])
    return jnp.array([w, x, y, z])


def wrong_cross(u, v):
    """AI negative: reverse every cross-product sign, accidentally computing v*u."""
    w = u[0] * v[0] - (u[1] * v[1] + u[2] * v[2] + u[3] * v[3])
    x = (u[0] * v[1] + u[1] * v[0]) - (u[2] * v[3] - u[3] * v[2])
    y = (u[0] * v[2] + u[2] * v[0]) - (u[3] * v[1] - u[1] * v[3])
    z = (u[0] * v[3] + u[3] * v[0]) - (u[1] * v[2] - u[2] * v[1])
    return jnp.array([w, x, y, z])


def function(variant="original", *, batch=None):
    target = (
        original()
        if variant == "original"
        else {"reordered": reordered, "regrouped": regrouped, "wrong_cross": wrong_cross}[variant]
    )
    # The source contract is (4,). Batching is explicitly an external JAX transform,
    # not an invented claim that the original function accepts (..., 4) directly.
    return target if batch is None else jax.vmap(target)


def samples(seed, *, batch=None):
    rng = np.random.default_rng(seed)
    shape = (4,) if batch is None else (batch, 4)
    return tuple(rng.normal(size=shape).astype(np.float32) for _ in range(2))


def reference(u, v):
    """Independent NumPy Hamilton product using left-multiplication matrices."""
    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    products = []
    for quaternion, right in zip(u.reshape(-1, 4), v.reshape(-1, 4), strict=True):
        w, x, y, z = quaternion
        left_matrix = np.array([[w, -x, -y, -z], [x, w, -z, y], [y, z, w, -x], [z, -y, x, w]])
        products.append(left_matrix @ right)
    return np.asarray(products).reshape(u.shape)
