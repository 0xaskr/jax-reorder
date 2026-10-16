"""锁定版本的 IR 字段/区域参数清单；候选升级只报告差异与拒绝状态。"""

import argparse
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax import lax
from jax._src import core
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import Status, UnsupportedJaxprError, compare_jaxprs
from jax_reorder.support import public_primitives

SNAPSHOT = Path(__file__).with_name("fixtures") / "jax-adapter-0.11.2.json"


def inventory():
    @jax.custom_jvp
    def jvp(x):
        return x * x

    @jvp.defjvp
    def jvp_rule(primals, tangents):
        (x,), (t,) = primals, tangents
        return x * x, (2 * x) * t

    @jax.custom_vjp
    def vjp(x):
        return x * x

    vjp.defvjp(lambda x: (x * x, x), lambda r, ct: ((2 * r) * ct,))
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("workers",))
    probes = {
        "jit": jax.jit(lax.sin),
        "cond": lambda x: lax.cond(x > 0, lax.sin, lax.cos, x),
        "scan": lambda x: lax.scan(lambda c, v: (c + v, c - v), x, jnp.arange(2, dtype=x.dtype)),
        "while": lambda x: lax.while_loop(lambda y: y < 2, lambda y: y + 1, x),
        "custom_jvp_call": jvp,
        "custom_vjp_call": vjp,
        "shard_map": jax.shard_map(lax.neg, mesh=mesh, in_specs=P(), out_specs=P()),
    }
    parameters = {}
    for name, function in probes.items():
        graph = jax.make_jaxpr(function)(jnp.float32(0.7))
        eq = next(eq for eq in graph.eqns if eq.primitive.name == name)
        parameters[name] = {
            key: f"{type(value).__module__}.{type(value).__qualname__}"
            for key, value in sorted(eq.params.items())
        }
    fields = {
        name: list(getattr(getattr(core, name), "__slots__", ()))
        for name in (
            "Jaxpr",
            "JaxprEqn",
            "Var",
            "Literal",
            "ShapedArray",
            "JaxprEqnContext",
            "AbstractToken",
        )
    }
    return {
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "python_major_minor": list(map(int, platform.python_version_tuple()[:2])),
        "private_fields": fields,
        "primitive_parameters": parameters,
        "public_lax": sorted(
            [primitive.name, list(paths)] for primitive, paths in public_primitives().items()
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    actual = inventory()
    expected = json.loads(SNAPSHOT.read_text())
    graph = jax.make_jaxpr(lax.neg)(jnp.float32(1))
    try:
        comparison = compare_jaxprs(graph, graph)
    except UnsupportedJaxprError as error:
        if not args.candidate:
            raise
        comparison = None
        failure = {"type": type(error).__name__, "path": error.path, "reason": error.reason}
    changed = [name for name in expected if actual.get(name) != expected[name]]
    report = {
        "inventory": actual,
        "changed_sections": changed,
        **({"comparison": comparison.status.value} if comparison else {"error": failure}),
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.candidate:
        if (jax.__version__, jaxlib.__version__) != ("0.11.2", "0.11.2"):
            assert comparison is None, "unaudited upgrade was silently accepted"
        print(json.dumps({"upgrade_requires_review": bool(changed), **report}, indent=2))
    else:
        assert not changed, (
            f"adapter inventory changed: {changed}; review before updating the snapshot"
        )
        assert comparison.status == Status.EQUIVALENT
        print("JAX/jaxlib 0.11.2 adapter inventory: unchanged; audited comparison passed")


if __name__ == "__main__":
    main()
