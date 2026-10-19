"""锁定版本的 IR 字段/区域参数清单；候选升级只报告差异与拒绝状态。

默认模式要求当前 JAX/jaxlib 0.11.2 的私有字段、各区域 primitive 参数、公开 lax 身份
与快照 fixtures/jax-adapter-0.11.2.json 逐项一致，并断言一次自比较为 equivalent。
--candidate 用于在独立环境中评估升级候选：不修改 uv.lock、快照或版本允许表，只输出
变更区域和比较/拒绝状态；未审查的新版本若被静默接受会触发断言。

前置条件：CPU 平台与 1 个 CPU 设备即可（shard_map 探测只用到单设备 mesh）。
脚本证明的是“适配面未漂移”，不是所有原语或业务路径都已覆盖。
"""

import argparse
import dataclasses
import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
from jax import lax
from jax._src import core
from jax._src.pallas import core as pallas_core
from jax._src.pallas import primitives as pallas_primitives
from jax._src.pallas.mosaic import core as tpu_core
from jax._src.state import indexing as state_indexing
from jax._src.state import types as state_types
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import Status, UnsupportedJaxprError, compare_jaxprs
from jax_reorder.core.support import public_primitives

SNAPSHOT = Path(__file__).with_name("fixtures") / "jax-adapter-0.11.2.json"


def dma_tree_schema(tree):
    """Snapshot static tree metadata, including nested NDIndexer trees."""

    def metadata(value):
        if type(value) is jax.tree_util.PyTreeDef:
            return dma_tree_schema(value)
        if type(value) is tuple:
            return [metadata(item) for item in value]
        assert value is None or type(value) in (int, bool, str), type(value)
        return value

    node = tree.node_data()
    return {
        "node": None if node is None else f"{node[0].__module__}.{node[0].__qualname__}",
        "metadata": None if node is None else metadata(node[1]),
        "children": [dma_tree_schema(child) for child in tree.children()],
    }


def remote_dma_inventory():
    """Trace one logical mesh device; recording remote schemas needs no execution."""
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("workers",))

    def kernel(x, y, send, receive):
        device = lax.axis_index("workers")
        pltpu.async_remote_copy(x, y, send, receive, (device,)).wait()

    spec = pl.BlockSpec((128,), lambda p: (p,))
    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((128,), jnp.float32),
        grid=(1,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
    )
    function = jax.shard_map(
        function, mesh=mesh, in_specs=P("workers"), out_specs=P("workers"), check_vma=False
    )
    graph = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((128,), jnp.float32))
    kernel = graph.eqns[0].params["jaxpr"].eqns[0].params["jaxpr"]
    result = []
    for eq in kernel.eqns:
        params = eq.params
        result.append(
            {
                "primitive": eq.primitive.name,
                "parameters": {
                    key: f"{type(value).__module__}.{type(value).__qualname__}"
                    for key, value in sorted(params.items())
                },
                "tree": dma_tree_schema(params["tree"]) if "tree" in params else None,
                "wait_flags": [params.get("is_wait_send"), params.get("insert_dummy_device")],
                "effects": sorted(
                    [type(effect).__name__, effect.name]
                    if type(effect) is core.NamedAxisEffect
                    else [type(effect).__name__, eq.invars.index(effect.input)]
                    for effect in eq.effects
                ),
            }
        )
    return result


def pallas_inventory():
    """Actual traced Pallas schemas, including Ref/indexer fields used by the slice."""

    def kernel(x, y, temporary, semaphore):
        lane = jnp.arange(128)
        index = pl.program_id(0) * 128 + lane
        data = x[:]
        y[:] = data
        value = pallas_primitives.load(x, (lane,), mask=index < 1024, other=0.0)
        pallas_primitives.store(y, (lane,), value, mask=index < 1024)
        temporary[:] = x[:]
        pltpu.async_copy(temporary, y, semaphore).wait()

    spec = pl.BlockSpec((128,), lambda p: (p,))
    function = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1024,), jnp.float32),
        grid=(8,),
        in_specs=(spec,),
        out_specs=spec,
        scratch_shapes=(pltpu.VMEM((128,), jnp.float32), pltpu.SemaphoreType.DMA),
    )
    graph = jax.make_jaxpr(function)(jax.ShapeDtypeStruct((1024,), jnp.float32))
    call = graph.eqns[0]

    def sliced_kernel(x, y, semaphore):
        pltpu.async_copy(x.at[128:256], y.at[:128], semaphore).wait()

    sliced = jax.make_jaxpr(
        pl.pallas_call(
            sliced_kernel,
            out_shape=jax.ShapeDtypeStruct((256,), jnp.float32),
            scratch_shapes=(pltpu.SemaphoreType.DMA,),
        )
    )(jax.ShapeDtypeStruct((256,), jnp.float32)).eqns[0]
    parameters = {}
    for eq in (call, *call.params["jaxpr"].eqns):
        parameters[eq.primitive.name] = {
            key: f"{type(value).__module__}.{type(value).__qualname__}"
            for key, value in sorted(eq.params.items())
        }
    classes = (
        pallas_core.GridMapping,
        pallas_core.BlockMapping,
        pallas_core.Blocked,
        state_indexing.NDIndexer,
        state_indexing.Slice,
        state_types.TransformedRef,
        tpu_core.CompilerParams,
    )
    return {
        "descriptor_fields": {
            cls.__name__: [f.name for f in dataclasses.fields(cls)] for cls in classes
        },
        "ref_slots": list(state_types.AbstractRef.__slots__),
        "effect_bindings": [
            list(pair)
            for pair in sorted(
                {(type(e).__name__, type(e.input).__name__) for e in call.params["jaxpr"].effects}
            )
        ],
        "index_map_source_type": type(
            call.params["grid_mapping"].block_mappings[0].index_map_jaxpr
        ).__name__,
        "primitive_parameters": parameters,
        "grid_dimension_semantics": [value.value for value in tpu_core.GridDimensionSemantics],
        "compiler_defaults": {
            field.name: getattr(tpu_core.CompilerParams(), field.name)
            for field in dataclasses.fields(tpu_core.CompilerParams)
        },
        "scratch_refs": [
            {
                "dtype_type": f"{type(aval.dtype).__module__}.{type(aval.dtype).__qualname__}",
                "shape": list(aval.shape),
                "memory_space": aval.memory_space.value,
                "kind": aval.kind,
            }
            for aval in call.params["grid_mapping"].scratch_avals
        ],
        "dma_effect_ports": {
            eq.primitive.name: [
                list(pair)
                for pair in sorted(
                    (type(effect).__name__, eq.invars.index(effect.input)) for effect in eq.effects
                )
            ]
            for eq in call.params["jaxpr"].eqns
            if eq.primitive.name in ("dma_start", "dma_wait")
        },
        "dma_tree_schemas": {
            kind: {
                eq.primitive.name: dma_tree_schema(eq.params["tree"])
                for eq in probe.params["jaxpr"].eqns
                if eq.primitive.name in ("dma_start", "dma_wait")
            }
            for kind, probe in (("whole", call), ("sliced", sliced))
        },
        "remote_dma": remote_dma_inventory(),
    }


def inventory():
    """追踪各区域探测函数，记录私有 IR 字段和每个区域 primitive 的参数类型。"""

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
    # 每个探测只负责让目标区域 primitive 出现在图中，参数结构才是快照要锁定的内容。
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
    # 记录私有 dataclass 的槽位顺序；字段增删会让快照差异暴露，而不是静默沿用旧编码。
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
        "pallas": pallas_inventory(),
        "public_lax": sorted(
            [primitive.name, list(paths)] for primitive, paths in public_primitives().items()
        ),
    }


def main():
    """默认核对快照并断言自比较通过；--candidate 只输出升级评估结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    actual = inventory()
    expected = json.loads(SNAPSHOT.read_text())
    graph = jax.make_jaxpr(lax.neg)(jnp.float32(1))
    # 候选环境中适配面可能已变，比较被显式拒绝也属于需要报告的结果；默认模式则直接失败。
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
