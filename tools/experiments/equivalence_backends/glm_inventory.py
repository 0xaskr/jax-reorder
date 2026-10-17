"""在禁止编译与 eager 执行的条件下清点真实 GLM 图的 primitive/参数。

运行（见 docs/development-docs/equivalence-backends/compiler.md）：
  python -m tools.experiments.equivalence_backends.glm_inventory \
      --json /tmp/jax-reorder-equivalence-eval/glm-inventory.json

追踪 examples/glm52_decode_dsa_mla.py 的 base 与全部固定改写，递归统计
嵌套图（cond/scan/jit 等）中的方程出现次数、primitive 集合与关键参数。
统计是语法出现次数，不是执行次数；不编译、不执行数组运算，也不判断
等价。输出 JSON；--profile/--dtype 选择签名，参数非法时 parser.error。
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
from jax._src import compiler, dispatch

from examples import glm52_decode_dsa_mla as layer
from jax_reorder import trace_function


def nested_graphs(value, path="root"):
    """递归遍历 jaxpr 及嵌套区域，按路径产出 (path, 图对象)。

    value 可为带 eqns 的 jaxpr、带 jaxpr 属性的 ClosedJaxpr、dict 或
    tuple/list；path 记录进入区域/参数的路径，便于在报告中定位。
    """
    if hasattr(value, "eqns"):
        yield path, value
        for index, equation in enumerate(value.eqns):
            for name, child in equation.params.items():
                yield from nested_graphs(child, f"{path}/{index}:{equation.primitive.name}/{name}")
    elif hasattr(value, "jaxpr"):
        yield from nested_graphs(value.jaxpr, path)
    elif isinstance(value, dict):
        for name, child in value.items():
            yield from nested_graphs(child, f"{path}/{name}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from nested_graphs(child, f"{path}/{index}")


def forbidden(*args, **kwargs):
    """在 tracing 期间拦截后端编译与 eager 执行，确保清单不运行模型。"""
    raise AssertionError("inventory must not compile or evaluate JAX array operations")


def inventory(graph):
    """统计一张图（含嵌套区域）的方程出现次数、primitive 与参数样例。

    graph 为追踪结果；返回 equation_occurrences、primitives、
    parameter_examples 与 regions。参数样例只保留影响索引/收缩/填充语义
    的键，用 repr 记录，不做数值执行。
    """
    counts = Counter()
    parameters = {}
    regions = []
    for path, child in nested_graphs(graph):
        regions.append({"path": path, "equations": len(child.eqns)})
        for equation in child.eqns:
            name = equation.primitive.name
            counts[name] += 1
            # 只保留影响索引、收缩、填充与归约语义的参数，避免输出过大。
            selected = {
                key: repr(value)
                for key, value in equation.params.items()
                if key
                in {
                    "axes",
                    "dimension",
                    "dimension_numbers",
                    "fill_value",
                    "k",
                    "length",
                    "mode",
                    "precision",
                    "recall_target",
                    "reduction_dimension",
                    "reverse",
                    "slice_sizes",
                }
            }
            if selected and selected not in parameters.setdefault(name, []):
                parameters[name].append(selected)
    return {
        "equation_occurrences": counts.total(),
        "primitives": dict(sorted(counts.items())),
        "parameter_examples": parameters,
        "regions": regions,
    }


def run(profile, dtype):
    """追踪 base 与全部固定改写，返回可 JSON 序列化的清单报告。

    profile/dtype 选择 GLM 签名；在禁止编译与 eager 执行的补丁下只做
    tracing。返回环境、来源提交、profile/dtype、primitive 并集与每张图的
    清单；不写文件、不判断等价。
    """
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        args = layer.signatures(profile, dtype)
        traces = {"base": trace_function(layer.function(profile), *args)}
        for case in layer.CASES:
            traces[case.name] = trace_function(layer.function(profile, case), *args)
    graphs = {name: inventory(trace.jaxpr) for name, trace in traces.items()}
    return {
        "jax_version": jax.__version__,
        "source_revision": layer.REVISION,
        "profile": profile,
        "dtype": str(jnp.dtype(dtype)),
        "compiled": False,
        "executed": False,
        # 该字段说明统计口径，防止把图内出现次数误读为运行时执行次数。
        "interpretation": "recursive syntactic occurrences, not executed iteration counts",
        "all_primitives": sorted({op for graph in graphs.values() for op in graph["primitives"]}),
        "graphs": graphs,
    }


def main():
    """解析参数、生成清单并按需写 JSON；始终打印摘要，不执行模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=layer.PROFILES, default="glm")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--json", type=Path)
    options = parser.parse_args()
    report = run(options.profile, jnp.dtype(options.dtype))
    if options.json:
        options.json.parent.mkdir(parents=True, exist_ok=True)
        options.json.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "profile": report["profile"],
                "graphs": len(report["graphs"]),
                "base_equation_occurrences": report["graphs"]["base"]["equation_occurrences"],
                "primitives": report["all_primitives"],
                "compiled": False,
                "executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
