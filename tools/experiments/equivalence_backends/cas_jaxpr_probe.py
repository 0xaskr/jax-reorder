"""把真实 GLM helper jaxpr 转成带索引的 SymPy 表达式，不做执行。

运行（见 docs/development-docs/equivalence-backends/cas.md）：
  PYTHONPATH=/tmp/jax-reorder-equivalence-eval/cas/venv/lib/python3.12/site-packages \
      .venv/bin/python -m tools.experiments.equivalence_backends.cas_jaxpr_probe \
      > /tmp/jax-reorder-equivalence-eval/cas/jaxpr-result.json

只支持正静态维度、无 effects、无捕获常量、float32 有限实数解释的
add/mul/broadcast_in_dim/dot_general/reduce_sum 与同 dtype 转换。它是本次
手写的有限适配器，不是生产转换器：负例只在均匀符号输入族上成立，不能
升级为整层不等价，也不覆盖 mask、softmax、top-k、区域或求导。追踪期间
阻断 JAX 编译与 eager 执行，同时阻断 SymPy 的随机判等。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from unittest.mock import patch

import jax
import jax.numpy as jnp
import sympy as sp
from jax._src import compiler, core, dispatch

from examples import glm52_decode_dsa_mla as layer


class Unsupported(ValueError):
    """适配器不支持的图特征；调用方据此把结论降为 unknown。"""

    pass


def forbidden(*args, **kwargs):
    """在 tracing/转换期间拦截 JAX 执行与数值判等。"""
    raise AssertionError("JAX execution and numerical equality heuristics are forbidden")


@dataclass
class Lowering:
    """把单张 jaxpr 逐方程转成自由索引 + 绑定求和的 SymPy 表达式。"""

    graph: object

    def __post_init__(self):
        """校验输入/方程与维度约束，并建立变量到方程/输入的映射。"""
        if self.graph.consts or self.graph.jaxpr.effects:
            raise Unsupported("captured constants and effects are outside this adapter")
        self.inputs = {var: index for index, var in enumerate(self.graph.jaxpr.invars)}
        self.equations = {}
        supported = {"add", "mul", "broadcast_in_dim", "dot_general", "reduce_sum"}
        supported.add("convert_element_type")
        for equation in self.graph.jaxpr.eqns:
            if equation.primitive.name not in supported or len(equation.outvars) != 1:
                raise Unsupported(f"unsupported primitive: {equation.primitive.name}")
            if equation.effects:
                raise Unsupported("effectful equation")
            self.equations[equation.outvars[0]] = equation
        for var in (*self.inputs, *self.equations):
            if any(not isinstance(size, int) or size <= 0 for size in var.aval.shape):
                raise Unsupported("only positive static dimensions; empty domains need a model")
            if str(var.aval.dtype) != "float32":
                raise Unsupported(f"only float32 interpreted as finite real: {var.aval.dtype}")

    def read(self, var, indices):
        """按给定自由索引返回变量的符号表达式。

        var 可为输入、字面量或方程输出；indices 为该变量的完整输出索引。
        不支持的 primitive、dtype 变化、空维度或秩不匹配都抛 Unsupported。
        """
        if isinstance(var, core.Literal):
            if var.aval.shape or not math.isfinite(float(var.val)):
                raise Unsupported("only finite scalar literals")
            return sp.Rational(float(var.val))
        if len(indices) != len(var.aval.shape):
            raise Unsupported("index rank mismatch")
        if var in self.inputs:
            name = f"input{self.inputs[var]}"
            if not indices:
                return sp.Symbol(name, real=True)
            return sp.IndexedBase(name, shape=var.aval.shape, real=True)[indices]
        equation = self.equations[var]
        op, args = equation.primitive.name, equation.invars
        if op in {"add", "mul"}:
            parts = []
            for arg in args:
                shape = arg.aval.shape
                aligned = indices[len(indices) - len(shape) :] if shape else ()
                mapped = tuple(
                    0 if size == 1 else ix for ix, size in zip(aligned, shape, strict=True)
                )
                parts.append(self.read(arg, mapped))
            return sp.Add(*parts) if op == "add" else sp.Mul(*parts)
        if op == "broadcast_in_dim":
            dimensions = equation.params["broadcast_dimensions"]
            mapped = tuple(
                0 if size == 1 else indices[axis]
                for size, axis in zip(args[0].aval.shape, dimensions, strict=True)
            )
            return self.read(args[0], mapped)
        if op == "convert_element_type":
            if args[0].aval.dtype != var.aval.dtype:
                raise Unsupported("dtype-changing conversion needs an explicit mathematical model")
            return self.read(args[0], indices)
        if op == "reduce_sum":
            axes = tuple(equation.params["axes"])
            source_shape = args[0].aval.shape
            mapped, limits, free = [], [], iter(indices)
            for axis, size in enumerate(source_shape):
                if axis in axes:
                    bound = sp.Dummy("reduce", integer=True)
                    mapped.append(bound)
                    limits.append((bound, 0, size - 1))
                else:
                    mapped.append(next(free))
            return sp.Sum(self.read(args[0], tuple(mapped)), *limits)
        (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = equation.params["dimension_numbers"]
        lhs, rhs = args
        lhs_indices = [None] * len(lhs.aval.shape)
        rhs_indices = [None] * len(rhs.aval.shape)
        free, limits = iter(indices), []
        for la, ra in zip(lhs_batch, rhs_batch, strict=True):
            lhs_indices[la] = rhs_indices[ra] = next(free)
        for mapping, contract, batch in (
            (lhs_indices, lhs_contract, lhs_batch),
            (rhs_indices, rhs_contract, rhs_batch),
        ):
            for axis in range(len(mapping)):
                if axis not in contract and axis not in batch:
                    mapping[axis] = next(free)
        for la, ra in zip(lhs_contract, rhs_contract, strict=True):
            if lhs.aval.shape[la] != rhs.aval.shape[ra]:
                raise Unsupported("contraction dimension mismatch")
            bound = sp.Dummy("contract", integer=True)
            lhs_indices[la] = rhs_indices[ra] = bound
            limits.append((bound, 0, lhs.aval.shape[la] - 1))
        product = self.read(lhs, tuple(lhs_indices)) * self.read(rhs, tuple(rhs_indices))
        return sp.Sum(product, *limits) if limits else product

    def outputs(self):
        """为每个输出构造自由索引表达式，展开并合并同类项后返回。"""
        outputs = []
        for var in self.graph.jaxpr.outvars:
            indices = tuple(sp.Symbol(f"out{axis}", integer=True) for axis in range(var.aval.ndim))
            expression = self.read(var, indices)
            outputs.append(sp.factor_terms(sp.expand(expression)).as_dummy())
        return outputs


def trace(function, args):
    """追踪函数，返回 (ClosedJaxpr, (pytree 结构, 叶子 shape/dtype 契约))。"""
    graph, output = jax.make_jaxpr(function, return_shape=True)(*args)
    leaves, tree = jax.tree.flatten(output)
    contract = [(tuple(leaf.shape), str(leaf.dtype)) for leaf in leaves]
    return graph, (str(tree), contract)


def compare(name, left, right, args, *, negative=False):
    """比较两个真实追踪图：先比对输出契约与支持域，再判定状态。

    返回报告 dict，status 为 equivalent / inequivalent / unknown。正例要求
    两侧索引表达式完全一致；negative=True 时额外在均匀符号输入族上检查
    非零残差，得到 inequivalent；其余情况一律 unknown，不伪造负例。
    """
    left_graph, left_contract = trace(left, args)
    right_graph, right_contract = trace(right, args)
    report = {
        "case": name,
        "input_shapes": [arg.shape for arg in args],
        "output_contract": left_contract,
        "primitives": [
            dict(Counter(eq.primitive.name for eq in g.jaxpr.eqns))
            for g in (left_graph, right_graph)
        ],
        "jaxprs": [str(left_graph), str(right_graph)],
    }
    if left_contract != right_contract:
        return report | {"status": "unknown", "reason": "output contract mismatch"}
    try:
        lhs, rhs = Lowering(left_graph).outputs(), Lowering(right_graph).outputs()
    except Unsupported as exc:
        return report | {"status": "unknown", "reason": str(exc)}
    report["indexed_outputs"] = [[str(expr) for expr in expressions] for expressions in (lhs, rhs)]
    if lhs == rhs:
        return report | {"status": "equivalent", "evidence": "identical exact indexed normal forms"}
    if negative:
        # 均匀符号张量族是这些输入的一个合法子域；替换后 Sum.doit 会闭合
        # 常量求和，不分配任何张量。只有该族出现非零残差才记为 inequivalent。
        parameters = {}

        def uniform(element):
            """把每个 Indexed 的基替换成同一符号，构造均匀符号输入族。"""
            name = str(element.base.label)
            return parameters.setdefault(name, sp.Symbol(f"uniform_{name}", real=True))

        residual = (lhs[0] - rhs[0]).replace(lambda item: isinstance(item, sp.Indexed), uniform)
        residual = sp.expand(residual.doit())
        if residual.has(sp.Sum):
            return report | {"status": "unknown", "reason": "uniform family sum remained"}
        if residual == 0:
            return report | {"status": "unknown", "reason": "uniform family did not distinguish"}
        if residual.is_Rational and residual != 0:
            return report | {
                "status": "inequivalent",
                "evidence": f"uniform family yields exact nonzero constant {residual}",
            }
        polynomial = sp.Poly(residual, *sorted(residual.free_symbols, key=str), domain=sp.QQ)
        if not polynomial.is_zero:
            return report | {
                "status": "inequivalent",
                "evidence": f"uniform symbolic input family yields nonzero polynomial {residual}",
            }
    return report | {"status": "unknown", "reason": "different normal forms without evidence"}


def run():
    """追踪 GLM helper 与最小控制用例，断言预期状态并返回报告。

    使用 examples/glm52_decode_dsa_mla.py 的 glm profile 真实 helper shape；
    断言只覆盖本探针声明支持域内的 7 个用例，结果写入报告。不执行模型。
    """
    p = layer.PROFILES["glm"]

    def abstract(shape):
        """构造 float32 静态形状描述，不分配数组。"""
        return jax.ShapeDtypeStruct(shape, jnp.float32)

    qk_args = (
        abstract((p.batch, p.heads, p.latent + p.rope)),
        abstract((p.batch, p.top_k, p.latent + p.rope)),
    )
    pv_args = (
        abstract((p.batch, p.heads, p.top_k)),
        abstract((p.batch, p.top_k, p.latent)),
    )
    small = (abstract((2, 2)),) * 3
    results = [
        compare(
            "glm_qk_scale",
            lambda q, k: layer._qk(q, k, layer.SCALE),
            lambda q, k: layer._qk_scale_first(q, k, layer.SCALE),
            qk_args,
        ),
        compare("glm_pv_reduce", layer._pv, layer._pv_reduce, pv_args),
        compare(
            "glm_qk_missing_scale",
            lambda q, k: layer._qk(q, k, layer.SCALE),
            lambda q, k: layer._qk_without_scale(q, k, layer.SCALE),
            qk_args,
            negative=True,
        ),
        compare(
            "distributivity", lambda x, y, z: (x + y) * z, lambda x, y, z: x * z + y * z, small
        ),
        compare("unsupported_exp", jnp.exp, lambda x: x, (abstract((2, 2)),)),
        compare("empty_domain", lambda x: x, lambda x: 2 * x, (abstract((0,)),), negative=True),
        compare(
            "constant_difference",
            lambda x: jnp.float32(1),
            lambda x: jnp.float32(2),
            (abstract(()),),
            negative=True,
        ),
    ]
    expected = [
        "equivalent",
        "equivalent",
        "inequivalent",
        "equivalent",
        "unknown",
        "unknown",
        "inequivalent",
    ]
    assert [result["status"] for result in results] == expected
    return {
        "jax_version": jax.__version__,
        "sympy_version": sp.__version__,
        "adapter": "handwritten restricted jaxpr-to-indexed-expression translator",
        "assumptions": [
            "finite real interpretation of float32 operations",
            "no dtype-changing casts",
        ],
        "scope": "actual GLM helpers at glm profile shapes; not complete DSA/MLA layer",
        "compiled": False,
        "executed": False,
        "results": results,
    }


def main():
    """在禁止 JAX 执行与 SymPy 随机判等的补丁下运行探针并打印 JSON。"""
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
        patch.object(sp.Expr, "equals", forbidden),
        patch.object(sp.Expr, "_random", forbidden),
    ):
        report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
