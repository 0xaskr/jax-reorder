"""把已校验 JAX 图转换为带索引的表达式；此处不执行代数规范化。

原语身份、参数模式、类型和轴映射是转换的共同信任基础。
张量中间值保持独立自由轴，读取时只替换当前表达式的自由索引。
"""

from dataclasses import dataclass, replace
from fractions import Fraction
from math import prod

import jax
import numpy as np
from jax import lax
from jax._src import ad_util, core, pjit
from jax._src.lax.lax import empty2_p
from jax._src.layout import AutoLayout

from .algebra_indices import add, divide, integer, map_index, scale
from .algebra_ir import Budget, Expr, Index, ratio
from .canonical import _ordered_equation
from .derivatives import CUSTOM_PRIMITIVES, semantic_parameters
from .errors import UnsupportedJaxprError
from .semantics import Encoder


def _reject_graph(_, path):
    """只有区域适配器可以解释子图，普通描述符不允许隐藏图。"""
    raise UnsupportedJaxprError(path, "nested graph requires an algebra region adapter")


def encode(value):
    """复用完整语义编码，不用 repr 或哈希摘要作为数学身份。"""
    return Encoder(_reject_graph, lambda _: None).encode(value, "algebra.metadata")


def _substitute(expr, indices, cache):
    """替换一个张量的自由轴，不进入 atom 的独立张量参数。"""
    key = expr, indices
    if key in cache:
        return cache[key]
    if expr.op in ("atom", "bag"):
        semantic, old = expr.data
        result = Expr(
            expr.op,
            (
                semantic,
                tuple(
                    map_index(i, lambda j: indices[j.number] if j.kind == "free" else j)
                    for i in old
                ),
            ),
            expr.args,
            expr.total,
        )
    elif expr.op == "constant":
        result = expr
    else:
        result = Expr(
            expr.op, expr.data, tuple(_substitute(a, indices, cache) for a in expr.args), expr.total
        )
    cache[key] = result
    return result


@dataclass(frozen=True)
class Tensor:
    """具有有序自由轴和完整抽象属性的张量表达式。"""

    signature: tuple
    shape: tuple[int, ...]
    dtype: object
    body: Expr
    nonfinite: bool = False


@dataclass(frozen=True)
class Lowered:
    """有序输出和必须保留的区域根，接口不包含原图临时变量名。"""

    signature: tuple
    outputs: tuple[Tensor, ...]
    barriers: tuple[Expr, ...]

    @property
    def roots(self):
        """保留每个输出位置；区域根单独参与比较，禁止消去循环边界。"""
        return (
            tuple(
                _substitute(
                    value.body,
                    tuple(
                        Index("zero") if n == 1 else Index("free", i)
                        for i, n in enumerate(value.shape)
                    ),
                    {},
                )
                for value in self.outputs
            )
            + self.barriers
        )


class Lowerer:
    """从输入向输出构造 DAG，索引范围保持符号。"""

    def __init__(self, budget: Budget, common_calls=None):
        """计数器只给临时求和索引新身份，不参与规范结果。"""
        self.budget = budget
        self.next_index = 0
        self.substitutions = {}
        self.rules = set()
        self.region_depth = 0
        self.index_domains = {}
        self.common_calls = common_calls
        self.read_shapes = {}
        self.nonfinite_graphs = {}
        self.finite_values = {}

    def _record_finiteness(self, value, finite, *, possible=False):
        """有限值事实不改变部分求值标记，因而不能授权删除内部未定义路径。"""
        if finite:
            value = replace(value, nonfinite=False)
        else:
            value = self._mark_nonfinite(value, possible)
        self.finite_values[value] = bool(finite)
        return value

    def _nonfinite_parameters(self, value):
        """gather fill_value 等参数也能产生非有限值，不能只检查 SSA 常量。"""
        if isinstance(value, (float, complex, np.number, np.ndarray)):
            array = np.asarray(value)
            return jax.dtypes.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array))
        if isinstance(value, dict):
            return any(self._nonfinite_parameters(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(self._nonfinite_parameters(v) for v in value)
        return False

    def _nonfinite_graph(self, graph):
        """检查共同调用中的非有限常量与复数极点；循环反馈采用保守上界。"""
        if graph in self.nonfinite_graphs:
            return self.nonfinite_graphs[graph]
        from .correspondence import _parameter_view

        def nonfinite(value, aval):
            return jax.dtypes.issubdtype(aval.dtype, np.inexact) and not np.all(
                np.isfinite(np.asarray(value, dtype=aval.dtype))
            )

        found = any(
            nonfinite(value, var.aval)
            for var, value in zip(graph.constvars, graph.consts, strict=True)
        )
        for eq in graph.eqns:
            self.budget.spend(path="algebra.nonfinite_graph")
            found |= eq.primitive is lax.tanh_p and any(
                jax.dtypes.issubdtype(var.aval.dtype, np.complexfloating) for var in eq.invars
            )
            found |= self._nonfinite_parameters(eq.params)
            found |= any(
                nonfinite(var.val, var.aval) for var in eq.invars if isinstance(var, core.Literal)
            )
            _, children = _parameter_view(
                eq, lambda _: None, derivative_scope="primal", parent=graph
            )
            for child in children.values():
                found |= self._nonfinite_graph(child)
        found |= any(
            nonfinite(var.val, var.aval) for var in graph.outvars if isinstance(var, core.Literal)
        )
        self.nonfinite_graphs[graph] = bool(found)
        return bool(found)

    def _mark_nonfinite(self, value, possible):
        """标记与 total 分开：部分实数函数不一定含无穷，非有限值不能消项。"""
        if possible and jax.dtypes.issubdtype(value.dtype, np.inexact):
            return replace(value, body=replace(value.body, total=False), nonfinite=True)
        return value

    def _axes(self, aval):
        """首期只接纳给定签名中的具体非负维度。"""
        if not isinstance(aval, core.ShapedArray) or any(
            type(d) is not int or d < 0 for d in aval.shape
        ):
            raise UnsupportedJaxprError("algebra.shape", "requires static shaped array values")
        return tuple(Index("free", i) for i, n in enumerate(aval.shape))

    def _tensor(self, aval, body):
        """记录完整抽象属性，dtype 仅用于判断实数规则的适用性。"""
        self._axes(aval)
        domains = dict(self.index_domains)
        domains.update((Index("free", i), (0, n)) for i, n in enumerate(aval.shape))
        return Tensor(
            encode(aval), tuple(aval.shape), aval.dtype, self._simplify_indices(body, domains, {})
        )

    def _simplify_indices(self, expression, domains, cache):
        """在已知输出域内化简坐标；不进入未知张量调用的独立参数。"""
        if expression in cache:
            return cache[expression]
        self.budget.spend(path="algebra.index_simplify")
        if expression.op in ("atom", "bag"):
            semantic, indices = expression.data
            result = Expr(
                expression.op,
                (semantic, tuple(map_index(i, lambda j: j, domains) for i in indices)),
                expression.args,
                expression.total,
            )
        else:
            result = Expr(
                expression.op,
                expression.data,
                tuple(self._simplify_indices(a, domains, cache) for a in expression.args),
                expression.total,
            )
        cache[expression] = result
        return result

    def _input(self, aval, identity, *, nonfinite=False):
        """读取外部输入或固定循环形参，身份由有序接口锚定。"""
        semantic = ("input", identity, encode(aval))
        self.read_shapes[semantic] = tuple(aval.shape)
        return self._record_finiteness(
            self._tensor(aval, Expr("atom", (semantic, self._axes(aval)))),
            identity[0] == "argument" and not nonfinite,
            possible=nonfinite,
        )

    def _constant(self, value, aval):
        """保留数组完整内容；有限实数标量按实际 dtype 值精确解释。"""
        array = np.asarray(value, dtype=aval.dtype)
        real = jax.dtypes.issubdtype(aval.dtype, np.floating)
        if jax.dtypes.issubdtype(aval.dtype, np.complexfloating) and not np.all(np.isfinite(array)):
            raise UnsupportedJaxprError("algebra.constant", "non-finite complex constant")
        if real and not np.all(np.isfinite(array)):
            if not np.all(np.isfinite(array) | np.isneginf(array)):
                raise UnsupportedJaxprError("algebra.constant", "non-finite constant")
            semantic = ("negative_infinity_constant", encode(aval), encode(array))
            self.read_shapes[semantic] = tuple(aval.shape)
            self.rules.add("negative_infinity_constant")
            return self._mark_nonfinite(
                self._tensor(aval, Expr("atom", (semantic, self._axes(aval)), total=False)),
                True,
            )
        if real and array.shape == ():
            # 锁定 CPU 浮点类型均可无损提升为 Python binary64 再取精确比值。
            body = Expr("constant", ratio(Fraction.from_float(float(array))))
        else:
            semantic = ("constant", encode(aval), encode(array))
            self.read_shapes[semantic] = tuple(aval.shape)
            body = Expr("atom", (semantic, self._axes(aval)))
        return self._record_finiteness(self._tensor(aval, body), bool(np.all(np.isfinite(array))))

    def _read(self, value, indices):
        """读取已构造张量，不枚举索引域中的元素。"""
        if len(indices) != len(value.shape):
            raise UnsupportedJaxprError("algebra.index", "read rank mismatch")
        self.budget.spend(path="algebra.read")
        return _substitute(value.body, indices, self.substitutions)

    def _broadcast_read(self, value, shape, indices):
        """按尾部对齐映射逐元素广播，长度 1 的轴读取下标 0。"""
        offset = len(shape) - len(value.shape)
        if offset < 0 or any(
            a not in (1, b) for a, b in zip(value.shape, shape[offset:], strict=True)
        ):
            raise UnsupportedJaxprError("algebra.broadcast", "incompatible broadcast mapping")
        return self._read(
            value,
            tuple(
                Index("zero") if a == 1 and b != 1 else i
                for a, b, i in zip(value.shape, shape[offset:], indices[offset:], strict=True)
            ),
        )

    def _sum(self, body, bindings):
        """为独立有限矩形域建立嵌套求和，规范化阶段再合并。"""
        for index, extent in reversed(bindings):
            body = Expr("sum", (index.number, extent), (body,), body.total)
        return body

    def _new_index(self):
        """创建不与外层求和混淆的临时索引。"""
        result = Index("sum", self.next_index)
        self.next_index += 1
        return result

    def _algebra_allowed(self, eq, values):
        """仅展开同一实数 dtype 的普通 CPU 数学路径。"""
        if len(eq.outvars) != 1:
            return False
        aval = eq.outvars[0].aval
        if not isinstance(aval, core.ShapedArray) or not jax.dtypes.issubdtype(
            aval.dtype, np.floating
        ):
            return False
        if any(value.dtype != aval.dtype for value in values):
            return False
        # 特殊放置、计算类型或布局不能在张量公式转换中被隐式丢弃。
        if eq.params.get("out_sharding") is not None or eq.ctx.compute_type is not None:
            return False
        if not eq.ctx.cur_abstract_mesh.empty or eq.ctx.xla_metadata is not None:
            return False
        if eq.ctx.cur_layout_mode is not core.LayoutMode.AUTO:
            return False
        for atom in (*eq.invars, *eq.outvars):
            abstract = atom.aval
            if (
                abstract.memory_space is not core.MemorySpace.Device
                or abstract.layout is not AutoLayout
                or not abstract.manual_axis_type.empty
                or not abstract.sharding.mesh.empty
            ):
                return False
        return True

    def _arithmetic(self, eq, values):
        """已审查的乘加和索引映射；没有适配的参数集合交给完整调用边界。"""
        if not self._algebra_allowed(eq, values):
            return None
        primitive, params = eq.primitive, eq.params
        if any(v.nonfinite for v in values) and primitive not in {
            lax.broadcast_in_dim_p,
            lax.transpose_p,
            lax.squeeze_p,
            lax.reshape_p,
            lax.slice_p,
        }:
            # 保留非有限运算的顺序与分组；x-x、0*x、分配律不适用于 NaN/Inf。
            return None
        aval = eq.outvars[0].aval
        indices = self._axes(aval)
        total = all(v.body.total for v in values)
        arithmetic = {lax.add_p: "add", ad_util.add_jaxvals_p: "add", lax.mul_p: "mul"}
        if primitive in arithmetic and (
            not params or (primitive is lax.mul_p and params == {"out_dtype": None})
        ):
            args = tuple(self._broadcast_read(v, aval.shape, indices) for v in values)
            body = Expr(arithmetic[primitive], args=args, total=total)
        elif primitive in (lax.sub_p, lax.neg_p) and not params:
            args = tuple(self._broadcast_read(v, aval.shape, indices) for v in values)
            negative = Expr("mul", args=(Expr("constant", (-1, 1)), args[-1]), total=total)
            body = (
                negative
                if primitive is lax.neg_p
                else Expr("add", args=(args[0], negative), total=total)
            )
        elif (
            primitive is lax.integer_pow_p
            and set(params) == {"y"}
            and type(params["y"]) is int
            and params["y"] >= 0
        ):
            exponent = params["y"]
            self.budget.spend(exponent, "algebra.power")
            if exponent == 0 and not total:
                raise UnsupportedJaxprError("algebra.domain", "cannot erase a partial power base")
            body = Expr("mul", args=(self._read(values[0], indices),) * exponent, total=total)
        elif (
            primitive is lax.broadcast_in_dim_p
            and set(params) <= {"shape", "broadcast_dimensions", "sharding"}
            and params.get("sharding") is None
        ):
            dimensions = params["broadcast_dimensions"]
            body = self._read(
                values[0],
                tuple(
                    Index("zero") if size == 1 and aval.shape[d] != 1 else indices[d]
                    for size, d in zip(values[0].shape, dimensions, strict=True)
                ),
            )
        elif primitive is lax.transpose_p and set(params) == {"permutation"}:
            mapped = [None] * len(indices)
            for output, original in enumerate(params["permutation"]):
                mapped[original] = indices[output]
            body = self._read(values[0], tuple(mapped))
        elif primitive is lax.squeeze_p and set(params) == {"dimensions"}:
            remaining = iter(indices)
            body = self._read(
                values[0],
                tuple(
                    Index("zero") if i in params["dimensions"] else next(remaining)
                    for i in range(len(values[0].shape))
                ),
            )
        elif (
            primitive is lax.reshape_p
            and set(params) <= {"new_sizes", "dimensions", "sharding"}
            and params.get("sharding") is None
        ):
            source = values[0]
            permutation = params.get("dimensions") or tuple(range(len(source.shape)))
            old = tuple(source.shape[i] for i in permutation)
            if prod(old) != prod(aval.shape):
                return None
            linear = add(
                *(scale(index, prod(aval.shape[i + 1 :])) for i, index in enumerate(indices))
            )
            mapped = [None] * len(old)
            for axis, size in enumerate(old):
                coordinate = divide(linear, max(1, prod(old[axis + 1 :])))
                mapped[permutation[axis]] = divide(coordinate, max(1, size), modulo=True)
            body = self._read(source, tuple(mapped))
        elif primitive is lax.slice_p and set(params) == {
            "start_indices",
            "limit_indices",
            "strides",
        }:
            strides = params["strides"] or (1,) * len(indices)
            body = self._read(
                values[0],
                tuple(
                    add(integer(start), scale(index, stride))
                    for index, start, stride in zip(
                        indices, params["start_indices"], strides, strict=True
                    )
                ),
            )
        elif primitive is lax.reduce_sum_p and set(params) <= {"axes", "out_sharding"}:
            axes = set(params["axes"])
            remaining = iter(indices)
            bindings, mapped = [], []
            for i, extent in enumerate(values[0].shape):
                if i in axes:
                    index = self._new_index()
                    bindings.append((index, extent))
                    self.index_domains[index] = (0, extent)
                    mapped.append(index)
                else:
                    mapped.append(next(remaining))
            body = self._sum(self._read(values[0], tuple(mapped)), bindings)
        elif (
            primitive is lax.dot_general_p
            and set(params)
            == {"dimension_numbers", "precision", "preferred_element_type", "out_sharding"}
            and params["precision"] is None
            and params["preferred_element_type"] in (None, aval.dtype)
        ):
            (lc, rc), (lb, rb) = params["dimension_numbers"]
            left, right = [None] * len(values[0].shape), [None] * len(values[1].shape)
            output = iter(indices)
            for i, j in zip(lb, rb, strict=True):
                left[i] = right[j] = next(output)
            for mapping in (left, right):
                contracted = lc if mapping is left else rc
                for i in range(len(mapping)):
                    if mapping[i] is None and i not in contracted:
                        mapping[i] = next(output)
            bindings = []
            for i, j in zip(lc, rc, strict=True):
                index = self._new_index()
                left[i] = right[j] = index
                bindings.append((index, values[0].shape[i]))
                self.index_domains[index] = (0, values[0].shape[i])
            args = (self._read(values[0], tuple(left)), self._read(values[1], tuple(right)))
            body = self._sum(Expr("mul", args=args, total=total), bindings)
        else:
            return None
        self.rules.add(str(primitive))
        return self._tensor(aval, body)

    def _opaque(self, eq, values, params):
        """保留纯算子及其参数，不通过名称接纳未知原语。"""
        # 能证明在有限实数上全定义的少量调用，允许安全删除其无用计算。
        total_primitives = {
            lax.sin_p,
            lax.cos_p,
            lax.tanh_p,
            lax.exp_p,
            lax.abs_p,
            lax.max_p,
            lax.min_p,
            lax.eq_p,
            lax.ne_p,
            lax.lt_p,
            lax.le_p,
            lax.gt_p,
            lax.ge_p,
            lax.select_n_p,
            lax.stop_gradient_p,
            lax.slice_p,
            lax.broadcast_in_dim_p,
            lax.reshape_p,
            lax.transpose_p,
            lax.squeeze_p,
            empty2_p,
        }
        total = eq.primitive in total_primitives and all(v.body.total for v in values)
        if eq.primitive is lax.tanh_p and not all(
            jax.dtypes.issubdtype(value.dtype, np.floating) for value in values
        ):
            # Complex tanh has poles, even though all input components are finite.
            total = False
        header = (
            "apply",
            eq.primitive,
            encode(params),
            encode(eq.ctx),
            tuple(v.signature for v in values),
            tuple(encode(v.aval) for v in eq.outvars),
        )
        return tuple(
            self._tensor(
                v.aval,
                Expr(
                    "atom",
                    ((header, port), self._axes(v.aval)),
                    tuple(a.body for a in values),
                    total,
                ),
            )
            for port, v in enumerate(eq.outvars)
        )

    def graph(self, graph, bindings=None):
        """转换全部节点并检查无用分支；绑定由受支持的区域适配器提供。"""
        if graph.effects or any(eq.effects or _ordered_equation(eq) for eq in graph.eqns):
            raise UnsupportedJaxprError("algebra.effects", "ordered or effectful algebra region")
        if bindings is None:
            bindings = tuple(
                self._input(v.aval, ("argument", i)) for i, v in enumerate(graph.invars)
            )
        from .algebra_finiteness import analyze
        from .algebra_jit import default_jit, transparent_jit
        from .algebra_padding import lower_padding
        from .algebra_piecewise import lower_piecewise
        from .algebra_scalar import lower_scalar
        from .algebra_scan import lower_scan

        environment = dict(zip(graph.invars, bindings, strict=True))
        environment.update(
            (v, self._constant(c, v.aval))
            for v, c in zip(graph.constvars, graph.consts, strict=True)
        )
        finiteness = analyze(self, graph, bindings)

        def record(eq, results, possible=False):
            return tuple(
                self._record_finiteness(value, finiteness[var].finite, possible=possible)
                for var, value in zip(eq.outvars, results, strict=True)
            )

        barriers = []
        inlined_jits = set()
        for eq in graph.eqns:
            self.budget.spend(path="algebra.lower")
            values = tuple(
                self._constant(v.val, v.aval) if isinstance(v, core.Literal) else environment[v]
                for v in eq.invars
            )
            params = semantic_parameters(eq, scope="primal", parent=graph)
            nonfinite = any(v.nonfinite for v in values) or self._nonfinite_parameters(params)
            if (
                self.common_calls is not None
                and default_jit(eq, params)
                and eq in self.common_calls.invocations
            ):
                # 已复核相同的完整子图可以用函数合同复用；实参和全部返回口仍保留。
                key, inputs, outputs = self.common_calls.invocations[eq]
                nonfinite |= self._nonfinite_graph(params["jaxpr"])
                header = ("common_jit", key)
                arguments = tuple(values[i].body for i in inputs)
                results = [None] * len(outputs)
                for port, mapped in enumerate(outputs):
                    var = eq.outvars[mapped]
                    self.read_shapes[(header, port)] = tuple(var.aval.shape)
                    results[mapped] = self._mark_nonfinite(
                        self._tensor(
                            var.aval,
                            Expr("atom", ((header, port), self._axes(var.aval)), arguments, True),
                        ),
                        nonfinite,
                    )
                barriers.append(Expr("atom", ((header, "invocation"), ()), arguments, False))
                results = record(eq, results, nonfinite)
                environment.update(zip(eq.outvars, results, strict=True))
                self.rules.add("common_jit")
                continue
            if transparent_jit(eq, values, params):
                # 组合前向与反向区域后，私有残差形状不再成为外部数学接口。
                # 子图完整转换，内部循环边界和部分函数检查不能随 JIT 一起消失。
                child = self.graph(params["jaxpr"], values)
                barriers.extend(child.barriers)
                environment.update(zip(eq.outvars, record(eq, child.outputs), strict=True))
                inlined_jits.add(eq)
                self.rules.add("transparent_jit")
                continue
            converted = self._arithmetic(eq, values)
            adapted = None
            if converted is None:
                for adapter in (lower_piecewise, lower_scalar, lower_padding, lower_scan):
                    if adapter in (lower_padding, lower_scan) and any(
                        value.nonfinite for value in values
                    ):
                        continue
                    adapted = adapter(self, eq, values, params)
                    if adapted is not None:
                        break
            if converted is not None:
                results = (converted,)
            elif adapted is not None:
                results = adapted
            elif eq.primitive in (
                pjit.jit_p,
                lax.cond_p,
                lax.scan_p,
                lax.while_p,
                *CUSTOM_PRIMITIVES,
            ):
                results, barrier = self._region(eq, values, params)
                barriers.append(barrier)
            else:
                results = self._opaque(eq, values, params)
            results = record(eq, results, nonfinite)
            environment.update(zip(eq.outvars, results, strict=True))
        outputs = tuple(
            self._constant(v.val, v.aval) if isinstance(v, core.Literal) else environment[v]
            for v in graph.outvars
        )
        used = set(v for v in graph.outvars if isinstance(v, core.Var))
        for eq in reversed(graph.eqns):
            if any(v in used for v in eq.outvars):
                used.update(v for v in eq.invars if isinstance(v, core.Var))
            elif any(not environment[v].body.total for v in eq.outvars):
                # 调用/循环已作为独立边界保留；其他部分函数不凭输出不可达而删除。
                if eq in inlined_jits or eq.primitive not in (
                    pjit.jit_p,
                    lax.cond_p,
                    lax.scan_p,
                    lax.while_p,
                    *CUSTOM_PRIMITIVES,
                ):
                    raise UnsupportedJaxprError(
                        "algebra.dead", "cannot ignore partial unused computation"
                    )
        signature = (
            tuple(encode(v.aval) for v in graph.invars),
            tuple(v.signature for v in outputs),
            graph.is_high,
        )
        return Lowered(signature, outputs, tuple(barriers))

    def _region(self, eq, values, params):
        """在对应控制结构内部转换；反馈形参由角色和嵌套深度锚定。"""
        depth = self.region_depth
        self.region_depth += 1
        try:
            parameters = dict(params)
            arguments = []
            from .correspondence import _parameter_view

            _, children = _parameter_view(
                eq, lambda _: None, parameters=params, derivative_scope="primal"
            )
            nonfinite = (
                any(v.nonfinite for v in values)
                or self._nonfinite_parameters(params)
                or any(self._nonfinite_graph(child) for child in children.values())
            )
            if eq.primitive is pjit.jit_p:
                child = parameters.pop("jaxpr")
                parameters.pop("name")  # JAX 已审查的诊断名称。
                names = ("in_shardings", "in_layouts", "donated_invars")
                descriptors = tuple(parameters.pop(name) for name in names)
                bound = tuple(
                    self._port(value, tuple(items[i] for items in descriptors))
                    for i, value in enumerate(values)
                )
                arguments.extend((self._pack(self.graph(child, bound)), self._bag(bound)))
            elif eq.primitive in CUSTOM_PRIMITIVES:
                child = parameters.pop("call_jaxpr")
                arguments.extend((self._pack(self.graph(child, values)), self._bag(values)))
            elif eq.primitive is lax.cond_p:
                branches = parameters.pop("branches")
                arguments.append(self._port(values[0], ("selector",)).body)
                for branch in branches:
                    arguments.append(self._pack(self.graph(branch, values[1:])))
                arguments.append(self._bag(values[1:]))
            elif eq.primitive is lax.while_p:
                condition, body = parameters.pop("cond_jaxpr"), parameters.pop("body_jaxpr")
                nc, nb = parameters.pop("cond_nconsts"), parameters.pop("body_nconsts")
                initial = values[nc + nb :]
                carry = tuple(
                    self._input(v.aval, ("while_carry", depth, i), nonfinite=nonfinite)
                    for i, v in enumerate(body.invars[nb:])
                )
                arguments.extend(
                    (
                        self._pack(self.graph(condition, (*values[:nc], *carry))),
                        self._pack(self.graph(body, (*values[nc : nc + nb], *carry))),
                        self._bag(values[:nc]),
                        self._bag(values[nc : nc + nb]),
                    )
                )
                arguments.extend(self._port(v, ("initial", i)).body for i, v in enumerate(initial))
            else:
                body = parameters.pop("jaxpr")
                ft_in = parameters.pop("ft_in")
                constants, carried, _ = ft_in.unpack()
                nc, nk = len(constants), len(carried)
                parameters["input_tree"] = ft_in.update(None for _ in ft_in.vals)
                bound = tuple(
                    self._port(v, ("scan_input", ft_in.vals[i])) for i, v in enumerate(values)
                )
                carry = tuple(
                    self._input(v.aval, ("scan_carry", depth, i), nonfinite=nonfinite)
                    for i, v in enumerate(body.invars[nc : nc + nk])
                )
                slices = tuple(
                    self._record_finiteness(
                        self._tensor(
                            var.aval,
                            Expr(
                                "atom",
                                (
                                    ("scan_slice", value.signature, encode(var.aval)),
                                    self._axes(var.aval),
                                ),
                                (value.body,),
                                value.body.total,
                            ),
                        ),
                        self.finite_values.get(value, False),
                        possible=value.nonfinite,
                    )
                    for var, value in zip(body.invars[nc + nk :], bound[nc + nk :], strict=True)
                )
                arguments.extend(
                    (
                        self._pack(self.graph(body, (*bound[:nc], *carry, *slices))),
                        self._bag(bound[:nc]),
                        self._bag(bound[nc + nk :]),
                    )
                )
                arguments.extend(
                    self._port(v, ("initial", i)).body for i, v in enumerate(bound[nc : nc + nk])
                )
            header = (
                "region",
                eq.primitive,
                encode(parameters),
                encode(eq.ctx),
                tuple(encode(v.aval) for v in eq.outvars),
            )
            barrier = Expr("atom", ((header, "boundary"), ()), tuple(arguments), False)
            outputs = tuple(
                self._tensor(
                    v.aval,
                    Expr("atom", ((header, port), self._axes(v.aval)), tuple(arguments), False),
                )
                for port, v in enumerate(eq.outvars)
            )
            self.rules.add(f"region:{eq.primitive}")
            return tuple(self._mark_nonfinite(v, nonfinite) for v in outputs), barrier
        finally:
            self.region_depth = depth

    def _port(self, value, metadata):
        """把逐端口语义绑定到实际值，使私有接口置换不会错配元数据。"""
        indices = tuple(
            Index("zero") if n == 1 else Index("free", i) for i, n in enumerate(value.shape)
        )
        body = Expr(
            "atom",
            (("port", value.signature, encode(metadata)), indices),
            (value.body,),
            value.body.total,
        )
        return self._record_finiteness(
            Tensor(value.signature, value.shape, value.dtype, body, value.nonfinite),
            self.finite_values.get(value, False),
        )

    def _bag(self, values):
        """私有捕获的集合保留每次绑定及其重数，不依赖捕获顺序。"""
        return Expr(
            "bag",
            (("bindings",), ()),
            tuple(self._port(v, ()).body for v in values),
            all(v.body.total for v in values),
        )

    def _pack(self, lowered):
        """打包子区域的有序输出与区域根；输入绑定已经代入各次读取。"""
        return Expr("atom", (("region_body", lowered.signature[1:]), ()), lowered.roots, False)
