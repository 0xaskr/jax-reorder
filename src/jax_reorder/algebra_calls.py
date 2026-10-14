"""完整结构相同的纯 JIT 用同一函数符号表示，避免展开共同非线性区域。"""

from types import SimpleNamespace

from jax._src import core, pjit

from .algebra_jit import default_jit
from .canonical import _ordered_equation
from .correspondence import _parameter_view
from .derivatives import semantic_parameters
from .errors import UnsupportedJaxprError
from .semantics import Encoder


class CommonCalls:
    """从两侧原图证据中的实际子图构造精确键，不使用函数名或摘要判等。"""

    def __init__(self, pairs, budget):
        self.budget = budget
        self.cache = {}
        self.members = {}
        for original, _, _ in pairs:
            self.key(original)
        self.shared = self.members[pairs[0][0]] & self.members[pairs[1][0]]
        self.invocations = {}
        pending, seen = list(pairs), set()
        while pending:
            original, adapted, mapping = pending.pop()
            if (original, adapted, mapping) in seen:
                continue
            seen.add((original, adapted, mapping))
            for eq, eqmap in zip(original.eqns, mapping.equations, strict=True):
                other = adapted.eqns[eqmap.right]
                children = dict(eqmap.children)
                if default_jit(eq, eq.params) and default_jit(other, other.params):
                    key = self.key(eq.params["jaxpr"])
                    if key in self.shared:
                        # 结构证据给出原始端口到适配端口的双射，不能用新槽位
                        # 直接调用旧函数符号。校验器会独立核对这些映射。
                        self.invocations[other] = (
                            key,
                            children["params.jaxpr"].inputs,
                            eqmap.outputs,
                        )
                if children:
                    _, left_children = _parameter_view(
                        eq, lambda _: None, derivative_scope="primal", parent=original
                    )
                    _, right_children = _parameter_view(
                        other, lambda _: None, derivative_scope="primal", parent=adapted
                    )
                    pending.extend(
                        (left_children[path], right_children[path], witness)
                        for path, witness in children.items()
                    )

    def key(self, graph):
        """按当前图的有序端口与实际定义位置编码全部节点及无根计算。"""
        if graph in self.cache:
            return self.cache[graph]
        if graph.effects or any(eq.effects or _ordered_equation(eq) for eq in graph.eqns):
            raise UnsupportedJaxprError("algebra.effects", "ordered or effectful common call")
        variables = {v: ("constant", i) for i, v in enumerate(graph.constvars)}
        variables.update((v, ("input", i)) for i, v in enumerate(graph.invars))
        calls = set()

        def nested(child, path):
            key = self.key(child)
            calls.update(self.members[child])
            return SimpleNamespace(graph=key)

        encoder = Encoder(nested, variables.__getitem__)

        def reference(var):
            if isinstance(var, core.Literal):
                return (
                    "literal",
                    encoder.encode(var.aval, "algebra.call.aval"),
                    encoder.literal(var.val, var.aval, "algebra.call.literal"),
                )
            return variables[var]

        def encoded(value):
            return encoder.encode(value, "algebra.call")

        equations = []
        for i, eq in enumerate(graph.eqns):
            self.budget.spend(path="algebra.call_graph")
            inputs = tuple(reference(v) for v in eq.invars)
            variables.update((v, ("node", i, j)) for j, v in enumerate(eq.outvars))
            params = dict(semantic_parameters(eq, scope="primal", parent=graph))
            if default_jit(eq, params):
                calls.add(self.key(params["jaxpr"]))
            if eq.primitive is pjit.jit_p:
                params.pop("name")
            equations.append(
                (
                    eq.primitive,
                    inputs,
                    tuple(encoded(v.aval) for v in eq.outvars),
                    encoded(params),
                    encoded(eq.ctx),
                )
            )
        key = (
            tuple(
                (encoded(v.aval), encoded(c))
                for v, c in zip(graph.constvars, graph.consts, strict=True)
            ),
            tuple(encoded(v.aval) for v in graph.invars),
            tuple(equations),
            tuple((reference(v), encoded(v.aval)) for v in graph.outvars),
            graph.is_high,
        )
        self.cache[graph] = key
        self.members[graph] = calls
        return key
