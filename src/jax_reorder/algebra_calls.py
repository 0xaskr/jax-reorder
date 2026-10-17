"""完整结构相同的纯 JIT 用同一函数符号表示，避免展开共同非线性区域。

两侧原图各自包含一个完整结构相同、执行契约默认且无非有限值的纯 JIT 时，
把调用内部展开成符号公式既昂贵又不利于合并。本模块用规范结构编码给这类
子图建立精确身份，让数学层把它表示成同一个不透明函数符号；身份来自图内容
和端口双射，不使用函数名、repr 或哈希摘要。
不变量：只有两侧共有（键出现在各自根图成员集合中）的调用才共享符号；
有序/effectful 图与非默认 JIT 契约不参与共享，含非有限常量的调用由
冻结阶段（algebra_transport）拒绝共享并保留完整展开边界。
"""

from types import SimpleNamespace

from jax._src import core, pjit

from .algebra_jit import default_jit
from .canonical import _ordered_equation
from .correspondence import _parameter_view
from .derivatives import semantic_parameters
from .errors import UnsupportedJaxprError
from .semantics import Encoder


class CommonCalls:
    """从两侧原图证据中的实际子图构造精确键，不使用函数名或摘要判等。

    pairs 是 (原图, 适配图, 原图到适配图的 GraphCorrespondence) 三元组。
    members 记录每张图（含嵌套调用）包含哪些调用的键；两个根图的交集
    shared 才是两侧完整相同的调用集合。invocations 以适配图中的方程对象
    为键，保存该调用的符号键及原图端口到适配端口的双射，供冻结阶段决定
    是否写成共同调用符号。cache 保证同一子图只编码一次。
    """

    def __init__(self, pairs, budget):
        """登记两侧根图，计算共有调用集合，并沿区域证据收集可替换的调用点。"""
        self.budget = budget
        self.cache = {}
        self.members = {}
        # 先为两侧根图计算成员集合，才能求交集识别“两侧共有”的调用。
        for original, _, _ in pairs:
            self.key(original)
        self.shared = self.members[pairs[0][0]] & self.members[pairs[1][0]]
        self.invocations = {}
        # 用显式栈沿区域证据遍历嵌套子图；三元组去重防止共享子图重复展开。
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
                        # 直接调用原函数符号；输入和输出映射必须成对应用。
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
        """按当前图的有序端口与实际定义位置编码全部节点及无根计算。

        编码包含常量内容、输入抽象值、每个方程（primitive、操作数引用、
        输出抽象值、语义参数、上下文）、返回值引用和 is_high 标记；常量按
        绑定位置编号，节点按方程位置编号，因此不依赖原始变量名。
        有序或 effectful 的图无法在不展开的情况下共享符号，直接拒绝。
        嵌套子图以自身键作为占位继续参与编码，并递归累计到 members。
        """
        if graph in self.cache:
            return self.cache[graph]
        if graph.effects or any(eq.effects or _ordered_equation(eq) for eq in graph.eqns):
            raise UnsupportedJaxprError("algebra.effects", "ordered or effectful common call")
        variables = {v: ("constant", i) for i, v in enumerate(graph.constvars)}
        variables.update((v, ("input", i)) for i, v in enumerate(graph.invars))
        calls = set()

        def nested(child, path):
            """把嵌套子图编成自身键，并把它的调用成员并入当前图的成员集合。"""
            # Encoder 约定嵌套子图返回带 .graph 属性的对象；这里用子图键代替
            # 规范化结果，使编码只看数学内容，不复制源图。
            key = self.key(child)
            calls.update(self.members[child])
            return SimpleNamespace(graph=key)

        encoder = Encoder(nested, variables.__getitem__)

        def reference(var):
            """把操作数编码成（字面量或已登记变量的）位置引用。"""
            # 字面量不占输入/常量槽位，按声明抽象值和内容单独编码；
            # 其余变量必须已在当前图的输入、常量或前面方程的输出中登记。
            if isinstance(var, core.Literal):
                return (
                    "literal",
                    encoder.encode(var.aval, "algebra.call.aval"),
                    encoder.literal(var.val, var.aval, "algebra.call.literal"),
                )
            return variables[var]

        def encoded(value):
            """用当前图的变量解析规则编码任意参数或抽象值。"""
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
                # jit 的 name 只是调试标签，不改变数学语义；不同名字的同一
                # 子图应能共享符号，因此从键中移除。
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
