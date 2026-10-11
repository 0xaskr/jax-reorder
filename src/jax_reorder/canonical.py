"""从输出反向 BFS 编号，按依赖深度重排，再精确比较计算图。

本模块维护两种顺序：规范结构中的节点按 BFS 编号存放，可执行 jaxpr
中的方程按（依赖深度，BFS 编号）存放。BFS 访问顺序不能直接作为执行顺序。
保留普通运算的操作数位置、结果端口和共享关系，不应用交换律或结合律。
cond 的内部实参和所有分支形参允许同步置换，外部函数输入顺序保持不变。
"""

import dataclasses
from collections import deque
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import jax
from jax import lax
from jax._src import config, core, pjit, shard_map
from jax._src.lax.lax import NoDCEEffect

from .errors import InvalidJaxprError, JaxprError, UnsupportedJaxprError
from .semantics import Encoder, Frozen, replace_nested
from .support import supported_primitive_identities


def _ordered_equation(equation):
    """判断方程及其嵌套子图是否包含需要保留相对顺序的操作。

    例如 jit 内部调用 rng_uniform，外层 jit 的 effects 仍可能为空。
    必须沿参数中的子图继续检查，否则外层重排可能交换两次有序调用。
    此处只识别顺序约束；子图是否合法由后续规范化流程单独检查。
    """
    pending = [equation]
    # 子图可以被多个参数共享；按对象身份去重，也避免畸形参数环导致死循环。
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, core.JaxprEqn):
            if value.primitive.name in {"rng_uniform", "dce_sink"}:
                return True
            pending.extend(value.params.values())
        elif isinstance(value, core.Jaxpr):
            pending.extend(value.eqns)
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return False


class Status(StrEnum):
    """本轮图比较结果；DIFFERENT 不表示数学公式已经被证明不等价。"""

    EQUIVALENT = "equivalent"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Issue:
    """阻止完整比较的问题：path 定位图内位置，reason 说明原因。"""

    path: str
    reason: str


@dataclass(frozen=True)
class Node:
    """一个方程的规范表示；相同算子的两次调用仍是两个独立节点。

    inputs 是按操作数位置排列的值引用（cond 内部绑定先同步规范化），
    outputs 是按结果端口排列的抽象值。
    params、effects、context 保存编码后的语义信息；order_after 保存
    张量使用关系之外的顺序前驱，其编号使用 BFS 编号而非原方程索引。
    """

    primitive: core.Primitive
    inputs: tuple
    outputs: tuple
    params: tuple
    effects: Frozen
    context: Frozen
    order_after: tuple[int, ...] = ()


@dataclass(frozen=True)
class Graph:
    """用于精确比较的图，不保存原始变量名或调试源位置。

    inputs 保存有序输入的抽象值，outputs 保存有序输出的值引用。
    constants 按首次使用顺序记录（抽象值，常量内容），nodes 按规范编号存放。
    未使用部分也会保留，但包含它们的结果不能宣称已获得完整规范形式。
    """

    inputs: tuple
    constants: tuple
    outputs: tuple
    nodes: tuple[Node, ...]
    effects: Frozen
    is_high: bool


@dataclass(frozen=True)
class Canonicalized:
    """规范化结果，同时提供可执行图和用于检查的结构信息。

    bfs_order[k] 是 BFS 编号 k 对应的原方程索引，只包含回溯覆盖的方程。
    equation_order[k] 是新方程列表第 k 项对应的原方程索引，包含保留的未使用部分。
    depths[i] 是原方程 i 的依赖深度；unused_equations / unused_constants
    也使用原图索引。value_references 将原 Var 映射到规范值引用。

    jaxpr 和变量映射只用于执行与诊断，不参与结果对象的结构相等比较。
    """

    jaxpr: core.Jaxpr = field(compare=False, repr=False)
    graph: Graph
    bfs_order: tuple[int, ...]
    equation_order: tuple[int, ...]
    depths: tuple[int, ...]
    unused_equations: tuple[int, ...]
    unused_constants: tuple[int, ...]
    issues: tuple[Issue, ...]
    value_references: Mapping[core.Var, tuple] = field(compare=False, repr=False)

    @property
    def complete(self) -> bool:
        """是否完成全图规范化；成功生成可执行图不一定意味着完整规范化。"""
        return not self.issues


@dataclass(frozen=True)
class Comparison:
    """图比较报告，包含状态、差异路径和两侧已完成的规范化结果。

    unknown 时查看 issues；different 时查看 difference。
    若一侧直接规范化失败，该侧结果为 None；部分完成时仍保留结果供诊断。
    """

    status: Status
    issues: tuple[Issue, ...] = ()
    difference: str | None = None
    left: Canonicalized | None = field(default=None, compare=False, repr=False)
    right: Canonicalized | None = field(default=None, compare=False, repr=False)


def _validate(graph: core.Jaxpr, path: str) -> None:
    """先按原方程顺序检查单图的 SSA 定义与使用关系。

    每个变量只能定义一次，且必须先定义后使用。字面量是直接值，无需绑定。
    此处不递归调用 JAX 类型检查，以便先由规范化器发现嵌套图中的环。
    """
    if not isinstance(graph, core.Jaxpr):
        raise InvalidJaxprError(path, "expected a jaxpr including its bound constants")
    defined: set[core.Var] = set()

    def bind(v, where):
        """登记一个定义；重复定义会破坏生产者索引的一一对应关系。"""
        if not isinstance(v, core.Var) or v in defined:
            raise InvalidJaxprError(where, "invalid binder or duplicate definition")
        defined.add(v)

    def use(v, where):
        """核对引用；前向引用和依赖环都不属于本接口接受的合法 jaxpr。"""
        if not isinstance(v, core.Literal) and (not isinstance(v, core.Var) or v not in defined):
            raise InvalidJaxprError(where, "undefined value or dependency cycle")

    if len(graph.constvars) != len(graph.consts):
        raise InvalidJaxprError(path, "constant binders and values must have equal lengths")
    for i, v in enumerate((*graph.constvars, *graph.invars)):
        bind(v, f"{path}.binders[{i}]")
    for i, eq in enumerate(graph.eqns):
        for v in eq.invars:
            use(v, f"{path}.eqns[{i}].inputs")
        for v in eq.outvars:
            bind(v, f"{path}.eqns[{i}].outputs")
    for v in graph.outvars:
        use(v, f"{path}.outputs")


class _Normalizer:
    """一次规范化调用的内部状态，供同一次调用中的所有嵌套子图共享。"""

    def __init__(self):
        self.primitives = supported_primitive_identities()
        # cache 复用共享子图的处理结果；active 仅记录当前递归路径，用于检测环。
        # 缓存不跨公开调用复用，避免再次调用时沿用已经改变的常量或图内容。
        # 同步置换会创建临时分支图。缓存持有图对象，避免释放后 id 被其他图复用。
        self.cache: dict[core.Jaxpr, Canonicalized] = {}
        self.active: set[core.Jaxpr] = set()

    def normalize(self, graph: core.Jaxpr, path: str) -> Canonicalized:
        """处理一个区域，复用共享子图，但拒绝递归引用自身的嵌套图。"""
        if not isinstance(graph, core.Jaxpr):
            raise InvalidJaxprError(path, "expected a jaxpr including its bound constants")
        if graph in self.active:
            raise InvalidJaxprError(path, "cyclic nested jaxpr")
        if graph in self.cache:
            return self.cache[graph]
        self.active.add(graph)
        try:
            result = self._normalize(graph, path)
            self.cache[graph] = result
            # 参数编码引用规范结构，参数替换引用新 jaxpr；两种图对象指向同一结果。
            self.cache[result.jaxpr] = result
            return result
        finally:
            self.active.remove(graph)

    def _cond_bindings(self, eq, path, issues):
        """按分支中的实际使用顺序，同步置换 cond 实参与全部分支形参。

        置换只改变内部槽位名称，变量引用不变。不能只排序实参，也不能各分支
        独立排序，否则可能把原本传错参数的计算误判为等价。
        """
        unknown = eq.params.keys() - {"branches", "branches_platforms"}
        if unknown:
            raise UnsupportedJaxprError(path, f"unsupported cond parameters {sorted(unknown)}")
        branches = eq.params.get("branches")
        if (
            type(branches) is not tuple
            or not branches
            or any(not isinstance(branch, core.Jaxpr) for branch in branches)
            or not eq.invars
            or any(len(branch.invars) != len(eq.invars) - 1 for branch in branches)
        ):
            raise InvalidJaxprError(path, "invalid cond branches or operand count")

        # 子分支先处理，保证嵌套 cond 的捕获顺序不会影响本层参数首次使用位置。
        normalized = [
            self.normalize(branch, f"{path}.params.branches[{i}]")
            for i, branch in enumerate(branches)
        ]
        slots: dict[int, None] = {}

        def discover(ref):
            if ref[0] == "input":
                slots.setdefault(ref[1], None)

        for result in normalized:
            for ref in result.graph.outputs:
                discover(ref)
            # nodes 已按 BFS 存放；输入引用的编号仍对应该分支原来的形参槽位。
            for node in result.graph.nodes:
                for ref in node.inputs:
                    discover(ref)
        unused = [i for i in range(len(eq.invars) - 1) if i not in slots]
        if len(unused) > 1:
            issues.append(Issue(path, "unused cond operands retain their original relative order"))
        order = (*slots, *unused)
        rebound = []
        for result in normalized:
            branch = result.jaxpr
            if order != tuple(range(len(branch.invars))):
                debug = branch.debug_info
                if debug.arg_names is not None:
                    names = debug.safe_arg_names(len(branch.invars))
                    debug = debug._replace(arg_names=tuple(names[i] for i in order))
                branch = branch.replace(invars=[branch.invars[i] for i in order], debug_info=debug)
            rebound.append(branch)
        return eq.replace(
            invars=[eq.invars[0], *(eq.invars[i + 1] for i in order)],
            params={**eq.params, "branches": tuple(rebound)},
        )

    def _params(self, eq, eq_path, encoder):
        """编码全部参数；仅对已核对语义的特定参数采用专门适配。"""
        encoded_params = []
        # 参数按名称取序，消除字典插入顺序差异；这不是按算子名称排列计算节点。
        for name, value in sorted(eq.params.items()):
            # jit 的函数名只用于诊断；比较键中排除它，返回的方程仍保留原参数。
            if eq.primitive is pjit.jit_p and name == "name":
                continue
            if (
                jax.__version__ == "0.11.2"
                and eq.primitive is lax.reduce_p
                and name == "computation"
                and isinstance(eq.params.get("jaxpr"), core.Jaxpr)
                and callable(value)
            ):
                # 仅在此版本中，lowering、JVP 和 batching 使用已追踪的 reducer
                # 子图。比较完整子图而不是 Python 函数身份，返回程序仍保留 callable。
                # 不能把这个例外推广为忽略所有回调，否则可能漏掉实际计算语义。
                encoded = ("represented_by_reducer_jaxpr",)
            else:
                encoded = encoder.encode(value, f"{eq_path}.params.{name}")
            encoded_params.append((name, encoded))
        return encoded_params

    def _normalize(self, graph: core.Jaxpr, path: str) -> Canonicalized:
        """依次建立依赖、BFS 编号、处理子图、计算深度并构造完整比较表示。"""
        _validate(graph, path)
        equations = list(graph.eqns)
        # 1. 原 Var 只用于查表。多结果方程还必须记录结果端口，不能只记生产者。
        producers = {
            v: (i, port) for i, eq in enumerate(equations) for port, v in enumerate(eq.outvars)
        }
        inputs = {v: i for i, v in enumerate(graph.invars)}
        constants = {v: i for i, v in enumerate(graph.constvars)}
        # 以下字典保持插入顺序：键是原索引，值是新编号。
        # 因此不需要依赖原变量名，也不会合并两个内容相同的调用或常量绑定。
        constant_ids: dict[int, int] = {}
        identifiers: dict[int, int] = {}
        queue: deque[int] = deque()
        extra_predecessors: dict[int, list[int]] = {}
        effect_roots: list[int] = []
        previous_ordered: int | None = None
        issues: list[Issue] = []

        # named-axis effect 表示所需的轴环境，NoDCEEffect 对应保留计算的标记。
        # 这里只接受已适配的这两类；未知 effect 即使出现在未使用分支也要拒绝。
        def check_effects(effects, where):
            for effect in effects:
                if not isinstance(effect, (core.NamedAxisEffect, NoDCEEffect)):
                    raise UnsupportedJaxprError(
                        where, f"unsupported effect {type(effect).__qualname__}"
                    )

        check_effects(graph.effects, f"{path}.effects")
        for i, eq in enumerate(equations):
            if eq.primitive not in self.primitives:
                raise UnsupportedJaxprError(
                    f"{path}.eqns[{i}]", f"unsupported primitive {eq.primitive.name}"
                )
            check_effects(eq.effects, f"{path}.eqns[{i}].effects")
            # rng_uniform、dce_sink 及包含它们的嵌套调用不能视作可任意换序的纯节点。
            # 给相邻的有序节点增加前驱边，同时作为额外根，避免输出回溯漏掉它们。
            if _ordered_equation(eq):
                effect_roots.append(i)
                if previous_ordered is not None:
                    extra_predecessors[i] = [previous_ordered]
                previous_ordered = i
            if eq.primitive is lax.cond_p:
                # 必须先改变父图看到的输入顺序，再编号其生产者；末尾只改比较键
                # 会留下由原捕获顺序决定的父图 BFS 编号，仍然造成误判。
                equations[i] = self._cond_bindings(eq, f"{path}.eqns[{i}]", issues)

        def visit(index):
            """节点第一次入队时分配编号，使共享前驱只出现一次。"""
            if index not in identifiers:
                identifiers[index] = len(identifiers)
                queue.append(index)

        def discover(atom):
            """发现一个值的来源；函数输入位置固定，常量绑定按首次引用编号。"""
            if isinstance(atom, core.Var):
                if atom in producers:
                    visit(producers[atom][0])
                elif atom in constants:
                    index = constants[atom]
                    constant_ids.setdefault(index, len(constant_ids))

        # 2. 输出位置决定根顺序，操作数位置决定前驱访问顺序；二者都不可排序或去重。
        # 输入/字面量可直接作为输出，不为它们额外制造计算节点。
        for atom in graph.outvars:
            discover(atom)
        for index in effect_roots:
            visit(index)
        while queue:
            index = queue.popleft()
            for atom in equations[index].invars:
                discover(atom)
            for predecessor in extra_predecessors.get(index, ()):
                visit(predecessor)
        bfs_order = tuple(identifiers)
        # 3. BFS 不可达的纯方程继续保留，并明确标记部分完成。
        # 保留原相对顺序且放在可达部分之后，仍满足原 SSA 图的依赖约束。
        unused = tuple(i for i in range(len(equations)) if i not in identifiers)
        if unused:
            issues.append(
                Issue(path, f"{len(unused)} unused equations retain their original relative order")
            )
        for index in unused:
            identifiers[index] = len(identifiers)
            for atom in equations[index].invars:
                if isinstance(atom, core.Var) and atom in constants:
                    old = constants[atom]
                    constant_ids.setdefault(old, len(constant_ids))
        unused_constants = tuple(i for i in range(len(graph.constvars)) if i not in constant_ids)
        if unused_constants:
            issues.append(Issue(path, f"{len(unused_constants)} unused constant bindings"))
        for index in unused_constants:
            constant_ids[index] = len(constant_ids)

        def reference(atom):
            """生成与原 Var 名称无关的值引用。

            引用分为 input、constant、literal 和 result 四类；result 同时包含
            生产者 BFS 编号和结果端口。区分这些类别才能保留绑定方式与共享关系。
            """
            if isinstance(atom, core.Literal):
                return (
                    "literal",
                    encoder.encode(atom.aval, path),
                    encoder.literal(atom.val, atom.aval, path),
                )
            if atom in inputs:
                return ("input", inputs[atom])
            if atom in constants:
                return ("constant", constant_ids[constants[atom]])
            index, port = producers[atom]
            return ("result", identifiers[index], port)

        def nested(child, child_path):
            """在子图边界内独立规范化；子图部分完成会使整个比较也不完整。"""
            result = self.normalize(child, child_path)
            issues.extend(result.issues)
            return result

        encoder = Encoder(nested, reference)
        frozen_params = []
        new_equations = []
        # 4. 编码全部方程，而非仅编码输出可达部分，避免通过忽略未知语义获得匹配。
        for i, eq in enumerate(equations):
            eq_path = f"{path}.eqns[{i}]"
            encoded_params = []
            with ExitStack() as scope:
                # 集合通信子图依赖外层 shard_map 的轴环境及 check_vma 配置。
                # 直接脱离该环境校验子图，会把合法的轴名称误报为未绑定。
                if eq.primitive is shard_map.shard_map_p:
                    scope.enter_context(
                        shard_map._extend_axis_env(
                            eq.params["mesh"], eq.params["newly_manual_axes"]
                        )
                    )
                    scope.enter_context(config._check_vma(eq.params["check_vma"]))
                encoded_params = self._params(eq, eq_path, encoder)
            frozen_params.append(tuple(encoded_params))
            # 编码过程已递归处理子图；此处只把参数中的子图替换成对应的新 jaxpr。
            # 没有需要替换的子图时，复用原方程对象，不修改原输入图。
            params = replace_nested(eq.params, lambda child: self.cache[child].jaxpr)
            new_equations.append(eq if params is eq.params else eq.replace(params=params))

        # 5. 原图已通过定义顺序检查，因此从前向后一次计算即可得到所有前驱深度。
        # 输入与常量深度视作 0，无计算前驱的节点深度为 1；附加顺序边同样计入。
        # 这是从输入出发的最长依赖深度，不是 BFS 到输出的最短距离。
        depths: list[int] = []
        for i, eq in enumerate(equations):
            predecessors = [
                producers[v][0] for v in eq.invars if isinstance(v, core.Var) and v in producers
            ]
            predecessors.extend(extra_predecessors.get(i, ()))
            depths.append(1 + max((depths[p] for p in predecessors), default=0))
        order = tuple(sorted(bfs_order, key=lambda i: (depths[i], identifiers[i]))) + unused
        # 6. 比较表示仍按 BFS 编号存放，让 result 引用可以直接定位节点。
        # 不要将此顺序与上面的可执行方程顺序 order 混用。
        ordered_nodes = []
        for i in identifiers:
            eq = equations[i]
            eq_path = f"{path}.eqns[{i}]"
            ordered_nodes.append(
                Node(
                    eq.primitive,
                    tuple(reference(v) for v in eq.invars),
                    tuple(
                        encoder.encode(v.aval, f"{eq_path}.outputs[{p}]")
                        for p, v in enumerate(eq.outvars)
                    ),
                    frozen_params[i],
                    encoder.encode(eq.effects, eq_path),
                    encoder.encode(eq.ctx, eq_path),
                    tuple(identifiers[p] for p in extra_predecessors.get(i, ())),
                )
            )
        form = Graph(
            tuple(
                encoder.encode(v.aval, f"{path}.inputs[{i}]") for i, v in enumerate(graph.invars)
            ),
            tuple(
                (
                    encoder.encode(graph.constvars[i].aval, path),
                    encoder.encode(graph.consts[i], path),
                )
                for i in constant_ids
            ),
            tuple(reference(v) for v in graph.outvars),
            tuple(ordered_nodes),
            encoder.encode(graph.effects, f"{path}.effects"),
            graph.is_high,
        )
        # 只替换方程列表；输入、输出、常量绑定、调试信息等仍由原 jaxpr 保留。
        reordered = graph.replace(eqns=[new_equations[i] for i in order])
        # 先完成嵌套结构与环检查，再交给 JAX 做最终类型/参数检查，避免 JAX 先递归溢出。
        try:
            core.check_jaxpr(reordered)
        except (core.JaxprTypeError, TypeError, ValueError, NameError) as exc:
            raise InvalidJaxprError(path, f"JAX type check failed: {exc}") from exc
        return Canonicalized(
            reordered,
            form,
            bfs_order,
            order,
            tuple(depths),
            unused,
            unused_constants,
            tuple(dict.fromkeys(issues)),
            MappingProxyType({v: reference(v) for v in (*inputs, *constants, *producers)}),
        )


def canonicalize(jaxpr: core.Jaxpr) -> Canonicalized:
    """规范化一个包含常量绑定的 CPU jaxpr，保留操作及其依赖关系。

    返回 Canonicalized，可分别查看可执行图、规范结构、编号和未完成原因。
    未使用纯分支保留并标记部分完成；非法图抛出 InvalidJaxprError，未知语义
    抛出 UnsupportedJaxprError。本函数不执行 CPU 编译，也不证明数值等价。

    调用方需保持追踪上下文一致，并另行核对展平前的函数输入输出 pytree。
    普通依赖链采用迭代遍历，嵌套区域与参数的递归仍受 Python 深度限制。
    """
    try:
        return _Normalizer().normalize(jaxpr, "jaxpr")
    except RecursionError as exc:
        raise UnsupportedJaxprError(
            "jaxpr", "nested graphs or parameters exceed the Python recursion limit"
        ) from exc


def _difference(left: Any, right: Any) -> str | None:
    """迭代比较完整编码结构，返回首个差异路径；相同则返回 None。

    不比较哈希或打印摘要。栈中倒序加入字段/元素，使取出时按声明顺序检查，
    保证报告位置稳定；编码后的叶子采用其已明确语义的精确相等比较。
    """
    pending = [("graph", left, right)]
    while pending:
        path, a, b = pending.pop()
        if type(a) is not type(b):
            return path
        if dataclasses.is_dataclass(a) and not isinstance(a, type):
            pending.extend(
                (f"{path}.{f.name}", getattr(a, f.name), getattr(b, f.name))
                for f in reversed(dataclasses.fields(a))
            )
        elif isinstance(a, tuple):
            if len(a) != len(b):
                return f"{path}.length"
            pending.extend((f"{path}[{i}]", a[i], b[i]) for i in reversed(range(len(a))))
        elif a != b:
            return path
    return None


def compare_jaxprs(left: core.Jaxpr, right: core.Jaxpr) -> Comparison:
    """比较两个完整计算图，返回 equivalent、different 或 unknown。

    两侧分别规范化，不使用对方的信息决定编号。只有两侧都完成处理后才比较
    完整规范结构；非法图、不支持语义、未使用分支等情况归为带原因的 unknown。
    different 只说明图中存在差异，不推出公式或一般 Python 函数不等价。
    """
    normalized = []
    issues = []
    for side, graph in (("left", left), ("right", right)):
        try:
            result = canonicalize(graph)
            normalized.append(result)
            issues.extend(Issue(f"{side}.{i.path}", i.reason) for i in result.issues)
        except JaxprError as exc:
            normalized.append(None)
            issues.append(Issue(f"{side}.{exc.path}", exc.reason))
    a, b = normalized
    # 一旦有未完成部分，就不能仅根据可达子图相同而返回 equivalent。
    if issues:
        return Comparison(Status.UNKNOWN, tuple(issues), left=a, right=b)
    difference = _difference(a.graph, b.graph)
    return Comparison(
        Status.DIFFERENT if difference else Status.EQUIVALENT,
        difference=difference,
        left=a,
        right=b,
    )
