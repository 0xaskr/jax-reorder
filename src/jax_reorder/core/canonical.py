"""把 jaxpr 规范化成与书写方式无关的结构，再逐字段比较两张计算图。

模块里同时存在两套顺序，用途不同，不能混用：
- 规范结构 Graph.nodes 按从输出出发的 BFS 编号存放，只用于比较；
- 可执行 jaxpr 的方程按“依赖深度、BFS 编号”排序，只用于执行。
所以 BFS 访问顺序不是执行顺序。

比较会保留算子、操作数位置、结果端口、共享关系、常量、抽象值和嵌套结构；
普通运算的操作数不交换，也不做结合律、分配律、常量折叠或 DCE。
jit、cond、shard_map 的私有端口和循环捕获形参允许同步置换，外部输入顺序
保持不变。非法图、未知语义和预算耗尽都直接抛出分类异常。
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import permutations
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import jax
import jaxlib
from jax import lax
from jax._src import config, core, flattree, pjit, shard_map
from jax._src.lax.lax import NoDCEEffect

from ..errors import (
    ComparisonBudgetError,
    ComparisonError,
    ComparisonFailureKind,
    ComparisonInternalError,
    ComparisonStage,
    InvalidJaxprError,
    UnsupportedJaxprError,
)
from ..status import Status
from ._typing import EncodedRegion, Parameters, ValueReference
from .adapters.pallas.structural import (
    captures_child,
    effect_supported,
    emit_pipeline_p,
    pallas_call_p,
    run_scoped_p,
)
from .adapters.pallas.structural import ordered as pallas_ordered
from .correspondence import GraphCorrespondence, build_correspondence, verify_correspondence
from .params import semantic_parameters
from .semantics import Encoder, Frozen, replace_nested
from .support import supported_primitive_identities
from .unrooted import DEFAULT_SEARCH_BUDGET, canonical_order, order_key, reorder_form

type NormalizationTask[T] = Generator[tuple[core.Jaxpr, str, bool], Canonicalized, T]


# 阅读主线：compare_jaxprs → canonicalize → _Normalizer.normalize → _normalize。
# 本模块负责生成规范表示；原图之间的对应关系由 correspondence 模块独立复核。
class _AmbiguousSlots(Exception):
    """把真正无锚点的内部槽位交给整张图的有界选择搜索。"""

    def __init__(self, ports: Iterable[int], path: str) -> None:
        """携带待排列的槽位和位置，由外层搜索展开，不直接作为用户错误报告。"""
        self.ports, self.path = tuple(ports), path


@dataclass(frozen=True, eq=False)
class _UsePath:
    """从外部有序根到一次使用的路径；共享前缀，避免长链复制整个路径元组。"""

    slot: int
    parent: _UsePath | None = None
    # 这里是到有序根的最短回溯距离，与保证可执行性的最长依赖深度不同。
    depth: int = 0

    def __lt__(self, other: _UsePath) -> bool:
        """先比较到有序根的距离，再比较同距离路径上从根开始的端口序列。"""
        if self.depth != other.depth:
            return self.depth < other.depth
        a, b = self, other
        result = False
        # 同深度时按根到使用点的端口序列比较，不按算子名或原始方程索引。
        # 链表从使用点指向根；持续覆盖 result，让更靠近根的首个差异决定结果。
        while a is not b:
            # 同深度路径同时到根；进入循环时两侧都有当前节点。
            a, b = cast(_UsePath, a), cast(_UsePath, b)
            if a.slot != b.slot:
                result = a.slot < b.slot
            a, b = a.parent, b.parent
        return result


def ordered_equation(equation: core.JaxprEqn) -> bool:
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
            if value.primitive.name in {"rng_uniform", "dce_sink"} or pallas_ordered(value):
                return True
            pending.extend(value.params.values())
        elif isinstance(value, core.Jaxpr):
            pending.extend(value.eqns)
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return False


@dataclass(frozen=True)
class Node:
    """一个方程的规范表示；相同算子的两次调用仍是两个独立节点。

    inputs 是按操作数位置排列的值引用（已适配的内部接口先同步规范化），
    outputs 是按结果端口排列的抽象值。
    params、effects、context 保存编码后的语义信息；order_after 保存
    张量使用关系之外的顺序前驱，其编号使用 BFS 编号而非原方程索引。
    """

    primitive: core.Primitive
    # 例如 ("input", 0) 引用函数第一个输入，("result", 2, 1) 引用节点 2 的第二个结果。
    # 操作数所在位置和引用中的结果端口分别表达“怎么用”和“从哪里来”。
    inputs: tuple[ValueReference, ...]
    outputs: tuple[Frozen, ...]
    # 参数按名称排序；effects/context 则保存编码器产生的完整语义结构。
    params: tuple[tuple[str, Frozen], ...]
    effects: Frozen
    context: Frozen
    order_after: tuple[int, ...] = ()


@dataclass(frozen=True)
class Graph:
    """用于精确比较的图，不保存原始变量名或调试源位置。

    inputs 保存有序输入的抽象值，outputs 保存有序输出的值引用。
    constants 记录（抽象值，常量内容），其可达前缀按首次使用顺序编号；
    nodes 的可达前缀按 BFS 编号，剩余常量和节点由完整图搜索确定顺序。
    未使用部分也参与精确搜索；不能把它们当作可忽略的尾部。
    """

    inputs: tuple[Frozen, ...]
    constants: tuple[tuple[Frozen, Frozen], ...]
    outputs: tuple[ValueReference, ...]
    nodes: tuple[Node, ...]
    effects: Frozen
    # JAX 的高层/低层 IR 标记也属于比较条件，不能混用两个抽象层级。
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
    value_references: Mapping[core.Var, ValueReference] = field(compare=False, repr=False)
    # lineage：中间图 →（原图，中间图每个方程对应的原方程索引）。
    # regions：各子图的规范化结果；semantic_views：方程用于比较的完整语义参数。
    # 后者可包含物化后的求导规则，不能拿它替换可执行方程里的原始 Python 回调。
    lineage: Mapping[core.Jaxpr, tuple[core.Jaxpr, tuple[int, ...]]] = field(
        default_factory=dict, compare=False, repr=False
    )
    regions: Mapping[core.Jaxpr, Canonicalized] = field(
        default_factory=dict, compare=False, repr=False
    )
    semantic_views: Mapping[core.JaxprEqn, Parameters] = field(
        default_factory=dict, compare=False, repr=False
    )
    # 这份结果是否把 custom_jvp / custom_vjp 携带的一阶规则算进了比较。
    include_derivative_rules: bool = True


@dataclass(frozen=True)
class Comparison:
    """图比较报告，包含状态、差异路径和两侧已完成的规范化结果。

    different 时查看 difference；失败通过异常传递。
    正常返回时两侧规范化与必要的对应校验均已完成。
    """

    status: Status
    difference: str | None = None
    left: Canonicalized | None = field(default=None, compare=False, repr=False)
    right: Canonicalized | None = field(default=None, compare=False, repr=False)
    correspondence: GraphCorrespondence | None = field(default=None, compare=False, repr=False)


def _validate(graph: core.Jaxpr, path: str) -> None:
    """先按原方程顺序检查单图的 SSA 定义与使用关系。

    每个变量只能定义一次，且必须先定义后使用。字面量是直接值，无需绑定。
    此处不递归调用 JAX 类型检查，以便先由规范化器发现嵌套图中的环。
    """
    if not isinstance(graph, core.Jaxpr):
        raise InvalidJaxprError(path, "expected a jaxpr including its bound constants")
    defined: set[core.Var] = set()

    def bind(v: core.Atom, where: str) -> None:
        """登记一个定义；重复定义会破坏生产者索引的一一对应关系。"""
        if not isinstance(v, core.Var) or v in defined:
            raise InvalidJaxprError(where, "invalid binder or duplicate definition")
        defined.add(v)

    def use(v: core.Atom, where: str) -> None:
        """核对引用；前向引用和依赖环都不属于本接口接受的合法 jaxpr。"""
        if not isinstance(v, core.Literal) and (not isinstance(v, core.Var) or v not in defined):
            raise InvalidJaxprError(where, "undefined value or dependency cycle")

    # 常量也有独立定义位置，绑定数量不一致时不能仅编码已有的那一部分。
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

    def __init__(
        self,
        include_derivative_rules: bool = True,
        search_budget: int = DEFAULT_SEARCH_BUDGET,
        decisions: tuple[tuple[int, ...], ...] = (),
    ) -> None:
        """初始化一次候选规范化的共享状态；每条槽位搜索分支使用独立实例。"""
        self.primitives = supported_primitive_identities()
        # cache 复用共享子图的处理结果；active 记录当前工作栈路径，用于检测环。
        # 缓存不跨公开调用复用，避免再次调用时沿用已经改变的常量或图内容。
        # 同步置换会创建临时分支图。缓存持有图对象，避免释放后 id 被其他图复用。
        self.cache: dict[core.Jaxpr, Canonicalized] = {}
        # 同一子图按“是否物化求导规则”不同而结果不同，必须分开缓存。
        self.scoped_cache: dict[tuple[core.Jaxpr, bool, bool], Canonicalized] = {}
        self.semantic_views: dict[core.JaxprEqn, Parameters] = {}
        self.include_derivative_rules = include_derivative_rules
        self.search_budget = search_budget
        # decisions 是外层搜索已选定的槽位置换前缀；遇到下一个歧义点时暂停并分叉。
        self.decisions = decisions
        self.choice_index = 0
        self.active: set[core.Jaxpr] = set()
        # 每张中间图的方程追溯到原始区域，供独立校验器检查原始图对应关系。
        self.lineage: dict[core.Jaxpr, tuple[core.Jaxpr, tuple[int, ...]]] = {}

    def _include_derivative_rules(self, path: str) -> bool:
        """按参数路径判断这个区域要不要物化自定义求导规则。"""
        # 规则内部只比前向，避免把一阶检查递归外推成高阶证明。
        return False if ".params.rules[" in path else self.include_derivative_rules

    def _free_slots(self, ports: Iterable[int], path: str) -> tuple[int, ...]:
        """无使用路径可排序时探索完整置换，不用原始位置作为语义平局规则。"""
        ports = tuple(ports)
        if len(ports) < 2:
            return ports
        if self.choice_index == len(self.decisions):
            raise _AmbiguousSlots(ports, path)
        selected = self.decisions[self.choice_index]
        self.choice_index += 1
        if sorted(selected) != sorted(ports):
            raise InvalidJaxprError(path, "inconsistent internal slot search")
        return selected

    def normalize(self, graph: core.Jaxpr, path: str) -> Canonicalized:
        """显式驱动区域生成器栈；子区域完成后将结果送回父区域。"""
        stack = []

        def push(child: core.Jaxpr, child_path: str, captures: bool = False) -> None:
            """登记当前路径并压入区域任务；共享子图可缓存，祖先回边则是环。"""
            if not isinstance(child, core.Jaxpr):
                raise InvalidJaxprError(
                    child_path, "expected a jaxpr including its bound constants"
                )
            if child in self.active:
                raise InvalidJaxprError(child_path, "cyclic nested jaxpr")
            self.active.add(child)
            stack.append(
                (child, child_path, captures, self._normalize(child, child_path, captures))
            )

        push(graph, path)
        value = None
        try:
            # 子生成器 yield（子图，路径）提出请求，return 交回 Canonicalized。
            # send 把子图结果送回父生成器；显式栈使深层 cond 不占用 Python 调用栈。
            while stack:
                current, current_path, captures, worker = stack[-1]
                try:
                    child, child_path, child_captures = worker.send(value)
                except StopIteration as finished:
                    result = finished.value
                    self.cache[current] = self.cache[result.jaxpr] = result
                    # 原图与返回图均指向同一次结果，后续同步换序会继续引用这些对象。
                    include = self._include_derivative_rules(current_path)
                    self.scoped_cache[(current, include, captures)] = result
                    self.scoped_cache[(result.jaxpr, include, captures)] = result
                    self.active.remove(current)
                    stack.pop()
                    value = result
                    continue
                if child in self.active:
                    raise InvalidJaxprError(child_path, "cyclic nested jaxpr")
                # 同一对象进入不同求导范围时不可共用结果，避免漏比或多比规则。
                key = (child, self._include_derivative_rules(child_path), child_captures)
                value = self.scoped_cache.get(key)
                if value is None:
                    push(child, child_path, child_captures)
            return cast(Canonicalized, value)
        finally:
            # 逆序关闭生成器，恢复跨 yield 保持的 mesh/轴配置上下文。
            for _, _, _, worker in reversed(stack):
                worker.close()
            self.active.clear()

    def _rebind_inputs(self, graph: core.Jaxpr, order: Sequence[int]) -> core.Jaxpr:
        """只调整形参槽位，变量及其引用保持不变，调试名称跟随变量移动。"""
        # order 为“新位置 → 旧位置”；恒等置换直接复用对象及已有缓存。
        if tuple(order) == tuple(range(len(graph.invars))):
            return graph
        debug = graph.debug_info
        if debug.arg_names is not None:
            names = debug.safe_arg_names(len(graph.invars))
            debug = debug._replace(arg_names=tuple(names[i] for i in order))
        rebound = graph.replace(invars=[graph.invars[i] for i in order], debug_info=debug)
        self.lineage[rebound] = self.lineage.get(graph, (graph, tuple(range(len(graph.eqns)))))
        return rebound

    def _rebind_outputs(self, graph: core.Jaxpr, order: Sequence[int]) -> core.Jaxpr:
        """按调用结果的置换重排子图输出槽位，只移动 Var 对象并登记方程来源。"""
        # 这里只重排返回列表；父方程的结果变量列表需由调用者采用相同置换。
        if tuple(order) == tuple(range(len(graph.outvars))):
            return graph
        debug = graph.debug_info
        if debug.result_paths is not None:
            paths = debug.safe_result_paths(len(graph.outvars))
            debug = debug._replace(result_paths=tuple(paths[i] for i in order))
        rebound = graph.replace(outvars=[graph.outvars[i] for i in order], debug_info=debug)
        self.lineage[rebound] = self.lineage.get(graph, (graph, tuple(range(len(graph.eqns)))))
        return rebound

    def _jit_bindings(
        self, eq: core.JaxprEqn, output_order: Sequence[int], path: str
    ) -> NormalizationTask[core.JaxprEqn]:
        """调用输入、输出和分片/布局/donation 描述同步置换，调用边界仍保留。"""
        # 参数集合按锁定的 JAX 0.11.2 核对：出现未知或缺失参数时明确失败，
        # 不能只按已知子集置换，否则升级后可能静默漏比新增语义。
        expected = {
            "jaxpr",
            "in_shardings",
            "out_shardings",
            "in_layouts",
            "out_layouts",
            "donated_invars",
            "ctx_mesh",
            "name",
            "keep_unused",
            "inline",
            "compiler_options_kvs",
        }
        unknown = eq.params.keys() - expected
        if unknown:
            raise UnsupportedJaxprError(path, f"unsupported jit parameters {sorted(unknown)}")
        if expected - eq.params.keys():
            raise InvalidJaxprError(path, "missing jit parameters")
        child = eq.params["jaxpr"]
        if (
            not isinstance(child, core.Jaxpr)
            or len(child.invars) != len(eq.invars)
            or len(child.outvars) != len(eq.outvars)
        ):
            raise InvalidJaxprError(path, "invalid jit interface")
        params = dict(eq.params)
        # 输入、输出元数据必须逐槽位对齐；长度不符时继续换序会错配 donation 等语义。
        for name in ("in_shardings", "in_layouts", "donated_invars"):
            if type(params[name]) is not tuple or len(params[name]) != len(eq.invars):
                raise InvalidJaxprError(path, f"invalid jit {name}")
        for name in ("out_shardings", "out_layouts"):
            if type(params[name]) is not tuple or len(params[name]) != len(eq.outvars):
                raise InvalidJaxprError(path, f"invalid jit {name}")
            params[name] = tuple(params[name][i] for i in output_order)
        # 先由调用结果的使用位置固定子图输出，再从子图内部使用位置确定输入顺序。
        # 父图实参、子图形参及逐端口元数据必须采用同一置换，才能保持调用语义。
        child = self._rebind_outputs(child, output_order)
        normalized = yield child, f"{path}.params.jaxpr", False
        input_order = self._capture_order(normalized, len(eq.invars), path)
        params["jaxpr"] = self._rebind_inputs(normalized.jaxpr, input_order)
        for name in ("in_shardings", "in_layouts", "donated_invars"):
            params[name] = tuple(params[name][i] for i in input_order)
        return eq.replace(
            invars=[eq.invars[i] for i in input_order],
            outvars=[eq.outvars[i] for i in output_order],
            params=params,
        )

    def _capture_order(
        self, result: Canonicalized, count: int, path: str, start: int = 0
    ) -> tuple[int, ...]:
        """按规范子图的首次使用位置排列捕获前缀，不能顺带改变 carry 位置。"""
        slots = {}
        # 直接返回的形参也是有序根，优先于节点中的使用；只移动指定角色区间。
        # 只在无根节点中使用的槽位同样没有固定使用路径，必须交给完整搜索。
        for ref in result.graph.outputs:
            if ref[0] == "input" and start <= ref[1] < start + count:
                slots.setdefault(ref[1], None)
        for node in result.graph.nodes[: len(result.bfs_order)]:
            # 无根后缀的排序依赖当前输入编号，不能反过来把它用作输入的锚点。
            # 剩余槽位由外层枚举，每次重绑后重新规范化全部节点与父图。
            for ref in node.inputs:
                if ref[0] == "input" and start <= ref[1] < start + count:
                    slots.setdefault(ref[1], None)
        unused = [i for i in range(start, start + count) if i not in slots]
        return (*slots, *self._free_slots(unused, path))

    def _output_order(
        self,
        eq: core.JaxprEqn,
        ports: Sequence[int],
        uses: Mapping[core.Var, _UsePath],
        path: str,
        *,
        feedback: Sequence[tuple[core.Jaxpr, int]] = (),
    ) -> tuple[int, ...]:
        """按父图最短使用路径排列已使用结果，把无锚点结果留给槽位搜索。"""
        used = [i for i in ports if eq.outvars[i] in uses]
        used.sort(key=lambda i: uses[eq.outvars[i]])
        unused = [i for i in ports if eq.outvars[i] not in uses]
        if not unused:
            return tuple(used)
        # Types and the consumer multiset of each feedback input move with its
        # output port. Refine before enumerating: eight ports need not cost 8!
        # merely because two outputs have the same immediate producer.
        # Operand positions at private calls can move, so the refinement uses
        # only consumer primitive multiplicity, never their raw port numbers.
        consumers = []
        for region, offset in feedback:
            counts: dict[core.Var, list[tuple[Any, ...]]] = {}
            for operation in region.eqns:
                key = order_key(operation.primitive)
                for value in operation.invars:
                    if isinstance(value, core.Var):
                        counts.setdefault(value, []).append(key)
            consumers.append(
                {i: tuple(sorted(counts.get(region.invars[offset + i], ()))) for i in unused}
            )
        children = (
            eq.params["branches"]
            if eq.primitive is lax.cond_p
            else (eq.params["body_jaxpr"],)
            if eq.primitive is lax.while_p
            else (eq.params["jaxpr"],)
        )
        categories = []
        encoder = Encoder(
            lambda child, loc: cast(EncodedRegion, SimpleNamespace(graph=("unrefined_region",))),
            lambda var: ("unrefined_variable",),
        )
        for child in children:
            producers = {v: e.primitive for e in child.eqns for v in e.outvars}
            categories.append(
                [
                    ("literal",)
                    if isinstance(v, core.Literal)
                    else ("producer", producers[v])
                    if v in producers
                    else ("input",)
                    if v in child.invars
                    else ("constant",)
                    for v in child.outvars
                ]
            )
        groups: dict[Any, list[int]] = {}
        for i in unused:
            key = order_key(
                (
                    tuple(category[i] for category in categories),
                    tuple(
                        encoder.encode(child.outvars[i].aval, f"{path}.type") for child in children
                    ),
                    tuple(consumer[i] for consumer in consumers),
                )
            )
            groups.setdefault(key, []).append(i)
        return (*used, *(i for key in sorted(groups) for i in self._free_slots(groups[key], path)))

    def _shard_bindings(
        self, eq: core.JaxprEqn, output_order: Sequence[int], path: str
    ) -> NormalizationTask[core.JaxprEqn]:
        """分片边界的私有端口同步置换，轴环境和逐位置 spec 都保持绑定。"""
        expected = {"jaxpr", "mesh", "in_specs", "out_specs", "check_vma", "newly_manual_axes"}
        if eq.params.keys() != expected:
            raise UnsupportedJaxprError(path, "shard_map parameter schema changed")
        params = dict(eq.params)
        child = params["jaxpr"]
        if (
            not isinstance(child, core.Jaxpr)
            or len(child.invars) != len(eq.invars)
            or len(child.outvars) != len(eq.outvars)
            or type(params["in_specs"]) is not tuple
            or type(params["out_specs"]) is not tuple
            or len(params["in_specs"]) != len(eq.invars)
            or len(params["out_specs"]) != len(eq.outvars)
        ):
            raise InvalidJaxprError(path, "invalid shard_map interface")
        child = self._rebind_outputs(child, output_order)
        # 上下文跨 yield 保持有效，子图抽象检查才能识别手动轴和变化性约束。
        with (
            shard_map._extend_axis_env(params["mesh"], params["newly_manual_axes"]),
            config._check_vma(params["check_vma"]),
        ):
            normalized = yield child, f"{path}.params.jaxpr", False
        inputs = self._capture_order(normalized, len(eq.invars), path)
        params["jaxpr"] = self._rebind_inputs(normalized.jaxpr, inputs)
        params["in_specs"] = tuple(params["in_specs"][i] for i in inputs)
        params["out_specs"] = tuple(params["out_specs"][i] for i in output_order)
        return eq.replace(
            invars=[eq.invars[i] for i in inputs],
            outvars=[eq.outvars[i] for i in output_order],
            params=params,
        )

    def _pallas_bindings(self, eq: core.JaxprEqn, path: str) -> NormalizationTask[core.JaxprEqn]:
        """Rename private allocations and scoped captures together with their bindings."""
        params = dict(semantic_parameters(eq))
        child = params["jaxpr"]
        captured = eq.primitive is run_scoped_p
        if captured and len(child.constvars) != len(eq.invars):
            raise InvalidJaxprError(path, "invalid scoped capture count")
        with ExitStack() as scope:
            if not captured:
                scope.enter_context(params["grid_mapping"].trace_env())
            result = yield child, f"{path}.params.jaxpr", captured
        start = 0 if captured else len(child.invars) - len(params["grid_mapping"].scratch_avals)
        if start < 0:
            raise InvalidJaxprError(path, "invalid scratch allocation count")
        resources = self._capture_order(result, len(child.invars) - start, path, start)
        rebound = self._rebind_inputs(result.jaxpr, (*range(start), *resources))
        executable = dict(eq.params)
        if not captured:
            executable["grid_mapping"] = dataclasses.replace(
                params["grid_mapping"],
                scratch_avals=tuple(
                    params["grid_mapping"].scratch_avals[i - start] for i in resources
                ),
            )
            executable["jaxpr"] = rebound
            return eq.replace(params=executable)
        # run_scoped's constvars are operand ports, not stored constant values.
        # Follow rooted uses; genuinely unanchored captures use the same bounded
        # interface search as private JAX call inputs.
        original = {result.value_references[v][1]: i for i, v in enumerate(child.constvars)}
        slots: dict[int, None] = {}
        for ref in (
            *result.graph.outputs,
            *(r for n in result.graph.nodes[: len(result.bfs_order)] for r in n.inputs),
        ):
            if ref[0] == "constant":
                slots.setdefault(original[ref[1]], None)
        constants = (
            *slots,
            *self._free_slots((i for i in range(len(child.constvars)) if i not in slots), path),
        )
        if constants != tuple(range(len(child.constvars))):
            rebound = rebound.replace(
                constvars=[child.constvars[i] for i in constants],
                consts=[child.consts[i] for i in constants],
            )
            self.lineage[rebound] = self.lineage[result.jaxpr]
        executable["jaxpr"] = rebound
        executable["ref_transforms"] = tuple(params["ref_transforms"][i] for i in resources)
        return eq.replace(invars=[eq.invars[i] for i in constants], params=executable)

    def _loop_bindings(
        self, eq: core.JaxprEqn, path: str, uses: Mapping[core.Var, _UsePath]
    ) -> NormalizationTask[core.JaxprEqn]:
        """同步循环捕获接口，分别处理 scan 的角色树与 while 的两组常量。

        反馈端口的初始化、body 形参、body 返回、condition 形参和最终结果
        必须联动；角色组之间不允许置换，也不合并任何共享绑定。
        """
        if eq.primitive is lax.scan_p:
            expected = {"reverse", "length", "jaxpr", "ft_in", "ft_out", "unroll"}
        else:
            expected = {"cond_nconsts", "body_nconsts", "cond_jaxpr", "body_jaxpr"}
        unknown = eq.params.keys() - expected
        if unknown:
            raise UnsupportedJaxprError(
                path, f"unsupported {eq.primitive} parameters {sorted(unknown)}"
            )
        if expected - eq.params.keys():
            raise InvalidJaxprError(path, f"missing {eq.primitive} parameters")
        params = dict(eq.params)

        if eq.primitive is lax.scan_p:
            body, ft_in, ft_out = params["jaxpr"], params["ft_in"], params["ft_out"]
            if not isinstance(body, core.Jaxpr) or not all(
                isinstance(tree, flattree.FlatTree) for tree in (ft_in, ft_out)
            ):
                raise InvalidJaxprError(path, "invalid scan body or interface trees")
            try:
                constants, carry, _ = ft_in.unpack()
                returned_carry, _ = ft_out.unpack()
            except (TypeError, ValueError, AssertionError) as exc:
                raise InvalidJaxprError(path, "invalid scan interface groups") from exc
            if (
                len(body.invars) != len(eq.invars)
                or len(ft_in) != len(eq.invars)
                or len(ft_out) != len(eq.outvars)
                or len(body.outvars) != len(eq.outvars)
                or len(carry) != len(returned_carry)
            ):
                raise InvalidJaxprError(path, "invalid scan operand or result count")
            nc, nk = len(constants), len(carry)
            # scan 输入为 [捕获常量, carry, xs]，输出为 [新 carry, ys]。
            # carry 的输入、输出构成跨迭代反馈边；换序时两端必须一起移动。
            carry_order = self._output_order(
                eq, range(nk), uses, f"{path}.carry", feedback=((body, nc),)
            )
            ys_order = self._output_order(eq, range(nk, len(eq.outvars)), uses, f"{path}.ys")
            out_order = (*carry_order, *ys_order)
            initial_order = (
                *range(nc),
                *(nc + i for i in carry_order),
                *range(nc + nk, len(eq.invars)),
            )
            body = self._rebind_outputs(self._rebind_inputs(body, initial_order), out_order)
            normalized = yield body, f"{path}.params.jaxpr", False
            captures = self._capture_order(normalized, nc, path)
            xs_order = self._capture_order(
                normalized, len(eq.invars) - nc - nk, path, start=nc + nk
            )
            order = (*captures, *(nc + i for i in carry_order), *xs_order)
            # body 的 carry 已调整一次，因此这里只再排列捕获与 xs。
            body_order = (*captures, *range(nc, nc + nk), *xs_order)
            params["jaxpr"] = self._rebind_inputs(normalized.jaxpr, body_order)
            # 附在叶子上的元数据与实际参数同步移动，carry/xs 的树结构完全保留。
            params["ft_in"] = ft_in.update(ft_in.vals[i] for i in order)
            params["ft_out"] = ft_out.update(ft_out.vals[i] for i in out_order)
            return eq.replace(
                invars=[eq.invars[i] for i in order],
                outvars=[eq.outvars[i] for i in out_order],
                params=params,
            )

        # while 父图实参为 [condition 捕获, body 捕获, carry]。
        # 两个子图各自只接收自己的捕获和同一组 carry，因此偏移量分别为 nc、nb。
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        condition, body = params["cond_jaxpr"], params["body_jaxpr"]
        carry_count = len(eq.outvars)
        if (
            type(nc) is not int
            or type(nb) is not int
            or nc < 0
            or nb < 0
            or not isinstance(condition, core.Jaxpr)
            or not isinstance(body, core.Jaxpr)
            or len(eq.invars) != nc + nb + carry_count
            or len(condition.invars) != nc + carry_count
            or len(body.invars) != nb + carry_count
            or len(condition.outvars) != 1
            or len(body.outvars) != carry_count
        ):
            raise InvalidJaxprError(path, "invalid while regions or operand count")
        carry_order = self._output_order(
            eq,
            range(carry_count),
            uses,
            f"{path}.carry",
            feedback=((body, nb), (condition, nc)),
        )
        condition = self._rebind_inputs(condition, (*range(nc), *(nc + i for i in carry_order)))
        body = self._rebind_inputs(body, (*range(nb), *(nb + i for i in carry_order)))
        body = self._rebind_outputs(body, carry_order)
        normalized_cond = yield condition, f"{path}.params.cond_jaxpr", False
        normalized_body = yield body, f"{path}.params.body_jaxpr", False
        cond_order = self._capture_order(normalized_cond, nc, f"{path}.condition")
        body_order = self._capture_order(normalized_body, nb, f"{path}.body")
        params["cond_jaxpr"] = self._rebind_inputs(
            normalized_cond.jaxpr, (*cond_order, *range(nc, nc + carry_count))
        )
        params["body_jaxpr"] = self._rebind_inputs(
            normalized_body.jaxpr, (*body_order, *range(nb, nb + carry_count))
        )
        order = (*cond_order, *(nc + i for i in body_order), *(nc + nb + i for i in carry_order))
        return eq.replace(
            invars=[eq.invars[i] for i in order],
            outvars=[eq.outvars[i] for i in carry_order],
            params=params,
        )

    def _cond_bindings(
        self,
        eq: core.JaxprEqn,
        path: str,
        output_order: Sequence[int] | None = None,
    ) -> NormalizationTask[core.JaxprEqn]:
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

        if output_order is None:
            output_order = tuple(range(len(eq.outvars)))
        if any(len(branch.outvars) != len(eq.outvars) for branch in branches):
            raise InvalidJaxprError(path, "invalid cond result count")

        # 子分支先处理，保证嵌套 cond 的捕获顺序不会影响本层参数首次使用位置。
        normalized = []
        for i, branch in enumerate(branches):
            result = yield (
                self._rebind_outputs(branch, output_order),
                f"{path}.params.branches[{i}]",
                False,
            )
            normalized.append(result)
        slots: dict[int, None] = {}

        def discover(ref: ValueReference) -> None:
            """按跨分支的首次出现顺序登记形参，只记录输入引用，不重编号常量。"""
            if ref[0] == "input":
                slots.setdefault(ref[1], None)

        # 分支顺序对应 selector 的取值，不能改变；所有分支共同确定唯一的形参顺序。
        for result in normalized:
            for ref in result.graph.outputs:
                discover(ref)
            # 全部分支的有序根共同锚定输入；无根后缀不得先于其他分支的
            # 实际使用位置定序，否则原输入编号会泄漏到共享槽位置换中。
            for node in result.graph.nodes[: len(result.bfs_order)]:
                for ref in node.inputs:
                    discover(ref)
        unused = [i for i in range(len(eq.invars) - 1) if i not in slots]
        order = (*slots, *self._free_slots(unused, path))
        rebound = []
        for result in normalized:
            rebound.append(self._rebind_inputs(result.jaxpr, order))
        # selector 只属于父 cond 的第 0 个操作数，不属于任何分支的形参列表。
        return eq.replace(
            invars=[eq.invars[0], *(eq.invars[i + 1] for i in order)],
            outvars=[eq.outvars[i] for i in output_order],
            params={**eq.params, "branches": tuple(rebound)},
        )

    def _params(
        self, eq: core.JaxprEqn, eq_path: str, encoder: Encoder, parent: core.Jaxpr
    ) -> list[tuple[str, Frozen]]:
        """编码全部参数；仅对已核对语义的特定参数采用专门适配。"""
        encoded_params = []
        # 参数按名称取序，消除字典插入顺序差异；这不是按算子名称排列计算节点。
        # semantic_parameters 会物化自定义求导规则，同一方程在发现与编码两轮
        # 中复用同一次结果，避免重复物化并保证两轮参数一致。
        params = self.semantic_views.get(eq)
        if params is None:
            params = semantic_parameters(
                eq, include_derivative_rules=self._include_derivative_rules(eq_path), parent=parent
            )
        self.semantic_views[eq] = params
        for name, value in sorted(params.items()):
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

    def _normalize(
        self, graph: core.Jaxpr, path: str, captures: bool
    ) -> NormalizationTask[Canonicalized]:
        """构造规范比较表示和可执行重排结果；输入图不被修改。

        先做 SSA 检查与值来源索引，沿有序根计算最短使用路径并同步重绑已适配的
        区域接口；再按输出根反向 BFS 编号、给无根部分补临时编号；随后编码全部
        方程、计算依赖深度并构造完整表示，未使用部分交给 canonical_order 搜索。
        返回的 graph 使用 BFS/搜索编号，reordered jaxpr 使用（依赖深度, 规范编号），
        两种顺序不能混用。
        """
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

        # named-axis effect 表示所需的轴环境，NoDCEEffect 对应保留计算的标记。
        # Pallas/Ref effects 由源适配器列出；未知 effect 在未使用分支也要拒绝。
        def check_effects(effects: Iterable[object], where: str) -> None:
            """在当前图及每个方程上检查 effect 类型，未知 effect 不能按纯计算处理。"""
            for effect in effects:
                if type(effect) not in (core.NamedAxisEffect, NoDCEEffect) and not effect_supported(
                    effect
                ):
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
            # 状态操作及包含它们的嵌套调用不能视作可任意换序的纯节点。
            # 给相邻的有序节点增加前驱边，同时作为额外根，避免输出回溯漏掉它们。
            if ordered_equation(eq):
                effect_roots.append(i)
                if previous_ordered is not None:
                    extra_predecessors[i] = [previous_ordered]
                previous_ordered = i
        # 消费者先处理，其输入槽位确定生产者的结果使用顺序。沿原图反向拓扑
        # 传播最短有序使用路径；不依赖无关计算在原方程列表中的排列。
        uses: dict[core.Var, _UsePath] = {}
        node_paths: dict[int, _UsePath] = {}

        def mark(atom: core.Atom, use_path: _UsePath) -> None:
            """保留结果值及其生产者的最短有序使用路径，供私有结果端口排序。"""
            if isinstance(atom, core.Var) and atom in producers:
                if atom not in uses or use_path < uses[atom]:
                    uses[atom] = use_path
                producer = producers[atom][0]
                if producer not in node_paths or use_path < node_paths[producer]:
                    node_paths[producer] = use_path

        for index, atom in enumerate(graph.outvars):
            mark(atom, _UsePath(index))
        for index, equation_index in enumerate(effect_roots):
            # 有序操作接在外部输出根之后编号，既覆盖无输出操作，也固定彼此顺序。
            use_path = _UsePath(len(graph.outvars) + index)
            if equation_index not in node_paths or use_path < node_paths[equation_index]:
                node_paths[equation_index] = use_path
        for i in reversed(range(len(equations))):
            # SSA 已保证生产者在消费者之前；逆序时消费者的私有输入槽位已处理完。
            eq = equations[i]
            if eq.primitive in (pjit.jit_p, lax.cond_p, shard_map.shard_map_p):
                output_order = self._output_order(
                    eq, range(len(eq.outvars)), uses, f"{path}.eqns[{i}].results"
                )
                if eq.primitive is pjit.jit_p:
                    eq = yield from self._jit_bindings(eq, output_order, f"{path}.eqns[{i}]")
                elif eq.primitive is shard_map.shard_map_p:
                    eq = yield from self._shard_bindings(eq, output_order, f"{path}.eqns[{i}]")
                else:
                    eq = yield from self._cond_bindings(eq, f"{path}.eqns[{i}]", output_order)
            elif eq.primitive in (lax.scan_p, lax.while_p):
                eq = yield from self._loop_bindings(eq, f"{path}.eqns[{i}]", uses)
            elif eq.primitive in (pallas_call_p, run_scoped_p):
                eq = yield from self._pallas_bindings(eq, f"{path}.eqns[{i}]")
            equations[i] = eq
            if i in node_paths:
                parent_path = node_paths[i]
                for port, atom in enumerate(eq.invars):
                    mark(atom, _UsePath(port, parent_path, parent_path.depth + 1))
        # 调用结果端口可能已经同步换序，引用必须使用新端口而非旧位置。
        producers = {
            v: (i, port) for i, eq in enumerate(equations) for port, v in enumerate(eq.outvars)
        }

        def visit(index: int) -> None:
            """节点第一次入队时分配编号，使共享前驱只出现一次。"""
            if index not in identifiers:
                identifiers[index] = len(identifiers)
                queue.append(index)

        def discover(atom: core.Atom) -> None:
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
        rooted_constants = len(constant_ids)
        # 3. 保留 BFS 不可达的纯方程，先按原顺序补临时编号。
        # 后续 canonical_order 对这部分做完整搜索，不能把临时编号当作最终规范顺序。
        unused = tuple(i for i in range(len(equations)) if i not in identifiers)
        for index in unused:
            identifiers[index] = len(identifiers)
            for atom in equations[index].invars:
                if isinstance(atom, core.Var) and atom in constants:
                    old = constants[atom]
                    constant_ids.setdefault(old, len(constant_ids))
        unused_constants = tuple(i for i in range(len(graph.constvars)) if i not in constant_ids)
        for index in unused_constants:
            constant_ids[index] = len(constant_ids)

        def reference(atom: core.Atom) -> ValueReference:
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

        def nested(child: core.Jaxpr, child_path: str) -> Canonicalized:
            """在子图边界内独立规范化；子图结果直接参与当前区域的编码。"""
            return self.scoped_cache[
                (child, self._include_derivative_rules(child_path), captures_child(eq, child_path))
            ]

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
                if eq.primitive in (pallas_call_p, emit_pipeline_p):
                    # Grid names and scalar-prefetch index maps use the call's
                    # tracing environment, not the ambient CPU axis context.
                    scope.enter_context(eq.params["grid_mapping"].trace_env())
                children = []

                def discover_child(
                    child: core.Jaxpr,
                    child_path: str,
                    children: list[tuple[core.Jaxpr, str]] = children,
                ) -> EncodedRegion:
                    """首次编码只登记子图任务，返回占位结果供参数容器继续遍历。"""
                    children.append((child, child_path))
                    return cast(EncodedRegion, SimpleNamespace(graph=None))

                # 先发现参数里的所有区域，再通过显式栈完成它们。同步编码器
                # 随后只读取缓存，避免参数编码递归调用区域规范化器。
                self._params(eq, eq_path, Encoder(discover_child, reference), graph)
                # 驱动器会 send 回处理结果，list 迭代器不支持 send，不能用 yield from。
                for child, child_path in children:  # noqa: UP028
                    yield child, child_path, captures_child(eq, child_path)
                encoded_params = self._params(eq, eq_path, encoder, graph)
            frozen_params.append(tuple(encoded_params))
            # 子图已由工作栈处理；此处只把参数中的子图替换成对应的新 jaxpr。
            # 没有需要替换的子图时，复用原方程对象，不修改原输入图。
            # 同一方程里的所有子图沿用同一范围；rules 内部固定只看前向。
            include_rules = self._include_derivative_rules(eq_path)
            capture_modes = {}
            for child, child_path in children:
                mode = captures_child(eq, child_path)
                if child in capture_modes and capture_modes[child] != mode:
                    raise UnsupportedJaxprError(eq_path, "shared region has mixed capture roles")
                capture_modes[child] = mode
            params = replace_nested(
                eq.params,
                lambda child, include=include_rules, modes=capture_modes: (
                    self.scoped_cache[(child, include, modes[child])].jaxpr
                ),
            )
            new_eq = eq if params is eq.params else eq.replace(params=params)
            new_equations.append(new_eq)
            self.semantic_views[new_eq] = {
                key: replace_nested(
                    value,
                    lambda child, key=key, include=include_rules, modes=capture_modes: (
                        self.scoped_cache[
                            (child, False if key == "rules" else include, modes[child])
                        ].jaxpr
                    ),
                )
                for key, value in self.semantic_views[eq].items()
            }

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
        # 6. 比较表示的可达前缀按 BFS 编号存放，无根后缀稍后再规范化；
        # result 引用使用该编号，不能与可执行方程顺序 order 混用。
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
                    ("captured_operand", i) if captures else encoder.encode(graph.consts[i], path),
                )
                for i in constant_ids
            ),
            tuple(reference(v) for v in graph.outvars),
            tuple(ordered_nodes),
            encoder.encode(graph.effects, f"{path}.effects"),
            graph.is_high,
        )
        if unused or unused_constants:
            # 可达前缀已有外部输出提供的锚点；搜索只能重命名剩余节点和常量。
            # 搜索返回“新位置 → 旧编号”，下方再合成“原图索引 → 新编号”。
            node_order, const_order = canonical_order(
                form, len(bfs_order), rooted_constants, budget=self.search_budget
            )
            form = reorder_form(form, node_order, const_order)
            original_nodes, original_constants = tuple(identifiers), tuple(constant_ids)
            identifiers = {original_nodes[old]: new for new, old in enumerate(node_order)}
            constant_ids = {original_constants[old]: new for new, old in enumerate(const_order)}
            # 可达部分不变；孤立部分同样按（依赖深度, 规范编号）排列，
            # 不能直接按搜索编号执行。
            order = tuple(sorted(bfs_order, key=lambda i: (depths[i], identifiers[i]))) + tuple(
                sorted(unused, key=lambda i: (depths[i], identifiers[i]))
            )
        # 只替换方程列表；输入、输出、常量绑定、调试信息等仍由原 jaxpr 保留。
        reordered = graph.replace(eqns=[new_equations[i] for i in order])
        # 中间图可能已换序多次，将本次执行顺序与既有来源映射合成，保留原方程索引。
        source, source_order = self.lineage.get(graph, (graph, tuple(range(len(graph.eqns)))))
        self.lineage[reordered] = (source, tuple(source_order[i] for i in order))
        # 先完成嵌套结构与环检查，再交给 JAX 做最终类型/参数检查，避免 JAX 先递归溢出。
        try:
            core.check_jaxpr(reordered)
        except (core.JaxprTypeError, TypeError, ValueError, NameError) as exc:
            raise InvalidJaxprError(path, f"JAX type check failed: {exc}") from exc
        # 变量映射覆盖输入、常量和方程结果，供对应关系模块把规范引用还原到原始绑定。
        return Canonicalized(
            reordered,
            form,
            bfs_order,
            order,
            tuple(depths),
            unused,
            unused_constants,
            MappingProxyType({v: reference(v) for v in (*inputs, *constants, *producers)}),
            include_derivative_rules=self._include_derivative_rules(path),
        )


def canonicalize(
    jaxpr: core.Jaxpr,
    *,
    include_derivative_rules: bool = True,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
) -> Canonicalized:
    """规范化包含常量绑定的 JAX/Pallas jaxpr，保留操作及其依赖关系。

    返回 Canonicalized，可分别查看可执行图、规范结构、编号和未完成原因。
    输入图不被修改；未使用纯分支参与完整比较。非法图抛出 InvalidJaxprError，
    未知语义抛出 UnsupportedJaxprError。本函数不执行 CPU 编译，也不证明数值等价。

    调用方需保持追踪上下文一致，并另行核对展平前的函数输入输出 pytree。
    依赖链、嵌套区域和参数使用显式栈；JAX 自身的检查仍可能受其资源限制。

    include_derivative_rules=True 时把 custom_jvp / custom_vjp 携带的一阶
    规则也物化进比较；False 只比较普通前向调用。search_budget 分别用于
    内部槽位候选搜索和每次无根图搜索；超限抛出 ComparisonBudgetError，
    不返回尚未证明是规范形式的局部最优结果。
    """
    try:
        if (jax.__version__, jaxlib.__version__) != ("0.11.2", "0.11.2"):
            raise UnsupportedJaxprError(
                "environment",
                "IR adapter requires audited JAX/jaxlib 0.11.2; "
                "run the version upgrade validation before extending support",
            )
        if type(include_derivative_rules) is not bool:
            raise ValueError("include_derivative_rules must be a bool")
        if type(search_budget) is not int or search_budget < 1:
            raise ValueError("search_budget must be a positive integer")
        # 每个待办项是一段内部槽位选择。重跑到下一处歧义时枚举所有选择，
        # 最后取完整图编码最小者；不能把首次成功的候选当作规范结果。
        pending: list[tuple[tuple[int, ...], ...]] = [()]
        attempts, best, best_key = 0, None, None
        while pending:
            decisions = pending.pop()
            attempts += 1
            if attempts > search_budget:
                raise ComparisonBudgetError(
                    "jaxpr.search",
                    f"internal interface search budget exceeded: {attempts} states, "
                    f"limit {search_budget}",
                    stage=ComparisonStage.STRUCTURAL,
                    used=attempts,
                    limit=search_budget,
                )
            normalizer = _Normalizer(include_derivative_rules, search_budget, decisions)
            try:
                result = normalizer.normalize(jaxpr, "jaxpr")
            except _AmbiguousSlots as choice:
                # 先估算阶乘数量，再生成置换，避免仅枚举候选就耗尽内存。
                # 预算不足必须拒绝；已找到的局部最优仍不能证明规范化完成。
                options = 1
                for size in range(2, len(choice.ports) + 1):
                    options *= size
                    if options > search_budget:
                        break
                if options + len(pending) + attempts > search_budget:
                    raise ComparisonBudgetError(
                        choice.path,
                        f"internal interface search budget exceeded: {len(choice.ports)} "
                        f"ambiguous slots, at least {options} alternatives, limit {search_budget}",
                        stage=ComparisonStage.STRUCTURAL,
                        used=attempts,
                        limit=search_budget,
                        alternatives=options,
                    ) from None
                pending.extend((*decisions, option) for option in permutations(choice.ports))
                continue
            candidate = dataclasses.replace(
                result,
                lineage=MappingProxyType(dict(normalizer.lineage)),
                regions=MappingProxyType(dict(normalizer.cache)),
                semantic_views=MappingProxyType(dict(normalizer.semantic_views)),
                include_derivative_rules=include_derivative_rules,
            )
            # 仅一个候选时省去大图排序键；出现第二个候选后才需要比较完整键。
            if best is None:
                best = candidate
            else:
                if best_key is None:
                    best_key = order_key(best.graph)
                key = order_key(candidate.graph)
                if key < best_key:
                    best, best_key = candidate, key
        return cast(Canonicalized, best)
    except RecursionError as exc:
        raise ComparisonBudgetError(
            "jaxpr",
            "nested graphs or parameters exceed the Python recursion limit",
            stage=ComparisonStage.STRUCTURAL,
            failure_kind=ComparisonFailureKind.RECURSION_LIMIT,
        ) from exc


def difference(left: Any, right: Any) -> str | None:
    """迭代比较完整编码结构，返回首个差异路径；相同则返回 None。

    不比较哈希或打印摘要。栈中倒序加入字段/元素，使取出时按声明顺序检查，
    保证报告位置稳定；编码后的叶子采用其已明确语义的精确相等比较。
    """
    # 差异路径定位的是规范 Graph，例如 nodes[0] 不一定是源 jaxpr 的第一个方程。
    pending = [("graph", left, right)]
    # 同一对子结构只展开一次，避免共享区域在比较时被重复展开为树。
    seen = set()
    while pending:
        path, a, b = pending.pop()
        if a is b:
            continue
        if type(a) is not type(b):
            return path
        pair = (id(a), id(b))
        if pair in seen:
            continue
        seen.add(pair)
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


def compare_jaxprs(
    left: core.Jaxpr,
    right: core.Jaxpr,
    *,
    include_derivative_rules: bool = True,
    search_budget: int = DEFAULT_SEARCH_BUDGET,
) -> Comparison:
    """结构阶段诊断：返回 equivalent/different；未完成检查直接抛出异常。

    两侧分别规范化，失败以 ComparisonError 子类抛出并在路径前补充 left/right。
    非法图、不支持语义、预算耗尽或对应证据复核失败都不会转成 different。
    different 仅表示完整图结构不同，不提供数学不等价结论。
    完整数学比较请使用 compare_algebraic_jaxprs 或默认 compare_functions。
    """
    if type(include_derivative_rules) is not bool:
        raise ValueError("include_derivative_rules must be a bool")
    if type(search_budget) is not int or search_budget < 1:
        raise ValueError("search_budget must be a positive integer")
    normalized = []
    for side, graph in (("left", left), ("right", right)):
        try:
            result = canonicalize(
                graph,
                include_derivative_rules=include_derivative_rules,
                search_budget=search_budget,
            )
            normalized.append(result)
        except ComparisonError as exc:
            raise exc.at(side) from exc
        except Exception as exc:
            raise ComparisonInternalError(
                f"{side}.jaxpr", f"{type(exc).__name__}: {exc}", stage=ComparisonStage.STRUCTURAL
            ) from exc
    a, b = normalized
    try:
        found = difference(a.graph, b.graph)
        correspondence = None
        if found is None:
            correspondence = build_correspondence(left, right, a, b)
            verify_correspondence(
                left, right, correspondence, include_derivative_rules=include_derivative_rules
            )
    except ComparisonError:
        # 已分类的比较错误保留原类型和诊断；其余异常包装成内部错误，保留异常链。
        raise
    except Exception as exc:
        raise ComparisonInternalError(
            "jaxpr.compare", f"{type(exc).__name__}: {exc}", stage=ComparisonStage.STRUCTURAL
        ) from exc
    return Comparison(
        Status.DIFFERENT if found else Status.EQUIVALENT,
        difference=found,
        left=a,
        right=b,
        correspondence=correspondence,
    )
