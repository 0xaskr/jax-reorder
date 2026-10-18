"""给没有输出根的纯计算也排出规范顺序。

从输出可达的部分已有固定的 BFS 编号；不可达部分仍通过共享输入和常量与它们
相连，因此这里做全图细化，再用有界精确搜索确定剩余节点和常量的排列。
搜索比较完整标签和连接，不用摘要当证据；最后还有原图双射校验兜底。
本模块只生成排列和重写后的 Graph，不判断两张图是否等价。
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from jax._src import core
from jax._src.random.prng import KeyTy

from ..errors import ComparisonBudgetError, ComparisonStage, UnsupportedJaxprError
from ._typing import ValueReference
from .semantics import Frozen

if TYPE_CHECKING:
    from .canonical import Graph


# 搜索预算按候选状态计数；达到预算也不能以当前最优候选代替完整结论。
DEFAULT_SEARCH_BUDGET = 10_000


def _variables(value: Frozen, transform: Callable[[ValueReference], Frozen]) -> Frozen:
    """只改当前区域的编码变量引用，子 Graph 保持自身的局部命名空间。

    transform 接收 Encoder 生成的规范值引用并返回新引用；普通字符串和数组
    内容不会被当成变量标签。
    """
    # ready=False 展开子项，ready=True 收集合成；不进入子 Graph 的局部变量空间。
    pending, values = [(False, value)], []
    while pending:
        ready, item = pending.pop()
        if type(item) is tuple and len(item) == 2 and item[0] == "variable":
            # 这是 Encoder 为参数内 Var 生成的标签；普通字符串和数组内容不重写。
            values.append(("variable", transform(item[1])))
        elif type(item) in (tuple, frozenset):
            parts = tuple(item)
            if not ready:
                pending.append((True, item))
                pending.extend((False, part) for part in reversed(parts))
            else:
                count = len(parts)
                children = values[-count:] if count else ()
                if count:
                    del values[-count:]
                values.append(type(item)(children))
        else:
            values.append(item)
    return values[0]


@dataclasses.dataclass(frozen=True, eq=False)
class _KeyRequest:
    """排序键的子任务占位符，让模板只展开一层，由工作栈继续计算子键。"""

    # 待展开的子值；order_key 的工作栈会为它单独计算键并回填。
    value: object


def order_key(
    value: object, *, cache: dict[int, tuple[object, tuple[Any, ...]]] | None = None
) -> tuple[Any, ...]:
    """为已编码语义给出可排序键；不代替最终的完整语义比较。

    类型/Primitive/PRNG 实现原本按身份区分，最后的身份项仅打破同名对象
    的平局。这是进程内规范表示，不是可跨环境持久化的序列化格式。
    cache 按对象身份复用已算出的键，并保留原对象以防 id 被回收后重用。
    """
    # 工作栈先处理子键再合成父键；无序集合排序，有序元组维持原元素位置。
    pending: list[tuple[str, Any]] = [("key", value)]
    values: list[Any] = []
    cache = {} if cache is None else cache
    while pending:
        action, item = pending.pop()
        if action == "key":
            if id(item) in cache:
                values.append(cache[id(item)][1])
            else:
                pending.append(("save", item))
                pending.append(("resolve", _key_template(item)))
        elif action == "save":
            # 保留原对象以防临时描述符被回收后 id 重用。
            cache[id(item)] = item, values[-1]
        elif action == "resolve":
            if isinstance(item, _KeyRequest):
                pending.append(("key", item.value))
            elif type(item) is tuple:
                pending.append(("join", len(item)))
                pending.extend(("resolve", v) for v in reversed(item))
            elif isinstance(item, frozenset):
                pending.append(("sort", len(item)))
                pending.extend(("resolve", v) for v in item)
            else:
                values.append(item)
        else:
            # join 保持原顺序，sort 只用于无序集合；空容器不能取 values[-0:]。
            parts = values[-item:] if item else ()
            if item:
                del values[-item:]
            values.append(tuple(sorted(parts)) if action == "sort" else tuple(parts))
    return values[0]


def _key_template(value: object) -> tuple[Any, ...]:
    """只描述一层，子值由 order_key 的工作栈展开。"""
    # 首项数字区分类型类别，使不同 Python 类型也能稳定比较排序键。
    if value is None:
        return (0,)
    if type(value) in (bool, int, str, bytes):
        return (1, type(value).__name__, value)
    if isinstance(value, type):
        return (2, value.__module__, value.__qualname__, id(value))
    if isinstance(value, core.Primitive):
        return (3, value.name, id(value))
    if isinstance(value, np.dtype):
        return (4, value.name, value.str, _KeyRequest(value.descr))
    if isinstance(value, KeyTy):
        # PRNG 名称相同仍可能实现不同，函数成员以身份区分，其余成员继续编码。
        return (
            5,
            tuple(
                (name, id(v) if callable(v) else _KeyRequest(v))
                for name, v in zip(value._impl._fields, value._impl, strict=True)
            ),
        )
    if isinstance(value, (tuple, list)):
        return (6, tuple(_KeyRequest(v) for v in value))
    if isinstance(value, frozenset):
        return (7, frozenset(_KeyRequest(v) for v in value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            8,
            _KeyRequest(type(value)),
            # 与数据类自身的相等契约一致，只纳入 compare=True 的语义字段。
            tuple(
                _KeyRequest(getattr(value, f.name)) for f in dataclasses.fields(value) if f.compare
            ),
        )
    # 没有排序适配的类型必须显式拒绝，不能用 repr 或对象 id 兜底，
    # 否则未知语义会被悄悄当成可比较的规范键。
    raise UnsupportedJaxprError("graph.search", f"no ordering adapter for encoded {type(value)}")


def canonical_order(
    graph: Graph, rooted_nodes: int, rooted_constants: int, *, budget: int = DEFAULT_SEARCH_BUDGET
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """返回旧节点/常量索引的规范排列；预算耗尽时显式失败。

    rooted_nodes/rooted_constants 是已固定前缀的长度。搜索先按语义标签分组，
    再用连接细化；仍有歧义时逐个拆分候选组，选取完整证书最小的排列。
    返回两组“新位置 → 旧索引”，分别对应计算节点和常量绑定，供 reorder_form
    重写连接；预算按探索状态而不是节点数计数，超限时整体失败。
    """
    n, nc = len(graph.nodes), len(graph.constants)
    # 搜索中把常量绑定也当作顶点：计算节点占 [0, n)，常量占 [n, n + nc)。
    # 输入端口和字面量保留在标签里，不能像内部顶点一样自由重新编号。
    labels, outgoing, incoming = [], [[] for _ in range(n + nc)], [[] for _ in range(n + nc)]
    has_parameter_references = False

    def anonymous(ref: ValueReference) -> tuple[Any, ...]:
        """从初始标签去掉可重编号的目标索引，同时保留引用种类和结果端口。"""
        nonlocal has_parameter_references
        if ref[0] in {"result", "constant"}:
            # 初始标签去掉临时编号；完整候选生成后仍会核对参数内引用的真实连接。
            has_parameter_references = True
            return (ref[0], "unbound", *ref[2:])
        return ref

    def edge(source: int, target: int, label: tuple[int, ...]) -> None:
        """同时登记正反邻接，供细化使用消费者和生产者两侧的信息。"""
        # 此处边从消费者指向生产者，与执行时的数据流方向相反。
        # label 保留操作数位置、生产者结果端口或附加顺序边位置。
        outgoing[source].append((label, target))
        incoming[target].append((label, source))

    for i, node in enumerate(graph.nodes):
        inputs = []
        for port, ref in enumerate(node.inputs):
            if ref[0] == "result":
                edge(i, ref[1], (0, port, ref[2]))
                inputs.append(("edge",))
            elif ref[0] == "constant":
                edge(i, n + ref[1], (1, port))
                inputs.append(("edge",))
            else:
                inputs.append(ref)
        for position, predecessor in enumerate(node.order_after):
            edge(i, predecessor, (2, position))
        # 可达节点把已有 BFS 编号写进标签，搜索过程中始终固定这些外部锚点。
        labels.append(
            (
                "node",
                i if i < rooted_nodes else None,
                node.primitive,
                tuple(inputs),
                _variables(node.outputs, anonymous),
                _variables(node.params, anonymous),
                _variables(node.effects, anonymous),
                _variables(node.context, anonymous),
            )
        )
    labels.extend(
        ("constant", i if i < rooted_constants else None, value)
        for i, value in enumerate(graph.constants)
    )
    # 完整标签相同才分到同一初始颜色；哈希集合用于去重，不把摘要当证据。
    keys = tuple(order_key(label) for label in labels)
    ranks = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    initial = tuple(ranks[key] for key in keys)

    def refine(colors: tuple[int, ...]) -> tuple[int, ...]:
        """反复按双向带标签邻接细分颜色，直到当前分组不再变化。"""
        # color 是候选类别编号。用双向邻接不断细分；同色只表示尚无法区分，
        # 不能据此宣称两个节点等价，未拆开的类别还需精确搜索。
        while True:
            signatures = tuple(
                (
                    colors[i],
                    tuple(sorted((label, colors[j]) for label, j in outgoing[i])),
                    tuple(sorted((label, colors[j]) for label, j in incoming[i])),
                )
                for i in range(len(labels))
            )
            ranks = {sig: rank for rank, sig in enumerate(sorted(set(signatures)))}
            refined = tuple(ranks[sig] for sig in signatures)
            # 新签名含旧颜色，因此已有分组不会被合并；编号稳定时达到不动点。
            if refined == colors:
                return colors
            colors = refined

    pending, best, best_order, visited = [initial], None, None, 0
    while pending:
        # 预算按探索状态计数；超限时整体失败，不返回已经找到的某个候选排列。
        visited += 1
        if visited > budget:
            raise ComparisonBudgetError(
                "graph.search",
                f"exact graph search budget exceeded: {n} nodes, {nc} constants, "
                f"{visited} states, limit {budget}",
                stage=ComparisonStage.STRUCTURAL,
                used=visited,
                limit=budget,
            )
        colors = refine(pending.pop())
        cells = defaultdict(list)
        for i, color in enumerate(colors):
            cells[color].append(i)
        # 按颜色选择下一组歧义，而不是按原方程列表中最早出现的节点选择；
        # 颜色编号来自标签排序，搜索顺序因此不受原始索引影响。
        ambiguous = next((cell for _, cell in sorted(cells.items()) if len(cell) > 1), None)
        if ambiguous is None:
            # 每个顶点已有唯一编号，才能构造包含全部标签和连接的完整比较证书。
            order = tuple(sorted(range(len(labels)), key=lambda i: colors[i]))
            numbers = {v: i for i, v in enumerate(order)}

            def renamed(
                ref: ValueReference, numbers: Mapping[int, int] = numbers
            ) -> ValueReference:
                """把参数引用投影到当前候选的统一顶点编号，供完整证书比较。"""
                if ref[0] == "result":
                    return ("result", numbers[ref[1]], ref[2])
                if ref[0] == "constant":
                    return ("constant", numbers[n + ref[1]])
                return ref

            parameter_keys = {}
            if has_parameter_references:
                for i, node in enumerate(graph.nodes):
                    parameter_keys[i] = order_key(
                        _variables((node.outputs, node.params, node.effects, node.context), renamed)
                    )
            # 证书包含初始语义标签、全部依赖和参数内引用；颜色相同本身不是证据。
            certificate = tuple(
                (
                    keys[i],
                    tuple(sorted((label, numbers[j]) for label, j in outgoing[i])),
                    parameter_keys.get(i, ()),
                )
                for i in order
            )
            if best is None or certificate < best:
                best, best_order = certificate, order
            continue
        # 完全相同的邻接表对应可直接交换的孪生顶点，只需探索一个代表。
        twins = set()
        for chosen in ambiguous:
            twin = (tuple(sorted(outgoing[chosen])), tuple(sorted(incoming[chosen])))
            # 参数中存在变量引用时，邻接表没有表达全部连接，因此禁用孪生剪枝。
            if twin in twins and not has_parameter_references:
                continue
            twins.add(twin)
            # 逐个尝试将同色类中的一个顶点单独分出，再继续细化与搜索。
            split = tuple((color, int(i != chosen)) for i, color in enumerate(colors))
            ranks = {sig: rank for rank, sig in enumerate(sorted(set(split)))}
            pending.append(tuple(ranks[sig] for sig in split))
    # 返回“新位置 → 旧索引”；可达前缀不变，常量编号从统一顶点编号还原。
    best_order = cast(tuple[int, ...], best_order)
    nodes = (*range(rooted_nodes), *(i for i in best_order if rooted_nodes <= i < n))
    constants = (
        *range(rooted_constants),
        *(i - n for i in best_order if i >= n + rooted_constants),
    )
    return nodes, constants


def reorder_form(graph: Graph, nodes: Sequence[int], constants: Sequence[int]) -> Graph:
    """按排列重写完整图中的全部连接，保持结果端口和共享次数。

    nodes/constants 是“新位置 → 旧索引”的排列，通常来自 canonical_order。
    返回新的 Graph，不修改输入，也不重新判断语义或删除未使用节点。
    """
    # 反转排列得到“旧索引 → 新编号”，所有值引用和顺序边必须一起改编号。
    node_ids = {old: new for new, old in enumerate(nodes)}
    constant_ids = {old: new for new, old in enumerate(constants)}

    def reference(ref: ValueReference) -> ValueReference:
        """按最终排列重写节点或常量引用，外部输入位置和结果端口保持不变。"""
        if ref[0] == "result":
            return ("result", node_ids[ref[1]], ref[2])
        if ref[0] == "constant":
            return ("constant", constant_ids[ref[1]])
        return ref

    return dataclasses.replace(
        graph,
        constants=tuple(graph.constants[i] for i in constants),
        outputs=tuple(reference(ref) for ref in graph.outputs),
        nodes=tuple(
            dataclasses.replace(
                graph.nodes[i],
                inputs=tuple(reference(ref) for ref in graph.nodes[i].inputs),
                params=_variables(graph.nodes[i].params, reference),
                outputs=_variables(graph.nodes[i].outputs, reference),
                effects=_variables(graph.nodes[i].effects, reference),
                context=_variables(graph.nodes[i].context, reference),
                order_after=tuple(node_ids[p] for p in graph.nodes[i].order_after),
            )
            for i in nodes
        ),
    )
