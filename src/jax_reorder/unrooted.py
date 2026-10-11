"""保留无输出根的纯计算，通过全图细化与有界精确搜索确定排列。

可达部分的 BFS 编号固定，未使用部分仍与其共享输入、常量和生产者。
摘要不作为相等证据：搜索比较完整标签和连接，最终还有原图双射校验。
"""

import dataclasses
from collections import defaultdict

import numpy as np
from jax._src import core
from jax._src.random.prng import KeyTy

from .errors import UnsupportedJaxprError

DEFAULT_SEARCH_BUDGET = 10_000


def _variables(value, transform):
    """只改当前区域的编码变量引用，子 Graph 保持自身的局部命名空间。"""
    # ready=False 展开子项，ready=True 收集合成；不进入子 Graph 的局部变量空间。
    pending, values = [(False, value)], []
    while pending:
        ready, item = pending.pop()
        if type(item) is tuple and len(item) == 2 and item[0] == "variable":
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
    value: object


def order_key(value):
    """为已编码语义给出可排序键；不代替最终的完整语义比较。

    类型/Primitive/PRNG 实现原本按身份区分，最后的身份项仅打破同名对象
    的平局。这是进程内规范表示，不是可跨环境持久化的序列化格式。
    """
    # 工作栈先处理子键再合成父键；无序集合排序，有序元组维持原元素位置。
    pending, values, cache = [("key", value)], [], {}
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
            parts = values[-item:] if item else ()
            if item:
                del values[-item:]
            values.append(tuple(sorted(parts)) if action == "sort" else tuple(parts))
    return values[0]


def _key_template(value):
    """只描述一层，子值由 order_key 的工作栈展开。"""
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
            tuple(
                _KeyRequest(getattr(value, f.name)) for f in dataclasses.fields(value) if f.compare
            ),
        )
    raise UnsupportedJaxprError("graph.search", f"no ordering adapter for encoded {type(value)}")


def canonical_order(graph, rooted_nodes, rooted_constants, *, budget=DEFAULT_SEARCH_BUDGET):
    """返回旧节点/常量索引的规范排列；预算耗尽时显式失败。"""
    n, nc = len(graph.nodes), len(graph.constants)
    # 搜索中把常量绑定也当作顶点：计算节点占 [0, n)，常量占 [n, n + nc)。
    # 输入端口和字面量保留在标签里，不能像内部顶点一样自由重新编号。
    labels, outgoing, incoming = [], [[] for _ in range(n + nc)], [[] for _ in range(n + nc)]
    has_parameter_references = False

    def anonymous(ref):
        nonlocal has_parameter_references
        if ref[0] in {"result", "constant"}:
            # 初始标签去掉临时编号；完整候选生成后仍会核对参数内引用的真实连接。
            has_parameter_references = True
            return (ref[0], "unbound", *ref[2:])
        return ref

    def edge(source, target, label):
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
    keys = tuple(order_key(label) for label in labels)
    ranks = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    initial = tuple(ranks[key] for key in keys)

    def refine(colors):
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
            if refined == colors:
                return colors
            colors = refined

    pending, best, best_order, visited = [initial], None, None, 0
    while pending:
        # 预算按探索状态计数；超限时整体失败，不返回已经找到的某个候选排列。
        visited += 1
        if visited > budget:
            raise UnsupportedJaxprError(
                "graph.search",
                f"exact graph search budget exceeded: {n} nodes, {nc} constants, "
                f"{visited} states, limit {budget}",
            )
        colors = refine(pending.pop())
        cells = defaultdict(list)
        for i, color in enumerate(colors):
            cells[color].append(i)
        ambiguous = next((cell for _, cell in sorted(cells.items()) if len(cell) > 1), None)
        if ambiguous is None:
            # 每个顶点已有唯一编号，才能构造包含全部标签和连接的完整比较证书。
            order = tuple(sorted(range(len(labels)), key=lambda i: colors[i]))
            numbers = {v: i for i, v in enumerate(order)}

            def renamed(ref, numbers=numbers):
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
    nodes = (*range(rooted_nodes), *(i for i in best_order if rooted_nodes <= i < n))
    constants = (
        *range(rooted_constants),
        *(i - n for i in best_order if i >= n + rooted_constants),
    )
    return nodes, constants


def reorder_form(graph, nodes, constants):
    """将完整图中的连接统一改编号，保持结果端口和共享次数。"""
    # 反转排列得到“旧索引 → 新编号”，所有值引用和顺序边必须一起改编号。
    node_ids = {old: new for new, old in enumerate(nodes)}
    constant_ids = {old: new for new, old in enumerate(constants)}

    def reference(ref):
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
