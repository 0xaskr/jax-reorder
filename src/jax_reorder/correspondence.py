"""原始图对应证据及独立校验，不重新执行规范化或 BFS。

证据构造读取规范化过程保存的来源；校验只读取两份原始图和对应表。
参数、常量的精确编码与比较器共享，属于需要单独测试的语义信任基础。
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jaxlib
from jax import lax
from jax._src import core, pjit, shard_map

from .derivatives import semantic_parameters
from .errors import CorrespondenceError
from .semantics import Encoder, encoded_equal
from .support import supported_primitive_identities


@dataclass(frozen=True)
class EquationCorrespondence:
    """left 图中某一方程的 right 索引，以及参数内每个子区域的对应证据。"""

    # 方程编号使用原图中的位置，不是规范化后的 BFS 编号或执行列表位置。
    right: int
    # 子图以参数路径定位，例如 params.branches[0]；元组排列顺序不参与语义
    # 比较，真正的区域顺序由参数容器结构和所对应的方程表达。
    children: tuple[tuple[str, "GraphCorrespondence"], ...] = ()
    # outputs[p] = q 表示左侧结果端口 p 对应右侧端口 q，与方程在列表中的位置无关。
    outputs: tuple[int, ...] = ()


@dataclass(frozen=True)
class GraphCorrespondence:
    """每个元组按 left 原始索引排列，元素给出 right 原始索引。

    输入映射只允许用于已审查的内部接口；顶层输入顺序固定。
    输出映射仅允许用于已审查的调用接口；顶层输出顺序同样固定。
    """

    # 这些映射均从左到右；例如 constants[0] = 2 表示左常量 0 对应右常量 2。
    inputs: tuple[int, ...]
    constants: tuple[int, ...]
    equations: tuple[EquationCorrespondence, ...]
    # 保存所有外部返回槽位，包括同一值重复返回的各个位置。
    outputs: tuple[int, ...] = ()


def _parameter_view(
    equation,
    variable,
    *,
    inputs=None,
    outputs=None,
    parameters=None,
    derivative_scope="all",
    parent=None,
):
    """保留参数内容，用子图路径作占位并另行交给图校验器。

    不读取规范图的参数编码。jit 名称和 reduce callable 是锁定版本中
    已审查的两个例外；其余值一律由完整编码检查。parameters 可传入已物化
    的语义视图；parent 提供捕获常量所在的原始区域；derivative_scope 控制
    规则适配器展开到什么范围。

    inputs/outputs 是从左侧槽位到当前右侧槽位的映射，只用于把右侧的逐位置
    元数据投影到左侧顺序。返回参数编码及“参数路径 → 原始子图”的映射。
    """
    regions = {}

    def nested(graph, path):
        """登记子图及其路径，让参数编码与子图内容校验分开进行。"""
        # 父参数只记录子图所在位置，内容随后逐区域核对；占位不意味着忽略子图。
        regions[path] = graph
        return SimpleNamespace(graph=("region", path))

    encoder = Encoder(nested, variable)
    values = []
    params = (
        semantic_parameters(equation, scope=derivative_scope, parent=parent)
        if parameters is None
        else parameters
    )
    for key, value in sorted(params.items()):
        if equation.primitive is pjit.jit_p and key == "name":
            continue
        # 逐位置元数据必须随槽位置换同步投影；inputs/outputs 为 None 时不做
        # 投影，父调用映射由上层另行校验。
        if equation.primitive is pjit.jit_p:
            if inputs is not None and key in {"in_shardings", "in_layouts", "donated_invars"}:
                value = tuple(value[i] for i in inputs)
            if outputs is not None and key in {"out_shardings", "out_layouts"}:
                value = tuple(value[i] for i in outputs)
        if equation.primitive is shard_map.shard_map_p:
            if inputs is not None and key == "in_specs":
                value = tuple(value[i] for i in inputs)
            if outputs is not None and key == "out_specs":
                value = tuple(value[i] for i in outputs)
        if equation.primitive is lax.scan_p:
            if inputs is not None and key == "ft_in":
                value = value.update(value.vals[i] for i in inputs)
            if outputs is not None and key == "ft_out":
                value = value.update(value.vals[i] for i in outputs)
        if (
            jax.__version__ == "0.11.2"
            and equation.primitive is lax.reduce_p
            and key == "computation"
            and isinstance(equation.params.get("jaxpr"), core.Jaxpr)
            and callable(value)
        ):
            # 锁定版本中 reduce 的 computation 由已追踪的 reducer 子图表达；
            # 比较该子图而不是 Python callable 身份，返回方程仍保留原 callable。
            encoded = ("represented_by_reducer_jaxpr",)
        else:
            encoded = encoder.encode(value, f"params.{key}")
        values.append((key, encoded))
    return tuple(values), regions


def build_correspondence(left, right, left_normalized, right_normalized):
    """利用规范化来源构造左右原始图的对应证据；构造成功本身不表示正确。

    left/right 是原始区域 jaxpr，left_normalized/right_normalized 是
    canonicalize 的结果。函数用 lineage 把可执行顺序还原为原方程索引，
    再逐区域生成 GraphCorrespondence；子图交给外部工作栈处理，避免递归。
    结果必须交给 verify_correspondence 复核后才能作为等价证据；来源缺失或
    绑定不一致时抛出 CorrespondenceError，不返回部分证据。
    """

    def ports(a, b, an, bn):
        """通过规范化后配对的值，恢复两份原始接口之间的端口映射。"""
        # 同一变量可作为多个返回值；逐个消费原端口，保留重复出现次数。
        ai, bi = defaultdict(deque), defaultdict(deque)
        for i, value in enumerate(a):
            ai[id(value)].append(i)
        for i, value in enumerate(b):
            bi[id(value)].append(i)
        result = [None] * len(a)
        for av, bv in zip(an, bn, strict=True):
            result[ai[id(av)].popleft()] = bi[id(bv)].popleft()
        return tuple(result)

    def build(a, b, an, bn, required_outputs=None):
        """为一个区域构造对应表，遇到子区域时交由外部生成器栈处理。

        required_outputs 由父调用已核对的结果端口提供，只用于重复返回同一
        Var 的区域；普通区域从两侧原始 outvars 恢复端口映射。
        """
        # a/b 是原区域，an/bn 是规范化后的可执行区域；不能混用这两套索引。
        # 来源表把可执行列表位置还原到原方程位置，规范节点编号不直接用于证据。
        source_a, order_a = left_normalized.lineage[an]
        source_b, order_b = right_normalized.lineage[bn]
        if source_a is not a or source_b is not b:
            raise CorrespondenceError(
                "correspondence", "region source does not match original graph"
            )
        ai = {v: i for i, v in enumerate(a.invars)}
        bi = {v: i for i, v in enumerate(b.invars)}
        inputs = [None] * len(a.invars)
        for av, bv in zip(an.invars, bn.invars, strict=True):
            inputs[ai[av]] = bi[bv]
        # 常量在可执行图中不换序，用首次引用编号恢复两侧原始绑定的对应。
        ar = left_normalized.regions[an].value_references
        br = right_normalized.regions[bn].value_references
        bc = {br[v]: i for i, v in enumerate(b.constvars)}
        constants = tuple(bc[ar[v]] for v in a.constvars)
        equations = [None] * len(a.eqns)
        for ia, ib, ae, be in zip(order_a, order_b, an.eqns, bn.eqns, strict=True):
            # 原方程参数给出待证明的区域，重排方程参数给出本次使用的规范区域；
            # 四份视图的路径必须一致，才能把规范化来源还原为原图证据。
            _, ac = _parameter_view(
                a.eqns[ia],
                lambda v: v,
                parameters=left_normalized.semantic_views.get(a.eqns[ia], a.eqns[ia].params),
            )
            _, bc = _parameter_view(
                b.eqns[ib],
                lambda v: v,
                parameters=right_normalized.semantic_views.get(b.eqns[ib], b.eqns[ib].params),
            )
            _, anc = _parameter_view(ae, lambda v: v, parameters=left_normalized.semantic_views[ae])
            _, bnc = _parameter_view(
                be, lambda v: v, parameters=right_normalized.semantic_views[be]
            )
            if ac.keys() != bc.keys() or ac.keys() != anc.keys() or ac.keys() != bnc.keys():
                raise CorrespondenceError("correspondence", "nested region paths do not match")
            output_ports = ports(a.eqns[ia].outvars, b.eqns[ib].outvars, ae.outvars, be.outvars)
            children = []
            for key in ac:
                # 子图可重复返回同一个值，按 Var 身份逐次消费无法还原唯一的槽位。
                # 此时父调用的结果端口仍是不同 SSA 定义，应以父端口约束子图返回。
                # 独立校验器继续逐项核对值关系和反馈，不能只接受这份映射。
                coupled = (
                    key == "params.jaxpr"
                    and ae.primitive in (pjit.jit_p, lax.scan_p, shard_map.shard_map_p)
                    or key.startswith("params.branches[")
                    and ae.primitive is lax.cond_p
                    or key == "params.body_jaxpr"
                    and ae.primitive is lax.while_p
                )
                child = yield (
                    ac[key],
                    bc[key],
                    anc[key],
                    bnc[key],
                    output_ports if coupled else None,
                )
                children.append((key, child))
            equations[ia] = EquationCorrespondence(
                ib,
                tuple(children),
                output_ports,
            )
        return GraphCorrespondence(
            tuple(inputs),
            constants,
            tuple(equations),
            ports(a.outvars, b.outvars, an.outvars, bn.outvars)
            if required_outputs is None
            else required_outputs,
        )

    try:
        # 生成器只描述一个区域的工作，嵌套深度由列表承载，不递归调用 build。
        root = (left, right, left_normalized.jaxpr, right_normalized.jaxpr, None)
        stack = [(root, build(*root))]
        completed = {}
        value = None
        while stack:
            try:
                children = stack[-1][1].send(value)
            except StopIteration as finished:
                value = finished.value
                completed[stack[-1][0]] = value
                stack.pop()
            else:
                # 相同的原图/规范图四元组复用证据；不同绑定仍由其父节点逐次核对。
                value = completed.get(children)
                if value is None:
                    stack.append((children, build(*children)))
        return value
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        # CorrespondenceError 是 ValueError 子类；已带路径的证据错误直接
        # 重抛，其他来源或绑定错误统一补上构造阶段的上下文。
        if isinstance(exc, CorrespondenceError):
            raise
        raise CorrespondenceError(
            "correspondence", f"cannot construct original graph mapping: {exc}"
        ) from exc


def verify_correspondence(left, right, correspondence, *, derivative_scope="all"):
    """逐区域复核原始图的双射与绑定；成功返回 None，否则抛出明确错误。

    本函数不接受规范化器对象，也不调用 canonicalize/compare_jaxprs。
    即使规范化器忽略了一个参数或错误地合并节点，原图校验仍应拒绝。
    接口校验、SSA 和支持域在此独立检查；完整 JAX 类型检查仍由调用方执行。
    derivative_scope 控制参数视图里物化的自定义求导规则范围；规则内部
    一律收窄为 primal，避免递归展开未请求的高阶规则。
    """

    def require(condition, path, reason):
        """将校验失败统一转成带路径的证据错误，避免泄漏无定位信息的断言。"""
        if not condition:
            raise CorrespondenceError(path, reason)

    def permutation(indices, left_size, right_size, path):
        """检查完整双射：数量相等、索引为整数，且每个目标位置恰好出现一次。"""
        # 必须覆盖双方每个位置一次；只验证已匹配子集会漏掉多余节点或共享变化。
        require(
            type(indices) is tuple
            and len(indices) == left_size == right_size
            and all(type(i) is int for i in indices)
            and sorted(indices) == list(range(right_size)),
            path,
            "mapping is not a complete bijection",
        )

    def reject_nested(_, path):
        """值描述符中不允许隐藏子图；合法区域必须经参数视图单独登记和校验。"""
        raise CorrespondenceError(path, "unexpected graph in a value descriptor")

    require(
        (jax.__version__, jaxlib.__version__) == ("0.11.2", "0.11.2"),
        "environment",
        "correspondence adapter requires audited JAX/jaxlib 0.11.2",
    )
    require(derivative_scope in ("all", "primal"), "scope", "invalid derivative scope")
    supported = supported_primitive_identities()

    def check_domain(graph, path):
        """证据不能给未定义值、重复定义或未解释的原语授予等价结论。"""
        require(isinstance(graph, core.Jaxpr), path, "expected a bound jaxpr")
        available = set()
        for binders, operands in (
            ((*graph.constvars, *graph.invars), ()),
            *((eq.outvars, eq.invars) for eq in graph.eqns),
            ((), graph.outvars),
        ):
            require(
                all(isinstance(v, core.Literal) or v in available for v in operands),
                path,
                "undefined value or dependency cycle",
            )
            for var in binders:
                require(
                    isinstance(var, core.Var) and var not in available,
                    path,
                    "invalid or duplicate definition",
                )
                available.add(var)
        require(
            all(eq.primitive in supported for eq in graph.eqns),
            path,
            "unsupported primitive in correspondence",
        )

    def ordered_positions(graph):
        """从原始区域识别隐式顺序约束，不能相信规范图已保存了这些边。

        返回需要保持相对顺序的方程索引；调用方经方程映射后与右侧逐项比较，
        确保两侧有序操作一一对应且顺序一致，而不是仅按名称放行。
        """
        positions = []
        for index, equation in enumerate(graph.eqns):
            work = [equation]
            visited = set()
            while work:
                value = work.pop()
                if id(value) in visited:
                    continue
                visited.add(id(value))
                if isinstance(value, core.JaxprEqn):
                    if value.primitive.name in {"rng_uniform", "dce_sink"}:
                        positions.append(index)
                        break
                    work.extend(value.params.values())
                elif isinstance(value, core.Jaxpr):
                    work.extend(value.eqns)
                elif isinstance(value, (tuple, list)):
                    work.extend(value)
                elif type(value) is dict:
                    work.extend(value.values())
        return positions

    # root 标记控制外部输入输出是否必须保持恒等映射；内部置换由父节点约束。
    pending = [(left, right, correspondence, "correspondence", True)]
    seen = set()
    while pending:
        a, b, mapping, path, root = pending.pop()
        # 共享区域可重复引用，同一个区域在不同捕获绑定下仍由父节点逐次核对。
        identity = (id(a), id(b), id(mapping), root)
        if identity in seen:
            continue
        seen.add(identity)
        check_domain(a, path)
        check_domain(b, path)
        require(isinstance(mapping, GraphCorrespondence), path, "invalid region correspondence")
        permutation(mapping.inputs, len(a.invars), len(b.invars), f"{path}.inputs")
        permutation(mapping.constants, len(a.constvars), len(b.constvars), f"{path}.constants")
        permutation(mapping.outputs, len(a.outvars), len(b.outvars), f"{path}.outputs")
        require(len(mapping.equations) == len(a.eqns), path, "missing equation mappings")
        require(
            all(isinstance(eq, EquationCorrespondence) for eq in mapping.equations),
            path,
            "invalid equation mapping",
        )
        permutation(tuple(eq.right for eq in mapping.equations), len(a.eqns), len(b.eqns), path)
        require(
            [mapping.equations[i].right for i in ordered_positions(a)] == ordered_positions(b),
            path,
            "implicit operation order differs",
        )
        if root:
            require(
                mapping.inputs == tuple(range(len(a.invars))), path, "external input order changed"
            )
            require(
                mapping.outputs == tuple(range(len(a.outvars))),
                path,
                "external output order changed",
            )
        # 先为全部定义建立左 Var → 右 Var 映射，再检查每条使用边。
        # 两侧原方程顺序可以不同，不能一边扫描配对方程一边假定对方已遇到生产者。
        values = {v: b.invars[mapping.inputs[i]] for i, v in enumerate(a.invars)}
        values.update({v: b.constvars[mapping.constants[i]] for i, v in enumerate(a.constvars)})
        for i, equation in enumerate(a.eqns):
            eqmap = mapping.equations[i]
            other = b.eqns[eqmap.right]
            permutation(eqmap.outputs, len(equation.outvars), len(other.outvars), path)
            if equation.primitive not in (
                pjit.jit_p,
                lax.cond_p,
                lax.scan_p,
                lax.while_p,
                shard_map.shard_map_p,
            ):
                require(
                    eqmap.outputs == tuple(range(len(equation.outvars))),
                    path,
                    "fixed result port changed",
                )
            values.update(
                (v, other.outvars[eqmap.outputs[p]]) for p, v in enumerate(equation.outvars)
            )
        # 不允许两个左侧定义映射到同一个右侧定义，否则会把重复计算误当成共享。
        require(len(values) == len(set(values.values())), path, "value sharing is not bijective")
        # 参数和抽象值也可能引用 Var；统一映射到右侧命名空间后，才能精确比较。
        ae = Encoder(reject_nested, lambda v, values=values: values[v])
        be = Encoder(reject_nested, lambda v: v)

        def same_value(av, bv, where, ae=ae, be=be, values=values):
            """字面量比较类型和精确内容，变量则要求遵守已建立的唯一绑定。"""
            if isinstance(av, core.Literal):
                require(isinstance(bv, core.Literal), where, "literal and variable differ")
                require(
                    encoded_equal(ae.encode(av.aval, where), be.encode(bv.aval, where)),
                    where,
                    "literal type differs",
                )
                require(
                    encoded_equal(
                        ae.literal(av.val, av.aval, where), be.literal(bv.val, bv.aval, where)
                    ),
                    where,
                    "literal content differs",
                )
            else:
                require(values.get(av) is bv, where, "value binding or sharing differs")

        require(a.is_high == b.is_high, path, "IR abstraction level differs")
        require(
            encoded_equal(ae.encode(a.effects, path), be.encode(b.effects, path)),
            path,
            "graph effects differ",
        )
        for av, bv in values.items():
            require(
                encoded_equal(ae.encode(av.aval, path), be.encode(bv.aval, path)),
                path,
                "abstract value differs",
            )
        require(
            len(a.consts) == len(a.constvars) and len(b.consts) == len(b.constvars),
            path,
            "invalid constant bindings",
        )
        for i, j in enumerate(mapping.constants):
            require(
                encoded_equal(ae.encode(a.consts[i], path), be.encode(b.consts[j], path)),
                path,
                "constant content differs",
            )
        require(len(a.outvars) == len(b.outvars), path, "output count differs")
        for i, av in enumerate(a.outvars):
            same_value(av, b.outvars[mapping.outputs[i]], f"{path}.outputs")

        for i, equation in enumerate(a.eqns):
            eqmap = mapping.equations[i]
            other = b.eqns[eqmap.right]
            where = f"{path}.eqns[{i}]"
            require(equation.primitive is other.primitive, where, "primitive identity differs")
            require(len(equation.invars) == len(other.invars), where, "operand count differs")
            require(
                encoded_equal(ae.encode(equation.effects, where), be.encode(other.effects, where)),
                where,
                "equation effects differ",
            )
            require(
                encoded_equal(ae.encode(equation.ctx, where), be.encode(other.ctx, where)),
                where,
                "equation context differs",
            )
            children = dict(eqmap.children)
            require(len(children) == len(eqmap.children), where, "duplicate child correspondence")
            input_order = None
            if equation.primitive in (pjit.jit_p, lax.scan_p, shard_map.shard_map_p):
                require("params.jaxpr" in children, where, "missing call child correspondence")
                input_order = children["params.jaxpr"].inputs
                permutation(input_order, len(equation.invars), len(other.invars), where)
            # 规则区域内部不再展开更高阶规则；其余区域按调用方传入的范围复核，
            # 与规范化时使用的 derivative_scope 保持一致。
            scope = "primal" if ".params.rules[" in where else derivative_scope
            # 此处重新读取原参数并物化求导规则，不接受规范化器提供的相等结论。
            ap, ac = _parameter_view(
                equation, lambda v, values=values: values[v], derivative_scope=scope, parent=a
            )
            bp, bc = _parameter_view(
                other,
                lambda v: v,
                inputs=input_order,
                outputs=eqmap.outputs,
                derivative_scope=scope,
                parent=b,
            )
            require(encoded_equal(ap, bp), where, "primitive parameters differ")
            require(
                children.keys() == ac.keys() == bc.keys(),
                where,
                "missing or extra child correspondence",
            )

            def bound_child(
                key,
                a_slots,
                b_slots,
                flexible,
                output_order=None,
                children=children,
                where=where,
                equation=equation,
                other=other,
                same_value=same_value,
            ):
                """结合子图形参映射复核父节点实际绑定，不使用规范化后的实参。

                flexible 之前的槽位可在该组内置换，其余位置必须固定。
                循环额外的角色分组和反馈关系由下方对应的原语分支继续校验。
                """
                child = children[key]
                # a_slots/b_slots 将子图形参位置投影到父方程操作数位置。
                # 例如 cond 的子图形参 0 对应父操作数 1，父操作数 0 是 selector。
                permutation(child.inputs, len(a_slots), len(b_slots), where)
                require(
                    child.outputs
                    == (tuple(range(len(child.outputs))) if output_order is None else output_order),
                    where,
                    "region result binding differs",
                )
                for ai, bi in enumerate(child.inputs):
                    require((ai < flexible) == (bi < flexible), where, "operand role changed")
                    if ai >= flexible:
                        require(ai == bi, where, "fixed region operand position changed")
                    same_value(equation.invars[a_slots[ai]], other.invars[b_slots[bi]], where)

            if equation.primitive in (pjit.jit_p, shard_map.shard_map_p):
                slots = tuple(range(len(equation.invars)))
                bound_child("params.jaxpr", slots, slots, len(slots), eqmap.outputs)
            elif equation.primitive is lax.cond_p:
                same_value(equation.invars[0], other.invars[0], where)
                slots = tuple(range(1, len(equation.invars)))
                for branch in range(len(equation.params["branches"])):
                    bound_child(
                        f"params.branches[{branch}]", slots, slots, len(slots), eqmap.outputs
                    )
            elif equation.primitive is lax.scan_p:
                consts, carry_vars, _ = equation.params["ft_in"].unpack()
                nc, nk = len(consts), len(carry_vars)
                slots = tuple(range(len(equation.invars)))
                bound_child("params.jaxpr", slots, slots, len(slots), eqmap.outputs)
                child = children["params.jaxpr"]
                for start, end in ((0, nc), (nc, nc + nk), (nc + nk, len(slots))):
                    require(
                        all(start <= child.inputs[j] < end for j in range(start, end)),
                        where,
                        "scan operand role changed",
                    )
                require(
                    all((i < nk) == (j < nk) for i, j in enumerate(eqmap.outputs)),
                    where,
                    "scan result role changed",
                )
                # 单次 body 匹配还不够：第 i 个新 carry 必须反馈到对应的下一轮输入。
                require(
                    all(child.inputs[nc + i] == nc + eqmap.outputs[i] for i in range(nk)),
                    where,
                    "scan feedback binding differs",
                )
            elif equation.primitive is lax.while_p:
                nc, nb = equation.params["cond_nconsts"], equation.params["body_nconsts"]
                carry = tuple(range(nc + nb, len(equation.invars)))
                cond_slots = (*range(nc), *carry)
                body_slots = (*range(nc, nc + nb), *carry)
                bound_child("params.cond_jaxpr", cond_slots, cond_slots, len(cond_slots))
                bound_child(
                    "params.body_jaxpr", body_slots, body_slots, len(body_slots), eqmap.outputs
                )
                for name, count in (("params.cond_jaxpr", nc), ("params.body_jaxpr", nb)):
                    child = children[name]
                    require(
                        all((i < count) == (j < count) for i, j in enumerate(child.inputs)),
                        where,
                        "while operand role changed",
                    )
                    # condition 与 body 捕获数量可以不同，但 carry 必须服从同一映射。
                    require(
                        all(
                            child.inputs[count + i] == count + eqmap.outputs[i]
                            for i in range(len(carry))
                        ),
                        where,
                        "while feedback binding differs",
                    )
            else:
                # 未专门适配的接口坚持逐操作数比较，不推断交换律或任意子图端口换序。
                for av, bv in zip(equation.invars, other.invars, strict=True):
                    same_value(av, bv, where)
                for child in children.values():
                    require(
                        child.inputs == tuple(range(len(child.inputs))),
                        where,
                        "unsupported internal input permutation",
                    )
                    require(
                        child.outputs == tuple(range(len(child.outputs))),
                        where,
                        "unsupported internal output permutation",
                    )
            for key, child in children.items():
                # 父子接口已核对，入栈后再检查该区域内部全部定义、操作及绑定。
                pending.append((ac[key], bc[key], child, f"{where}.{key}", False))
