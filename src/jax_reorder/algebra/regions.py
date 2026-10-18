"""把 cond/scan/while 区域绑定到与外围图相同的 SymPy 构造器。

区域实参、分支结果、carry 和序列切片都由同一个 SymbolicBuilder 构造，因此
自由坐标、定义条件和有限性事实能穿过区域边界继续保留。一般 scan 只做有界
符号展开，while 只接受能符号确定的终止；无法表达的区域明确抛出
UnsupportedJaxprError，不猜测循环行为。
"""

import sympy as sp

from ..errors import UnsupportedJaxprError
from .expression import SymbolicTensor, coordinates


def independent_scan(builder, eq, values):
    """把“独立加法累加”的 scan 表示成初值加有限和。

    满足以下条件时返回替代输出：每个变化的 carry 都只更新为
    carry + 与变化 carry 无关的增量，且序列输出只读取当前切片。这样长度不必
    展开，carry 的最终值就是初值加上逐项增量之和。条件不满足时返回 None，
    交给下面的有界展开路径；两种情况都不改变区域语义。
    """
    params = eq.params
    nc, nk, length = params["num_consts"], params["num_carry"], params["length"]
    # 只有全定义且有限的值才能参与实数差值；否则无法证明增量可加。
    if any(value.defined != sp.true or not value.finite for value in values):
        return None
    time, _, _ = builder.bound(length)
    states, bases = [], []
    # 每个 carry 用独立 Dummy 作为“状态身份”，数组状态再包一层 IndexedBase，
    # 以便在 body 表达式中识别对状态的读取。
    for index, initial in enumerate(values[nc : nc + nk]):
        if str(initial.dtype) not in ("float16", "bfloat16", "float32", "float64"):
            return None
        base = sp.Dummy(f"carry_{index}", real=True)
        expression = (
            sp.IndexedBase(base, shape=initial.shape, real=True)[coordinates(initial.shape)]
            if initial.shape
            else base
        )
        bases.append(base)
        states.append(SymbolicTensor(initial.shape, initial.dtype, expression))
    slices = []
    for value in values[nc + nk :]:
        # 序列输入沿首轴取第 time 个切片；time 是本次扫描的求和绑定变量。
        expression, domain = value.read((time, *coordinates(value.shape[1:])))
        slices.append(
            SymbolicTensor(value.shape[1:], value.dtype, expression, domain, value.finite)
        )
    body = builder.graph(params["jaxpr"], [*values[:nc], *states, *slices])
    # body 输出出现未定义或非有限值时，差值不能按实数加法消项。
    if any(
        value.defined != sp.true or value.expression.has(sp.nan, sp.zoo, sp.oo, -sp.oo)
        for value in body
    ):
        return None
    fixed = {
        base: initial
        for base, state, updated, initial in zip(
            bases, states, body[:nk], values[nc : nc + nk], strict=True
        )
        if state.expression == updated.expression
    }

    def substitute(expression):
        """用初值替换未变化 carry 的读取，使增量只依赖常量、输入和序列切片。"""
        replacements = {
            base: initial.expression for base, initial in fixed.items() if not initial.shape
        }
        for read in expression.atoms(sp.Indexed):
            if read.base.label in fixed:
                replacements[read] = fixed[read.base.label].read(read.indices)[0]
        return expression.xreplace(replacements)

    changing = [base for base in bases if base not in fixed]
    increments = []
    for state, updated in zip(states, body[:nk], strict=True):
        builder.budget.spend(path="algebra.scan.update")
        delta = sp.simplify(substitute(updated.expression - state.expression))
        # 增量仍依赖变化中的 carry 时不是独立累加，交回一般展开路径。
        if delta.has(*changing):
            return None
        increments.append(delta)
    # 序列输出若依赖变化中的 carry，就不能只保留最终 carry 的闭式。
    if any(substitute(value.expression).has(*changing) for value in body[nk:]):
        return None
    outputs = []
    for initial, delta in zip(values[nc : nc + nk], increments, strict=True):
        # length 为 0 时循环不执行，carry 保持初值。
        expression = (
            initial.expression + builder.summation(delta, ((time, 0, length - 1),))
            if length
            else initial.expression
        )
        outputs.append(SymbolicTensor(initial.shape, initial.dtype, expression))
    for var, value in zip(eq.outvars[nk:], body[nk:], strict=True):
        # 展开时序列输出按时间轴逐项记录；最终坐标轴就是该时间下标。
        axes = coordinates(var.aval.shape)
        expression = substitute(value.read(axes[1:])[0]).xreplace({time: axes[0]})
        outputs.append(SymbolicTensor(tuple(var.aval.shape), var.aval.dtype, expression))
    builder.rules.add("scan:independent_accumulation")
    return outputs


def region(builder, eq, values, path):
    """分派 cond/scan/while，并返回与区域输出端口对应的 SymbolicTensor。

    cond 把各分支结果合成带选择条件的 Piecewise；一般 scan 在长度上限内逐步
    展开，独立加法累加另走闭式；while 只在条件能符号判定时展开。区域输出保留
    分支/carry 的定义条件和有限性，未知控制流显式拒绝。
    """
    params, name = eq.params, eq.primitive.name
    if name == "cond":
        selector = values[0]
        branches = [builder.graph(graph, values[1:]) for graph in params["branches"]]
        outputs = []
        for var, alternatives in zip(eq.outvars, zip(*branches, strict=True), strict=True):
            arms, domains = [], []
            for index, value in enumerate(alternatives):
                # 每个分支按选择值编号加守卫；输出条件要求选择值有定义且至少
                # 一个分支守卫成立。默认分支只是 Piecewise 的占位，不参与条件。
                guard = sp.Eq(selector.expression, index)
                arms.append((value.expression, guard))
                domains.append(sp.And(guard, value.defined))
            outputs.append(
                SymbolicTensor(
                    tuple(var.aval.shape),
                    var.aval.dtype,
                    sp.Piecewise(
                        *arms, (sp.false if str(var.aval.dtype) == "bool" else sp.S.Zero, True)
                    ),
                    sp.And(selector.defined, sp.Or(*domains)),
                    all(value.finite for value in alternatives),
                )
            )
        return outputs
    if name == "scan":
        independent = independent_scan(builder, eq, values)
        if independent is not None:
            return independent
        count, length, ncarry = params["num_consts"], params["length"], params["num_carry"]
        constants = values[:count]
        carry = values[count : count + ncarry]
        sequences = values[count + ncarry :]
        # 展开是符号展开：每步都扣预算，绝不执行 JAX 循环；超过上限就拒绝。
        if length > 64:
            raise UnsupportedJaxprError(path, "scan length exceeds bounded symbolic unrolling")
        emissions = [[] for _ in eq.outvars[ncarry:]]
        order = reversed(range(length)) if params["reverse"] else range(length)
        for index in order:
            builder.budget.spend(path=path)
            slices = []
            for value in sequences:
                # 序列输入按时间下标取值；reverse 只改变遍历顺序，记录的实际
                # 下标仍是原序列位置。
                expression, domain = value.read((sp.Integer(index), *coordinates(value.shape[1:])))
                slices.append(
                    SymbolicTensor(value.shape[1:], value.dtype, expression, domain, value.finite)
                )
            outputs = builder.graph(params["jaxpr"], [*constants, *carry, *slices])
            carry = outputs[:ncarry]
            for target, value in zip(emissions, outputs[ncarry:], strict=True):
                target.append((index, value))
        outputs = list(carry)
        for var, emitted in zip(eq.outvars[ncarry:], emissions, strict=True):
            # 每个 ys 端口按实际时间下标合成 Piecewise；长度 0 时没有分支，
            # 条件保持 true，值取占位零。
            axes = coordinates(var.aval.shape)
            arms, domains = [], []
            for index, value in emitted:
                expression, domain = value.read(axes[1:])
                guard = sp.Eq(axes[0], index)
                arms.append((expression, guard))
                domains.append(sp.And(guard, domain))
            outputs.append(
                SymbolicTensor(
                    tuple(var.aval.shape),
                    var.aval.dtype,
                    sp.Piecewise(
                        *arms, (sp.false if str(var.aval.dtype) == "bool" else sp.S.Zero, True)
                    )
                    if arms
                    else sp.S.Zero,
                    sp.Or(*domains) if arms else sp.true,
                    all(value.finite for _, value in emitted),
                )
            )
        return outputs
    if name == "while":
        nc, nb = params["cond_nconsts"], params["body_nconsts"]
        condition_constants, body_constants = values[:nc], values[nc : nc + nb]
        carry = values[nc + nb :]
        # 最多接受 64 次 body 展开：循环判定为 false 时返回当前 carry；条件既非
        # 真也非假表示终止依赖数据，不能用有限展开代表原循环。
        for _ in range(65):
            predicate = builder.graph(params["cond_jaxpr"], [*condition_constants, *carry])[0]
            truth = sp.simplify(predicate.expression)
            # 谓词自身未定义时，后续真值判断没有依据。
            if predicate.defined != sp.true:
                raise UnsupportedJaxprError(path, "loop predicate conditions are not established")
            if truth == sp.false:
                return carry
            if truth != sp.true:
                raise UnsupportedJaxprError(
                    path, "data-dependent loop termination is not represented"
                )
            builder.budget.spend(path=path)
            carry = builder.graph(params["body_jaxpr"], [*body_constants, *carry])
        raise UnsupportedJaxprError(path, "while loop exceeds bounded symbolic unrolling")
    # 其他区域（如 shard_map）没有语义适配，必须显式拒绝。
    raise UnsupportedJaxprError(path, "unsupported region")
