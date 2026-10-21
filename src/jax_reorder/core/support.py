"""按 Primitive 对象身份（而不是名字）识别支持哪些原语。

公开的 lax 函数和实际出现在 jaxpr 里的原语并不一一对应，所以还要登记包装
函数产生的内部原语。被识别只说明对象身份在范围内；参数、effects 和子图
仍要逐项解释和校验。CPU 上的真实支持情况由原语清单和测试证明，不能仅凭
出现在名单里就宣称全面支持。
"""

import sys

from jax import lax
from jax._src import ad_util, pjit, shard_map
from jax._src import core as jax_core
from jax.extend import core

from .adapters.pallas.structural import TPU_PRIMITIVES
from .derivatives import CUSTOM_PRIMITIVES


def public_primitives() -> dict[core.Primitive, tuple[str, ...]]:
    """返回公开原语到全部导出路径的映射，供清单与遗漏检查使用。

    字典键是 Primitive 对象，值是导出名称；别名归到同一个原语下，名称排序
    只用于稳定展示，不参与计算图的节点排序或等价判定。
    """
    names: dict[core.Primitive, list[str]] = {}
    # 锁定版本中公开的 lax 子模块为 linalg；同一对象可能通过多个名字导出。
    for module in (lax, lax.linalg):
        for name, value in vars(module).items():
            if not name.startswith("_") and isinstance(value, core.Primitive):
                names.setdefault(value, []).append(f"{module.__name__}.{name}")
    return {primitive: tuple(sorted(paths)) for primitive, paths in names.items()}


def supported_primitive_identities() -> frozenset[core.Primitive]:
    """收集当前环境中可进入语义校验流程的原语身份。

    每次创建规范化器或复核对应证据时重新获取，以包含追踪时已加载的 lax 内部模块。
    同名自定义 Primitive 不会自动匹配，避免通过名称冒充受支持的原语。
    """
    # dict 迭代得到原语对象，先以公开导出作为支持身份集合的起点。
    result = set(public_primitives())
    # 公开包装函数可能追踪出未公开导出的辅助原语，不能只查 lax 顶层属性。
    # 先取得模块表快照，避免遍历期间模块加载使字典大小发生变化。
    for name, module in tuple(sys.modules.items()):
        if module is not None and name.startswith("jax._src.lax."):
            result.update(v for v in vars(module).values() if isinstance(v, core.Primitive))
    # jit、分片和布局等包装节点必须保留其边界，不能为了比较而直接删除或内联。
    result.update(v for v in vars(pjit).values() if isinstance(v, core.Primitive))
    # program_order 是额外调度 API，其子图带指令顺序语义；当前未适配，也不属于
    # 已支持的公开 lax 包装路径，因此明确移除，避免把它当普通纯子图重排。
    result.discard(pjit.program_order_p)
    result.add(shard_map.shard_map_p)
    # check_vma=True 的通信组合会插入 pvary，改变手动轴的变化类型。
    # 它定义在 core，未被上面的 lax 模块扫描发现；仍保留节点、axes 和抽象值。
    result.add(jax_core.pvary_p)
    # 一阶自动求导用 add_any 累加余切向量。它没有隐藏参数，操作数顺序仍保留。
    # 高级自定义抽象类型依旧必须通过 Encoder 的独立审查。
    result.add(ad_util.add_jaxvals_p)
    # 自定义求导调用由 derivatives 模块展开规则，其 Python callable 不按名称比较。
    result.update(CUSTOM_PRIMITIVES)
    result.update(TPU_PRIMITIVES)
    # 返回不可变快照，使一次规范化期间的身份集合不受调用方修改影响。
    return frozenset(result)
