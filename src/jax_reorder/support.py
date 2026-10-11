"""按 Primitive 对象身份发现原语，不建立 dtype、shape 或参数值白名单。

公开 lax 函数与实际追踪原语并非一一对应，因此也需要登记包装函数产生的
内部原语。被发现只表示身份在范围内；参数、effects、子图仍须逐项解释和校验。
实际 CPU 执行证据由原语清单及测试提供，不能仅凭出现在集合中就宣称全面支持。
"""

import sys

from jax import lax
from jax._src import core as jax_core
from jax._src import pjit, shard_map
from jax.extend import core


def public_primitives() -> dict[core.Primitive, tuple[str, ...]]:
    """返回公开原语到全部导出路径的映射，供清单与遗漏检查使用。"""
    names: dict[core.Primitive, list[str]] = {}
    # 锁定版本中公开的 lax 子模块为 linalg；同一对象可能通过多个名字导出。
    for module in (lax, lax.linalg):
        for name, value in vars(module).items():
            if not name.startswith("_") and isinstance(value, core.Primitive):
                names.setdefault(value, []).append(f"{module.__name__}.{name}")
    return {primitive: tuple(sorted(paths)) for primitive, paths in names.items()}


def supported_primitive_identities() -> frozenset[core.Primitive]:
    """收集当前环境中可进入语义校验流程的原语身份。

    集合在每次创建规范化器时获取，以包含追踪时已加载的 lax 内部模块。
    同名自定义 Primitive 不会自动匹配，避免通过名称冒充受支持的原语。
    """
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
    return frozenset(result)
