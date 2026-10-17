"""默认纯 JIT 的有序参数代入；特殊执行语义继续保留调用边界。

只有“不改变求值位置和放置方式”的纯 jit 才能被当作普通调用处理：参数按
原顺序代入，内部图继续参与转换；否则保留原区域语义，不做跨边界展开。
本模块只做契约判断，不改写图，也不把非默认属性近似成默认值。
"""

from jax._src import pjit
from jax._src.api import Inline
from jax._src.sharding_impls import UNSPECIFIED

from .algebra_semantics import plain_equation


def default_jit(eq, params):
    """完整核对已审核默认 JIT 的执行属性。

    返回 True 当且仅当方程是 pjit.jit_p、参数集合恰好是锁定版本审查过的
    那一组、方程本身是普通执行属性，且每个参数都是默认值、子图为纯图。
    参数集合用等值比较：多出或缺少键都意味着版本/适配未审查，一律拒绝。
    分片、布局、捐赠、mesh、keep_unused、inline、编译选项任一非默认都可能
    改变执行语义或设备放置，因此不接受“看起来等价”的近似。
    子图带 effects 或是 high-level 图时也不能当作纯函数共享。
    """
    required = {
        "jaxpr",
        "name",
        "in_shardings",
        "out_shardings",
        "in_layouts",
        "out_layouts",
        "donated_invars",
        "ctx_mesh",
        "keep_unused",
        "inline",
        "compiler_options_kvs",
    }
    if eq.primitive is not pjit.jit_p or set(params) != required or not plain_equation(eq):
        return False
    if (
        # UNSPECIFIED/None/False/AUTO/空 mesh/空 kv 分别是这些字段的默认值；
        # 用 is 比较，避免 == 把等价对象误当成同一默认标记。
        any(item is not UNSPECIFIED for item in (*params["in_shardings"], *params["out_shardings"]))
        or any(item is not None for item in (*params["in_layouts"], *params["out_layouts"]))
        or any(item is not False for item in params["donated_invars"])
        or not params["ctx_mesh"].empty
        or params["keep_unused"] is not False
        or params["inline"] is not Inline.AUTO
        or params["compiler_options_kvs"] != ()
    ):
        return False
    child = params["jaxpr"]
    # 有序 effects 或跨抽象层级的高层图不能作为普通调用展开。
    if child.effects or child.is_high:
        return False
    return True
