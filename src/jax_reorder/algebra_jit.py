"""默认纯 JIT 的有序参数代入；特殊执行语义继续保留调用边界。"""

from jax._src import pjit
from jax._src.api import Inline
from jax._src.sharding_impls import UNSPECIFIED

from .algebra_semantics import plain_equation


def default_jit(eq, params):
    """完整核对已审核默认 JIT 的执行属性。"""
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
    if child.effects or child.is_high:
        return False
    return True
