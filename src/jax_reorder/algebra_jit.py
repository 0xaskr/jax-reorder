"""默认纯 JIT 的有序参数代入；特殊执行语义继续保留调用边界。"""

from jax._src import core, pjit
from jax._src.api import Inline
from jax._src.sharding_impls import UNSPECIFIED

from .algebra_lower import encode
from .algebra_piecewise import _plain


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
    if eq.primitive is not pjit.jit_p or set(params) != required or not _plain(eq):
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


def transparent_jit(eq, values, params):
    """只消去已审核的默认编译边界，不删除已求值的部分函数实参。"""
    if not default_jit(eq, params):
        return False
    child = params["jaxpr"]
    if tuple(encode(var.aval) for var in child.invars) != tuple(
        value.signature for value in values
    ) or tuple(encode(var.aval) for var in child.outvars) != tuple(
        encode(var.aval) for var in eq.outvars
    ):
        return False
    used = set(var for var in child.outvars if isinstance(var, core.Var))
    for node in reversed(child.eqns):
        if any(var in used for var in node.outvars):
            used.update(var for var in node.invars if isinstance(var, core.Var))
    return all(
        var in used or value.body.total for var, value in zip(child.invars, values, strict=True)
    )
