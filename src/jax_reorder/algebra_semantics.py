"""Metadata identity and execution-contract checks used before symbolic conversion."""

from jax._src import core
from jax._src.layout import AutoLayout

from .errors import UnsupportedJaxprError
from .semantics import Encoder


def _reject_graph(_, path):
    """只有区域适配器可以解释子图，普通描述符不允许隐藏图。"""
    raise UnsupportedJaxprError(path, "nested graph requires an algebra region adapter")


def encode(value):
    """复用完整语义编码，不用 repr 或哈希摘要作为数学身份。"""
    return Encoder(_reject_graph, lambda _: None).encode(value, "algebra.metadata")


def plain_equation(eq):
    """仅允许默认执行属性；不能在数学适配中丢失布局或放置约束。"""
    if (
        eq.effects
        or eq.ctx.compute_type is not None
        or not eq.ctx.cur_abstract_mesh.empty
        or eq.ctx.xla_metadata is not None
        or eq.ctx.cur_layout_mode is not core.LayoutMode.AUTO
    ):
        return False
    for var in (*eq.invars, *eq.outvars):
        aval = var.aval
        if (
            not isinstance(aval, core.ShapedArray)
            or any(type(size) is not int or size < 0 for size in aval.shape)
            or aval.memory_space is not core.MemorySpace.Device
            or aval.layout is not AutoLayout
            or not aval.manual_axis_type.empty
            or not aval.sharding.mesh.empty
        ):
            return False
    return True
