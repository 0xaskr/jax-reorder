"""符号转换前使用的元数据身份编码与执行契约检查。

数学比较的第一步是把图冻结成独立进程的输入，这要求共享同一套精确语义
编码：参数、抽象值和常量按类型与完整内容比较，而不用 repr 或哈希代替。
本模块还集中判断“默认执行属性”：布局、mesh、内存空间和非默认上下文一旦
出现，就不能被当成普通实数表达式改写，必须拒绝或保留区域边界。
这里不执行数组，也不解释嵌套子图；子图需要由区域适配器显式登记。
"""

from jax._src import core
from jax._src.layout import AutoLayout

from ...core.semantics import Encoder
from ...errors import UnsupportedJaxprError


def _reject_graph(_, path):
    """只有区域适配器可以解释子图，普通描述符不允许隐藏图。

    传入该回调后，任何未显式处理的嵌套 jaxpr 都会在编码时抛出带路径的
    UnsupportedJaxprError，而不是被静默压成字符串或省略。
    """
    raise UnsupportedJaxprError(path, "nested graph requires an algebra region adapter")


def encode(value):
    """复用完整语义编码，不用 repr 或哈希摘要作为数学身份。

    变量解析回调返回 None：单独编码的参数/抽象值不引用当前图的变量。
    编码失败（未知描述符、无法解释的子图）直接传出异常。
    """
    return Encoder(_reject_graph, lambda _: None).encode(value, "algebra.metadata")


def plain_equation(eq):
    """仅允许默认执行属性；不能在数学适配中丢失布局或放置约束。

    方程级：无 effects、无指定计算结果类型、空 mesh、无 XLA 元数据且布局
    模式为 AUTO。变量级：每个输入输出必须是维度为非负整数的 ShapedArray，
    使用默认设备内存、AutoLayout、无 manual axis 和空分片 mesh。
    token 等非 ShapedArray 抽象值同样返回 False，交给上层拒绝。
    """
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
            # 维度必须是已给定的非负 Python 整数；符号维度无法静态适配。
            or any(type(size) is not int or size < 0 for size in aval.shape)
            or aval.memory_space is not core.MemorySpace.Device
            or aval.layout is not AutoLayout
            or not aval.manual_axis_type.empty
            or not aval.sharding.mesh.empty
        ):
            return False
    return True
