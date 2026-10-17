"""把参数、抽象值与常量编码成可精确比较的内存结构。

编码带有类型或类别标签，避免 Python 中 True == 1 等规则混淆不同语义。
数值数组保留 dtype、shape 和完整内容；嵌套图交由规范化器处理。
这不是持久化格式，也不会用 repr、打印摘要或哈希代替完整比较。
未知类型必须明确拒绝，不能为了得到相等结果而丢弃字段。
"""

import dataclasses
import enum
import struct
from collections.abc import Callable
from typing import Any

import jax
import numpy as np
from jax._src import core, flattree, layout, mesh, named_sharding
from jax._src.hashable_array import HashableArray
from jax._src.lax.lax import NoDCEEffect
from jax._src.numpy.scalar_types import _ScalarMeta
from jax._src.random.prng import KeyTy, PRNGKeyArray

from .errors import UnsupportedJaxprError

# 编码结果主要由带标签的元组、frozenset、字节串以及已适配的描述符组成。
# Any 避免为不同参数类型建立庞大的联合类型，不表示允许任意未知对象原样透传。
type Frozen = Any


def encoded_equal(left, right):
    """原图校验使用的值相等检查，深层容器不依赖 Python 递归比较。

    常规浅描述符使用内建精确比较；深容器回退到联合内容编号。哈希仅用于
    字典查找，字典仍比较完整键；不把哈希碰撞当成相等。
    """
    try:
        return left == right
    except RecursionError:
        pass
    # 两侧共用内容编号表：容器类型和全部子项编号完全相同，才会得到同一编号。
    # cache 仅避免重复展开共享对象；跨两侧的相等依据仍是完整内容，而非对象 id。
    interned, cache = {}, {}

    def number(root):
        """后序遍历一个编码值，以类型和子项内容编号构造可复用的结构编号。"""
        pending = [(False, root)]
        while pending:
            ready, value = pending.pop()
            if id(value) in cache:
                continue
            if type(value) in (tuple, frozenset):
                children = tuple(value)
                kind = type(value)
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                # 与数据类自身的相等契约一致，只收集 compare=True 的语义字段。
                children = tuple(
                    getattr(value, f.name) for f in dataclasses.fields(value) if f.compare
                )
                kind = type(value)
            else:
                signature = ("leaf", type(value), value)
                cache[id(value)] = interned.setdefault(signature, len(interned))
                continue
            if not ready:
                # 父容器晚于全部子项出栈，回填时才有完整的子项编号。
                pending.append((True, value))
                pending.extend((False, child) for child in reversed(children))
                continue
            ids = tuple(cache[id(child)] for child in children)
            signature = ("container", kind, frozenset(ids) if kind is frozenset else ids)
            cache[id(value)] = interned.setdefault(signature, len(interned))
        return cache[id(root)]

    return number(left) == number(right)


@dataclasses.dataclass(frozen=True, eq=False)
class _EncodeRequest:
    """描述编码的一个子任务；实际处理由显式栈完成。"""

    value: Any
    path: str


class Encoder:
    """由当前图提供引用解析规则的语义编码器。

    nested(child, path) 返回子图的规范化结果；variable(var) 返回当前图内的
    规范值引用。通过回调把图遍历与值编码分开，不让本模块自行分配节点编号。
    """

    def __init__(self, nested: Callable, variable: Callable):
        """注入区域和变量解析器，每个编码器独立维护当前容器路径。"""
        self.nested = nested
        self.variable = variable
        self.active: set[int] = set()
        # 调度期间的 encode 调用只生成占位请求，避免描述符层级形成 Python 递归。
        self.scheduling = False

    def fields(self, obj: Any, names: tuple[str, ...], path: str) -> Frozen:
        """保存对象实际类型及指定字段，字段顺序由已核对的类型适配规则确定。"""
        return (
            type(obj),
            tuple((name, self.encode(getattr(obj, name), f"{path}.{name}")) for name in names),
        )

    def literal(self, value: Any, aval: core.AbstractValue, path: str) -> Frozen:
        """按抽象值声明的 dtype 编码字面量，而不是按其 Python 存储类型编码。"""
        # JAX 字面量可能存成 Python 标量、TypedFloat/TypedInt 或 NumPy 标量。
        # 相同声明类型和数值不应仅因这些存储形式不同而被判为不同。
        if isinstance(aval, core.ShapedArray):
            return self.encode(np.asarray(value, dtype=aval.dtype), path)
        return self.encode(value, path)

    def encode(self, value: Any, path: str) -> Frozen:
        """显式遍历嵌套描述符，检查当前路径上的环，不消耗 Python 递归深度。"""
        if self.scheduling:
            # _encode/fields 中的递归调用只登记子任务，由外层工作栈统一展开。
            return _EncodeRequest(value, path)
        pending = [("encode", value, path)]
        # values 是后序结果栈；join 只消费属于当前容器的末尾若干项。
        values = []
        try:
            # encode 生成一层模板，resolve 展开子任务，join 收集结果，leave 退出路径。
            # active 只保留当前祖先；合法的共享参数可被重复访问，只有祖先回边才算环。
            while pending:
                action, item, location = pending.pop()
                if action == "encode":
                    if id(item) in self.active:
                        raise UnsupportedJaxprError(location, "cyclic parameter container")
                    self.active.add(id(item))
                    self.scheduling = True
                    try:
                        template = self._encode(item, location)
                    finally:
                        self.scheduling = False
                    pending.append(("leave", item, None))
                    pending.append(("resolve", template, None))
                elif action == "leave":
                    self.active.remove(id(item))
                elif action == "resolve":
                    if isinstance(item, _EncodeRequest):
                        pending.append(("encode", item.value, item.path))
                    elif type(item) in (tuple, frozenset):
                        parts = tuple(item)
                        pending.append(("join", (type(item), len(parts)), None))
                        pending.extend(("resolve", child, None) for child in reversed(parts))
                    else:
                        values.append(item)
                else:
                    kind, count = item
                    # count 为 0 时不能使用 values[-0:]，否则会误取走此前的全部结果。
                    children = values[-count:] if count else ()
                    if count:
                        del values[-count:]
                    values.append(kind(children))
            return values[0]
        finally:
            # 失败时也清空当前路径，使同一个编码器的后续调用不继承半完成状态。
            self.scheduling = False
            self.active.clear()

    def _encode(self, value: Any, path: str) -> Frozen:
        """按已核对的类型适配规则分派；每个分支保存足以区分语义的完整字段。

        已审查描述符的子类、未知容器类型和非 JAX 模块的数据类都会被显式拒绝，
        不提供 repr 或显示文本回退。
        """
        # 已审查描述符的子类可能新增语义字段，不能只编码基类已知字段。
        exact_types = (
            core.JaxprEqnContext,
            core.ShapedArray,
            core.AbstractToken,
            core.ManualAxisType,
            mesh.AbstractMesh,
            mesh.Mesh,
            jax.sharding.NamedSharding,
            jax.sharding.SingleDeviceSharding,
            jax.sharding.PartitionSpec,
            layout.Layout,
            HashableArray,
            NoDCEEffect,
        )
        for known in exact_types:
            if isinstance(value, known) and type(value) is not known:
                raise UnsupportedJaxprError(path, f"unreviewed descriptor subclass {type(value)}")
        # 图、变量、原语不能按字符串比较：分别使用规范子图、规范引用和原语身份。
        if isinstance(value, core.Jaxpr):
            return ("jaxpr", self.nested(value, path).graph)
        if isinstance(value, core.Var):
            # 变量只能由所属图解释；这里不猜测它是输入、常量还是某个方程的结果。
            return ("variable", self.variable(value))
        if isinstance(value, core.Primitive):
            return ("primitive", value)
        if isinstance(value, NoDCEEffect):
            # 已核对的 NoDCEEffect 没有附加语义字段；顺序处理由图规范化器负责。
            return ("no_dce_effect",)
        if isinstance(value, HashableArray):
            # 仍读取完整数组内容，不依赖包装对象的哈希或可能合并 NaN 情况的比较。
            return self.fields(value, ("val",), path)
        if isinstance(value, _ScalarMeta) or (
            isinstance(value, type)
            and (issubclass(value, np.generic) or value in (bool, int, float, complex))
        ):
            # 某些合法参数直接传 jnp.float32 等 dtype 类，而不是 np.dtype 实例。
            return ("dtype", np.dtype(value))
        if value is None:
            # 空值使用独立标签，不能与空 tuple、空数组或整数零合并。
            return ("none",)
        if isinstance(value, enum.Enum):
            # 保留枚举类型，不能把不同枚举类中数值相同的成员合并。
            return ("enum", type(value), self.encode(value.value, path))
        if type(value) in (bool, int, str, bytes):
            return (type(value), value)
        if type(value) is float:
            # 普通浮点相等会合并 +0/-0，NaN 又不等于自身；字节编码保留位模式。
            return (float, struct.pack(">d", value))
        if type(value) is complex:
            return (complex, struct.pack(">dd", value.real, value.imag))
        if isinstance(value, np.dtype):
            # 不只保存 dtype.str：不同 ml_dtypes（如 float8 格式）可能得到相同字符串。
            return ("dtype", value)
        if isinstance(value, KeyTy):
            # KeyTy 的相等比较包含完整 PRNGImpl（含函数身份），是已核对的描述符语义。
            # 只保存名称会漏掉同名但实现不同的随机数生成器。
            return ("key_dtype", value)
        if isinstance(value, PRNGKeyArray):
            # key 数组不能直接当普通 NumPy 数组读取；同时保存实现类型和底层 key 数据。
            return (
                "key_array",
                self.encode(value.dtype, path),
                self.encode(jax.random.key_data(value), path),
            )
        if isinstance(value, (np.ndarray, np.generic, jax.Array)):
            if isinstance(value, jax.Array) and any(d.platform != "cpu" for d in value.devices()):
                raise UnsupportedJaxprError(path, "only CPU arrays are currently supported")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise UnsupportedJaxprError(path, "object arrays have opaque element semantics")
            # 按逻辑 C 顺序读取元素，保存完整内容而非显示截断后的文本。
            # shape、dtype 另存，避免形状不同或类型不同的相同字节被误判相等。
            return ("array", array.dtype, array.shape, array.tobytes(order="C"))
        if isinstance(value, core.AbstractToken):
            return ("token",)
        if isinstance(value, jax.tree_util.PyTreeDef):
            # 自定义 pytree 的辅助数据可能是不透明对象，不能直接信任其 __eq__。
            # 分别保存节点类型、辅助数据和有序子树；未知辅助数据会走显式拒绝路径。
            node = value.node_data()
            if node is None:
                return ("pytree_leaf",)
            node_type, metadata = node
            return (
                "pytree",
                node_type,
                self.encode(metadata, f"{path}.metadata"),
                tuple(
                    self.encode(c, f"{path}.children[{i}]") for i, c in enumerate(value.children())
                ),
            )
        # JAX 0.11.2 的 scan 等参数使用 FlatTree；各变体的语义字段并不相同。
        # 升级 JAX 时需要重新核对这些字段，不能把内部对象的打印形式当作契约。
        flat_fields = {
            flattree.FTTuple: ("elts",),
            flattree.FTDict: ("keys", "_vals"),
            flattree.FTPyTree: ("xs", "_tree"),
            flattree.FTSingleton: ("val",),
            flattree.FTStatic: ("val",),
            flattree.FTFiltered: ("val",),
        }
        if type(value) in flat_fields:
            return self.fields(value, flat_fields[type(value)], path)
        if isinstance(value, core.ShapedArray):
            # 抽象值不只有 dtype/shape，分片、布局、弱类型等也影响语义。
            # _stripped_weak_type 是派生缓存，不作为额外语义字段保存。
            return self.fields(
                value,
                (
                    "shape",
                    "dtype",
                    "weak_type",
                    "sharding",
                    "manual_axis_type",
                    "memory_space",
                    "layout",
                ),
                path,
            )
        if isinstance(value, core.ManualAxisType):
            # 手动轴上的变化性与归约状态属于抽象类型，通信图比较必须保留。
            return self.fields(value, ("varying", "unreduced", "reduced", "unreduced_kind"), path)
        if isinstance(value, core.JaxprEqnContext):
            # 保存所有已设置的上下文字段；弱引用槽不是执行语义，未设置槽不能读取。
            names = tuple(
                name for name in value.__slots__ if name != "__weakref__" and hasattr(value, name)
            )
            return self.fields(value, names, path)
        if isinstance(value, mesh.AbstractMesh):
            return self.fields(
                value, ("axis_sizes", "axis_names", "axis_types", "abstract_device"), path
            )
        if isinstance(value, mesh.Mesh):
            # 相同 mesh 形状不代表相同设备排列；设备顺序和轴类型都必须保留。
            devices = tuple(self.encode(v, f"{path}.devices") for v in value.devices.flat)
            return (
                "mesh",
                value.devices.shape,
                devices,
                self.encode(value.axis_names, path),
                self.encode(value.axis_types, path),
            )
        if isinstance(value, jax.Device):
            if value.platform != "cpu":
                raise UnsupportedJaxprError(path, "only CPU device placement is supported")
            return ("device", value.platform, value.process_index, value.id, value.device_kind)
        if isinstance(value, jax.sharding.NamedSharding):
            # mesh、分区规则、内存种类和逻辑设备顺序共同决定分片描述。
            return self.fields(value, ("mesh", "spec", "memory_kind", "_logical_device_ids"), path)
        if isinstance(value, jax.sharding.SingleDeviceSharding):
            return self.fields(value, ("_device", "memory_kind"), path)
        if isinstance(value, jax.sharding.PartitionSpec):
            return self.fields(
                value, ("partitions", "unreduced", "reduced", "unreduced_kind"), path
            )
        if value is jax.sharding.PartitionSpec.UNCONSTRAINED:
            return ("unconstrained_partition",)
        if value is layout.AutoLayout or value is named_sharding.UNSPECIFIED:
            # 自动布局与未指定分片是已知单例，不需要编码其实现缓存或打印文本。
            return ("singleton", type(value))
        if isinstance(value, layout.Layout):
            return self.fields(
                value, ("major_to_minor", "tiling", "sub_byte_element_size_in_bits"), path
            )
        if isinstance(value, slice):
            return (slice, self.encode((value.start, value.stop, value.step), path))
        if isinstance(value, range):
            return (range, value.start, value.stop, value.step)
        if isinstance(value, tuple):
            # 保留有序元素及实际类型，因此 namedtuple 不会退化成普通 tuple。
            return (type(value), tuple(self.encode(v, f"{path}[{i}]") for i, v in enumerate(value)))
        if type(value) is list:
            return (list, tuple(self.encode(v, f"{path}[{i}]") for i, v in enumerate(value)))
        if type(value) is dict:
            # 字典插入顺序不是这里的参数语义；键和值本身仍需逐项完整编码。
            return (
                dict,
                frozenset(
                    (self.encode(k, path), self.encode(v, f"{path}[{k!r}]"))
                    for k, v in value.items()
                ),
            )
        if type(value) in (set, frozenset):
            # 无序容器保持无序语义，并区分 set 与 frozenset 的类型。
            return (type(value), frozenset(self.encode(v, path) for v in value))
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            # JAX 描述符逐字段编码；不据此接纳任意用户 dataclass 或隐藏的回调语义。
            if type(value).__module__.startswith(("jax.", "jaxlib.")):
                return self.fields(value, tuple(f.name for f in dataclasses.fields(value)), path)
        # 不使用 repr 回退：不同对象可能有相同显示文本，却承载不同计算语义。
        raise UnsupportedJaxprError(
            path,
            "opaque parameter or abstract value of type "
            f"{type(value).__module__}.{type(value).__qualname__}",
        )


def replace_nested(value: Any, replace: Callable) -> Any:
    """替换参数容器里的子 jaxpr，保留容器类型及其余对象。

    迭代处理 tuple/namedtuple、list 和 dict 中的值，其他描述符原样保留。
    调用前须先完成语义编码和环检查；本函数只负责重建必要的运行时参数容器。
    """
    # 先展开再回填，避免深层参数递归；只替换已编码检查过的子图，不重新解释语义。
    pending, values = [(False, value)], []
    while pending:
        ready, item = pending.pop()
        if isinstance(item, core.Jaxpr):
            values.append(replace(item))
        elif isinstance(item, tuple) or type(item) in (list, dict):
            children = tuple(item.values()) if type(item) is dict else tuple(item)
            if not ready:
                pending.append((True, item))
                pending.extend((False, child) for child in reversed(children))
                continue
            count = len(children)
            changed = values[-count:] if count else []
            if count:
                del values[-count:]
            # 子项均未替换时保留原容器身份，避免制造无意义的新参数和方程对象。
            if all(a is b for a, b in zip(children, changed, strict=True)):
                result = item
            elif type(item) is dict:
                result = dict(zip(item, changed, strict=True))
            elif hasattr(item, "_fields"):
                result = type(item)(*changed)
            else:
                result = type(item)(changed)
            values.append(result)
        else:
            values.append(item)
    return values[0]
