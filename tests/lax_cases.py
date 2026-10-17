"""CPU lax 清单的可执行固定用例，与重排算法本身分开维护。

每个 Case 用 ``expected`` 声明它必须实际追踪出的 primitive 名称，避免被 JAX 优化掉的
包装函数被算作覆盖。这里的参数是可运行的代表值，不是对合法 dtype/shape 域的收窄；
在 CPU 上 JAX 支持的全部合法组合仍属于目标域，表中数值不构成白名单。
``lax_inventory.py`` 读取本文件重新生成原语清单，执行证据由 pytest 运行这些图提供。

前置条件：conftest.py 固定 CPU 平台并暴露 2 个 host 设备，集合通信用例依赖后者。
本文件只描述输入与期望 primitive，不证明任何比较结论，也不覆盖所有原语参数组合。
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lax import windowed_reductions as windows
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P


@dataclass(frozen=True)
class Case:
    """一条固定原语用例：调用方式、实参、期望 primitive 名和确定性要求。

    ``deterministic=False`` 只用于 rng_uniform 这类每次追踪结果不同的用例，
    数值对照退化为 shape/dtype 与有限性检查。
    """

    name: str
    fn: Callable
    args: tuple
    expected: str
    deterministic: bool = True


def trace_case(case):
    """把用例追踪成单个 jaxpr，供清单脚本和 pytest 共用同一份输入定义。"""
    return jax.make_jaxpr(case.fn)(*case.args)


def cases():
    """构建全部固定用例；同一对象列表同时驱动清单生成与 CPU 执行测试。"""
    # 固定的浮点、整数、复数、对称矩阵输入；数值经过挑选以避开定义域边缘。
    f = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32)
    g = jnp.array([1.25, 1.5, 1.75], dtype=jnp.float32)
    i = jnp.array([1, 2, 3], dtype=jnp.int32)
    u = jnp.array([1, 2, 3], dtype=jnp.uint32)
    c = jnp.array([1 + 2j, 3 - 4j, -2 + 0.5j], dtype=jnp.complex64)
    matrix = jnp.array([[4, 1], [1, 3]], dtype=jnp.float32)
    result = []

    def add(name, fn, args, expected=None, *, deterministic=True):
        """追加一条用例；expected 默认与用例名相同，仅在名称是包装用例时显式传入。"""
        result.append(Case(name, fn, tuple(args), expected or name, deterministic))

    # 一元逐元素原语：全部使用同一组 float32 值，便于比较规范化前后的执行结果。
    for name in (
        "abs acos acosh asin asinh atan atanh bessel_i0e bessel_i1e cbrt ceil cos cosh "
        "digamma erf erf_inv erfc exp exp2 expm1 floor is_finite lgamma log log1p log2 "
        "logistic neg one_minus_square round rsqrt sign sin sinh sqrt square tan tanh"
    ).split():
        add(name, getattr(lax, name), (f,))
    # 二元逐元素原语：两个不同的 float32 向量，避免常量折叠成单值。
    for name in (
        "add atan2 div igamma igamma_grad_a igammac max min mul nextafter pow rem sub zeta"
    ).split():
        add(name, getattr(lax, name), (f, g))
    # 比较原语返回布尔数组，其 dtype 和元素类型也参与结构比较。
    for name in ("eq ge gt le lt ne").split():
        add(name, getattr(lax, name), (f, g))
    # 无符号整数输入：clz / population_count 只接受整数类型。
    for name in ("clz population_count").split():
        add(name, getattr(lax, name), (u,))
    # 按位原语使用 int32 且第二个操作数加一，避免出现 0 位移等退化参数。
    for name, api in (("and", "bitwise_and"), ("or", "bitwise_or"), ("xor", "bitwise_xor")):
        add(name, getattr(lax, api), (i, i + 1))
    add("not", lax.bitwise_not, (i,))
    # 位移与 mulhi 使用相同操作数，只验证 primitive 与参数编码，不追求特定数值。
    for name in ("mulhi shift_left shift_right_arithmetic shift_right_logical").split():
        add(name, getattr(lax, name), (i, i))
    # 复数原语：conj/imag/real 读同一复数数组，complex 由两个实数向量合成。
    for name in ("conj imag real").split():
        add(name, getattr(lax, name), (c,))
    add("complex", lax.complex, (f, g))
    add("regularized_incomplete_beta", lax.betainc, (g, g, f))
    add("polygamma", lax.polygamma, (jnp.ones_like(f), g))
    # 显式静态参数：精度、dtype、shape、切片与广播维都编码在 primitive 参数里。
    add("integer_pow", partial(lax.integer_pow, y=3), (f,))
    add("clamp", lax.clamp, (jnp.zeros_like(f), f, jnp.ones_like(f)))
    add("convert_element_type", partial(lax.convert_element_type, new_dtype=jnp.float16), (f,))
    add("bitcast_convert_type", partial(lax.bitcast_convert_type, new_dtype=jnp.int32), (f,))
    add("reduce_precision", partial(lax.reduce_precision, exponent_bits=5, mantissa_bits=7), (f,))
    add(
        "broadcast_in_dim",
        partial(lax.broadcast_in_dim, shape=(2, 3), broadcast_dimensions=(1,)),
        (f,),
    )
    # 形状变换只改变下标映射，供结构层核对输出 aval 与参数。
    add("reshape", partial(lax.reshape, new_sizes=(1, 3)), (f,))
    add("transpose", partial(lax.transpose, permutation=(1, 0)), (matrix,))
    add("squeeze", partial(lax.squeeze, dimensions=(0,)), (f[None, :],))
    add("concatenate", lambda a, b: lax.concatenate((a, b), dimension=0), (f, g))
    add("stack", lambda a, b: lax.stack((a, b), axis=0), (f, g))
    add("unstack", partial(lax.unstack, axis=0), (matrix,))
    add("split", partial(lax.split, sizes=(1, 2), axis=0), (f,))
    add("tile", partial(lax.tile, reps=(2,)), (f,))
    add("rev", partial(lax.rev, dimensions=(0,)), (f,))
    add("slice", partial(lax.slice, start_indices=(0,), limit_indices=(2,), strides=(1,)), (f,))
    # 动态切片/更新的起点是运行期标量输入，因此参数位置与静态 slice 用例不同。
    add("dynamic_slice", lambda x, start: lax.dynamic_slice(x, (start,), (2,)), (f, jnp.int32(1)))
    add(
        "dynamic_update_slice",
        lambda x, y, start: lax.dynamic_update_slice(x, y, (start,)),
        (f, g[:1], jnp.int32(1)),
    )
    # pad 的填充值作为独立实参传入，验证常量叶子与 padding_config 一起被保留。
    add("pad", partial(lax.pad, padding_config=((1, 2, 1),)), (f, jnp.float32(0)))
    add("iota", lambda: lax.iota(jnp.int32, 4), ())
    # select_n 的第一个输入是谓词下标，后两个是候选值分支。
    add("select_n", lax.select_n, (jnp.array([0, 1, 0]), f, g))
    # 归约覆盖浮点求和/积/极值与整数布尔归约，轴固定为 0。
    for name in ("reduce_sum reduce_prod reduce_max reduce_min").split():
        add(name, partial(getattr(lax, name), axes=(0,)), (matrix,))
    for name in ("reduce_and reduce_or reduce_xor").split():
        add(name, partial(getattr(lax, name), axes=(0,)), (i,))
    # 累积原语带 reverse=True，额外覆盖方向参数。
    for name in ("cumlogsumexp cummax cummin cumprod cumsum").split():
        add(name, partial(getattr(lax, name), axis=0, reverse=True), (f,))
    # argmax/argmin 输出整数下标，index_dtype 是显式静态参数。
    for name in ("argmax argmin").split():
        add(name, partial(getattr(lax, name), axis=0, index_dtype=jnp.int32), (f,))
    # 收缩类原语：dot/conv 使用固定 dimension_numbers，FFT 使用复数输入。
    add(
        "dot_general",
        partial(lax.dot_general, dimension_numbers=(((1,), (0,)), ((), ()))),
        (matrix, matrix),
    )
    add(
        "conv_general_dilated",
        partial(
            lax.conv_general_dilated,
            window_strides=(1,),
            padding="SAME",
            dimension_numbers=("NWC", "WIO", "NWC"),
        ),
        (jnp.ones((1, 4, 1)), jnp.ones((2, 1, 2))),
    )
    add("fft", partial(lax.fft, fft_type=lax.FftType.FFT, fft_lengths=(3,)), (c,))
    add("top_k", partial(lax.top_k, k=2), (f,))
    # approx_max_k 是近似算法，输入只要求能追踪和结构保留。
    add("approx_top_k", partial(lax.approx_max_k, k=2), (f,))
    # sort 同时接收数值键和整数载荷，验证多元组输入与 num_keys 参数。
    add("sort", partial(lax.sort, dimension=0, is_stable=True, num_keys=1), ((g, i),))
    # 直接绑定 primitive 的用例绕过公开包装函数，避免包装层改变名称或参数。
    add("copy", lax.copy_p.bind, (f,))
    add("stage", lax.stage_p.bind, (f,))
    add("stop_gradient", lax.stop_gradient, (f,))
    add("optimization_barrier", lax.optimization_barrier, ((f, i),))
    # RNG：bit_generator 输出 key；rng_uniform 每次追踪结果不同，故标记为非确定性。
    add(
        "rng_bit_generator",
        partial(lax.rng_bit_generator, shape=(3,), dtype=jnp.uint32),
        (jnp.array([0, 1, 2, 3], dtype=jnp.uint32),),
    )
    add(
        "rng_uniform",
        partial(lax.rng_uniform, shape=(3,)),
        (jnp.float32(0), jnp.float32(1)),
        deterministic=False,
    )
    # token 与有序 effect：create_token/after_all 建立顺序依赖，重排时不能交换。
    add("create_token", lambda x: lax.create_token(x), (f,))
    add("after_all", lambda x: lax.after_all(lax.create_token(x), lax.create_token(x)), (f,))
    # 控制流区域：cond/scan/while 各自带子图，规范化需要保持区域接口。
    add("cond", lambda x: lax.cond(x[0] > 0, lax.sin, lax.cos, x), (f,))
    add("scan", lambda x: lax.scan(lambda a, b: (a + b, a * b), jnp.float32(0), x), (f,))
    add("while", lambda x: lax.while_loop(lambda y: y < 3, lambda y: y + 1, x), (jnp.int32(0),))
    # 不透明求解器：solve 回调只在执行期使用，追踪时只出现一次调用。
    add(
        "custom_linear_solve",
        lambda x: lax.custom_linear_solve(
            lambda y: y * 2, x, solve=lambda _, b: b / 2, symmetric=True
        ),
        (f,),
    )
    # gather/scatter 使用固定索引映射；scatter 的六个变体在同一组 (f, 索引, 更新值) 上运行。
    gather_dims = lax.GatherDimensionNumbers(
        offset_dims=(), collapsed_slice_dims=(0,), start_index_map=(0,)
    )
    add(
        "gather",
        partial(lax.gather, dimension_numbers=gather_dims, slice_sizes=(1,), mode="clip"),
        (f, jnp.array([[2], [0]])),
    )
    scatter_dims = lax.ScatterDimensionNumbers(
        update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
    )
    for api in ("scatter scatter_add scatter_sub scatter_mul scatter_min scatter_max").split():
        # 用例名用连字符与清单文档一致；实际 API 名仍然带下划线。
        add(
            api.replace("_", "-"),
            partial(
                getattr(lax, api),
                dimension_numbers=scatter_dims,
                indices_are_sorted=True,
                unique_indices=True,
                mode="clip",
            ),
            (f, jnp.array([[0], [2]]), jnp.array([0.3, 0.7])),
        )
    # 窗口归约绑定私有实现，公开 API 在追踪时会被改写成 reduce_window 组合。
    for name in ("reduce_window_sum", "reduce_window_max", "reduce_window_min"):
        add(
            name,
            partial(
                getattr(windows, "_" + name),
                window_dimensions=(2,),
                window_strides=(1,),
                padding=((0, 0),),
            ),
            (f,),
        )
    # 通用 reduce 与 reduce_window 用自定义二元组合函数，验证 reduce 子图被递归处理。
    add(
        "reduce_window",
        lambda x: lax.reduce_window(x, jnp.float32(1), lax.mul, (2,), (1,), "VALID"),
        (f,),
    )
    add(
        "reduce",
        lambda x: lax.reduce(
            (x, x), (jnp.float32(0), jnp.float32(1)), lambda a, b: (a[0] + b[0], a[1] * b[1]), (0,)
        ),
        (f,),
    )
    # 线性代数：统一使用对称正定矩阵，使 eigh/cholesky/triangular_solve 落在合法域内。
    add("cholesky", lax.linalg.cholesky, (matrix,))
    add("eig", lax.linalg.eig, (matrix,))
    add("eigh", lax.linalg.eigh, (matrix,))
    add("hessenberg", lax.linalg.hessenberg, (matrix,))
    add("lu", lax.linalg.lu, (matrix,))
    add("qr", lax.linalg.qr, (matrix,))
    add("schur", lax.linalg.schur, (matrix,))
    add("svd", lax.linalg.svd, (matrix,))
    add(
        "triangular_solve",
        partial(lax.linalg.triangular_solve, left_side=True, lower=True),
        (jnp.tril(matrix), matrix),
    )
    add("tridiagonal", lax.linalg.tridiagonal, (matrix,))
    add(
        "tridiagonal_solve",
        lax.linalg.tridiagonal_solve,
        (
            jnp.array([0.0, 1.0, 1.0]),
            jnp.array([4.0, 4.0, 4.0]),
            jnp.array([1.0, 1.0, 0.0]),
            jnp.ones((3, 1)),
        ),
    )
    add(
        "lu_pivots_to_permutation",
        partial(lax.linalg.lu_pivots_to_permutation, permutation_size=3),
        (jnp.array([2, 1, 2], dtype=jnp.int32),),
    )
    add(
        "cholesky_update",
        lax.linalg.cholesky_update,
        (jnp.linalg.cholesky(matrix).T, jnp.array([0.1, 0.2])),
    )
    add("householder_product", lax.linalg.householder_product, (matrix, jnp.array([0.1, 0.2])))
    add("ormqr", lax.linalg.ormqr, (matrix, jnp.array([0.1, 0.2]), matrix))
    add("symmetric_product", lax.linalg.symmetric_product, (matrix, matrix))
    # eq_to/le_to/lt_to 是内部比较变体，直接绑定对应 primitive。
    for name in ("eq_to", "le_to", "lt_to"):
        add(name, getattr(lax, name + "_p").bind, (f, g))
    # 设备放置只绑定单个 CPU 设备，不要求多设备。
    add("device_put", partial(jax.device_put, device=jax.devices("cpu")[0]), (f,))
    add(
        "sharding_constraint",
        partial(
            lax.with_sharding_constraint,
            shardings=jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0]),
        ),
        (f,),
    )
    # select_and_* 系列绑定窗口化实现的私有函数，避免公开 API 组合改写掉目标 primitive。
    add(
        "select_and_scatter_add",
        partial(
            windows._select_and_scatter_add,
            select_prim=lax.ge_p,
            window_dimensions=(2,),
            window_strides=(1,),
            padding=((0, 0),),
        ),
        (f[:2], g),
    )
    add(
        "select_and_gather_add",
        partial(
            windows._select_and_gather_add,
            select_prim=lax.ge_p,
            window_dimensions=(2,),
            window_strides=(1,),
            padding=((0, 0),),
            base_dilation=(1,),
            window_dilation=(1,),
        ),
        (f, g),
    )
    add(
        "select_and_scatter",
        lambda x, source: windows._select_and_scatter(
            x, lax.ge, (2,), (1,), ((0, 0),), source, jnp.float32(0), lax.add
        ),
        (g, f[:2]),
    )

    # 集合通信用例：设备数来自实际 CPU 设备列表（conftest 固定为 2），
    # 输入按设备维切分，使 shard_map 的 in_specs 可以落在每个设备上。
    cpu_mesh = Mesh(np.array(jax.devices("cpu")), ("workers",))
    count = len(jax.devices("cpu"))
    collective_input = jnp.arange(count * 4, dtype=jnp.float32).reshape(count * 2, 2)
    # psum/pmin/pmax 输出聚合到副本，因此 out_specs 不带设备轴。
    for name in ("psum", "pmin", "pmax"):
        body = partial(getattr(lax, name), axis_name="workers")
        add(
            name,
            jax.shard_map(
                body, mesh=cpu_mesh, in_specs=P("workers", None), out_specs=P(), check_vma=False
            ),
            (collective_input,),
        )
    # axis_index 读取设备坐标，保持每设备一片；all_gather 反向合并设备轴。
    add(
        "axis_index",
        jax.shard_map(
            lambda x: x + lax.axis_index("workers"),
            mesh=cpu_mesh,
            in_specs=P("workers", None),
            out_specs=P("workers", None),
            check_vma=False,
        ),
        (collective_input,),
    )
    add(
        "all_gather",
        jax.shard_map(
            partial(lax.all_gather, axis_name="workers", axis=0, tiled=True),
            mesh=cpu_mesh,
            in_specs=P("workers", None),
            out_specs=P(),
            check_vma=False,
        ),
        (collective_input,),
    )
    # all_to_all 与 ppermute 在设备间交换分片，输出仍按设备轴布局。
    add(
        "all_to_all",
        jax.shard_map(
            partial(lax.all_to_all, axis_name="workers", split_axis=0, concat_axis=0, tiled=True),
            mesh=cpu_mesh,
            in_specs=P("workers", None),
            out_specs=P("workers", None),
            check_vma=False,
        ),
        (collective_input,),
    )
    add(
        "ppermute",
        jax.shard_map(
            partial(
                lax.ppermute, axis_name="workers", perm=[(n, (n + 1) % count) for n in range(count)]
            ),
            mesh=cpu_mesh,
            in_specs=P("workers", None),
            out_specs=P("workers", None),
            check_vma=False,
        ),
        (collective_input,),
    )

    def ragged(x):
        """构造每个设备固定大小的收发描述；形态合法即可，重点是 primitive 被追踪出来。"""
        offsets = jnp.arange(count, dtype=jnp.int32)
        sizes = jnp.ones(count, dtype=jnp.int32)
        return lax.ragged_all_to_all(
            x, jnp.zeros_like(x), offsets, sizes, offsets, sizes, axis_name="workers"
        )

    add(
        "ragged_all_to_all",
        jax.shard_map(
            ragged,
            mesh=cpu_mesh,
            in_specs=P("workers", None),
            out_specs=P("workers", None),
            check_vma=False,
        ),
        (collective_input,),
    )
    # pbroadcast 在 CPU 没有 lowering，reduce_scatter 由 psum_scatter 产生；
    # 二者只保留结构证据，执行拒绝在 test_lax_primitives.py 中显式断言。
    for name, body in (
        ("pbroadcast", partial(lax.pbroadcast, axis_name=("workers",), source=0)),
        ("reduce_scatter", partial(lax.psum_scatter, axis_name="workers", tiled=True)),
    ):
        add(
            name,
            jax.shard_map(
                body,
                mesh=cpu_mesh,
                in_specs=P("workers", None),
                out_specs=P("workers", None),
                check_vma=False,
            ),
            (collective_input,),
        )
    # 以下辅助 primitive 来自公开包装函数，用例名可与目标 primitive 不同。
    add("empty", lambda: lax.empty((2, 3), jnp.float32), ())
    add("composite", lax.composite(lambda x: lax.neg(x), "jax_reorder.neg"), (f,))
    add(
        "composite_attributes",
        lambda x: lax.composite(lambda x, bias: x + bias, "jax_reorder.bias")(
            x, bias=np.ones(3, np.float32)
        ),
        (f,),
        "composite",
    )
    add(
        "platform_index",
        lambda x: lax.platform_dependent(x, cpu=lambda x: lax.neg(x), default=lambda x: lax.sin(x)),
        (f,),
    )
    # dce_sink 的两个变体只在 prevent_mlir_dce 参数上不同，因此共享 expected 名称。
    add("dce_sink", lambda x: (lax.dce_sink(x), x)[1], (f,))
    add("dce_sink_mlir", lambda x: (lax.dce_sink(x, prevent_mlir_dce=True), x)[1], (f,), "dce_sink")
    # check_vma=True 触发 psum_invariant 这条附加不变式原语。
    add(
        "psum_invariant",
        jax.shard_map(
            partial(lax.psum, axis_name="workers"),
            mesh=cpu_mesh,
            in_specs=P("workers", None),
            out_specs=P(),
            check_vma=True,
        ),
        (collective_input,),
    )
    # 较新的收缩原语：scaled_dot 与带分组尺寸的 ragged_dot_general。
    add("scaled_dot", lax.scaled_dot, (matrix, matrix))
    add(
        "ragged_dot_general",
        partial(
            lax.ragged_dot_general,
            ragged_dot_dimension_numbers=lax.RaggedDotDimensionNumbers(
                (((1,), (1,)), ((), ())), (0,), (0,)
            ),
        ),
        (jnp.ones((4, 3)), jnp.ones((2, 3, 2)), jnp.array([2, 2], jnp.int32)),
    )
    return result
