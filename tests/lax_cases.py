"""Runnable, typed fixtures for the CPU lax inventory, separate from the algorithm.

These are representative values, not restrictions on the accepted dtype/shape
domain. Cases name the primitive they must actually trace, preventing optimized
away wrappers from being counted as coverage.
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
    name: str
    fn: Callable
    args: tuple
    expected: str
    deterministic: bool = True


def trace_case(case):
    return jax.make_jaxpr(case.fn)(*case.args)


def cases():
    f = jnp.array([0.25, 0.5, 0.75], dtype=jnp.float32)
    g = jnp.array([1.25, 1.5, 1.75], dtype=jnp.float32)
    i = jnp.array([1, 2, 3], dtype=jnp.int32)
    u = jnp.array([1, 2, 3], dtype=jnp.uint32)
    c = jnp.array([1 + 2j, 3 - 4j, -2 + 0.5j], dtype=jnp.complex64)
    matrix = jnp.array([[4, 1], [1, 3]], dtype=jnp.float32)
    result = []

    def add(name, fn, args, expected=None, *, deterministic=True):
        result.append(Case(name, fn, tuple(args), expected or name, deterministic))

    for name in (
        "abs acos acosh asin asinh atan atanh bessel_i0e bessel_i1e cbrt ceil cos cosh "
        "digamma erf erf_inv erfc exp exp2 expm1 floor is_finite lgamma log log1p log2 "
        "logistic neg one_minus_square round rsqrt sign sin sinh sqrt square tan tanh"
    ).split():
        add(name, getattr(lax, name), (f,))
    for name in (
        "add atan2 div igamma igamma_grad_a igammac max min mul nextafter pow rem sub zeta"
    ).split():
        add(name, getattr(lax, name), (f, g))
    for name in ("eq ge gt le lt ne").split():
        add(name, getattr(lax, name), (f, g))
    for name in ("clz population_count").split():
        add(name, getattr(lax, name), (u,))
    for name, api in (("and", "bitwise_and"), ("or", "bitwise_or"), ("xor", "bitwise_xor")):
        add(name, getattr(lax, api), (i, i + 1))
    add("not", lax.bitwise_not, (i,))
    for name in ("mulhi shift_left shift_right_arithmetic shift_right_logical").split():
        add(name, getattr(lax, name), (i, i))
    for name in ("conj imag real").split():
        add(name, getattr(lax, name), (c,))
    add("complex", lax.complex, (f, g))
    add("regularized_incomplete_beta", lax.betainc, (g, g, f))
    add("polygamma", lax.polygamma, (jnp.ones_like(f), g))
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
    add("dynamic_slice", lambda x, start: lax.dynamic_slice(x, (start,), (2,)), (f, jnp.int32(1)))
    add(
        "dynamic_update_slice",
        lambda x, y, start: lax.dynamic_update_slice(x, y, (start,)),
        (f, g[:1], jnp.int32(1)),
    )
    add("pad", partial(lax.pad, padding_config=((1, 2, 1),)), (f, jnp.float32(0)))
    add("iota", lambda: lax.iota(jnp.int32, 4), ())
    add("select_n", lax.select_n, (jnp.array([0, 1, 0]), f, g))
    for name in ("reduce_sum reduce_prod reduce_max reduce_min").split():
        add(name, partial(getattr(lax, name), axes=(0,)), (matrix,))
    for name in ("reduce_and reduce_or reduce_xor").split():
        add(name, partial(getattr(lax, name), axes=(0,)), (i,))
    for name in ("cumlogsumexp cummax cummin cumprod cumsum").split():
        add(name, partial(getattr(lax, name), axis=0, reverse=True), (f,))
    for name in ("argmax argmin").split():
        add(name, partial(getattr(lax, name), axis=0, index_dtype=jnp.int32), (f,))
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
    add("approx_top_k", partial(lax.approx_max_k, k=2), (f,))
    add("sort", partial(lax.sort, dimension=0, is_stable=True, num_keys=1), ((g, i),))
    add("copy", lax.copy_p.bind, (f,))
    add("stage", lax.stage_p.bind, (f,))
    add("stop_gradient", lax.stop_gradient, (f,))
    add("optimization_barrier", lax.optimization_barrier, ((f, i),))
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
    add("create_token", lambda x: lax.create_token(x), (f,))
    add("after_all", lambda x: lax.after_all(lax.create_token(x), lax.create_token(x)), (f,))
    add("cond", lambda x: lax.cond(x[0] > 0, lax.sin, lax.cos, x), (f,))
    add("scan", lambda x: lax.scan(lambda a, b: (a + b, a * b), jnp.float32(0), x), (f,))
    add("while", lambda x: lax.while_loop(lambda y: y < 3, lambda y: y + 1, x), (jnp.int32(0),))
    add(
        "custom_linear_solve",
        lambda x: lax.custom_linear_solve(
            lambda y: y * 2, x, solve=lambda _, b: b / 2, symmetric=True
        ),
        (f,),
    )
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
    for name in ("eq_to", "le_to", "lt_to"):
        add(name, getattr(lax, name + "_p").bind, (f, g))
    add("device_put", partial(jax.device_put, device=jax.devices("cpu")[0]), (f,))
    add(
        "sharding_constraint",
        partial(
            lax.with_sharding_constraint,
            shardings=jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0]),
        ),
        (f,),
    )
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

    cpu_mesh = Mesh(np.array(jax.devices("cpu")), ("workers",))
    count = len(jax.devices("cpu"))
    collective_input = jnp.arange(count * 4, dtype=jnp.float32).reshape(count * 2, 2)
    for name in ("psum", "pmin", "pmax"):
        body = partial(getattr(lax, name), axis_name="workers")
        add(
            name,
            jax.shard_map(
                body, mesh=cpu_mesh, in_specs=P("workers", None), out_specs=P(), check_vma=False
            ),
            (collective_input,),
        )
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
    add("dce_sink", lambda x: (lax.dce_sink(x), x)[1], (f,))
    add("dce_sink_mlir", lambda x: (lax.dce_sink(x, prevent_mlir_dce=True), x)[1], (f,), "dce_sink")
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
