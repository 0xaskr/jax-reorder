# Array mappings and explicit traversal adapted from johnviljoen/jax2sympy,
# revision 2cb6a6568716b26af0abd542a941f87ae0580e25.
# MIT License
#
# Copyright (c) 2026 John Viljoen
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Explicit per-element JAX conversion retained for reference and future adapters.

This interface preserves the original formula-only baseline. Production comparison
uses SymbolicBuilder, which also tracks definition conditions and resource bounds.
"""

# ruff: noqa: E731, B007, B905, E741
import jax
import numpy as np
import sympy as sy
from jax.extend.core import Literal

from .algebra_symbolic import _exact_array as exact_array

# elementwise
_sym_add = lambda a, b: a + b
_sym_sub = lambda a, b: a - b
_sym_mul = lambda a, b: a * b
_sym_div = lambda a, b: a / b
_sym_pow = lambda a, b: a**b
_sym_integer_pow = lambda a, exp: a**exp
_sym_neg = lambda a: -a
_sym_exp = lambda a: np.vectorize(sy.exp, otypes=[object])(a)
_sym_log = lambda a: np.vectorize(sy.log)(a)
_sym_sin = lambda a: np.vectorize(sy.sin)(a)
_sym_cos = lambda a: np.vectorize(sy.cos)(a)
_sym_tan = lambda a: np.vectorize(sy.tan)(a)
_sym_asin = lambda a: np.vectorize(sy.asin)(a)
_sym_acos = lambda a: np.vectorize(sy.acos)(a)
_sym_atan = lambda a: np.vectorize(sy.atan)(a)
_sym_atan2 = lambda a, b: np.vectorize(sy.atan2)(
    a, b
)  # UNTESTED (added 2026-07-14, no numeric regression coverage yet)
_sym_sinh = lambda a: np.vectorize(sy.sinh)(a)
_sym_cosh = lambda a: np.vectorize(sy.cosh)(a)
_sym_tanh = lambda a: np.vectorize(sy.tanh)(a)
_sym_sqrt = lambda a: np.vectorize(sy.sqrt)(a)
_sym_sign = lambda a: np.vectorize(sy.sign)(a)
_sym_square = lambda a: a**2
_sym_abs = lambda a: np.vectorize(sy.Abs)(a)
_sym_eq = lambda a, b: np.vectorize(sy.Eq)(a, b)
_sym_ne = lambda a, b: np.vectorize(sy.Ne)(a, b)
_sym_lt = lambda a, b: np.vectorize(sy.StrictLessThan)(a, b)
_sym_le = lambda a, b: np.vectorize(sy.LessThan)(a, b)
_sym_gt = lambda a, b: np.vectorize(sy.StrictGreaterThan)(a, b)
_sym_ge = lambda a, b: np.vectorize(sy.GreaterThan)(a, b)
_sym_and = lambda a, b: np.vectorize(sy.And)(a, b)
_sym_or = lambda a, b: np.vectorize(sy.Or)(a, b)
_sym_not = lambda a: np.vectorize(sy.Not)(a)
_sym_max = lambda a, b: np.vectorize(sy.Max)(a, b)
_sym_min = lambda a, b: np.vectorize(sy.Min)(a, b)
_sym_logaddexp = lambda a, b: np.vectorize(lambda x, y: sy.log(sy.exp(x) + sy.exp(y)))(a, b)


# array
def _sym_reduce_sum(a, eqn):
    assert len(a) == 1
    return np.sum(a[0], axis=eqn.params["axes"])


def _sym_reduce_or(a, eqn):
    # print("WARNING: _sym_reduce_or not validated and is in use")
    assert len(a) == 1
    arr = a[0]
    axes = eqn.params["axes"]
    if axes == ():
        # Reduce over all axes - return single Or of all elements
        return sy.Or(*arr.flat)
    else:
        return np.apply_along_axis(lambda x: sy.Or(*x), axes[0], arr)


def _sym_transpose(a, eqn):
    # print("WARNING: _sym_transpose not validated and is in use")
    perm = (
        eqn.params.get("permutation")  # canonical in JAX
        or eqn.params.get("axes")  # NumPy nomenclature, just in case
        or eqn.params.get("dims")  # older/internal spelling
    )
    if perm is None:
        perm = tuple(range(a.ndim))[::-1]
    return np.transpose(a, axes=perm)


# add any for our purposes is just add
def _sym_add_any(inexprs):
    # Use broadcasting to add arrays sequentially
    result = inexprs[0]
    for array in inexprs[1:]:
        result = np.add(result, array)  # np.add handles broadcasting
    return result


def _sym_select_n(inexprs):
    # boolean = inexprs[0]
    # true_arr = inexprs[1]
    # false_arr = inexprs[2]
    # return np.where(boolean, true_arr, false_arr)

    pred, false_val, true_val = inexprs

    # Fast-path: real booleans → we can still delegate to NumPy
    if pred.dtype != object and pred.dtype == bool:
        return np.where(pred, true_val, false_val)

    # print("WARNING: _sym_select_n symbolic path not validated and in use")

    # General symbolic path: element-wise Piecewise
    piecewise_fn = np.vectorize(
        lambda p, t, f: sy.Piecewise((t, p), (f, True)),
        otypes=[object],
    )
    return piecewise_fn(pred, true_val, false_val)


def _sym_pad(inexprs, eqn):

    assert len(inexprs) == 2

    inexpr = inexprs[0]
    padding_value = inexprs[1]
    padding_config = eqn.params["padding_config"]

    # Create the padded shape
    padded_shape = [
        low + (inexpr.shape[i] - 1) * (interior + 1) + 1 + high
        for i, (low, high, interior) in enumerate(padding_config)
    ]

    # Create an array filled with the padding value
    result = np.full(padded_shape, padding_value, dtype=inexpr.dtype)

    # Calculate slices to place the original array into the padded result
    insert_slices = tuple(
        slice(low, low + inexpr.shape[i] * (interior + 1), interior + 1)
        for i, (low, high, interior) in enumerate(padding_config)
    )

    # Place the operand in the result array
    result[insert_slices] = inexpr
    return result


_sym_convert_element_type = lambda a: a


def _sym_reshape(a, new_sizes, dimensions):
    # if 0 in dimensions:
    #     print("WARNING: using _sym_reshape with 0 dimensions is not validated and is happening")
    if dimensions is None or 0 in dimensions:
        a = a.reshape(*new_sizes)
    else:
        raise NotImplementedError
    return a


def _sym_squeeze(a, dimensions):
    slices = [0 if i in dimensions else slice(None) for i in range(a.ndim)]
    return a[tuple(slices)]


# builds an array on device - this can be simplified
def _sym_iota(eqn):
    shape = eqn.params["shape"]
    dimension = eqn.params["dimension"]
    return (
        np.arange(shape[dimension])
        .reshape([1 if i != dimension else shape[dimension] for i in range(len(shape))])
        .repeat(np.prod(shape) // shape[dimension], axis=dimension)
        .reshape(shape)
    )


def _sym_split(inexprs, eqn):
    # print("WARNING: check functionality of split")
    sizes = eqn.params["sizes"]
    axis = eqn.params["axis"]
    assert len(inexprs) == 1
    operand = inexprs[0]
    indices = np.cumsum(sizes[:-1])  # Calculate split indices
    return np.split(operand, indices, axis=axis)


def _sym_slice(a, start_indices, limit_indices, strides):
    axis_slice = []
    if strides is not None:
        for start, limit, stride in zip(start_indices, limit_indices, strides):
            axis_slice.append([start, limit, stride])
    else:
        for start, limit in zip(start_indices, limit_indices):
            axis_slice.append([start, limit, 1])
    return a[tuple([slice(s[0], s[1], s[2]) for s in axis_slice])]


def _sym_dynamic_slice(inexprs, eqn):
    # inexprs[0] is the array, inexprs[1:] are the start indices
    a = inexprs[0]
    start_indices = inexprs[1:]
    slice_sizes = eqn.params["slice_sizes"]
    # Build slices: start[i] : start[i] + size[i]
    slices = tuple(
        slice(
            int(start.flat[0]) if hasattr(start, "flat") else int(start),
            int(start.flat[0]) + size if hasattr(start, "flat") else int(start) + size,
        )
        for start, size in zip(start_indices, slice_sizes)
    )
    return a[slices]


def _sym_dot_general(a, b, eqn):
    (ca, cb), (ba, bb) = eqn.params["dimension_numbers"]
    la, lb = list(range(a.ndim)), list(range(a.ndim, a.ndim + b.ndim))
    for left, right in zip(ca + ba, cb + bb, strict=True):
        lb[right] = la[left]
    output = [la[i] for i in ba]
    output += [la[i] for i in range(a.ndim) if i not in ca + ba]
    output += [lb[i] for i in range(b.ndim) if i not in cb + bb]
    return np.einsum(a, la, b, lb, output)


def _sym_reduce_extreme(a, eqn, function):
    result = a[0]
    for axis in sorted(eqn.params["axes"], reverse=True):
        result = np.apply_along_axis(lambda row: function(*row), axis, result)
    return result


def _sym_gather(operand, start_indices, dimension_numbers, slice_sizes, mode):

    # Unpack dimension numbers
    offset_dims = dimension_numbers.offset_dims
    # only can handle one offset dim rn
    assert len(offset_dims) == 1
    collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
    start_index_map = dimension_numbers.start_index_map

    # Handle index_vector_dim (assume last dimension if not specified)
    index_vector_dim = len(start_indices.shape) - 1

    # Initialize the output shape
    batch_dims = [dim for dim in range(len(start_indices.shape)) if dim != index_vector_dim]
    batch_shape = [start_indices.shape[d] for d in batch_dims]
    offset_shape = [
        slice_sizes[dim] for dim in range(len(slice_sizes)) if dim not in collapsed_slice_dims
    ]
    output_shape = batch_shape + offset_shape
    output = np.empty(output_shape, dtype=operand.dtype)

    # Precompute mapping for remapped_offset_dims
    remapped_offset_dims = []
    operand_rank = len(operand.shape)
    assert operand_rank == 1
    available_dims = [dim for dim in range(operand_rank) if dim not in collapsed_slice_dims]
    assert len(offset_dims) == 1
    for idx, offset_dim in enumerate(offset_dims):
        remapped_offset_dims.append(available_dims[idx])

    # Iterate over all indices in the output array
    for out_index in np.ndindex(*output_shape):
        # Step 1: Extract batch indices G
        G = [out_index[d] for d in range(len(batch_dims))]

        # Step 2: Compute S using start_indices
        S = [0] * len(start_index_map)
        for i, map_dim in enumerate(start_index_map):
            combined_index = tuple(G[:index_vector_dim] + [i] + G[index_vector_dim:])
            S[i] = start_indices[combined_index]

        # Step 3: Compute Sin
        Sin = [0] * operand_rank
        for i, map_dim in enumerate(start_index_map):
            Sin[map_dim] = S[i]

        # Step 4: Compute Oin
        Oin = [0] * operand_rank
        for i, offset_dim in enumerate(offset_dims):
            Oin[remapped_offset_dims[i]] = out_index[len(batch_dims) + i]

        # NOTE MODIFIED: In = tuple(Sin[d] + Oin[d] for d in range(operand_rank))
        # Step 5: Compute In = Sin + Oin
        In = tuple(Sin[d] + Oin[d] for d in range(operand_rank))
        In = (In[0] - offset_dim * operand.shape[0],)  # the offset

        # Step 6: Handle out-of-bounds indices
        if mode == jax.lax.GatherScatterMode.CLIP:
            In = tuple(max(0, min(idx, dim - 1)) for idx, dim in zip(In, operand.shape))
        elif mode == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
            pass  # Assume indices are valid
        else:
            if any(idx < 0 or idx >= dim for idx, dim in zip(In, operand.shape)):
                raise IndexError(f"Out-of-bounds index: {In}")

        # Step 7: Populate the output array
        output[out_index] = operand[In]

    return output


def _sym_scatter(operand, scatter_indices, updates, dimension_numbers, mode, combine=None):
    # combine=None replaces (lax.scatter); scatter-add passes an additive
    # combine so colliding updates accumulate.

    # Unpack dimension numbers
    update_window_dims = dimension_numbers.update_window_dims
    inserted_window_dims = dimension_numbers.inserted_window_dims
    scatter_dims_to_operand_dims = dimension_numbers.scatter_dims_to_operand_dims

    # Initialize output array with operand contents
    output = operand.copy()

    # Determine update_scatter_dims: dimensions in updates not used for the window
    update_scatter_dims = [i for i in range(updates.ndim) if i not in update_window_dims]

    def build_window_dims_to_operand_dims(update_window_dims, inserted_window_dims, operand_rank):
        """
        Build mapping from update_window_dims to operand dimensions, excluding inserted_window_dims.
        """
        # Get all operand dimensions excluding inserted_window_dims
        available_dims = [i for i in range(operand_rank) if i not in inserted_window_dims]
        mapping = {dim: available_dims[idx] for idx, dim in enumerate(update_window_dims)}
        return mapping

    for update_index in np.ndindex(*updates.shape):
        # print(update_index)
        # Step 1: Extract G from update_index
        G = [update_index[dim] for dim in update_scatter_dims]

        # Step 2: Compute S using scatter_indices and Combine(G, i)
        S = [0] * len(scatter_dims_to_operand_dims)
        for i, scatter_dim in enumerate(scatter_dims_to_operand_dims):
            # Combine G and scatter_indices
            combined = tuple(G[:scatter_dim] + [i] + G[scatter_dim:])
            S[i] = scatter_indices[combined]

        # Step 3: Compute Sin
        Sin = [0] * operand.ndim
        for i, scatter_dim in enumerate(scatter_dims_to_operand_dims):
            Sin[scatter_dim] = S[i]

        # Step 4: Compute Win
        Win = [0] * operand.ndim
        window_mapping = build_window_dims_to_operand_dims(
            update_window_dims, inserted_window_dims, operand.ndim
        )
        for i in update_window_dims:
            Win[window_mapping[i]] = update_index[i]

        # Step 5: Calculate final index I = Win + Sin
        I = tuple(int(Win[d] + Sin[d]) for d in range(operand.ndim))

        # Step 6: Handle out-of-bounds indices
        if mode == jax.lax.GatherScatterMode.CLIP:
            I = tuple(max(0, min(idx, dim - 1)) for idx, dim in zip(I, operand.shape))
        elif mode == jax.lax.GatherScatterMode.FILL_OR_DROP:
            if any(idx < 0 or idx >= dim for idx, dim in zip(I, operand.shape)):
                continue  # Skip out-of-bounds updates
        elif mode == jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS:
            pass  # Assume all indices are in bounds
        else:
            if any(idx < 0 or idx >= dim for idx, dim in zip(I, operand.shape)):
                raise IndexError(f"Out-of-bounds index: {I}")

        # Step 7: Apply the update
        if combine is None:
            output[I] = updates[update_index]
        else:
            output[I] = combine(output[I], updates[update_index])

    return output


def _sym_broadcast_in_dim(a, shape, broadcast_dimensions):
    assert len(a) == 1
    a = a[0]
    reshaped_shape = [1] * len(shape)
    for i, dim in enumerate(broadcast_dimensions):
        reshaped_shape[dim] = a.shape[i]
    a_reshaped = np.reshape(a, reshaped_shape)
    return np.broadcast_to(a_reshaped, shape)


def _sym_concatenate(arr_list, dimension=0):
    return np.concat(arr_list, axis=dimension)


def _sym_stack(arr_list, axis=0):
    # jax >= 0.9 emits a first-class 'stack' primitive (jnp.stack no longer
    # lowers to concatenate+reshape as it did on <= 0.8). Validated by
    # tests/test_jax_upgrade_surface.py against dense jax autodiff.
    return np.stack(arr_list, axis=axis)


def _sym_cumsum(a, eqn):
    # UNTESTED (added 2026-07-14, no numeric regression coverage yet)
    assert len(a) == 1
    arr = a[0]
    axis = eqn.params["axis"]
    reverse = eqn.params.get("reverse", False)
    if reverse:
        arr = np.flip(arr, axis=axis)
    out = np.cumsum(arr, axis=axis)
    return np.flip(out, axis=axis) if reverse else out


primitive_to_sympy_op = {
    "add": lambda inexprs, eqn: _sym_add(*inexprs),
    "sub": lambda inexprs, eqn: _sym_sub(*inexprs),
    "mul": lambda inexprs, eqn: _sym_mul(*inexprs),
    "div": lambda inexprs, eqn: _sym_div(*inexprs),
    "neg": lambda inexprs, eqn: _sym_neg(*inexprs),
    "pow": lambda inexprs, eqn: _sym_pow(*inexprs),
    "integer_pow": lambda inexprs, eqn: _sym_integer_pow(inexprs[0], eqn.params["y"]),
    "sign": lambda inexprs, eqn: _sym_sign(*inexprs),
    "square": lambda inexprs, eqn: _sym_square(*inexprs),
    "abs": lambda inexprs, eqn: _sym_abs(*inexprs),
    "exp": lambda inexprs, eqn: _sym_exp(*inexprs),
    "log": lambda inexprs, eqn: _sym_log(*inexprs),
    "sin": lambda inexprs, eqn: _sym_sin(*inexprs),
    "cos": lambda inexprs, eqn: _sym_cos(*inexprs),
    "tan": lambda inexprs, eqn: _sym_tan(*inexprs),
    "asin": lambda inexprs, eqn: _sym_asin(*inexprs),
    "acos": lambda inexprs, eqn: _sym_acos(*inexprs),
    "atan": lambda inexprs, eqn: _sym_atan(*inexprs),
    "sinh": lambda inexprs, eqn: _sym_sinh(*inexprs),
    "cosh": lambda inexprs, eqn: _sym_cosh(*inexprs),
    "tanh": lambda inexprs, eqn: _sym_tanh(*inexprs),
    "sqrt": lambda inexprs, eqn: _sym_sqrt(*inexprs),
    "eq": lambda inexprs, eqn: _sym_eq(*inexprs),
    "ne": lambda inexprs, eqn: _sym_ne(*inexprs),
    "lt": lambda inexprs, eqn: _sym_lt(*inexprs),
    "le": lambda inexprs, eqn: _sym_le(*inexprs),
    "gt": lambda inexprs, eqn: _sym_gt(*inexprs),
    "ge": lambda inexprs, eqn: _sym_ge(*inexprs),
    "not": lambda inexprs, eqn: _sym_not(*inexprs),
    "and": lambda inexprs, eqn: _sym_and(*inexprs),
    "or": lambda inexprs, eqn: _sym_or(*inexprs),
    "max": lambda inexprs, eqn: _sym_max(*inexprs),
    "min": lambda inexprs, eqn: _sym_min(*inexprs),
    "logaddexp": lambda inexprs, eqn: _sym_logaddexp(*inexprs),
    "broadcast_in_dim": lambda inexprs, eqn: _sym_broadcast_in_dim(
        inexprs,
        shape=eqn.params["shape"],
        broadcast_dimensions=eqn.params["broadcast_dimensions"],
    ),
    "concatenate": lambda inexprs, eqn: _sym_concatenate(inexprs, eqn.params["dimension"]),
    "stack": lambda inexprs, eqn: _sym_stack(inexprs, eqn.params["axis"]),
    "dynamic_slice": lambda inexprs, eqn: _sym_dynamic_slice(inexprs, eqn),
    "slice": lambda inexprs, eqn: _sym_slice(
        *inexprs,
        eqn.params["start_indices"],
        eqn.params["limit_indices"],
        eqn.params.get("strides", None),
    ),
    "squeeze": lambda inexprs, eqn: _sym_squeeze(*inexprs, eqn.params["dimensions"]),
    "scatter": lambda inexprs, eqn: _sym_scatter(
        *inexprs,
        dimension_numbers=eqn.params["dimension_numbers"],
        mode=eqn.params["mode"],
    ),
    # UNTESTED (added 2026-07-14, no numeric regression coverage yet)
    "scatter-add": lambda inexprs, eqn: _sym_scatter(
        *inexprs,
        dimension_numbers=eqn.params["dimension_numbers"],
        mode=eqn.params["mode"],
        combine=lambda old, new: old + new,
    ),
    "gather": lambda inexprs, eqn: _sym_gather(
        *inexprs,
        dimension_numbers=eqn.params["dimension_numbers"],
        slice_sizes=eqn.params["slice_sizes"],
        mode=eqn.params["mode"],
    ),
    "add_any": lambda inexprs, eqn: _sym_add_any(inexprs),
    "pad": lambda inexprs, eqn: _sym_pad(inexprs, eqn),
    "select_n": lambda inexprs, eqn: _sym_select_n(inexprs),
    "reduce_sum": lambda inexprs, eqn: _sym_reduce_sum(inexprs, eqn),
    "reduce_max": lambda inexprs, eqn: _sym_reduce_extreme(inexprs, eqn, sy.Max),
    "reduce_min": lambda inexprs, eqn: _sym_reduce_extreme(inexprs, eqn, sy.Min),
    "reduce_or": lambda inexprs, eqn: _sym_reduce_or(inexprs, eqn),
    # UNTESTED (added 2026-07-14, no numeric regression coverage yet)
    "atan2": lambda inexprs, eqn: _sym_atan2(*inexprs),
    # UNTESTED (added 2026-07-14, no numeric regression coverage yet)
    "cumsum": lambda inexprs, eqn: _sym_cumsum(inexprs, eqn),
    "debug_print": lambda inexprs, eqn: [],  # no-op, side effect only
    "iota": lambda inexprs, eqn: _sym_iota(eqn),
    "split": lambda inexprs, eqn: _sym_split(inexprs, eqn),
    "convert_element_type": lambda inexprs, eqn: _sym_convert_element_type(inexprs[0]),
    "dot_general": lambda inexprs, eqn: _sym_dot_general(inexprs[0], inexprs[1], eqn),
    "transpose": lambda inexprs, eqn: _sym_transpose(inexprs[0], eqn),
    "tile": lambda inexprs, eqn: np.tile(inexprs[0], eqn.params["reps"]),
    "reshape": lambda inexprs, eqn: _sym_reshape(
        *inexprs,  # the expression being reshaped
        eqn.params["new_sizes"],
        eqn.params["dimensions"],
    ),
}


def get_sym_invar(invars, var2sym):
    inexprs = []
    for inv in invars:
        if isinstance(inv, Literal):
            inexprs.append(exact_array(inv.val))
        else:
            inexprs.append(var2sym[inv])
    return inexprs


def get_sym_outvar(inexprs, eqn):
    prim_name = eqn.primitive.name
    if prim_name not in primitive_to_sympy_op:
        raise NotImplementedError(f"No Sympy translation rule for JAX primitive '{prim_name}'.")
    out_expr = primitive_to_sympy_op[prim_name](inexprs, eqn)
    if not isinstance(out_expr, list):  # multi var outputs are put in a list
        return [out_expr]
    else:
        return out_expr


def jaxpr_to_sympy_expressions(jaxpr, var2sym=None, x_cnt=0, c_cnt=0):
    if var2sym is None:
        var2sym = {}

    def append_vars(vars, var2sym, sym_name, cnt):
        for i, var in enumerate(vars):  # go through all inputs to the jaxpr
            assert var not in var2sym  # double check we are not overwriting anything
            shape = var.aval.shape  # the abstract shape of the input
            num_elements = int(np.prod(shape))
            kind = np.dtype(var.aval.dtype).kind
            assumptions = (
                {"real": True} if kind == "f" else {"integer": True} if kind in "iu" else {}
            )
            symbols_list = sy.symbols(
                [f"{sym_name}{i + cnt}_{j}" for j in range(num_elements)], **assumptions
            )
            symbolic_array = np.array(symbols_list).reshape(shape)
            var2sym[var] = symbolic_array
        return var2sym, cnt + len(vars)

    def substitute_symbol(symbol, sym_map):
        return symbol.subs(sym_map)

    var2sym, x_cnt = append_vars(jaxpr.jaxpr.invars, var2sym, "x", cnt=x_cnt)
    for var, value in zip(jaxpr.jaxpr.constvars, jaxpr.consts, strict=True):
        var2sym[var] = exact_array(value)
    c_cnt += len(jaxpr.jaxpr.constvars)

    for i, eqn in enumerate(jaxpr.eqns):
        # print(f"eqn number: {i}")
        # print(f"eqn: {eqn}")
        if eqn.primitive.name in ("pjit", "jit"):
            _jaxpr = eqn.params.get("jaxpr", None)
            sub_syms_out, sub_var2sym, x_cnt, c_cnt = jaxpr_to_sympy_expressions(
                _jaxpr, var2sym={}, x_cnt=x_cnt, c_cnt=c_cnt
            )
            syms_in = get_sym_invar(eqn.invars, var2sym)
            iterator = iter(sub_var2sym)
            sub_syms_in = [sub_var2sym[next(iterator)] for _ in range(len(syms_in))]
            sym_map = {}
            for sym_in, sub_sym_in in zip(syms_in, sub_syms_in):
                sym_map.update({key: value for key, value in zip(sub_sym_in.flat, sym_in.flat)})
            sub_syms_out = [np.vectorize(substitute_symbol)(f, sym_map) for f in sub_syms_out]
            exprs = sub_syms_out
        else:
            inexprs = get_sym_invar(eqn.invars, var2sym)
            exprs = get_sym_outvar(inexprs, eqn)
        exprs = [
            np.asarray(expr) for expr in exprs
        ]  # check we dont lose the array with squeezing and indexing
        for outvar, expr in zip(eqn.outvars, exprs):
            # print(f"expression added to var2sym: {expr} under: {outvar}")
            assert outvar.aval.shape == expr.shape
            var2sym[outvar] = expr

    out_syms = get_sym_invar(jaxpr.jaxpr.outvars, var2sym)
    return out_syms, var2sym, x_cnt, c_cnt
