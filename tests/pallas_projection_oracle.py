"""Independent NumPy evaluation of a projected value at concrete output points.

Tensor operations execute as whole NumPy tensors, separate from both the core
state interpreter and the SymPy builder. This is numerical validation, not an
equivalence proof.
"""

from functools import reduce

import numpy as np

from jax_reorder.core.index_domain import IndexExpr
from jax_reorder.core.memory import Predicate, ValueExpr


def at_point(root, inputs, point):
    inputs = tuple(np.asarray(x) for x in inputs)

    def scope(bindings):
        cache = {}

        def emit(value):
            if id(value) in cache:
                return cache[id(value)]
            if type(value) is bool:
                return value
            if isinstance(value, IndexExpr):
                args = [emit(a) for a in value.args]
                result = {
                    "constant": lambda: int(value.value),
                    "axis": lambda: bindings[str(value.value)],
                    "sum": lambda: sum(args),
                    "scale": lambda: int(value.value) * args[0],
                    "div": lambda: args[0] // int(value.value),
                    "mod": lambda: args[0] % int(value.value),
                    "lookup": lambda: np.asarray(args[1:])[args[0]],
                }[value.kind]()
            elif isinstance(value, Predicate | ValueExpr):
                name = value.operation
                params = (
                    {p.name: p.value for p in value.parameters}
                    if isinstance(value, ValueExpr)
                    else {}
                )
                if name.startswith("tensor."):
                    bodies = []
                    for body, shape, dtype, names in zip(
                        value.operands[: len(params["__input_shapes"])],
                        params["__input_shapes"],
                        params["__input_dtypes"],
                        params["__input_axes"],
                        strict=True,
                    ):
                        local = dict(zip(names, np.indices(shape), strict=True))
                        bodies.append(
                            np.asarray(
                                np.broadcast_to(scope({**bindings, **local})(body), shape),
                                dtype=dtype,
                            )
                        )
                    if name == "tensor.dot_general":
                        (ca, cb), (ba, bb) = params["dimension_numbers"]
                        a, b = bodies
                        la = list(range(a.ndim))
                        lb = list(range(a.ndim, a.ndim + b.ndim))
                        for i, j in zip(ca + ba, cb + bb, strict=True):
                            lb[j] = la[i]
                        out = (
                            [la[i] for i in ba]
                            + [la[i] for i in range(a.ndim) if i not in ca + ba]
                            + [lb[i] for i in range(b.ndim) if i not in cb + bb]
                        )
                        tensor = np.einsum(a, la, b, lb, out)
                    else:
                        fn = np.sum if name == "tensor.reduce_sum" else np.max
                        tensor = fn(bodies[0], axis=params["axes"])
                    coordinates = tuple(emit(a) for a in value.operands[len(bodies) :])
                    result = np.asarray(tensor)[coordinates]
                elif name == "literal":
                    result = np.frombuffer(value.data, dtype=value.type.dtype)[0]
                elif name == "input":
                    result = inputs[value.input][tuple(emit(a) for a in value.operands)]
                elif name == "select_n" and np.ndim(selector := emit(value.operands[0])) == 0:
                    # Storage pieces have local domains. Evaluating an unchosen
                    # tile at this output point would invent invalid coordinates.
                    result = emit(value.operands[int(selector) + 1])
                else:
                    args = [emit(a) for a in value.operands]
                    if name == "convert_element_type":
                        result = np.asarray(args[0]).astype(params["new_dtype"])
                    elif name == "select_n":
                        assert len(args) == 3
                        result = np.where(args[0], args[2], args[1])
                    elif name in ("and", "or"):
                        result = reduce(
                            np.bitwise_and if name == "and" else np.bitwise_or,
                            args,
                        )
                    elif name == "div" and np.dtype(value.type.dtype).kind in "iu":
                        # NumPy floor division and JAX lax.div differ for negative
                        # operands. Correct with an exact integer remainder.
                        a, b = args
                        with np.errstate(divide="raise", invalid="raise", over="ignore"):
                            quotient = np.floor_divide(a, b)
                            correction = (np.signbit(a) != np.signbit(b)) & (
                                np.remainder(a, b) != 0
                            )
                            result = np.asarray(quotient + correction, dtype=value.type.dtype)
                    else:
                        result = {
                            "add": np.add,
                            "sub": np.subtract,
                            "mul": np.multiply,
                            "div": np.divide,
                            "rem": np.fmod,
                            "neg": np.negative,
                            "abs": np.abs,
                            "max": np.maximum,
                            "min": np.minimum,
                            "exp": np.exp,
                            "exp2": np.exp2,
                            "eq": np.equal,
                            "ne": np.not_equal,
                            "lt": np.less,
                            "le": np.less_equal,
                            "gt": np.greater,
                            "ge": np.greater_equal,
                            "not": np.bitwise_not,
                            "xor": np.bitwise_xor,
                            "sign": np.sign,
                        }[name](*args)
            else:
                raise AssertionError(type(value))
            cache[id(value)] = result
            return result

        return emit

    return scope({f"output:{i}": k for i, k in enumerate(point)})(root)
