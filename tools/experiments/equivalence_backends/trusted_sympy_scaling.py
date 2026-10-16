"""Compare explicit and indexed construction, never executing the traced JAX program."""

import json
import time
import tracemalloc

import jax
import jax.numpy as jnp
import numpy as np
import sympy as sp

from jax_reorder._vendor.jax2sympy import jaxpr_to_sympy_expressions
from jax_reorder.algebra_symbolic import SymbolicBuilder
from jax_reorder.algebra_transport import freeze


def main():
    jax.devices()
    rows = []
    for size in (8, 32, 128, 512):
        graph = jax.make_jaxpr(lambda x: jnp.sum(x * x))(jax.ShapeDtypeStruct((size,), jnp.float32))
        frozen = freeze(graph)
        for mode in ("explicit", "indexed"):
            tracemalloc.start()
            started = time.perf_counter()
            if mode == "explicit":
                outputs = jaxpr_to_sympy_expressions(graph)[0]
                expression = sp.sympify(np.asarray(outputs[0]).item())
            else:
                expression = SymbolicBuilder().graph(frozen)[0].expression
            elapsed = time.perf_counter() - started
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rows.append(
                {
                    "size": size,
                    "mode": mode,
                    "milliseconds": round(elapsed * 1000, 3),
                    "python_peak_bytes": peak,
                    "unique_nodes": len(set(sp.preorder_traversal(expression))),
                    "indexed_reads": len(expression.atoms(sp.Indexed)),
                    "free_symbols": len(expression.free_symbols),
                }
            )
    print(json.dumps({"jax": jax.__version__, "sympy": sp.__version__, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
