# Project context

## Requested scope

`jax-reorder` compares JAX programs through canonicalized jaxprs.

1. Normalize variable names and the order of independent equations.
2. Support commutative and associative normalization as an explicit algebraic mode.
3. Warn when floating-point expressions are changed by algebraic rewrites.
4. A match should provide a useful sufficient condition within its stated scope;
   a mismatch is not a proof that two programs compute different functions.

## Comparison contract to implement

- Structural mode preserves operations, operand order, data dependencies,
  constants, abstract values, primitive parameters, and ordered outputs.
- Algebraic mode has a distinct match category and records the rewrites used.
  Floating-point reassociation can cause substantial differences, overflow,
  cancellation, and different NaN behavior; warnings must not promise a small error.
- Function-level comparison must account for input and output pytree structure.
- Comparisons are specialized to the traced input signature and static arguments.
  Python side effects outside the trace and arbitrary input signatures are out of scope.
- Include closed-over constant values, not just printed jaxpr text. Any digest is
  an indexing aid, not a replacement for exact canonical representation comparison.
- Specify support for nested jaxprs, effects, and primitive parameters before
  accepting them. Unsupported constructs should produce an explicit result/error.

## Initial milestone

The repository currently contains a uv environment definition, package skeleton,
lint configuration, CI smoke check, and a runnable tracing example. The reorder
algorithm and comparison API are intentionally left for the next task.

When implementing, start with structural normalization and soundness tests, then
add audited algebraic rules and precision warnings. Record any change to this
contract in the README and tests.
