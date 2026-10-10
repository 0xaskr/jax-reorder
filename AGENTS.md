# Working on jax-reorder

Read `.agents/README.md` for the project requirements and semantic boundaries.

- Use Python 3.12 and uv. Run `uv sync --locked` to prepare the environment.
- If the sandbox makes the default uv cache read-only, set
  `UV_CACHE_DIR=/tmp/jax-reorder-uv-cache` for uv commands.
- Keep implementation in `src/jax_reorder/` and runnable examples in `examples/`.
- Use `uv add` / `uv add --dev` for dependencies and commit `uv.lock`.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and
  `uv run python examples/inspect_jaxpr.py` before committing.
- Add meaningful tests under `tests/` when implementing canonicalization; then
  enable `uv run pytest` in CI. There are no algorithm tests at the scaffold stage.
- Preserve dependency order, input/output ordering, constants, dtypes, shapes,
  primitive parameters, and nested computation semantics.
- Keep structural matches separate from matches requiring algebraic rewrites.
  Floating-point commutative or associative rewrites must emit a warning.
- Unsupported effects or opaque parameters must fail explicitly; never silently
  ignore semantics to obtain a match.
- Do not claim general Python equivalence or bitwise floating-point equivalence
  from algebraic normalization.
- Do not commit virtual environments, credentials, or generated build artifacts.
