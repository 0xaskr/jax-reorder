#!/usr/bin/env bash
set -euo pipefail

# 只验证固定上游 DeepSeek-V4.1-Flash mHC 层；参数直接交给层验证入口。
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/jax-reorder-uv-cache}"
export JAX_PLATFORMS=cpu

cd -- "$repo_root"
exec uv run --locked python -m examples.deepseek_v41.runner \
    --json "${TMPDIR:-/tmp}/jax-reorder-deepseek-v41.json" "$@"
