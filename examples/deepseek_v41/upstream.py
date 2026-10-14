"""校验上游文件后加载原始定义，保留其装饰器、签名和计算体。"""

import __future__

import ast
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "deepseek_v41"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
REVISION = MANIFEST["revision"]


def verify_snapshot():
    """逐文件校验固定来源，包括许可证和实际模型调用上下文。"""
    for entry in MANIFEST["files"]:
        name = entry["path"]
        path = ROOT / (name if name == "LICENSE" else name + ".txt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"upstream snapshot checksum mismatch: {name}")


@functools.cache
def original():
    """只加载 mHC 的完整定义；避免引入无关的模型加载与量化依赖。"""
    verify_snapshot()
    path = ROOT / "deepseek_v41/reference.py.txt"
    names = {"hc_split_sinkhorn", "hc_mixes", "hc_pre", "hc_post"}
    definitions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise ValueError("incomplete upstream mHC definitions")
    namespace = {"jax": jax, "jnp": jnp, "functools": functools}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return SimpleNamespace(**{name: namespace[name] for name in names})
