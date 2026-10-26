"""校验上游文件后加载原始定义，保留其装饰器、签名和计算体。

固定来源是 Inferact/tpu-megakernels PR #3 的 DeepSeek-V4.1-Flash 参考实现；
提交、PR 状态与逐文件 SHA-256 记录在 tests/fixtures/deepseek_v41/manifest.json。
本模块只加载 mHC 的四个函数并核对来源，不验证数值等价性，也不代表上游
其他函数或配置可用。运行入口见 `python -m tests.deepseek_v41.runner`。
"""

import __future__

import ast
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

# 快照目录与 manifest 均随仓库分发，换版本必须同时更新清单和许可证。
ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "deepseek_v41"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
# 报告引用该提交，保证每条结论都能追溯到同一份源码。
REVISION = MANIFEST["revision"]


def verify_snapshot():
    """逐文件校验固定来源，包括许可证和实际模型调用上下文。

    无参数；按 manifest 列出的每个文件重新计算 SHA-256，不返回摘要。
    校验失败抛出 ValueError，调用方据此停止比较；校验通过只表示来源
    未变，不构成任何数值或等价性证据。
    """
    for entry in MANIFEST["files"]:
        name = entry["path"]
        # 许可证按原文件名保存，其余源码追加 .txt，避免被当作可导入源码。
        path = ROOT / (name if name == "LICENSE" else name + ".txt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"upstream snapshot checksum mismatch: {name}")


@functools.cache
def original():
    """只加载 mHC 的完整定义；避免引入无关的模型加载与量化依赖。

    返回 SimpleNamespace，包含 hc_split_sinkhorn、hc_mixes、hc_pre 和
    hc_post 四个原始函数。首次调用先执行 verify_snapshot，再用原装饰器
    执行抽取出的 AST；结果由 functools.cache 缓存，不重复校验或执行。
    """
    verify_snapshot()
    path = ROOT / "deepseek_v41/reference.py.txt"
    # post 依赖 hc_mixes 生成的系数；四者一起加载才能保留真实调用上下文。
    names = {"hc_split_sinkhorn", "hc_mixes", "hc_pre", "hc_post"}
    definitions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in definitions} != names:
        raise ValueError("incomplete upstream mHC definitions")
    # 原文件顶层只用到这三个模块；缺失依赖应在执行时显式失败而不是被静默替代。
    namespace = {"jax": jax, "jnp": jnp, "functools": functools}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(path),
            "exec",
            # 源码启用 from __future__ import annotations，编译时须保持同一语义。
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return SimpleNamespace(**{name: namespace[name] for name in names})
