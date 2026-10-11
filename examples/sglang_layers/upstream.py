"""从已校验快照加载指定上游定义；仅提取基准，不生成或改写候选。"""

import __future__

import ast
import dataclasses
import functools
import hashlib
import json
import math
import sys
import types
from collections.abc import Mapping
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.nnx.nn import dtypes
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec

ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "sglang_jax"
MANIFEST = json.loads((ROOT / "manifest.json").read_text())
REVISION = MANIFEST["revision"]
PREFIX = "python/sgl_jax/srt/"
SOURCES = {entry["path"]: entry for entry in MANIFEST["files"]}


@functools.cache
def source(path):
    """完整文件摘要不符就失败，不能悄悄用另一个版本的原方法作基准。"""
    full = PREFIX + path
    data = (ROOT / (full + ".txt")).read_bytes()
    assert hashlib.sha256(data).hexdigest() == SOURCES[full]["sha256"], full
    return data.decode()


def load(path, name, **dependencies):
    """逐字保留定义的 AST，包括方法、分组、参数、装饰器与控制流。

    不运行服务器相关模块导入，只提供选中层实际使用的依赖。源文件位置及
    行号保持原样；候选函数完全在独立文件中手写，不走此加载入口。
    """

    def matches(item):
        if isinstance(item, (ast.ClassDef, ast.FunctionDef)):
            return item.name == name
        if isinstance(item, ast.Assign):
            return any(
                isinstance(target, ast.Name) and target.id == name for target in item.targets
            )
        return False

    definition = next(item for item in ast.parse(source(path)).body if matches(item))
    module_name = "_sglang_jax_fixture_" + path.replace("/", "_").replace(".", "_") + name
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    module.__dict__.update(
        jax=jax,
        jnp=jnp,
        np=np,
        nnx=nnx,
        lax=lax,
        math=math,
        functools=functools,
        partial=functools.partial,
        dataclasses=dataclasses,
        dataclass=dataclasses.dataclass,
        Mesh=Mesh,
        NamedSharding=NamedSharding,
        PartitionSpec=PartitionSpec,
        P=PartitionSpec,
        Precision=lax.Precision,
        **dependencies,
    )
    code = compile(
        ast.Module(body=[definition], type_ignores=[]),
        str(ROOT / (PREFIX + path + ".txt")),
        "exec",
        flags=__future__.annotations.compiler_flag,
    )
    exec(code, module.__dict__)
    return module.__dict__[name]


@functools.cache
def dependencies():
    """使用实际线性层、激活与分片辅助函数，日志配置固定为关闭调试打印。"""
    named_scope = load("utils/profiling_utils.py", "named_scope")
    log_shardings = load(
        "utils/debug_utils.py",
        "log_shardings",
        FRAMEWORK_LOG_LEVEL=0,
        FrameworkLogLevel=types.SimpleNamespace(DEBUG=1),
    )
    should_scatter = load("utils/parallel_utils.py", "should_scatter")
    return {
        "DtypeConfig": load(
            "configs/dtype_config.py",
            "DtypeConfig",
            STR_DTYPE_TO_JAX_DTYPE=load("configs/dtype_config.py", "STR_DTYPE_TO_JAX_DTYPE"),
        ),
        "named_scope": named_scope,
        "log_shardings": log_shardings,
        "LinearBase": load("layers/linear.py", "LinearBase", named_scope=named_scope),
        "GeluAndMul": load("layers/activation.py", "GeluAndMul", named_scope=named_scope),
        "ACT2FN": load("layers/activation.py", "ACT2FN"),
        "config_value": load("configs/mimo.py", "config_value", Mapping=Mapping),
        "Embed": load("layers/embeddings.py", "Embed", dtypes=dtypes, named_scope=named_scope),
        "VisionShardSpecs": load("multimodal/layers/vision_sharding.py", "VisionShardSpecs"),
        "make_reduce_sharding": load(
            "utils/parallel_utils.py", "make_reduce_sharding", should_scatter=should_scatter
        ),
        "swigluoai": load("models/minimax_m3.py", "swigluoai"),
        "get_act_fn": load("multimodal/layers/mlp.py", "get_act_fn"),
        "_resolve_rngs": load("multimodal/layers/mlp.py", "_resolve_rngs"),
    }


@functools.cache
def layer_class(path, name):
    deps = dict(dependencies())
    if name in ("UMT5FFN", "T5FFN"):
        deps["ACT_FN"] = load(path, "ACT_FN", gelu_new=load(path, "gelu_new"))
    if name == "CLIPMLP":
        deps["quick_gelu"] = load(path, "quick_gelu")
        deps["ACT_FN"] = load(path, "ACT_FN", quick_gelu=deps["quick_gelu"])
    return load(path, name, **deps)


def model_registry(directory="models/"):
    """独立读取所有模型 EntryClass，清单不得反向依赖测试覆盖表。"""
    result = {}
    for path in sorted(SOURCES):
        if not path.startswith(PREFIX + directory) or not path.endswith(".py"):
            continue
        relative = path.removeprefix(PREFIX)
        for item in ast.parse(source(relative)).body:
            if isinstance(item, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "EntryClass"
                for target in item.targets
            ):
                entries = item.value.elts if isinstance(item.value, ast.List) else [item.value]
                for entry in entries:
                    assert isinstance(entry, ast.Name) and entry.id not in result
                    result[entry.id] = relative
    return result


@functools.cache
def parsed(path):
    return ast.parse(source(path))


@functools.cache
def definition(path, name, seen=frozenset()):
    """静态解析上游类及其 import 别名，用于核对真实构造/继承路径。"""
    if (path, name) in seen or PREFIX + path not in SOURCES:
        return None
    seen = seen | {(path, name)}
    for item in ast.walk(parsed(path)):
        if isinstance(item, ast.ClassDef) and item.name == name:
            return path, item
        if (
            isinstance(item, ast.ImportFrom)
            and item.module
            and item.module.startswith("sgl_jax.srt.")
        ):
            for alias in item.names:
                if (alias.asname or alias.name) == name:
                    target = item.module.removeprefix("sgl_jax.srt.").replace(".", "/") + ".py"
                    return definition(target, alias.name, seen)
    return None


def component_registry():
    path = "multimodal/manager/stage.py"
    fn = next(
        item
        for item in parsed(path).body
        if isinstance(item, ast.FunctionDef) and item.name == "get_model_class"
    )
    mapping = next(item.value for item in fn.body if isinstance(item, ast.Assign))
    return {
        ast.literal_eval(k): (path, v.id) for k, v in zip(mapping.keys, mapping.values, strict=True)
    }


def layer_path(path, name, target_path, target_name):
    """沿显式类构造和继承边找出来源链；不能仅按模型名字猜测层归属。"""
    pending = [(path, name, ())]
    visited = set()
    while pending:
        path, name, chain = pending.pop(0)
        resolved = definition(path, name)
        if resolved is None:
            continue
        path, node = resolved
        ident = (path, node.name)
        if ident in visited:
            continue
        visited.add(ident)
        chain = (*chain, ident)
        if ident == (target_path, target_name):
            return chain
        refs = [base.id for base in node.bases if isinstance(base, ast.Name)]
        refs += [
            item.func.id
            for item in ast.walk(node)
            if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
        ]
        pending.extend((path, ref, chain) for ref in dict.fromkeys(refs))
    return None


def multimodal_registries():
    """只读取静态注册键，不执行配置工厂或依赖模型名进行猜测。"""
    result = {}
    for path in (
        "multimodal/models/static_configs/yaml_registry.py",
        "multimodal/configs/config_registry.py",
    ):
        for item in ast.parse(source(path)).body:
            if not isinstance(item, ast.ClassDef):
                continue
            for statement in item.body:
                if (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.target.id == "_REGISTRY"
                ):
                    result[item.name] = tuple(ast.literal_eval(k) for k in statement.value.keys)
    return result
