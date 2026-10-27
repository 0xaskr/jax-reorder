"""Verify pinned files and load explicitly selected, unmodified source definitions.

Each fixtures/upstream_tpu/<repo_key>/manifest.json contains repository, revision,
license and a files mapping. Each file entry has path, snapshot, sha256 and git_blob.
Snapshots remain text assets; this loader never imports the enclosing project.
"""

import ast
import hashlib
import itertools
import json
import sys
import types
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "upstream_tpu"
_LOAD_IDS = itertools.count()


def _directory(repo_key: str) -> Path:
    if not repo_key or Path(repo_key).name != repo_key or repo_key in {".", ".."}:
        raise ValueError(f"invalid upstream repository key: {repo_key!r}")
    directory = (FIXTURES / repo_key).resolve()
    if not directory.is_relative_to(FIXTURES.resolve()):
        raise ValueError(f"upstream repository escapes fixture directory: {repo_key!r}")
    return directory


def read_manifest(repo_key: str) -> dict[str, Any]:
    """Read one repository's metadata without executing upstream code."""
    path = _directory(repo_key) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"upstream manifest must be an object: {path}")
    for field in ("repository", "revision", "license"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ValueError(f"upstream manifest requires nonempty {field!r}: {path}")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError(f"upstream manifest requires a nonempty files mapping: {path}")
    return manifest


def _source(repo_key: str, file_key: str) -> tuple[Path, str]:
    manifest = read_manifest(repo_key)
    entry = manifest["files"].get(file_key)
    if not isinstance(entry, dict):
        raise ValueError(f"missing upstream file entry: {repo_key}/{file_key}")
    for field in ("path", "snapshot", "sha256", "git_blob"):
        if not isinstance(entry.get(field), str) or not entry[field]:
            raise ValueError(f"upstream file requires {field!r}: {repo_key}/{file_key}")
    directory = _directory(repo_key)
    relative = Path(entry["snapshot"])
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory):
        raise ValueError(f"upstream snapshot escapes repository directory: {entry['snapshot']}")
    data = path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    git_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if sha256 != entry["sha256"]:
        raise ValueError(f"upstream SHA-256 mismatch: {repo_key}/{file_key} ({path})")
    if git_blob != entry["git_blob"]:
        raise ValueError(f"upstream Git blob mismatch: {repo_key}/{file_key} ({path})")
    return path, data.decode("utf-8")


def read_source(repo_key: str, file_key: str) -> str:
    """Return source only after both content digests match the manifest."""
    return _source(repo_key, file_key)[1]


def _bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
        return {target.id for target in node.targets}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id} if node.value is not None else set()
    return set()


def _execute_definitions(
    path: Path,
    tree: ast.Module,
    selected: list[ast.stmt],
    namespace: Mapping[str, Any],
) -> dict[str, Any]:
    # Source future imports are preserved; caller/compiler future flags cannot leak in.
    code = compile(
        ast.Module(body=selected, type_ignores=tree.type_ignores),
        str(path),
        "exec",
        dont_inherit=True,
    )
    module_name = f"_upstream_tpu_fixture_{next(_LOAD_IDS)}"
    module = types.ModuleType(module_name)
    module.__dict__.update(namespace)
    module.__dict__.update(__name__=module_name, __file__=str(path), __package__="")
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module.__dict__


def load_definitions(
    repo_key: str,
    file_key: str,
    names: Collection[str],
    namespace: Mapping[str, Any],
) -> dict[str, Any]:
    """Load complete top-level definitions and explicitly named source assignments.

    Dependencies must be supplied explicitly or selected from the same file. Bodies,
    decorators and assignment expressions are retained verbatim as AST nodes. All
    targets of a shared assignment must be selected; ambiguous redefinitions fail.
    The returned objects retain their module globals, including injected dependencies.
    A unique module stays registered so decorators such as dataclass can resolve it.
    This isolates imports, not execution: selected decorators and assignments execute.
    """
    if isinstance(names, str):
        raise ValueError("upstream definition names must be a collection, not a string")
    requested = tuple(names)
    if not requested or any(
        not isinstance(name, str) or not name.isidentifier() for name in requested
    ):
        raise ValueError("upstream definition names must be nonempty Python identifiers")
    wanted = set(requested)
    if len(wanted) != len(requested):
        raise ValueError("upstream definition names must be unique")
    path, source = _source(repo_key, file_key)
    tree = ast.parse(source, filename=str(path))
    selected, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            selected.append(node)
            continue
        bound = _bound_names(node)
        if not bound & wanted:
            continue
        if not bound <= wanted:
            raise ValueError(f"select every target of the upstream assignment: {sorted(bound)}")
        if bound & found:
            raise ValueError(f"ambiguous upstream definitions: {sorted(bound & found)}")
        selected.append(node)
        found.update(bound)
    if found != wanted:
        raise ValueError(
            f"missing top-level upstream definitions in {path}: {sorted(wanted - found)}"
        )

    loaded = _execute_definitions(path, tree, selected, namespace)
    return {name: loaded[name] for name in requested}


def load_nested_definition(
    repo_key: str,
    file_key: str,
    qualified_name: str,
    namespace: Mapping[str, Any],
) -> Any:
    """Load one complete nested function with explicitly supplied closure dependencies.

    The dotted path selects direct lexical children, never a same-named function
    elsewhere or one hidden in a conditional definition. Enclosing training code
    does not execute. The original function node, decorators, defaults and future
    imports remain intact; captured values are supplied as isolated module globals.
    This supports selected calculation helpers, not general closure reconstruction.
    """
    if not isinstance(qualified_name, str):
        raise ValueError("upstream nested definition path must be a dotted string")
    parts = qualified_name.split(".")
    if len(parts) < 2 or any(not part.isidentifier() for part in parts):
        raise ValueError("upstream nested definition path requires two or more identifiers")
    path, source = _source(repo_key, file_key)
    tree = ast.parse(source, filename=str(path))
    body = tree.body
    node = None
    for part in parts:
        matches = [item for item in body if part in _bound_names(item)]
        if len(matches) != 1:
            reason = "missing" if not matches else "ambiguous"
            raise ValueError(f"{reason} upstream nested definition: {qualified_name} in {path}")
        node = matches[0]
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            raise ValueError(f"upstream nested path is not a definition: {qualified_name}")
        body = node.body
    if not isinstance(node, ast.FunctionDef):
        raise ValueError(f"upstream nested target must be a function: {qualified_name}")
    selected = [
        item
        for item in tree.body
        if isinstance(item, ast.ImportFrom) and item.module == "__future__"
    ]
    loaded = _execute_definitions(path, tree, [*selected, node], namespace)
    return loaded[node.name]
