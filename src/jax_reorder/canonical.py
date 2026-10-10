"""Output-rooted BFS numbering, dependency-depth ordering and exact comparison."""

import dataclasses
from collections import deque
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import jax
from jax import lax
from jax._src import config, core, pjit, shard_map
from jax._src.lax.lax import NoDCEEffect

from .errors import InvalidJaxprError, JaxprError, UnsupportedJaxprError
from .semantics import Encoder, Frozen, replace_nested
from .support import supported_primitive_identities


def _ordered_equation(equation):
    """Include implicit ordering hidden inside a call/control-flow boundary."""
    pending = [equation]
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, core.JaxprEqn):
            if value.primitive.name in {"rng_uniform", "dce_sink"}:
                return True
            pending.extend(value.params.values())
        elif isinstance(value, core.Jaxpr):
            pending.extend(value.eqns)
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return False


class Status(StrEnum):
    EQUIVALENT = "equivalent"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Issue:
    path: str
    reason: str


@dataclass(frozen=True)
class Node:
    primitive: core.Primitive
    inputs: tuple
    outputs: tuple
    params: tuple
    effects: Frozen
    context: Frozen
    order_after: tuple[int, ...] = ()


@dataclass(frozen=True)
class Graph:
    inputs: tuple
    constants: tuple
    outputs: tuple
    nodes: tuple[Node, ...]
    effects: Frozen
    is_high: bool


@dataclass(frozen=True)
class Canonicalized:
    jaxpr: core.Jaxpr = field(compare=False, repr=False)
    graph: Graph
    bfs_order: tuple[int, ...]
    equation_order: tuple[int, ...]
    depths: tuple[int, ...]
    unused_equations: tuple[int, ...]
    unused_constants: tuple[int, ...]
    issues: tuple[Issue, ...]
    value_references: Mapping[core.Var, tuple] = field(compare=False, repr=False)

    @property
    def complete(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class Comparison:
    status: Status
    issues: tuple[Issue, ...] = ()
    difference: str | None = None
    left: Canonicalized | None = field(default=None, compare=False, repr=False)
    right: Canonicalized | None = field(default=None, compare=False, repr=False)


def _validate(graph: core.Jaxpr, path: str) -> None:
    if not isinstance(graph, core.Jaxpr):
        raise InvalidJaxprError(path, "expected a jaxpr including its bound constants")
    defined: set[core.Var] = set()

    def bind(v, where):
        if not isinstance(v, core.Var) or v in defined:
            raise InvalidJaxprError(where, "invalid binder or duplicate definition")
        defined.add(v)

    def use(v, where):
        if not isinstance(v, core.Literal) and (not isinstance(v, core.Var) or v not in defined):
            raise InvalidJaxprError(where, "undefined value or dependency cycle")

    if len(graph.constvars) != len(graph.consts):
        raise InvalidJaxprError(path, "constant binders and values must have equal lengths")
    for i, v in enumerate((*graph.constvars, *graph.invars)):
        bind(v, f"{path}.binders[{i}]")
    for i, eq in enumerate(graph.eqns):
        for v in eq.invars:
            use(v, f"{path}.eqns[{i}].inputs")
        for v in eq.outvars:
            bind(v, f"{path}.eqns[{i}].outputs")
    for v in graph.outvars:
        use(v, f"{path}.outputs")


class _Normalizer:
    def __init__(self):
        self.primitives = supported_primitive_identities()
        self.cache: dict[int, Canonicalized] = {}
        self.active: set[int] = set()

    def normalize(self, graph: core.Jaxpr, path: str) -> Canonicalized:
        if id(graph) in self.active:
            raise InvalidJaxprError(path, "cyclic nested jaxpr")
        if id(graph) in self.cache:
            return self.cache[id(graph)]
        self.active.add(id(graph))
        try:
            result = self._normalize(graph, path)
            self.cache[id(graph)] = result
            self.cache[id(result.jaxpr)] = result
            return result
        finally:
            self.active.remove(id(graph))

    def _params(self, eq, eq_path, encoder):
        encoded_params = []
        for name, value in sorted(eq.params.items()):
            # JIT's function name labels diagnostics, not the computation.
            if eq.primitive is pjit.jit_p and name == "name":
                continue
            if (
                jax.__version__ == "0.11.2"
                and eq.primitive is lax.reduce_p
                and name == "computation"
                and isinstance(eq.params.get("jaxpr"), core.Jaxpr)
                and callable(value)
            ):
                # In this version, lowering, JVP and batching use the reducer
                # jaxpr. Preserve the callable in the returned program and
                # compare its already-traced meaning through that subgraph.
                encoded = ("represented_by_reducer_jaxpr",)
            else:
                encoded = encoder.encode(value, f"{eq_path}.params.{name}")
            encoded_params.append((name, encoded))
        return encoded_params

    def _normalize(self, graph: core.Jaxpr, path: str) -> Canonicalized:
        _validate(graph, path)
        equations = graph.eqns
        producers = {
            v: (i, port) for i, eq in enumerate(equations) for port, v in enumerate(eq.outvars)
        }
        inputs = {v: i for i, v in enumerate(graph.invars)}
        constants = {v: i for i, v in enumerate(graph.constvars)}
        constant_ids: dict[int, int] = {}
        identifiers: dict[int, int] = {}
        queue: deque[int] = deque()
        extra_predecessors: dict[int, list[int]] = {}
        effect_roots: list[int] = []
        previous_ordered: int | None = None
        issues: list[Issue] = []

        # Named-axis effects describe the required axis environment, not ordered
        # host IO. All other effects require an explicit semantic adapter.
        def check_effects(effects, where):
            for effect in effects:
                if not isinstance(effect, (core.NamedAxisEffect, NoDCEEffect)):
                    raise UnsupportedJaxprError(
                        where, f"unsupported effect {type(effect).__qualname__}"
                    )

        check_effects(graph.effects, f"{path}.effects")
        for i, eq in enumerate(equations):
            if eq.primitive not in self.primitives:
                raise UnsupportedJaxprError(
                    f"{path}.eqns[{i}]", f"unsupported primitive {eq.primitive.name}"
                )
            check_effects(eq.effects, f"{path}.eqns[{i}].effects")
            # Stateful RNG and explicit program-order nodes cannot be treated as
            # independent pure operations even when their effects set is empty.
            if _ordered_equation(eq):
                effect_roots.append(i)
                if previous_ordered is not None:
                    extra_predecessors[i] = [previous_ordered]
                previous_ordered = i

        def visit(index):
            if index not in identifiers:
                identifiers[index] = len(identifiers)
                queue.append(index)

        def discover(atom):
            if isinstance(atom, core.Var):
                if atom in producers:
                    visit(producers[atom][0])
                elif atom in constants:
                    index = constants[atom]
                    constant_ids.setdefault(index, len(constant_ids))

        for atom in graph.outvars:
            discover(atom)
        for index in effect_roots:
            visit(index)
        while queue:
            index = queue.popleft()
            for atom in equations[index].invars:
                discover(atom)
            for predecessor in extra_predecessors.get(index, ()):
                visit(predecessor)
        bfs_order = tuple(identifiers)
        unused = tuple(i for i in range(len(equations)) if i not in identifiers)
        if unused:
            issues.append(
                Issue(path, f"{len(unused)} unused equations retain their original relative order")
            )
        for index in unused:
            identifiers[index] = len(identifiers)
            for atom in equations[index].invars:
                if isinstance(atom, core.Var) and atom in constants:
                    old = constants[atom]
                    constant_ids.setdefault(old, len(constant_ids))
        unused_constants = tuple(i for i in range(len(graph.constvars)) if i not in constant_ids)
        if unused_constants:
            issues.append(Issue(path, f"{len(unused_constants)} unused constant bindings"))
        for index in unused_constants:
            constant_ids[index] = len(constant_ids)

        def reference(atom):
            if isinstance(atom, core.Literal):
                return (
                    "literal",
                    encoder.encode(atom.aval, path),
                    encoder.literal(atom.val, atom.aval, path),
                )
            if atom in inputs:
                return ("input", inputs[atom])
            if atom in constants:
                return ("constant", constant_ids[constants[atom]])
            index, port = producers[atom]
            return ("result", identifiers[index], port)

        def nested(child, child_path):
            result = self.normalize(child, child_path)
            issues.extend(result.issues)
            return result

        encoder = Encoder(nested, reference)
        frozen_params = []
        new_equations = []
        # Validate and encode every equation, including unused ones.
        for i, eq in enumerate(equations):
            eq_path = f"{path}.eqns[{i}]"
            encoded_params = []
            with ExitStack() as scope:
                # A collective's body is only well-typed inside the axis
                # environment established by its enclosing shard_map.
                if eq.primitive is shard_map.shard_map_p:
                    scope.enter_context(
                        shard_map._extend_axis_env(
                            eq.params["mesh"], eq.params["newly_manual_axes"]
                        )
                    )
                    scope.enter_context(config._check_vma(eq.params["check_vma"]))
                encoded_params = self._params(eq, eq_path, encoder)
            frozen_params.append(tuple(encoded_params))
            params = replace_nested(eq.params, lambda child: self.cache[id(child)].jaxpr)
            new_equations.append(eq if params is eq.params else eq.replace(params=params))

        depths: list[int] = []
        for i, eq in enumerate(equations):
            predecessors = [
                producers[v][0] for v in eq.invars if isinstance(v, core.Var) and v in producers
            ]
            predecessors.extend(extra_predecessors.get(i, ()))
            depths.append(1 + max((depths[p] for p in predecessors), default=0))
        order = tuple(sorted(bfs_order, key=lambda i: (depths[i], identifiers[i]))) + unused
        ordered_nodes = []
        for i in identifiers:
            eq = equations[i]
            eq_path = f"{path}.eqns[{i}]"
            ordered_nodes.append(
                Node(
                    eq.primitive,
                    tuple(reference(v) for v in eq.invars),
                    tuple(
                        encoder.encode(v.aval, f"{eq_path}.outputs[{p}]")
                        for p, v in enumerate(eq.outvars)
                    ),
                    frozen_params[i],
                    encoder.encode(eq.effects, eq_path),
                    encoder.encode(eq.ctx, eq_path),
                    tuple(identifiers[p] for p in extra_predecessors.get(i, ())),
                )
            )
        form = Graph(
            tuple(
                encoder.encode(v.aval, f"{path}.inputs[{i}]") for i, v in enumerate(graph.invars)
            ),
            tuple(
                (
                    encoder.encode(graph.constvars[i].aval, path),
                    encoder.encode(graph.consts[i], path),
                )
                for i in constant_ids
            ),
            tuple(reference(v) for v in graph.outvars),
            tuple(ordered_nodes),
            encoder.encode(graph.effects, f"{path}.effects"),
            graph.is_high,
        )
        reordered = graph.replace(eqns=[new_equations[i] for i in order])
        # Check only after recursive structural validation has rejected cycles.
        try:
            core.check_jaxpr(reordered)
        except (core.JaxprTypeError, TypeError, ValueError, NameError) as exc:
            raise InvalidJaxprError(path, f"JAX type check failed: {exc}") from exc
        return Canonicalized(
            reordered,
            form,
            bfs_order,
            order,
            tuple(depths),
            unused,
            unused_constants,
            tuple(dict.fromkeys(issues)),
            MappingProxyType({v: reference(v) for v in (*inputs, *constants, *producers)}),
        )


def canonicalize(jaxpr: core.Jaxpr) -> Canonicalized:
    """Normalize a CPU jaxpr without changing its operations or dependencies.

    Unsupported semantics and malformed graphs raise explicit JaxprError
    subclasses. Unused pure branches are retained and reported as incomplete.
    """
    try:
        return _Normalizer().normalize(jaxpr, "jaxpr")
    except RecursionError as exc:
        raise UnsupportedJaxprError(
            "jaxpr", "nested graphs or parameters exceed the Python recursion limit"
        ) from exc


def _difference(left: Any, right: Any) -> str | None:
    pending = [("graph", left, right)]
    while pending:
        path, a, b = pending.pop()
        if type(a) is not type(b):
            return path
        if dataclasses.is_dataclass(a) and not isinstance(a, type):
            pending.extend(
                (f"{path}.{f.name}", getattr(a, f.name), getattr(b, f.name))
                for f in reversed(dataclasses.fields(a))
            )
        elif isinstance(a, tuple):
            if len(a) != len(b):
                return f"{path}.length"
            pending.extend((f"{path}[{i}]", a[i], b[i]) for i in reversed(range(len(a))))
        elif a != b:
            return path
    return None


def compare_jaxprs(left: core.Jaxpr, right: core.Jaxpr) -> Comparison:
    """Compare complete graphs; a structural mismatch is not an algebraic proof."""
    normalized = []
    issues = []
    for side, graph in (("left", left), ("right", right)):
        try:
            result = canonicalize(graph)
            normalized.append(result)
            issues.extend(Issue(f"{side}.{i.path}", i.reason) for i in result.issues)
        except JaxprError as exc:
            normalized.append(None)
            issues.append(Issue(f"{side}.{exc.path}", exc.reason))
    a, b = normalized
    if issues:
        return Comparison(Status.UNKNOWN, tuple(issues), left=a, right=b)
    difference = _difference(a.graph, b.graph)
    return Comparison(
        Status.DIFFERENT if difference else Status.EQUIVALENT,
        difference=difference,
        left=a,
        right=b,
    )
