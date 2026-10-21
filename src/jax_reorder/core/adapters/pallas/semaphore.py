"""Reviewed semaphore slots for the locked TPU backend."""

from __future__ import annotations

from typing import Any, Literal, cast

import jax
from jax._src import core
from jax._src.pallas import core as pc
from jax._src.pallas import primitives as pp
from jax._src.pallas.mosaic import core as mc
from jax._src.state import indexing
from jax._src.state import types as st

from ...model import (
    Effect,
    GlobalEffectView,
    LiteralValue,
    Operand,
    SemaphoreView,
    TensorIndex,
    ValueRef,
)
from ..attributes import require

SEMAPHORE_SCHEMAS = {
    pp.semaphore_signal_p: frozenset({"args_tree", "device_id_type"}),
    pp.semaphore_wait_p: frozenset({"args_tree"}),
    pp.semaphore_read_p: frozenset({"args_tree"}),
}


def _tree(tree: Any, path: str) -> None:
    require(type(tree) is jax.tree_util.PyTreeDef, path, "unknown semaphore tree descriptor")
    node = tree.node_data()
    require(
        node is None
        or (node[0] in (list, tuple, type(None)) and node[1] is None)
        or (node[0] is indexing.NDIndexer and type(node[1]) is tuple and len(node[1]) == 3),
        path,
        "unsupported semaphore transform/target tree",
    )
    if node is not None and node[0] is indexing.NDIndexer:
        _tree(node[1][0], path)
    for child in tree.children():
        _tree(child, path)


def adapt_semaphore(
    eq: Any, operands: tuple[Operand, ...], effects: tuple[Effect, ...], path: str
) -> SemaphoreView:
    p = eq.params
    require(set(p) == SEMAPHORE_SCHEMAS[eq.primitive], path, "unaudited semaphore schema")
    _tree(p["args_tree"], path)
    require(p["args_tree"].num_leaves == len(operands), path, "semaphore operand count differs")
    args = p["args_tree"].unflatten(operands)
    phase = {
        pp.semaphore_signal_p: "signal",
        pp.semaphore_wait_p: "wait",
        pp.semaphore_read_p: "read",
    }[eq.primitive]
    arity = {"signal": 5, "wait": 4, "read": 2}[phase]
    require(type(args) is list and len(args) == arity, path, "unknown semaphore interface")
    args = cast(list[Any], args)
    require(
        isinstance(args[0], ValueRef) and type(args[1]) is tuple,
        path,
        "semaphore must be a Ref with reviewed indexing",
    )
    avals = dict(zip(operands, (v.aval for v in eq.invars), strict=True))
    sem = avals[args[0]]
    kinds = (pc.Semaphore, mc.DMASemaphore) if phase == "read" else (pc.Semaphore,)
    require(
        type(sem) is st.AbstractRef
        and type(sem.dtype) in kinds
        and sem.memory_space is mc.MemorySpace.SEMAPHORE,
        path,
        "incompatible semaphore type",
    )
    index = None
    if args[1]:
        require(len(args[1]) == 1, path, "semaphore requires one scalar element selection")
        ix = args[1][0]
        require(
            type(ix) is indexing.NDIndexer
            and ix.shape == sem.shape
            and ix.int_indexer_shape == ()
            and len(ix.indices) == len(sem.shape)
            and all(
                isinstance(i, ValueRef | LiteralValue)
                and avals[i].shape == ()
                and str(avals[i].dtype) in ("int32", "int64")
                for i in ix.indices
            ),
            path,
            "semaphore index must select a scalar element",
        )
        ix = cast(indexing.NDIndexer, ix)
        index = TensorIndex(tuple(cast(Operand, i) for i in ix.indices), tuple(ix.shape))
    require(index is not None or sem.shape == (), path, "semaphore array needs an element index")
    if phase == "read":
        require(
            not effects
            and len(eq.outvars) == 1
            and eq.outvars[0].aval.shape == ()
            and str(eq.outvars[0].aval.dtype) == "int32",
            path,
            "invalid semaphore read result/effects",
        )
        return SemaphoreView("read", args[0], index=index)
    require(not eq.outvars, path, "semaphore command unexpectedly returns a value")
    amount = args[2]
    require(
        isinstance(amount, ValueRef | LiteralValue)
        and avals[amount].shape == ()
        and str(avals[amount].dtype) == "int32",
        path,
        "semaphore amount must be scalar int32",
    )
    expected = {GlobalEffectView("semaphore")}
    coordinates = None
    core_index = None
    if phase == "signal":
        require(
            p["device_id_type"] is pp.DeviceIdType.MESH,
            path,
            "unsupported semaphore device/core policy",
        )
        core_index = args[4]
        require(
            core_index is None
            or (
                isinstance(core_index, ValueRef | LiteralValue)
                and avals[core_index].shape == ()
                and str(avals[core_index].dtype) == "int32"
            ),
            path,
            "semaphore core target must be scalar int32",
        )
        target = args[3]
        if target is not None:
            coordinates = target if type(target) is tuple else (target,)
            require(
                bool(coordinates)
                and all(
                    isinstance(c, ValueRef | LiteralValue)
                    and avals[c].shape == ()
                    and str(avals[c].dtype) == "int32"
                    for c in coordinates
                ),
                path,
                "semaphore target must contain scalar mesh coordinates",
            )
            expected.add(GlobalEffectView("communication"))
    else:
        # Both the locked TPU lowering and TPU interpreter reject decrement=False.
        raw = eq.invars[-1]
        require(
            type(raw) is core.Literal
            and type(raw.aval) is core.ShapedArray
            and raw.aval.shape == ()
            and str(raw.aval.dtype) == "bool"
            and bool(raw.val),
            path,
            "TPU semaphore wait requires decrement=True",
        )
    require(set(effects) == expected, path, "inconsistent semaphore effects")
    return SemaphoreView(
        cast(Literal["signal", "wait", "read"], phase),
        args[0],
        amount,
        coordinates,
        core_index,
        index,
    )
