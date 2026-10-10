"""Execute and preserve each targeted primitive before claiming coverage."""

import jax
import numpy as np
import pytest
from jax._src import core
from lax_cases import cases, trace_case

from jax_reorder import Status, canonicalize, compare_jaxprs

CPU_REJECTIONS = {
    "ragged_all_to_all": (jax.errors.JaxRuntimeError, "ragged-all-to-all.*not supported"),
    "pbroadcast": (NotImplementedError, "pbroadcast.*not found for platform cpu"),
}


def nested_primitives(graph):
    found = set()
    pending = [graph]
    while pending:
        value = pending.pop()
        if isinstance(value, core.Jaxpr):
            for equation in value.eqns:
                found.add(equation.primitive)
                pending.extend(equation.params.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
    return found


@pytest.mark.parametrize("case", cases(), ids=lambda case: case.name)
def test_lax_cpu_primitive(case):
    original = trace_case(case)
    names = {p.name for p in nested_primitives(original)}
    if case.name == "stage":
        # Public stage is eliminated by JAX 0.11.2 itself. Do not count it as
        # executed IR coverage: its raw abstract-eval rule is also invalid.
        assert not names
    else:
        assert case.expected in names, f"fixture did not trace {case.expected}: {names}"
    normalized = canonicalize(original)
    assert normalized.complete, normalized.issues
    assert compare_jaxprs(original, normalized.jaxpr).status == Status.EQUIVALENT
    core.check_jaxpr(normalized.jaxpr)
    # JIT the complete graph, so control flow and named-axis environments use
    # their CPU lowering. The tested primitive must survive tracing above.
    flat_args = jax.tree.leaves(case.args)
    if case.name in CPU_REJECTIONS:
        # A real CPU backend rejection, not a skipped project limitation.
        error, message = CPU_REJECTIONS[case.name]
        with pytest.raises(error, match=message):
            jax.jit(lambda *args: core.eval_jaxpr(original, original.consts, *args))(*flat_args)
        return
    before = jax.jit(lambda *args: core.eval_jaxpr(original, original.consts, *args))(*flat_args)
    after = jax.jit(
        lambda *args: core.eval_jaxpr(normalized.jaxpr, normalized.jaxpr.consts, *args)
    )(*flat_args)
    assert len(before) == len(after)
    for a, b in zip(before, after, strict=True):
        if isinstance(a, core.Token):
            assert isinstance(b, core.Token)
        elif case.deterministic:
            np.testing.assert_array_equal(a, b)
        else:
            assert a.shape == b.shape and a.dtype == b.dtype
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))


@pytest.mark.parametrize(
    "case", [c for c in cases() if c.name not in CPU_REJECTIONS], ids=lambda case: case.name
)
def test_each_primitive_with_an_independent_branch(case):
    def before(extra, *args):
        value = case.fn(*args)
        side = jax.lax.neg(extra)
        return value, side

    def after(extra, *args):
        side = jax.lax.neg(extra)
        value = case.fn(*args)
        return value, side

    args = (np.float32(2), *case.args)
    left, right = jax.make_jaxpr(before)(*args), jax.make_jaxpr(after)(*args)
    comparison = compare_jaxprs(left, right)
    assert comparison.status == Status.EQUIVALENT, comparison
    assert canonicalize(comparison.right.jaxpr).graph == comparison.right.graph


def test_public_inventory_has_no_unexplained_omissions():
    from jax_reorder.support import public_primitives

    found = set().union(*(nested_primitives(trace_case(c)) for c in cases()))
    # stage is erased by JAX tracing; ragged_all_to_all is present but its CPU
    # rejection is asserted explicitly above, not counted as CPU execution.
    assert set(public_primitives()) - found == {jax.lax.stage_p}


@pytest.mark.parametrize("which", ["psend", "precv"])
def test_send_receive_are_rejected_by_jax_cpu(which):
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    mesh = Mesh(np.array(jax.devices("cpu")), ("workers",))
    if which == "psend":

        def body(x):
            jax.lax.psend(x, "workers", [(0, 1)])
            return x
    else:

        def body(x):
            return jax.lax.precv(
                jax.lax.create_token(x), jax.ShapeDtypeStruct((1,), np.float32), "workers", [(0, 1)]
            )

    fn = jax.shard_map(body, mesh=mesh, in_specs=P("workers"), out_specs=P(), check_vma=False)
    with pytest.raises(NotImplementedError, match=f"{which} is currently only implemented on GPU"):
        jax.jit(fn)(np.ones(len(jax.devices("cpu")), dtype=np.float32))
