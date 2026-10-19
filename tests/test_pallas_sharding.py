"""Shard-local retiling keeps device maps, partitions and TPU execution policy."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_reorder import ComparisonContractError, CorrespondenceError, Status, UnsupportedJaxprError
from jax_reorder.core.adapters.jax import adapt
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.normalization import (
    compare_program_views,
    normalize_view,
    verify_normalized_evidence,
)
from tests.test_pallas_semantic_slice import call, view

PARTITIONED = P("d")


def device_mesh(reverse=False):
    devices = jax.devices("cpu")[:2]
    assert len(devices) == 2
    return Mesh(np.array(devices[::-1] if reverse else devices), ("d",))


def sharded(
    block,
    *,
    reverse=False,
    mesh=None,
    check_vma=True,
    in_spec=PARTITIONED,
    out_spec=PARTITIONED,
    outer_jit=False,
    inner_jit=False,
    interpret=False,
    size=2048,
    constant=False,
):
    def local(x):
        programs = x.shape[0] // block
        spec = pl.BlockSpec((block,), lambda p: (programs - 1 - p if reverse else p,))

        def kernel(a, b):
            b[:] = jnp.ones((block,), jnp.float32) if constant else a[:] + 1

        out = jax.ShapeDtypeStruct(
            x.shape, x.dtype, manual_axis_type=jax.typeof(x).manual_axis_type
        )
        return pl.pallas_call(
            kernel,
            out_shape=out,
            grid=(programs,),
            in_specs=(spec,),
            out_specs=spec,
            interpret=interpret,
        )(x)

    f = jax.shard_map(
        jax.jit(local) if inner_jit else local,
        mesh=mesh or device_mesh(),
        in_specs=in_spec,
        out_specs=out_spec,
        check_vma=check_vma,
    )
    return jax.jit(f) if outer_jit else f


def sharded_view(block, **kwargs):
    return adapt(
        jax.make_jaxpr(sharded(block, **kwargs))(
            jax.ShapeDtypeStruct((kwargs.get("size", 2048),), jnp.float32)
        )
    ).view


@pytest.mark.parametrize("check_vma", [False, True])
@pytest.mark.parametrize("wrapper", [{}, {"outer_jit": True}, {"inner_jit": True}])
def test_shard_local_retile_preserves_all_call_boundaries(check_vma, wrapper):
    a = sharded_view(128, check_vma=check_vma, **wrapper)
    b = sharded_view(256, reverse=True, check_vma=check_vma, **wrapper)
    assert compare_program_views(a, a).equivalence_kind == "structural"
    result = compare_program_views(a, b)
    assert result.status == Status.EQUIVALENT and result.equivalence_kind == "normalized"
    boundary = next(b for b in result.evidence.left.boundaries if b.shard_map is not None)
    assert boundary.shard_map.check_vma is check_vma
    assert len(boundary.shard_map.mesh.devices) == 2
    assert boundary.shard_map.input_partitions == ((("d",),),)
    assert result.evidence.left.shard_dependencies == (((0,),),)
    verify_normalized_evidence(a, b, result.evidence, budget=AnalysisBudget())


def test_device_permutation_is_a_contract_difference():
    a, b = sharded_view(128), sharded_view(256, mesh=device_mesh(True))
    with pytest.raises(ComparisonContractError, match="execution contracts"):
        compare_program_views(a, b)


def test_partition_contract_is_not_inferred_from_same_mesh():
    a, b = sharded_view(128), sharded_view(256, in_spec=P(), out_spec=P())
    with pytest.raises(ComparisonContractError, match="execution contracts"):
        compare_program_views(a, b)


def test_false_replication_claim_is_rejected_even_without_vma_checking():
    a = sharded_view(128, out_spec=P(), check_vma=False)
    with pytest.raises(UnsupportedJaxprError, match="replication is not proved"):
        normalize_view(a, budget=AnalysisBudget())


def test_actual_replication_is_proved_from_inputs():
    a, b = (
        sharded_view(128, in_spec=P(), out_spec=P()),
        sharded_view(256, in_spec=P(), out_spec=P()),
    )
    assert compare_program_views(a, b).equivalence_kind == "normalized"


def test_unused_partitioned_input_does_not_prevent_constant_replication():
    a, b = (
        sharded_view(block, out_spec=P(), check_vma=False, constant=True) for block in (128, 256)
    )
    result = compare_program_views(a, b)
    assert result.equivalence_kind == "normalized"
    assert result.evidence.left.shard_dependencies == (((),),)


def test_replication_witness_is_checked_during_replay():
    a, b = sharded_view(128), sharded_view(256)
    result = compare_program_views(a, b)
    forged = replace(
        result.evidence, left=replace(result.evidence.left, shard_dependencies=(((),),))
    )
    with pytest.raises(CorrespondenceError, match="does not replay"):
        verify_normalized_evidence(a, b, forged, budget=AnalysisBudget())


def test_cpu_shard_execution_is_separate_from_symbolic_proof():
    x = np.arange(2048, dtype=np.float32)
    for block in (128, 256):
        actual = sharded(block, reverse=block == 256, check_vma=False, interpret=True)(x)
        np.testing.assert_array_equal(actual, x + 1)
    assert (
        compare_program_views(sharded_view(128), sharded_view(256, reverse=True)).equivalence_kind
        == "normalized"
    )


@pytest.mark.parametrize("semantics", ["parallel", "arbitrary", pltpu.PARALLEL, pltpu.ARBITRARY])
def test_order_and_parallelism_can_change_only_after_independence_proof(semantics):
    params = pltpu.CompilerParams(dimension_semantics=(semantics,))
    a, b = view(compiler_params=params), view(256, True)
    assert call(a).grid[0].semantics in ("parallel", "arbitrary")
    assert compare_program_views(a, b).equivalence_kind == "normalized"


@pytest.mark.parametrize(
    "options",
    [
        {"shape_invariant_numerics": False},
        {"disable_bounds_checks": True},
        {"skip_device_barrier": True},
        {"collective_id": 1},
        {"has_side_effects": True},
        {"vmem_limit_bytes": 4096},
    ],
)
def test_nondefault_compiler_policy_has_structural_facts_but_no_implicit_retile_rule(options):
    params = pltpu.CompilerParams(**options)
    a, b = view(compiler_params=params), view(256, compiler_params=params)
    assert not call(a).compiler.default_execution_options
    assert compare_program_views(a, a).equivalence_kind == "structural"
    with pytest.raises(UnsupportedJaxprError, match="compiler execution options"):
        compare_program_views(a, b)


def test_numeric_policy_change_is_a_contract_mismatch():
    a = view(compiler_params=pltpu.CompilerParams(shape_invariant_numerics=False))
    with pytest.raises(ComparisonContractError):
        compare_program_views(a, view(256))


def test_unknown_compiler_flags_and_descriptor_subclasses_are_rejected():
    with pytest.raises(UnsupportedJaxprError, match="per-key semantic review"):
        view(compiler_params=pltpu.CompilerParams(flags={"unreviewed_flag": True}))

    class Extended(pltpu.CompilerParams):
        pass

    with pytest.raises(UnsupportedJaxprError, match="reviewed Mosaic"):
        view(compiler_params=Extended())


def test_explicit_empty_flags_preserve_source_schema_and_allow_default_policy_retile():
    params = pltpu.CompilerParams(flags={})
    a, b = view(compiler_params=params), view(256, compiler_params=params)
    assert call(a).compiler.default_execution_options
    assert compare_program_views(a, b).equivalence_kind == "normalized"


def test_core_specific_scheduling_does_not_assume_hardware_binding():
    a = view(compiler_params=pltpu.CompilerParams(dimension_semantics=(pltpu.CORE_PARALLEL,)))
    with pytest.raises(UnsupportedJaxprError, match="hardware binding"):
        normalize_view(a, budget=AnalysisBudget())
