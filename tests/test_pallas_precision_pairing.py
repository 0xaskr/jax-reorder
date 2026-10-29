"""Non-default precision is paired by actual contraction operands and types."""

import jax
import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental import pallas as pl

from jax_reorder import Status, UnsupportedJaxprError, compare_algebraic_jaxprs


def mixed(*, move=False, swap_policy=False, preset=False, multiplier=2.0):
    signature = jax.ShapeDtypeStruct((4, 4), jnp.float32)

    def kernel(a, b, c, y):
        highest = lax.DotAlgorithmPreset.F32_F32_F32 if preset else lax.Precision.HIGHEST
        x = a[...] * multiplier if move else a[...]
        first = jnp.matmul(x, b[...], precision=None if swap_policy else highest)
        second = jnp.matmul(a[...], c[...], precision=highest if swap_policy else None)
        y[...] = (first if move else multiplier * first) + second

    return jax.make_jaxpr(pl.pallas_call(kernel, out_shape=signature))(
        signature, signature, signature
    )


@pytest.mark.parametrize("factor,status", [(2.0, Status.EQUIVALENT), (3.0, Status.INEQUIVALENT)])
def test_mixed_policies_and_real_scale_positive_negative(factor, status):
    result = compare_algebraic_jaxprs(
        mixed(), mixed(move=True, multiplier=factor), scope="device_local"
    )
    assert result.status is status and result.warnings


def test_same_policy_set_cannot_hide_policies_swapped_to_other_operands():
    with pytest.raises(UnsupportedJaxprError, match="operand/type correspondence"):
        compare_algebraic_jaxprs(mixed(), mixed(swap_policy=True), scope="device_local")


def test_matching_dot_inventory_cannot_hide_swapped_output_policies():
    signature = jax.ShapeDtypeStruct((4, 4), jnp.float32)

    def function(swapped):
        def kernel(a, b, first, second):
            first[...] = jnp.matmul(
                a[...], b[...], precision=None if swapped else lax.Precision.HIGHEST
            )
            second[...] = jnp.matmul(
                a[...], b[...], precision=lax.Precision.HIGHEST if swapped else None
            )

        return pl.pallas_call(kernel, out_shape=(signature, signature))

    graphs = [jax.make_jaxpr(function(s))(signature, signature) for s in (False, True)]
    with pytest.raises(UnsupportedJaxprError, match="operand/type correspondence"):
        compare_algebraic_jaxprs(*graphs, scope="device_local")


def test_dot_algorithm_preset_remains_outside_reviewed_real_precision():
    with pytest.raises(UnsupportedJaxprError, match="non-default dot precision"):
        compare_algebraic_jaxprs(mixed(), mixed(move=True, preset=True), scope="device_local")


def test_ordinary_jax_highest_precision_is_not_implicitly_admitted():
    signature = jax.ShapeDtypeStruct((4, 4), jnp.float32)
    graphs = [
        jax.make_jaxpr(
            lambda a, b, move=move: (
                jnp.matmul(a * 2 if move else a, b, precision=lax.Precision.HIGHEST)
                * (1 if move else 2)
            )
        )(signature, signature)
        for move in (False, True)
    ]
    with pytest.raises(UnsupportedJaxprError, match="non-default dot precision"):
        compare_algebraic_jaxprs(*graphs)


def test_matching_precision_does_not_hide_different_accumulator_types():
    signature = jax.ShapeDtypeStruct((4, 4), jnp.bfloat16)
    output = jax.ShapeDtypeStruct((4, 4), jnp.float32)

    def function(accumulator):
        def kernel(a, b, y):
            y[...] = jnp.matmul(
                a[...],
                b[...],
                precision=lax.Precision.HIGHEST,
                preferred_element_type=accumulator,
            ).astype(jnp.float32)

        return pl.pallas_call(kernel, out_shape=output)

    graphs = [
        jax.make_jaxpr(function(accumulator))(signature, signature)
        for accumulator in (jnp.float32, jnp.bfloat16)
    ]
    with pytest.raises(UnsupportedJaxprError, match="operand/type correspondence"):
        compare_algebraic_jaxprs(*graphs, scope="device_local")
