"""Pinned AlphaFold 3 masked means preserve the source's clamped denominator."""

import numbers
from collections import abc
from collections.abc import Sequence
from functools import cache

import jax.numpy as jnp
import numpy as np

from tests.upstream_tpu.sources import load_definitions

REVISION = "5e23d45bff3a4c1993516ead2ec848cb42d64c1a"
EPSILON = 0.125
CASES = (
    ("reordered", True, True),
    ("broadcast_mask", False, True),
    ("wrong_broadcast", False, False),
    ("wrong_additive_epsilon", False, False),
)
SCENARIOS = ("rows", "all_axes", "keepdims", "multiple_axes")


@cache
def original():
    return load_definitions(
        "alphafold3",
        "utils",
        ("mask_mean",),
        {"abc": abc, "Sequence": Sequence, "numbers": numbers, "jnp": jnp},
    )["mask_mean"]


def settings(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    return {
        "rows": (-1, False),
        "all_axes": (None, False),
        "keepdims": (1, True),
        "multiple_axes": ((1, 2), True),
    }[scenario]


def function(variant=None, *, scenario="rows", epsilon=EPSILON):
    axis, keepdims = settings(scenario)
    if variant is None:

        def baseline(mask, value):
            return original()(mask, value, axis=axis, keepdims=keepdims, eps=epsilon)

        return baseline
    if variant not in {row[0] for row in CASES}:
        raise ValueError(variant)

    def candidate(mask, value):
        axes = (
            tuple(range(value.ndim))
            if axis is None
            else ((axis,) if isinstance(axis, int) else axis)
        )
        factor = float(np.prod([value.shape[a] if mask.shape[a] == 1 else 1 for a in axes]))
        if variant == "broadcast_mask":
            mask = jnp.broadcast_to(mask, value.shape)
            count = jnp.sum(mask, axis=axes, keepdims=keepdims)
        else:
            count = jnp.sum(mask, axis=axes, keepdims=keepdims) * (
                1.0 if variant == "wrong_broadcast" else factor
            )
        denominator = (
            count + epsilon if variant == "wrong_additive_epsilon" else jnp.maximum(count, epsilon)
        )
        return jnp.sum(mask * value, axis=axes, keepdims=keepdims) / denominator

    return candidate


def inputs(seed, *, scenario="rows", zero=False, tiny=False):
    rng = np.random.default_rng(seed)
    shapes = {
        "rows": ((2, 1), (2, 3)),
        "all_axes": ((1, 3), (2, 3)),
        "keepdims": ((2, 1), (2, 3)),
        "multiple_axes": ((2, 1, 3), (2, 2, 3)),
    }
    mask_shape, value_shape = shapes[scenario]
    mask = rng.uniform(0.2, 1.0, size=mask_shape).astype(np.float32)
    if zero:
        mask.fill(0)
    elif tiny:
        mask.fill(0.01)
    values = rng.uniform(0.5, 2.0, size=value_shape).astype(np.float32)
    return mask, values


def reference(mask, value, *, scenario="rows", epsilon=EPSILON):
    axis, keepdims = settings(scenario)
    values = np.asarray(value, np.float64)
    weights = np.broadcast_to(np.asarray(mask, np.float64), values.shape)
    count = np.sum(weights, axis=axis, keepdims=keepdims)
    return np.sum(weights * values, axis=axis, keepdims=keepdims) / np.clip(count, epsilon, None)
