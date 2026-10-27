"""Pinned AlphaFold 2 masked means, fixed AI rewrites and independent references."""

import collections
import numbers
from functools import cache

import jax.numpy as jnp
import numpy as np

from tests.upstream_tpu.sources import load_definitions

REVISION = "c77e5d2a8961d1a353632c462914ff0a32a950f6"
EPSILON = 0.125
CASES = (
    ("reordered", True, True),
    ("broadcast_mask", False, True),
    ("wrong_broadcast", False, False),
    ("wrong_clamped_epsilon", False, False),
)
SCENARIOS = ("rows", "all_axes", "drop_channel", "multiple_axes")


@cache
def original():
    return load_definitions(
        "alphafold",
        "utils",
        ("mask_mean",),
        {"collections": collections, "numbers": numbers, "jnp": jnp},
    )["mask_mean"]


def settings(scenario):
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    return {
        "rows": (-1, False),
        "all_axes": (None, False),
        "drop_channel": (1, True),
        "multiple_axes": ((1, 2), False),
    }[scenario]


def function(variant=None, *, scenario="rows", epsilon=EPSILON):
    axis, drop = settings(scenario)
    if variant is None:

        def baseline(mask, value):
            return original()(mask, value, axis=axis, drop_mask_channel=drop, eps=epsilon)

        return baseline
    if variant not in {row[0] for row in CASES}:
        raise ValueError(variant)

    def candidate(mask, value):
        if drop:
            mask = mask[..., 0]
        axes = (
            tuple(range(value.ndim))
            if axis is None
            else ((axis,) if isinstance(axis, int) else axis)
        )
        factor = float(np.prod([value.shape[a] if mask.shape[a] == 1 else 1 for a in axes]))
        if variant == "broadcast_mask":
            mask = jnp.broadcast_to(mask, value.shape)
            count = jnp.sum(mask, axis=axes)
        else:
            count = jnp.sum(mask, axis=axes) * (1.0 if variant == "wrong_broadcast" else factor)
        denominator = (
            jnp.maximum(count, epsilon) if variant == "wrong_clamped_epsilon" else count + epsilon
        )
        return jnp.sum(mask * value, axis=axes) / denominator

    return candidate


def inputs(seed, *, scenario="rows", zero=False):
    rng = np.random.default_rng(seed)
    shapes = {
        "rows": ((2, 1), (2, 3)),
        "all_axes": ((1, 3), (2, 3)),
        "drop_channel": ((2, 1, 1), (2, 3)),
        "multiple_axes": ((2, 1, 3), (2, 2, 3)),
    }
    mask_shape, value_shape = shapes[scenario]
    mask = rng.uniform(0.2, 1.0, size=mask_shape).astype(np.float32)
    if zero:
        mask.fill(0)
    values = rng.uniform(0.5, 2.0, size=value_shape).astype(np.float32)
    return mask, values


def reference(mask, value, *, scenario="rows", epsilon=EPSILON):
    axis, drop = settings(scenario)
    mask = np.asarray(mask[..., 0] if drop else mask, np.float64)
    values = np.asarray(value, np.float64)
    weights = np.broadcast_to(mask, values.shape)
    return np.sum(weights * values, axis=axis) / (np.sum(weights, axis=axis) + epsilon)
