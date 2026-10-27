"""Original legacy ControlNet helpers with fixed AI candidates and NumPy oracles.

The pinned script contains actual Flax TPU training code. Only its complete SNR
and gradient-norm definitions execute here; no model imports or training scaffold
are replaced. The scheduler dependency supplies a concrete cumulative-alpha array.
"""

from functools import cache
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from tests.upstream_tpu.sources import load_nested_definition

REVISION = "3b56af0c5268049588cf470e38955f4c18002498"
SNR_CASES = (
    ("reordered", True, True),
    ("ratio_of_squares", False, True),
    ("wrong_scale", False, False),
)
NORM_CASES = (
    ("reordered", True, True),
    ("sum_squares", False, True),
    ("wrong_mean", False, False),
)


def schedule_values():
    betas = np.linspace(0.01, 0.15, 8, dtype=np.float32)
    return np.cumprod(1.0 - betas, dtype=np.float32)


@cache
def originals():
    state = SimpleNamespace(common=SimpleNamespace(alphas_cumprod=jnp.asarray(schedule_values())))
    namespace = {"jax": jax, "jnp": jnp, "noise_scheduler_state": state}
    return (
        load_nested_definition("diffusers", "training", "main.compute_snr", namespace),
        load_nested_definition("diffusers", "training", "main.train_step.l2", namespace),
    )


def snr_function(variant=None):
    if variant is None:
        return originals()[0]
    if variant not in {row[0] for row in SNR_CASES}:
        raise ValueError(variant)
    alphas = jnp.asarray(schedule_values())

    def candidate(timesteps):
        # Move the independent noise branch first; the original gather stays intact.
        sigma = ((1.0 - alphas) ** 0.5)[timesteps]
        alpha = (alphas**0.5)[timesteps]
        if variant == "ratio_of_squares":
            return alpha**2 / sigma**2
        snr = (alpha / sigma) ** 2
        return snr * 1.5 if variant == "wrong_scale" else snr

    return candidate


def snr_inputs(seed):
    rng = np.random.default_rng(seed)
    # Include both schedule endpoints as well as seeded interior steps.
    return np.asarray([0, *rng.integers(1, 7, size=2), 7], dtype=np.int32)


def snr_reference(timesteps):
    alphas = schedule_values().astype(np.float64)[np.asarray(timesteps)]
    return alphas / (1.0 - alphas)


def norm_function(variant=None):
    if variant is None:
        return originals()[1]
    if variant not in {row[0] for row in NORM_CASES}:
        raise ValueError(variant)

    def candidate(xs):
        leaves = jax.tree_util.tree_leaves(xs)
        if variant == "reordered":
            # Evaluate leaves in reverse order, then restore their reduction order.
            terms = [jnp.vdot(value, value) for value in reversed(leaves)]
            return jnp.sqrt(sum(reversed(terms)))
        reductions = [
            jnp.mean(jnp.square(value)) if variant == "wrong_mean" else jnp.sum(jnp.square(value))
            for value in leaves
        ]
        return jnp.sqrt(sum(reductions))

    return candidate


def norm_inputs(seed):
    rng = np.random.default_rng(seed)
    return {
        "weights": rng.normal(size=(2, 3)).astype(np.float32),
        "bias": rng.normal(size=(4,)).astype(np.float32),
        "nested": (rng.uniform(0.2, 1.4, size=(2,)).astype(np.float32),),
    }


def norm_reference(xs):
    # Explicit NumPy assembly avoids the source's tree traversal and dot reductions.
    values = np.concatenate(
        [np.asarray(xs[name], np.float64).ravel() for name in ("weights", "bias")]
        + [np.asarray(xs["nested"][0], np.float64).ravel()]
    )
    return np.linalg.norm(values)
