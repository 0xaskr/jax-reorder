"""Real FAVOR+ features and ALX distributed Gramian with fixed AI candidates."""

from functools import cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tests.upstream_tpu.sources import load_definitions


@cache
def originals():
    namespace = dict(jax=jax, jnp=jnp, lax=lax)
    performer = load_definitions(
        "google_research", "performer", ("exp_softmax_kernel_transformation",), namespace
    )["exp_softmax_kernel_transformation"]
    gramian = load_definitions("google_research", "als", ("compute_gramain",), namespace)[
        "compute_gramain"
    ]
    return performer, gramian


def performer(data, projection, *, is_query):
    return originals()[0](data, None, is_query, projection_matrix=projection)


def candidate(data, projection, *, is_query, variant):
    """Reorder independent statistics, distribute a scale, or change stabilization."""
    normalizer = 1.0 / jnp.sqrt(jnp.sqrt(data.shape[-1]))
    ratio = 1.0 / jnp.sqrt(projection.shape[0])
    diagonal = jnp.sum(jnp.square(data), axis=data.ndim - 1)
    diagonal = (diagonal / 2.0) * normalizer * normalizer
    diagonal = jnp.expand_dims(diagonal, axis=data.ndim - 1)
    projected = jnp.einsum("...lhd,md->...lhm", normalizer * data, projection)
    axes = (projected.ndim - 1,) if is_query else None
    stabilization = jnp.max(projected, axis=axes, keepdims=True)
    features = jnp.exp(projected - stabilization - diagonal)
    epsilon = 0.25 if variant == "wrong_epsilon" else 0.000001
    if variant == "distributed_scale":
        return ratio * features + ratio * epsilon
    return ratio * (features + epsilon)


def performer_reference(data, projection, *, is_query):
    data, projection = np.asarray(data, np.float64), np.asarray(projection, np.float64)
    normalizer = data.shape[-1] ** -0.25
    projected = np.einsum("...lhd,md->...lhm", normalizer * data, projection)
    diagonal = np.sum(data * data, axis=-1, keepdims=True) * normalizer**2 / 2
    stabilization = np.max(projected, axis=-1 if is_query else None, keepdims=True)
    return (np.exp(projected - stabilization - diagonal) + 0.000001) / np.sqrt(projection.shape[0])


def performer_samples(seed):
    rng = np.random.default_rng(seed)
    return tuple(
        rng.normal(scale=0.4, size=shape).astype(np.float32) for shape in ((1, 2, 1, 4), (3, 4))
    )


def gramian_einsum(embeddings):
    return lax.psum(jnp.einsum("ni,nj->ij", embeddings, embeddings), "i")


def gramian_wrong_average(embeddings):
    return lax.pmean(jnp.matmul(embeddings.T, embeddings), "i")


def distributed_gramian(variant):
    functions = {
        "original": originals()[1],
        "einsum": gramian_einsum,
        "wrong_average": gramian_wrong_average,
    }
    mesh = Mesh(np.asarray(jax.devices("cpu")[:2]), ("i",))
    return jax.shard_map(
        functions[variant], mesh=mesh, in_specs=P("i", None), out_specs=P(None, None)
    )


def performer_functions(is_query, variant):
    return partial(performer, is_query=is_query), partial(
        candidate, is_query=is_query, variant=variant
    )
