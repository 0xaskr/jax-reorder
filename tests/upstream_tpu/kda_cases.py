"""Pinned complete KDA backward mega kernels and fixed AI dependency rewrites.

Only import/environment plumbing is supplied here. The launcher, DMA pipeline,
ref writes, grid and six public gradient outputs come from the original source.
Every revision loads its own helpers. No computation is replaced with a stub.
"""

import ast
import dataclasses
import functools
import inspect
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int

from tests.upstream_tpu.sources import load_definitions, read_manifest, read_source

REPO = "pallas_kda"
REVISION = "a861cfb60717d74dff52ed9d508cc7eab1782c1b"
HISTORY = ("oct09", "sep08", "jul30")
OUTPUTS = ("dq", "dk", "dv", "db", "dg", "dh0")

# These labels describe source mathematics, independently of checker capability.
AI_EXPECTATIONS = {
    "dav_reordered": "equivalent",
    "dav_scale_inside": "algebraically_equivalent",
    "dav_wrong_scale": "inequivalent",
    "dv_wrong_transpose": "inequivalent",
    "cumsum_forward": "inequivalent",
}
CANDIDATES = (
    ("latest", "original"),
    *((r, "original") for r in HISTORY),
    *(("latest", name) for name in AI_EXPECTATIONS),
)


@functools.partial(jax.jit, static_argnames=["scale"])
def dav_reordered(bdo, bvn, bAqk, scale, segment_ids=None):
    """AI positive: compute dv0 before the independent dAqk product."""
    if segment_ids is not None:
        raise ValueError("AI candidates in this campaign require dense input")
    bt = bdo.shape[1]
    causal = jnp.arange(bt)[:, None] >= jnp.arange(bt)[None, :]
    masked_a = jnp.where(causal[None], bAqk, 0)
    dv0 = jnp.matmul(
        masked_a.transpose(0, 2, 1),
        bdo.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    da = jnp.matmul(
        bdo.astype(jnp.float32),
        bvn.transpose(0, 2, 1),
        preferred_element_type=jnp.float32,
    )
    return jnp.where(causal[None], da * scale, 0.0), dv0


@functools.partial(jax.jit, static_argnames=["scale"])
def dav_scale_inside(bdo, bvn, bAqk, scale, segment_ids=None):
    """AI mathematical positive: distribute the dAqk scale into the lhs of dot."""
    if segment_ids is not None:
        raise ValueError("AI candidates in this campaign require dense input")
    bt = bdo.shape[1]
    causal = jnp.arange(bt)[:, None] >= jnp.arange(bt)[None, :]
    da = jnp.matmul(
        bdo.astype(jnp.float32) * scale,
        bvn.transpose(0, 2, 1),
        preferred_element_type=jnp.float32,
    )
    masked_a = jnp.where(causal[None], bAqk, 0)
    dv0 = jnp.matmul(
        masked_a.transpose(0, 2, 1),
        bdo.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    )
    return jnp.where(causal[None], da, 0.0), dv0


@functools.partial(jax.jit, static_argnames=["scale"])
def dav_wrong_scale(bdo, bvn, bAqk, scale, segment_ids=None):
    """AI negative: double only the intra dAqk scale; keep the dv0 gradient."""
    return dav_reordered(bdo, bvn, bAqk, 2.0 * scale, segment_ids)


@functools.partial(jax.jit, static_argnames=["scale"])
def dv_wrong_transpose(bdo, bvn, bAqk, scale, segment_ids=None):
    """AI negative: use A rather than A.T in the dv0 contraction."""
    da, _ = dav_reordered(bdo, bvn, bAqk, scale, segment_ids)
    bt = bdo.shape[1]
    causal = jnp.arange(bt)[:, None] >= jnp.arange(bt)[None, :]
    masked_a = jnp.where(causal[None], bAqk, 0)
    dv0 = jnp.matmul(masked_a, bdo.astype(jnp.float32), preferred_element_type=jnp.float32)
    return da, dv0


@jax.jit
def cumsum_forward(dg_total):
    """AI negative: accumulate gate gradients from the start instead of the end."""
    return jax.lax.associative_scan(jnp.add, dg_total, axis=1)


def _all_definitions(key):
    tree = ast.parse(read_source(REPO, key))
    names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                names.append(node.target.id)
    return tuple(names)


@functools.cache
def source_namespace(revision="latest", variant="original"):
    if revision not in ("latest", *HISTORY):
        raise ValueError(f"unknown pinned KDA revision: {revision}")
    if variant != "original" and (revision != "latest" or variant not in AI_EXPECTATIONS):
        raise ValueError(f"unsupported KDA candidate: {revision}/{variant}")
    namespace = dict(
        jax=jax,
        jnp=jnp,
        np=np,
        pl=pl,
        pltpu=pltpu,
        functools=functools,
        partial=functools.partial,
        math=math,
        dataclass=dataclasses.dataclass,
        Array=Array,
        Float=Float,
        Int=Int,
        # Fixed source defaults, independent of the developer shell's tuning knobs.
        os=SimpleNamespace(environ={}),
    )

    def load(kind, names=None):
        key = f"{revision}_{kind}"
        values = load_definitions(REPO, key, names or _all_definitions(key), namespace)
        namespace.update(values)
        return values

    hardware = load("hardware")
    hardware["set_tpu_config"](hardware["TPU_V7"])
    load("cp", ("CPContext",))
    load("oputils", ("get_interpret",))
    # Upstream cdiv is overloaded under the same name; pl.cdiv is the same
    # ceiling-division formula for the nonnegative static tiling sizes used here.
    namespace["cdiv"] = pl.cdiv
    load("utils", ("align_up", "assert_shape", "assert_shape_or_none"))
    files = read_manifest(REPO)["files"]
    if f"{revision}_cost" in files:
        load("cost")
    if f"{revision}_segmented" in files:
        load(
            "segmented",
            (
                "_expand_token_mask",
                "fixed_width_prefix_sum",
                "segmented_prefix_sum",
                "segmented_reverse_sum",
                "segmented_suffix_last",
                "same_segment_causal_mask",
            ),
        )
    # Pure helpers contain the mathematical work; sequential helper writes
    # into the actual mega kernel's output refs.
    load(
        "fusion",
        (
            "_chunk_segment_metadata",
            "_chunk_window_has_valid",
            "_matmul_in_dtype",
            "compute_m1_recompute",
            "compute_wy_backward",
            "compute_intra_backward_sequential_to_chunk_refs",
            "compute_reverse_cumsum_dg_scan",
        ),
    )
    if variant == "cumsum_forward":
        namespace["compute_reverse_cumsum_dg_scan"] = cumsum_forward
    names = _all_definitions(f"{revision}_mega")
    if variant in AI_EXPECTATIONS and variant != "cumsum_forward":
        namespace["compute_dav"] = globals()[variant]
        names = tuple(n for n in names if n != "compute_dav")
    load("mega", names)
    return namespace


@dataclasses.dataclass(frozen=True)
class Configuration:
    chunks: int = 1
    dtype: str = "float32"
    recompute: bool = False
    cast_gate: bool = False
    chunk_batch: int = 1

    @property
    def name(self):
        suffix = "_recompute" if self.recompute else "_saved"
        if self.cast_gate:
            suffix += "_gate_bf16"
        return f"nt{self.chunks}_{self.dtype}_cb{self.chunk_batch}{suffix}"


DEFAULT_CONFIG = Configuration()
CONFIGURATIONS = (
    DEFAULT_CONFIG,
    Configuration(chunks=2),
    Configuration(chunks=2, dtype="bfloat16"),
    Configuration(chunks=2, recompute=True),
    Configuration(chunks=2, dtype="bfloat16", cast_gate=True),
    Configuration(chunks=3, chunk_batch=2),
)


def signatures(config=DEFAULT_CONFIG):
    dtype = jnp.dtype(config.dtype)
    token = (1, 1, config.chunks * 64, 128)
    # q, k, v, beta, Aqk, Akk, g, h, do, dht, initial_state.
    shapes = (
        token,
        token,
        token,
        token[:3],
        (1, 1, config.chunks * 64, 128),
        (1, 1, config.chunks * 64, 128),
        token,
        (1, 1, config.chunks, 128, 128),
        token,
        (1, 1, 1, 128, 128),
        (1, 1, 1, 128, 128),
    )
    return tuple(
        jax.ShapeDtypeStruct(shape, jnp.float32 if i in (6, 7, 9, 10) else dtype)
        for i, shape in enumerate(shapes)
    )


def supports_configuration(revision, config):
    parameters = inspect.signature(source_namespace(revision)["chunk_kda_bwd_fusion"]).parameters
    return not config.cast_gate or "cast_dg_to_q_dtype" in parameters


@functools.cache
def function(revision="latest", variant="original", config=DEFAULT_CONFIG):
    launcher = source_namespace(revision, variant)["chunk_kda_bwd_fusion"]
    parameters = inspect.signature(launcher).parameters
    if config.cast_gate and "cast_dg_to_q_dtype" not in parameters:
        raise ValueError(f"{revision} has no cast_dg_to_q_dtype option")
    options = dict(
        scale=128**-0.5,
        mini_batch=1,
        disable_recompute=not config.recompute,
        has_initial_state=True,
        return_dh0=True,
        cp_active=False,
        chunk_batch=config.chunk_batch,
    )
    if config.cast_gate:
        options["cast_dg_to_q_dtype"] = True
    return functools.partial(launcher, **options)


@functools.cache
def trace(revision="latest", variant="original", config=DEFAULT_CONFIG):
    return jax.make_jaxpr(function(revision, variant, config))(*signatures(config))


def pallas_calls(value):
    if isinstance(value, core.ClosedJaxpr):
        value = value.jaxpr
    if isinstance(value, core.Jaxpr):
        for equation in value.eqns:
            if equation.primitive.name == "pallas_call":
                yield equation
            yield from pallas_calls(equation.params)
    elif isinstance(value, dict):
        for child in value.values():
            yield from pallas_calls(child)
    elif isinstance(value, tuple | list):
        for child in value:
            yield from pallas_calls(child)


def primitive_counts(value):
    counts = {}
    seen = set()

    def visit(item):
        if isinstance(item, core.ClosedJaxpr):
            item = item.jaxpr
        if isinstance(item, core.Jaxpr):
            if id(item) in seen:
                return
            seen.add(id(item))
            for equation in item.eqns:
                name = equation.primitive.name
                counts[name] = counts.get(name, 0) + 1
                visit(equation.params)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, tuple | list):
            for child in item:
                visit(child)

    visit(value)
    return dict(sorted(counts.items()))
