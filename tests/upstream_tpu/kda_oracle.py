"""Independent dense KDA recurrence and official TPU CPU interpreter evidence.

The NumPy oracle uses token-by-token state updates and their manual adjoints.
It does not call any mega/WY/intra helper or the graph comparator. Public dg is
with respect to natural-log gate increments; the kernel takes cumulative log2
coordinates. Samples construct consistent saved states and triangular factors.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from tests.upstream_tpu import kda_cases as cases


def samples(seed=7, config=cases.DEFAULT_CONFIG):
    rng = np.random.default_rng(seed)
    t, k, v = config.chunks * 64, 128, 128

    def quantize(x):
        return np.asarray(jnp.asarray(x, dtype=config.dtype), dtype=np.float64)

    q, key = (quantize(rng.normal(0, 0.08, (t, k))) for _ in range(2))
    value, do = (quantize(rng.normal(0, 0.1, (t, v))) for _ in range(2))
    beta = quantize(rng.uniform(0.3, 0.7, t))
    raw_gate = rng.uniform(-0.03, -0.01, (t, k))
    # Make the input coordinates exact f32 first, then define the oracle gate
    # increments from those coordinates to avoid a separate coordinate error.
    g = np.cumsum(raw_gate.reshape(config.chunks, 64, k), axis=1) / np.log(2.0)
    g = g.astype(np.float32)
    gate = np.diff(g.astype(np.float64), axis=1, prepend=np.zeros((config.chunks, 1, k)))
    gate = gate.reshape(t, k) * np.log(2.0)
    initial = rng.normal(0, 0.02, (k, v)).astype(np.float32).astype(np.float64)
    dht = rng.normal(0, 0.02, (k, v)).astype(np.float32).astype(np.float64)
    states, decayed, residuals, output_states = [], [], [], []
    state = initial.copy()
    for i in range(t):
        if i % 64 == 0:
            states.append(state.copy())
        p = np.exp(gate[i])[:, None] * state
        residual = value[i] - key[i] @ p
        state = p + beta[i] * np.outer(key[i], residual)
        decayed.append(p)
        residuals.append(residual)
        output_states.append(state)

    aqk, akk = [], []
    for chunk in range(config.chunks):
        start, stop = chunk * 64, (chunk + 1) * 64
        kc, qc, bc = key[start:stop], q[start:stop], beta[start:stop]
        decay = np.exp2(g[chunk, :, None, :].astype(np.float64) - g[chunk, None, :, :])
        qk = np.sum(qc[:, None, :] * kc[None, :, :] * decay, axis=-1) * 128**-0.5
        kk = bc[:, None] * np.sum(kc[:, None, :] * kc[None, :, :] * decay, axis=-1)
        aqk.append(np.pad(np.tril(qk), ((0, 0), (0, 64))))
        inverse = np.linalg.solve(np.eye(64) + np.tril(kk, -1), np.eye(64))
        akk.append(np.pad(inverse, ((0, 0), (0, 64))))

    gradients = [
        np.zeros_like(q),
        np.zeros_like(key),
        np.zeros_like(value),
        np.zeros_like(beta),
        np.zeros_like(gate),
    ]
    dq, dk, dv, db, dg = gradients
    dh = dht.copy()
    for i in reversed(range(t)):
        dq[i] = (output_states[i] @ do[i]) * 128**-0.5
        dh = dh + np.outer(q[i] * 128**-0.5, do[i])
        kt_dh = key[i] @ dh
        dv[i] = beta[i] * kt_dh
        db[i] = kt_dh @ residuals[i]
        dk[i] = beta[i] * (dh @ residuals[i]) - decayed[i] @ dv[i]
        dp = dh - np.outer(key[i], dv[i])
        dg[i] = np.sum(dp * decayed[i], axis=-1)
        dh = dp * np.exp(gate[i])[:, None]
    reference = tuple(x[None, None] for x in gradients) + (dh[None, None, None],)
    inputs = (
        q,
        key,
        value,
        beta,
        np.concatenate(aqk),
        np.concatenate(akk),
        g.reshape(t, k),
        np.stack(states),
        do,
        dht,
        initial,
    )
    arguments = tuple(
        jnp.asarray(value.reshape(signature.shape), signature.dtype)
        for value, signature in zip(inputs, cases.signatures(config), strict=True)
    )
    return arguments, reference


def differences(actual, reference):
    metrics = {}
    for name, observed, expected in zip(cases.OUTPUTS, actual, reference, strict=True):
        a, b = np.asarray(observed, np.float64), np.asarray(expected, np.float64)
        if a.shape != b.shape:
            raise AssertionError(f"{name} output shape mismatch: {a.shape} != {b.shape}")
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        error = a - b
        metrics[name] = {
            "finite": finite,
            "max_abs": float(np.max(np.abs(error))) if finite else None,
            "relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(b), 1e-30))
            if finite
            else None,
            "close": bool(np.allclose(a, b, rtol=3e-4, atol=3e-6)) if finite else False,
            "exact": bool(np.array_equal(a, b)),
        }
    return metrics


def interpreter_results(config=cases.DEFAULT_CONFIG, seeds=(7, 23), variants=None):
    """Execute actual launchers; return metrics without promoting samples to proofs."""
    from jax._src.pallas.mosaic.interpret import interpret_pallas_call as interpreter

    variants = variants or cases.CANDIDATES
    params = pltpu.InterpretParams(
        detect_races=True,
        skip_floating_point_ops=False,
        uninitialized_memory="nan",
        out_of_bounds_reads="raise",
        buffer_bounds="logical",
        num_cores_or_threads=1,
    )
    compiled = {}
    for revision, variant in variants:
        if not cases.supports_configuration(revision, config):
            continue
        with pltpu.force_tpu_interpret_mode(params):
            compiled[(revision, variant)] = (
                jax.jit(cases.function(revision, variant, config))
                .lower(*cases.signatures(config))
                .compile()
            )
    print("READY", flush=True)
    results = []
    for seed in seeds:
        args, reference = samples(seed, config)
        base = None
        for (revision, variant), function in compiled.items():
            actual = jax.block_until_ready(function(*args))
            jax.effects_barrier()
            race = interpreter.races is None or interpreter.races.races_found
            if (revision, variant) == ("latest", "original"):
                base = actual
            if base is None:
                raise ValueError("interpreter variants must start with the latest baseline")
            results.append(
                {
                    "revision": revision,
                    "variant": variant,
                    "seed": seed,
                    "races": bool(race),
                    "vs_base": differences(actual, base),
                    "vs_recurrence": differences(actual, reference),
                }
            )
    return results


def worker(config=None, output=None, variants=None, seeds=(7, 23)):
    from pathlib import Path

    settings = cases.Configuration(**(config or {}))
    results = interpreter_results(settings, seeds, variants)
    if output is not None:
        Path(output).write_text(
            json.dumps(
                {
                    "configuration": dataclasses.asdict(settings),
                    "results": results,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
    print("RESULT " + json.dumps({"case": "kda", "status": "ok"}), flush=True)


def validate_results(config, results):
    """Assert labels on complete outputs; historical measurements remain unlabelled."""
    for row in results:
        assert not row["races"], row
        assert all(metric["finite"] for metric in row["vs_base"].values()), row
        if config.dtype == "float32" and row["variant"] in (
            "original",
            "dav_reordered",
            "dav_scale_inside",
        ):
            assert all(metric["close"] for metric in row["vs_recurrence"].values()), row
        if row["variant"] == "dav_reordered":
            assert all(metric["exact"] for metric in row["vs_base"].values()), row
        if row["variant"] == "dav_scale_inside":
            assert all(metric["close"] for metric in row["vs_base"].values()), row
        if cases.AI_EXPECTATIONS.get(row["variant"]) == "inequivalent":
            assert any(not metric["close"] for metric in row["vs_base"].values()), row


def regression_worker(config=None):
    settings = cases.Configuration(**(config or {}))
    results = interpreter_results(settings)
    validate_results(settings, results)
    # The Oct 9 implementation retains an f32 intra cotangent until reverse
    # cumsum; latest intentionally rounds via the BF16 output ref first.
    if settings.cast_gate:
        older = [r for r in results if r["revision"] == "oct09"]
        assert older and all(not r["vs_base"]["dg"]["close"] for r in older)
        assert all(
            metric["exact"]
            for row in older
            for name, metric in row["vs_base"].items()
            if name != "dg"
        )
    print("RESULT " + json.dumps({"case": "kda_regression", "status": "ok"}), flush=True)
