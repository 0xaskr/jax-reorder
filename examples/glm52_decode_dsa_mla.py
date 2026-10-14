"""GLM-5.2 decode DSA/MLA：开源纯 JAX reference + 手写 5 正例 / 5 反例。

运行：uv run python -m examples.glm52_decode_dsa_mla --json /tmp/glm52.json

边界：query 已投影并施加 RoPE，paged cache 已包含当前 token；这里执行
DSA 索引、query absorption、稀疏 MLA、value expansion 和输出投影。
只验证 FP32 的单层 decode，不代表完整模型或 BF16 舍入路径。
来源及 Apache-2.0 许可证见 tests/fixtures/sglang_jax/。
"""

import argparse
import functools
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from jax_reorder import Status, compare_functions, verify_algebraic_proof

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/sglang_jax"
SOURCE = "python/sgl_jax/srt/kernels/dsa/ref.py"
REVISION = "c98ced4a7a64e83261f6672937ce41f0095828ba"
SCALE = 256**-0.5
TOP_K = 4
PROFILES = {
    # batch, heads, latent, rope, nope, value, hidden, index_heads, index_dim
    "small": (2, 2, 8, 4, 6, 8, 10, 3, 8),
    "glm": (2, 2, 512, 64, 192, 256, 16, 32, 128),
}


@functools.cache
def upstream():
    """执行完整、未改写的纯 JAX 模块；哈希不符即停止。"""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if manifest["revision"] != REVISION:
        raise ValueError("unexpected upstream revision")
    entry = next(item for item in manifest["files"] if item["path"] == SOURCE)
    path = ROOT / (SOURCE + ".txt")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError("upstream reference checksum mismatch")
    namespace = {"__name__": "_glm52_dsa_reference"}
    exec(compile(data, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def select(q_idx, weights, index_cache, metadata):
    lens, pages, cu_q, cu_kv, distribution = metadata
    return upstream().streamindex_topk_ref(
        q_idx,
        weights,
        index_cache,
        lens,
        pages,
        cu_q,
        cu_kv,
        distribution,
        k=TOP_K,
        pages_per_seq=pages.size // lens.size,
        one_token_per_seq=True,
    )


def mla(q, cache, indices, metadata, latent, *, scale=SCALE):
    lens, pages, cu_q, cu_kv, distribution = metadata
    return upstream().sparse_mla_ref(
        q,
        cache,
        lens,
        indices,
        pages,
        cu_q,
        cu_kv,
        distribution,
        sm_scale=scale,
        pages_per_seq=pages.size // lens.size,
        v_dim=latent,
    )


def base(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """原始 reference 的组合；两次吸收投影遵循 Glm5Attention._forward_mqa。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    output = values.reshape(values.shape[0], -1) @ wo
    return output, ids


def query_first(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """E1：query absorption 与索引器相互独立，先计算 query。"""
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    ids = select(q_idx, weights, index_cache, metadata)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    return values.reshape(values.shape[0], -1) @ wo, ids


def query_broadcast(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """E2：query absorption 的 einsum 改成逐点乘积与归约。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.sum(q_nope[:, None, :, :] * uk[None, :, :, :], axis=-1)
    q = jnp.concatenate((q_latent.transpose(0, 2, 1), q_rope), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    return values.reshape(values.shape[0], -1) @ wo, ids


def value_broadcast(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """E3：value expansion 的 einsum 改成逐点乘积与归约。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    values = jnp.sum(latent[..., None] * uv.transpose(1, 0, 2)[None, ...], axis=2)
    return values.reshape(values.shape[0], -1) @ wo, ids


def fused_output(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """E4：先合并 value/output 权重，再将 latent 一次投影到输出。"""
    by_head = wo.reshape(uv.shape[1], uv.shape[2], wo.shape[-1])
    combined = jnp.einsum("rhd,hdo->hro", uv, by_head)
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    return jnp.einsum("thr,hro->to", latent, combined), ids


def headwise_output(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """E5：输出投影先逐 head 求值，再把各 head 的输出相加。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    by_head = wo.reshape(uv.shape[1], uv.shape[2], wo.shape[-1])
    partial = jnp.einsum("thd,hdo->tho", values, by_head)
    return jnp.sum(partial, axis=1), ids


def wrong_scale(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """N1：错误地用 absorbed 维度计算 softmax scale。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0], scale=q.shape[-1] ** -0.5)
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    return values.reshape(values.shape[0], -1) @ wo, ids


def wrong_rope(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """N2：删除 rotary query 对 logits 的贡献。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, jnp.zeros_like(q_rope)), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    return values.reshape(values.shape[0], -1) @ wo, ids


def wrong_selection(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """N3：计算 top-k 却用前 k 个位置参与注意力（连线错误）。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    first = jnp.broadcast_to(jnp.arange(TOP_K, dtype=jnp.int32), ids.shape)
    latent = mla(q, cache, first, metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    return values.reshape(values.shape[0], -1) @ wo, ids


def wrong_pages(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """N4：把逻辑页号当物理页号，索引器仍使用正确页表。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    lens, pages, cu_q, cu_kv, distribution = metadata
    bad_metadata = lens, jnp.arange(pages.size, dtype=jnp.int32), cu_q, cu_kv, distribution
    latent = mla(q, cache, ids, bad_metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv)
    return values.reshape(values.shape[0], -1) @ wo, ids


def wrong_value(q_nope, q_rope, q_idx, weights, cache, index_cache, uk, uv, wo, metadata):
    """N5：value 权重的 head 对应关系倒置，保持形状不变。"""
    ids = select(q_idx, weights, index_cache, metadata)
    q_latent = jnp.einsum("thd,rhd->thr", q_nope, uk)
    q = jnp.concatenate((q_latent, q_rope), axis=-1)
    latent = mla(q, cache, ids, metadata, uk.shape[0])
    values = jnp.einsum("thr,rhd->thd", latent, uv[:, ::-1, :])
    return values.reshape(values.shape[0], -1) @ wo, ids


EQUIVALENT = (query_first, query_broadcast, value_broadcast, fused_output, headwise_output)
INEQUIVALENT = (wrong_scale, wrong_rope, wrong_selection, wrong_pages, wrong_value)


def inputs(profile="glm", seed=17, *, lengths=(11, 2)):
    batch, heads, latent, rope, nope, value, hidden, index_heads, index_dim = PROFILES[profile]
    rng = np.random.default_rng(seed)

    def normal(shape, scale=1.0):
        return jnp.asarray(rng.normal(size=shape).astype(np.float32) * scale)

    # 每序列三页，每页四 token；打乱物理页防止错误页表侥幸通过。
    page_size, pages_per_seq = 4, 3
    pages = jnp.array([4, 1, 5, 0, 3, 2], dtype=jnp.int32)
    metadata = (
        jnp.asarray(lengths, dtype=jnp.int32),
        pages,
        jnp.arange(batch + 1, dtype=jnp.int32),
        jnp.arange(batch + 1, dtype=jnp.int32) * (page_size * pages_per_seq),
        jnp.array([batch, batch, batch], dtype=jnp.int32),
    )
    return (
        normal((batch, heads, nope)),
        normal((batch, heads, rope)),
        normal((batch, index_heads, index_dim)),
        normal((batch, index_heads)),
        normal((6, page_size, latent + rope)),
        normal((6, page_size, index_dim)),
        normal((latent, heads, nope), nope**-0.5),
        normal((latent, heads, value), latent**-0.5),
        normal((heads * value, hidden), (heads * value) ** -0.5),
        metadata,
    )


def numpy_reference(args):
    """独立 FP64 参照：逐序列/逐头计算，不调用任何 JAX 层 helper。"""
    qn, qr, qi, weights, cache, icache, uk, uv, wo = [
        np.asarray(x, dtype=np.float64) for x in args[:9]
    ]
    lens, pages, _, cu_kv, _ = map(np.asarray, args[9])
    result = np.zeros((qn.shape[0], wo.shape[-1]), dtype=np.float64)
    ids = np.full((qn.shape[0], TOP_K), -1, dtype=np.int32)
    for t, length in enumerate(lens):
        page_start = int(cu_kv[t]) // cache.shape[1]
        physical = pages[page_start : page_start + pages.size // lens.size]
        keys = icache[physical].reshape(-1, icache.shape[-1])[:length]
        scores = np.zeros(length, dtype=np.float64)
        for h in range(qi.shape[1]):
            scores += weights[t, h] * np.maximum(keys @ qi[t, h], 0)
        chosen = np.argsort(-scores, kind="stable")[:TOP_K]
        ids[t, : len(chosen)] = chosen
        if not len(chosen):
            continue
        selected = cache[physical].reshape(-1, cache.shape[-1])[chosen]
        values = []
        for h in range(qn.shape[1]):
            packed_q = np.concatenate((uk[:, h, :] @ qn[t, h], qr[t, h]))
            scores_h = selected @ packed_q * SCALE
            exp = np.exp(scores_h - scores_h.max())
            latent = (exp / exp.sum()) @ selected[:, : uk.shape[0]]
            values.append(latent @ uv[:, h, :])
        result[t] = np.concatenate(values) @ wo
    return result, ids


def check_numeric(function, args):
    expected, expected_ids = numpy_reference(args)
    actual, actual_ids = jax.jit(function)(*args)
    actual = np.asarray(actual)
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)
    return float(np.max(np.abs(actual - expected)))


def compare(function, args):
    report = compare_functions(base, function, *args, check_cpu=True)
    mode = report.modes[0]
    if function in EQUIVALENT:
        if report.status != Status.EQUIVALENT:
            raise AssertionError((function.__name__, report.status, mode.issues))
        if function is query_first:
            if mode.equivalence_kind != "structural":
                raise AssertionError("schedule-only variant must be structural")
        else:
            if mode.equivalence_kind != "algebraic" or not mode.algebra.warnings:
                raise AssertionError("algebraic variant must retain floating-point warning")
            verify_algebraic_proof(mode.left.jaxpr, mode.right.jaxpr, mode.algebra.proof)
    elif report.status == Status.EQUIVALENT or mode.comparison.status != Status.DIFFERENT:
        raise AssertionError((function.__name__, "expected structural mismatch", mode.issues))
    if not report.cpu_compiled:
        raise AssertionError((function.__name__, "CPU compilation failed"))
    return {
        "structural": mode.comparison.status.value,
        "status": report.status.value,
        "kind": mode.equivalence_kind,
        "cpu_compiled": report.cpu_compiled,
        "warnings": mode.algebra.warnings if mode.algebra else (),
        "issues": [{"path": x.path, "reason": x.reason} for x in mode.issues],
    }


def counterexample(function, args):
    left = np.asarray(jax.jit(base)(*args)[0])
    right = np.asarray(jax.jit(function)(*args)[0])
    mask = ~np.isclose(left, right, atol=2e-5, rtol=2e-4)
    if not np.isfinite(left).all() or not np.isfinite(right).all() or not mask.any():
        raise AssertionError((function.__name__, "missing finite output counterexample"))
    index = tuple(int(i) for i in np.argwhere(mask)[0])
    return {
        "index": index,
        "base": float(left[index]),
        "candidate": float(right[index]),
        "max_abs_difference": float(np.max(np.abs(left - right))),
    }


def run(profile="glm", trials=2):
    if trials < 1:
        raise ValueError("trials must be positive")
    args = inputs(profile)
    result = {
        "source": f"https://github.com/sgl-project/sglang-jax/blob/{REVISION}/{SOURCE}",
        "profile": profile,
        "dimensions": PROFILES[profile],
        "dtype": "float32",
        "jax": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "scope": "forward decode; projected queries and updated caches; CPU; no Pallas",
        "equivalent": [],
        "inequivalent": [],
    }
    for seed in range(17, 17 + trials):
        check_numeric(base, inputs(profile, seed))
    for function in EQUIVALENT:
        proof = compare(function, args)
        samples = [
            {"seed": seed, "max_abs_error": check_numeric(function, inputs(profile, seed))}
            for seed in range(17, 17 + trials)
        ]
        result["equivalent"].append({"name": function.__name__, "proof": proof, "samples": samples})
        print(f"PASS equivalent {function.__name__}: {proof['kind']}", flush=True)
    for function in INEQUIVALENT:
        proof = compare(function, args)
        witnesses = [
            {"seed": seed, **counterexample(function, inputs(profile, seed))}
            for seed in range(17, 17 + trials)
        ]
        result["inequivalent"].append(
            {"name": function.__name__, "comparison": proof, "counterexamples": witnesses}
        )
        print(
            f"PASS inequivalent {function.__name__}: numerical witness; graph={proof['status']}",
            flush=True,
        )
    print("5 equivalent + 5 inequivalent passed; algebra uses ideal reals, not bitwise equality.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="glm")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--json", type=Path)
    options = parser.parse_args()
    # 上游 approx_max_k 在 TPU 允许近似选择；此脚本的 exact NumPy top-k 参照只测 CPU。
    with jax.default_device(jax.devices("cpu")[0]):
        result = run(options.profile, options.trials)
    if options.json:
        options.json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
