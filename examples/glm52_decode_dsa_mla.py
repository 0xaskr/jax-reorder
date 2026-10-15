"""用 GLM-5.2 纯 JAX DSA/MLA 核心的 5+5 手写改写检查 jaxpr 比较器。

uv run python -m examples.glm52_decode_dsa_mla --json /tmp/glm52-jaxpr.json
只构造 ShapeDtypeStruct、追踪和比较；不编译、不执行 layer、不采样数值。
退出码：0=比较完成且无缺口，1=预期关系冲突，2=有 unknown 能力缺口。
来源及 Apache-2.0 许可证见 tests/fixtures/sglang_jax/。
"""

import argparse
import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from jax_reorder import Status, compare_function_traces, trace_function

ROOT = Path(__file__).resolve().parents[1] / "tests/fixtures/sglang_jax"
SOURCE = "python/sgl_jax/srt/kernels/dsa/ref.py"
REVISION = "c98ced4a7a64e83261f6672937ce41f0095828ba"
SCALE = 256**-0.5


@dataclass(frozen=True)
class Profile:
    batch: int
    heads: int
    latent: int
    rope: int
    index_heads: int
    index_dim: int
    page_size: int
    pages_per_seq: int
    top_k: int


PROFILES = {
    "small": Profile(2, 2, 8, 4, 3, 8, 4, 3, 4),
    "glm": Profile(2, 64, 512, 64, 32, 128, 128, 32, 2048),
}


@functools.cache
def upstream():
    """校验后加载原始 reference，保留全部定义和装饰器。"""
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


def _qk(q, keys, scale):
    logits = jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), keys.astype(jnp.float32))
    return logits * scale


def _qk_scale_first(q, keys, scale):
    """E3：把 (Q Kᵀ) s 改为 (Q s) Kᵀ，cast 位置保持不变。"""
    return jnp.einsum("thd,tkd->thk", q.astype(jnp.float32) * scale, keys.astype(jnp.float32))


def _qk_without_scale(q, keys, scale):
    """N1：遗漏 softmax 温度；两个不同 logits 的相对概率会改变。"""
    return jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), keys.astype(jnp.float32))


def _pv(probabilities, values):
    return jnp.einsum("thk,tkd->thd", probabilities, values.astype(jnp.float32))


def _pv_reduce(probabilities, values):
    """E4：把 P V 改为广播乘积及沿选中 token 维度求和。"""
    return jnp.sum(probabilities[..., None] * values[:, None, :, :].astype(jnp.float32), axis=2)


def _valid_indices(indices, length, in_seq):
    return (indices >= 0) & (indices < length) & in_seq[:, None]


def _negative_index_is_valid(indices, length, in_seq):
    """N2：-1 padding 被当作第 0 个 token，重复参与 softmax 与 value 归约。"""
    return (indices < length) & in_seq[:, None]


def _key_softmax(logits):
    return jax.nn.softmax(logits, axis=-1)


def _head_softmax(logits):
    """N3：错误地沿 attention head 轴归一化。"""
    return jax.nn.softmax(logits, axis=1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "pages_per_seq",
        "v_dim",
        "qk",
        "pv",
        "validity",
        "softmax",
        "reorder",
    ),
)
def _mla_core(
    q,
    cache_kv,
    kv_lens,
    topk_indices,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    *,
    sm_scale,
    pages_per_seq,
    v_dim,
    qk=_qk,
    pv=_pv,
    validity=_valid_indices,
    softmax=_key_softmax,
    reorder=False,
):
    """手写原 sparse_mla_ref 骨架；公式 helper 在追踪时展开到循环体。"""
    tokens, heads, _ = q.shape
    _, page_size, width = cache_kv.shape
    max_kv = pages_per_seq * page_size
    out = jnp.zeros((tokens, heads, v_dim), dtype=q.dtype)

    def body(seq_id, out):
        q_start, q_end = cu_q_lens[seq_id], cu_q_lens[seq_id + 1]
        if reorder:
            # E1：这些节点不依赖 paged KV gather，可提前计算而不改变操作数。
            q_pos = jnp.arange(tokens)
            in_seq = (q_pos >= q_start) & (q_pos < q_end)
            kv_len = kv_lens[seq_id]
            idx = jnp.where(topk_indices >= 0, topk_indices, 0)
            valid = validity(topk_indices, kv_len, in_seq)
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        kv_flat = cache_kv[seq_pages].reshape(max_kv, width)
        if not reorder:
            q_pos = jnp.arange(tokens)
            in_seq = (q_pos >= q_start) & (q_pos < q_end)
            kv_len = kv_lens[seq_id]
            idx = jnp.where(topk_indices >= 0, topk_indices, 0)
            valid = validity(topk_indices, kv_len, in_seq)
        selected = kv_flat[idx]
        logits = qk(q, selected, sm_scale)
        logits = jnp.where(valid[:, None, :], logits, float("-inf"))
        probabilities = softmax(logits)
        probabilities = jnp.where(valid[:, None, :], probabilities, 0.0)
        values = selected[..., :v_dim]
        output = pv(probabilities, values).astype(q.dtype)
        return jnp.where(in_seq[:, None, None], output, out)

    return jax.lax.fori_loop(0, kv_lens.shape[0], body, out)


def _head_scores(weights, products):
    return jnp.einsum("th,thk->tk", weights, jax.nn.relu(products))


def _head_scores_reduce(weights, products):
    """E5：将 DSA 的逐 head 加权点积改为乘积再归约，保留 ReLU 位置。"""
    return jnp.sum(weights[:, :, None] * jax.nn.relu(products), axis=1)


def _head_scores_without_relu(weights, products):
    """N4：负的 QK 项也参与打分，可能改变 top-k 排名。"""
    return jnp.einsum("th,thk->tk", weights, products)


def _visible(positions, length):
    return (positions < length)[None, :]


def _inclusive_visible(positions, length):
    """N5：把第 length 个无效位置也纳入 indexer 候选集。"""
    return (positions <= length)[None, :]


@functools.partial(
    jax.jit,
    static_argnames=("k", "pages_per_seq", "one_token_per_seq", "score", "visible", "reorder"),
)
def _dsa_core(
    q,
    weights,
    cache_kv,
    seq_lens,
    page_indices,
    cu_q_lens,
    cu_kv_lens,
    distribution,
    *,
    k,
    pages_per_seq,
    one_token_per_seq=True,
    score=_head_scores,
    visible=_visible,
    reorder=False,
):
    """手写 streamindex_topk_ref 的 decode 分支，保留两级 top-k 和 -1 padding。"""
    if not one_token_per_seq:
        raise ValueError("this candidate covers one-token decode only")
    tokens, _, width = q.shape
    page_size = cache_kv.shape[1]
    max_kv = pages_per_seq * page_size
    weights32 = weights.astype(jnp.float32)
    out = jnp.full((tokens, k), -1, dtype=jnp.int32)

    def body(seq_id, out):
        # 原始 decode 分支保留 compression_ratio=1 的整数除法。
        length = seq_lens[seq_id] // 1
        if reorder:
            # E2：query/weight 的读取移到独立的页表读取之前。
            query = jax.lax.dynamic_slice_in_dim(q, seq_id, 1, axis=0)
            gates = jax.lax.dynamic_slice_in_dim(weights32, seq_id, 1, axis=0)
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        keys = cache_kv[seq_pages].reshape(max_kv, width)
        if not reorder:
            query = jax.lax.dynamic_slice_in_dim(q, seq_id, 1, axis=0)
            gates = jax.lax.dynamic_slice_in_dim(weights32, seq_id, 1, axis=0)
        products = jnp.einsum("thd,kd->thk", query, keys, preferred_element_type=jnp.float32)
        scores = score(gates, products)
        mask = visible(jnp.arange(max_kv), length)
        scores = jnp.where(mask, scores, float("-inf"))
        candidate_values, candidate_ids = jax.lax.approx_max_k(
            scores, k, recall_target=0.70, aggregate_to_topk=False
        )
        values, selection = jax.lax.top_k(candidate_values, k)
        indices = jnp.take_along_axis(candidate_ids, selection, axis=-1)
        n_valid = mask.sum(-1, keepdims=True)
        indices = jnp.where(
            (jnp.arange(k)[None, :] < n_valid) & (values > float("-inf")), indices, -1
        )
        return jax.lax.dynamic_update_slice_in_dim(out, indices, seq_id, axis=0)

    return jax.lax.fori_loop(0, seq_lens.shape[0], body, out)


def mla_schedule(*args, **kwargs):
    return _mla_core(*args, **kwargs, reorder=True)


def dsa_schedule(*args, **kwargs):
    return _dsa_core(*args, **kwargs, reorder=True)


def mla_scale_first(*args, **kwargs):
    return _mla_core(*args, **kwargs, qk=_qk_scale_first)


def mla_value_reduce(*args, **kwargs):
    return _mla_core(*args, **kwargs, pv=_pv_reduce)


def dsa_head_reduce(*args, **kwargs):
    return _dsa_core(*args, **kwargs, score=_head_scores_reduce)


def mla_missing_scale(*args, **kwargs):
    return _mla_core(*args, **kwargs, qk=_qk_without_scale)


def mla_unmasked_padding(*args, **kwargs):
    return _mla_core(*args, **kwargs, validity=_negative_index_is_valid)


def mla_wrong_softmax_axis(*args, **kwargs):
    return _mla_core(*args, **kwargs, softmax=_head_softmax)


def dsa_missing_relu(*args, **kwargs):
    return _dsa_core(*args, **kwargs, score=_head_scores_without_relu)


def dsa_inclusive_length(*args, **kwargs):
    return _dsa_core(*args, **kwargs, visible=_inclusive_visible)


@dataclass(frozen=True)
class Case:
    function: object
    target: str
    expected: str
    rationale: str

    @property
    def name(self):
        return self.function.__name__


CASES = (
    Case(mla_schedule, "mla", "equivalent", "MLA 循环内 mask/索引计算提前，依赖保持不变"),
    Case(dsa_schedule, "dsa", "equivalent", "DSA 循环内 query/gate 读取提前，依赖保持不变"),
    Case(mla_scale_first, "mla", "equivalent", "理想实数：(Q Kᵀ)s = (Qs)Kᵀ"),
    Case(mla_value_reduce, "mla", "equivalent", "理想实数：P V = sum_k(P_k V_k)"),
    Case(dsa_head_reduce, "dsa", "equivalent", "理想实数：head 点积 = 加权逐项求和；ReLU 不动"),
    Case(
        mla_missing_scale,
        "mla",
        "inequivalent",
        "logits 为 [0,1] 时 softmax([0,s]) 与 softmax([0,1]) 不同",
    ),
    Case(
        mla_unmasked_padding,
        "mla",
        "inequivalent",
        "均匀 logits、V=[0,1]，两个 -1 padding 被算为 V0：输出 1/2 变为 1/4",
    ),
    Case(
        mla_wrong_softmax_axis,
        "mla",
        "inequivalent",
        "均匀 logits、V 全 1 时，沿 head 归一化的输出为 K/H，而正确输出为 1",
    ),
    Case(
        dsa_missing_relu,
        "dsa",
        "inequivalent",
        "两 head 权重为 1，两个位置乘积 [2,-10] 与 [1,0]，删 ReLU 会颠倒排名",
    ),
    Case(
        dsa_inclusive_length,
        "dsa",
        "inequivalent",
        "无效位置 length 的高分可进入 top-k，改变索引输出及注意力候选",
    ),
)


def signatures(profile="glm", dtype=jnp.float32):
    p = PROFILES[profile]
    pages = p.batch * p.pages_per_seq
    abstract = jax.ShapeDtypeStruct
    metadata = (
        abstract((p.batch,), jnp.int32),
        abstract((pages,), jnp.int32),
        abstract((p.batch + 1,), jnp.int32),
        abstract((p.batch + 1,), jnp.int32),
        abstract((3,), jnp.int32),
    )
    return (
        abstract((p.batch, p.index_heads, p.index_dim), dtype),
        abstract((p.batch, p.index_heads), dtype),
        abstract((pages, p.page_size, p.index_dim), dtype),
        abstract((p.batch, p.heads, p.latent + p.rope), dtype),
        abstract((pages, p.page_size, p.latent + p.rope), dtype),
        metadata,
    )


def function(profile="glm", case=None):
    """只替换指定核心；原始基线直接调用两个未修改的上游 reference。"""
    p = PROFILES[profile]
    indexer = upstream().streamindex_topk_ref
    attention = upstream().sparse_mla_ref
    if case is not None:
        if case.target == "dsa":
            indexer = case.function
        else:
            attention = case.function

    def layer(q_idx, weights, index_cache, q, cache, metadata):
        lens, pages, cu_q, cu_kv, distribution = metadata
        indices = indexer(
            q_idx,
            weights,
            index_cache,
            lens,
            pages,
            cu_q,
            cu_kv,
            distribution,
            k=p.top_k,
            pages_per_seq=p.pages_per_seq,
            one_token_per_seq=True,
        )
        output = attention(
            q,
            cache,
            lens,
            indices,
            pages,
            cu_q,
            cu_kv,
            distribution,
            sm_scale=SCALE,
            pages_per_seq=p.pages_per_seq,
            v_dim=p.latent,
        )
        return output, indices

    return layer


def classify(expected, status):
    if status == Status.UNKNOWN:
        return "coverage_gap"
    if expected == "equivalent":
        return "proven_equivalent" if status == Status.EQUIVALENT else "unexpected_mismatch"
    if status == Status.INEQUIVALENT:
        return "proven_inequivalent"
    return "false_acceptance" if status == Status.EQUIVALENT else "structural_mismatch_only"


def compare(case, base_trace, args, profile="glm"):
    candidate = trace_function(function(profile, case), *args)
    comparison = compare_function_traces(base_trace, candidate)
    record = {
        "name": case.name,
        "target": case.target,
        "expected_relation": case.expected,
        "rationale": case.rationale,
        "status": comparison.status.value,
        "structural": comparison.comparison.status.value if comparison.comparison else None,
        "kind": comparison.equivalence_kind,
        "outcome": classify(case.expected, comparison.status),
        "failure_kind": comparison.failure_kind,
        "issues": [{"path": x.path, "reason": x.reason} for x in comparison.issues],
        "difference": comparison.difference,
        "warnings": comparison.algebra.warnings if comparison.algebra else (),
        "rules": comparison.algebra.rules if comparison.algebra else (),
    }
    return record, candidate


def exit_code(report):
    outcomes = {row["outcome"] for row in report["comparisons"]}
    if outcomes & {"false_acceptance", "unexpected_mismatch"}:
        return 1
    return 2 if outcomes & {"coverage_gap", "structural_mismatch_only"} else 0


def run(profile="glm", dtype=jnp.float32, jaxpr_dir=None):
    args = signatures(profile, dtype)
    base_trace = trace_function(function(profile), *args)
    if jaxpr_dir is not None:
        jaxpr_dir = Path(jaxpr_dir)
        jaxpr_dir.mkdir(parents=True, exist_ok=True)
        (jaxpr_dir / "base.jaxpr").write_text(str(base_trace.jaxpr) + "\n")
    result = {
        "source": f"https://github.com/sgl-project/sglang-jax/blob/{REVISION}/{SOURCE}",
        "profile": profile,
        "dtype": str(jnp.dtype(dtype)),
        "jax": jax.__version__,
        "mode": "forward",
        "compiled": False,
        "executed": False,
        "input_signature": [
            {"shape": list(x.shape), "dtype": str(x.dtype)} for x in jax.tree.leaves(args)
        ],
        "scope": (
            "decode DSA/MLA core; projected queries and updated caches; abstract jaxpr comparison"
        ),
        "comparisons": [],
    }
    for case in CASES:
        record, candidate = compare(case, base_trace, args, profile)
        result["comparisons"].append(record)
        if jaxpr_dir is not None:
            (jaxpr_dir / f"{case.name}.jaxpr").write_text(str(candidate.jaxpr) + "\n")
        print(
            f"{case.name}: expected={case.expected} structural={record['structural']} "
            f"result={record['status']} [{record['outcome']}]",
            flush=True,
        )
        for issue in record["issues"]:
            print(f"  {issue['path']}: {issue['reason']}", flush=True)
    result["summary"] = {
        name: sum(row["outcome"] == name for row in result["comparisons"])
        for name in (
            "proven_equivalent",
            "proven_inequivalent",
            "structural_mismatch_only",
            "coverage_gap",
            "false_acceptance",
            "unexpected_mismatch",
        )
    }
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="glm")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--jaxpr-dir", type=Path, help="保存 base 和 10 个候选的 jaxpr 文本")
    options = parser.parse_args()
    report = run(options.profile, jnp.dtype(options.dtype), options.jaxpr_dir)
    if options.json:
        options.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
