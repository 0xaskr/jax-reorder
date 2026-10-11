"""逐层验收：固定 AI 预判、精确图比较、独立参照、数值反例与重排执行。"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import nnx
from jax._src import core

from jax_reorder import Status, canonicalize, compare_jaxprs

from .builders import PROFILES, baseline, build, cpu_mesh, sample, split
from .candidates import FAMILIES, REASONS
from .catalog import COMPONENT_LAYERS, CONFIG_LAYERS, LAYERS, MODEL_LAYERS, MULTIMODAL_LAYERS
from .oracles import reference
from .upstream import REVISION


def values(tree):
    return [np.asarray(value) for value in jax.tree.leaves(tree)]


def close(left, right, *, low_precision=False, oracle=False):
    """结构、shape 和 dtype 先独立检查；容差仅用于数值验证。"""
    a, b = values(left), values(right)
    assert len(a) == len(b)
    rtol, atol = ((0.035, 0.035) if low_precision else (3e-5, 3e-5)) if oracle else (2e-5, 2e-5)
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_allclose(x.astype(np.float64), y.astype(np.float64), rtol=rtol, atol=atol)


def snapshot(tree):
    return [
        {"shape": list(v.shape), "dtype": str(v.dtype), "values": v.astype(np.float64).tolist()}
        for v in values(tree)
    ]


def verify(spec, profile, *, show_jaxpr=False):
    mesh = cpu_mesh()
    results = []
    with jax.set_mesh(mesh):
        model = build(spec, profile, mesh)
        graphdef, params, rest = split(model)

        def base(p, *x):
            return baseline(spec, nnx.merge(graphdef, p, rest), *x)

        samples = [sample(spec, profile, mesh, params, seed) for seed in range(5)]
        base_graph = jax.make_jaxpr(base)(*samples[0])
        compiled_base = jax.jit(base)
        original = str(base_graph)
        expected_statuses = (Status.EQUIVALENT, Status.DIFFERENT)
        for fn, expected, reason in zip(
            FAMILIES[spec.family], expected_statuses, REASONS[spec.family], strict=True
        ):

            def candidate(p, *x, fn=fn):
                return fn(nnx.merge(graphdef, p, rest), *x)

            candidate_graph = jax.make_jaxpr(candidate)(*samples[0])
            if show_jaxpr:
                print(
                    f"\n{spec.key}/{profile.name}/{fn.__name__}\nBASE\n{base_graph}\nCANDIDATE\n{candidate_graph}"
                )
            report = compare_jaxprs(base_graph, candidate_graph)
            assert report.status == expected, (spec.key, profile.name, fn.__name__, report)
            assert str(base_graph) == original
            normalized = (report.left, report.right)
            for item in normalized:
                assert item is not None and item.complete
                core.check_jaxpr(item.jaxpr)
                assert canonicalize(item.jaxpr).graph == item.graph
            executables = [
                jax.jit(lambda *args, g=item.jaxpr: core.eval_jaxpr(g, g.consts, *args))
                for item in normalized
            ]
            compiled_candidate = jax.jit(candidate)
            witness = None
            for seed, args in enumerate(samples):
                base_output = compiled_base(*args)
                output = compiled_candidate(*args)
                oracle = reference(spec, nnx.merge(graphdef, args[0], rest), *args[1:])
                close(base_output, oracle, low_precision=profile.name == "bf16", oracle=True)
                assert jax.tree.structure(base_output) == jax.tree.structure(output)
                flat = jax.tree.leaves(args)
                close(base_output, executables[0](*flat))
                close(output, executables[1](*flat))
                if expected == Status.EQUIVALENT:
                    close(base_output, output)
                elif witness is None:
                    a, b = values(base_output), values(output)
                    if any(
                        not np.allclose(
                            x.astype(np.float64), y.astype(np.float64), rtol=2e-3, atol=2e-3
                        )
                        for x, y in zip(a, b, strict=True)
                    ):
                        witness = {
                            "seed": seed,
                            "inputs": snapshot(args),
                            "base_outputs": snapshot(base_output),
                            "candidate_outputs": snapshot(output),
                        }
            if expected == Status.DIFFERENT:
                assert witness is not None, (spec.key, profile.name, fn.__name__, "缺少数值反例")
            results.append(
                {
                    "layer": spec.key,
                    "profile": profile.name,
                    "candidate": fn.__name__,
                    "expected": expected.value,
                    "reason": reason,
                    "actual": report.status.value,
                    "difference": report.difference,
                    "samples": len(samples),
                    "raw_order_changed": str(candidate_graph) != original,
                    "witness": witness,
                }
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=[s.key for s in LAYERS])
    parser.add_argument("--profile", choices=[p.name for p in PROFILES])
    parser.add_argument("--report", type=Path)
    parser.add_argument("--show-jaxpr", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    results = []
    for spec in LAYERS:
        if args.layer and spec.key != args.layer:
            continue
        for profile in PROFILES:
            if args.profile and profile.name != args.profile:
                continue
            results.extend(verify(spec, profile, show_jaxpr=args.show_jaxpr))
            print(
                f"PASS {spec.key}/{profile.name}: equivalent + different，5 组独立输入/权重",
                flush=True,
            )
    if args.report:
        report = {
            "revision": REVISION,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "models": MODEL_LAYERS,
            "multimodal_models": MULTIMODAL_LAYERS,
            "configs": CONFIG_LAYERS,
            "components": COMPONENT_LAYERS,
            "results": results,
        }
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"通过 {len(results)} 个候选/签名，{sum(r['samples'] for r in results)} 组数值验证。")


if __name__ == "__main__":
    main()
