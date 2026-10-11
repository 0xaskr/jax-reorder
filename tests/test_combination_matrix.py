"""交叉原语、精度、拓扑三元组与独立数值参照；AI 标签可被测试否定。"""

import jax
import numpy as np
import pytest
from jax._src import core

from examples.combination_matrix import CASES, reference, sample
from examples.full_layers.runner import numerical_witness, raw_schedule
from examples.sglang_layers.runner import close
from jax_reorder import Status, canonicalize, compare_jaxprs


@pytest.mark.parametrize(
    "case,dtype",
    [(case, dtype) for case in CASES for dtype in case.dtypes],
    ids=lambda x: x.key if hasattr(x, "key") else np.dtype(x).name,
)
def test_fixed_cross_family_triples(case, dtype):
    with jax.enable_x64():
        inputs = [sample(case, dtype, seed) for seed in (0, 1, 2)]
        graphs = [jax.make_jaxpr(fn)(*inputs[0]) for fn in case.programs]
        snapshots = tuple(map(str, graphs))
        assert case.primitives <= {eq.primitive.name for eq in graphs[0].eqns}
        # 固定三元组的每一对都比较，检查自反、对称及传递一致性。
        for a in graphs[:3]:
            for b in graphs[:3]:
                match = compare_jaxprs(a, b)
                assert match.status == Status.EQUIVALENT and match.correspondence is not None
        assert all(raw_schedule(graphs[0]) != raw_schedule(g) for g in graphs[1:3])
        assert compare_jaxprs(graphs[0], graphs[3]).status == Status.DIFFERENT
        witness = None
        for graph, fn in zip(graphs, case.programs, strict=True):
            core.check_jaxpr(graph)
            normal = canonicalize(graph)
            core.check_jaxpr(normal.jaxpr)
            assert normal.graph == canonicalize(normal.jaxpr).graph
            compiled = jax.jit(fn)
            reordered = jax.jit(lambda *xs, g=normal.jaxpr: core.eval_jaxpr(g, g.consts, *xs))
            for args in inputs:
                value = compiled(*args)
                close(value, reordered(*args))
                if fn is not case.programs[-1]:
                    close(
                        value,
                        reference(case, args),
                        low_precision=np.dtype(dtype).name == "bfloat16",
                        oracle=True,
                    )
                else:
                    witness = witness or numerical_witness(case.programs[0](*args), value)
        assert witness is not None
        assert snapshots == tuple(map(str, graphs))
