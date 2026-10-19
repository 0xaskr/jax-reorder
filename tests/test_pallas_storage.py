"""A small dense oracle checks the interval store independently of DMA rules."""

import random

import pytest

from jax_reorder import UnsupportedJaxprError
from jax_reorder.core.index_domain import AnalysisBudget
from jax_reorder.core.model import Origin
from jax_reorder.core.storage import Span, Storage


def test_interval_values_initialization_and_versions_against_dense_oracle():
    rng = random.Random(400)
    for _ in range(40):
        size = 16
        storage = Storage(size, None, Origin("initial"), AnalysisBudget(100000))
        dense, versions = [None] * size, [0] * size
        for step in range(1, 40):
            start = rng.randrange(size)
            stop = rng.randrange(start + 1, size + 1)
            value = rng.randrange(20)
            storage.write(Span(start, stop), value, Origin(f"write{step}"), overwrite=True)
            dense[start:stop] = [value] * (stop - start)
            versions[start:stop] = [step] * (stop - start)
            actual = storage.read(Span(0, size), "oracle", initialized=False)
            assert [p.value for p in actual for _ in range(p.span.size)] == dense
            assert [p.version for p in actual for _ in range(p.span.size)] == versions
            if None in dense:
                with pytest.raises(UnsupportedJaxprError, match="uninitialized"):
                    storage.read(Span(0, size), "oracle")


def test_unique_output_writes_allow_a_partition_but_not_overlap():
    storage = Storage(256, None, Origin("output"), AnalysisBudget())
    storage.write(Span(128, 256), 2, Origin("upper"), overwrite=False)
    storage.write(Span(0, 128), 1, Origin("lower"), overwrite=False)
    assert [p.value for p in storage.read(Span(0, 256), "read")] == [1, 2]
    with pytest.raises(UnsupportedJaxprError, match="overlap"):
        storage.write(Span(127, 129), 3, Origin("bad"), overwrite=False)
