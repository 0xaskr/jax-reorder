"""Exact owner partitions preserve coverage and retain large symbolic ranges."""

from itertools import product

import pytest

from jax_reorder import ComparisonBudgetError, UnsupportedJaxprError
from jax_reorder.core.index_domain import (
    AnalysisBudget,
    add,
    axis,
    bounds,
    divide,
    integer,
    modulo,
    scale,
)
from jax_reorder.core.tensor_owner_partition import partition_owner_ranges


@pytest.mark.parametrize("reverse", [False, True])
def test_complete_owners_coalesce_before_large_grid_replay(reverse):
    count = 1_000_000
    owner = axis("program:0")
    expr = add(integer(count - 1), scale(owner, -1)) if reverse else owner
    budget = AnalysisBudget(1000)
    partitions = partition_owner_ranges(
        ((expr, count - 1),), {"program:0": (0, count - 1)}, ("program:0",), budget, "test"
    )
    assert len(partitions) == 2
    assert sorted(part.intervals[0][1][1] - part.intervals[0][1][0] + 1 for part in partitions) == [
        1,
        count - 1,
    ]


@pytest.mark.parametrize("mixed", [False, True])
def test_every_owner_has_one_partition_and_all_tail_requirements_agree(mixed):
    ranges = {"program:0": (0, 11)} if mixed else {"program:0": (0, 2), "program:1": (0, 3)}
    first = axis("program:0")
    requirements = (
        ((divide(first, 4), 2), (modulo(first, 4), 3))
        if mixed
        else ((first, 2), (axis("program:1"), 3))
    )
    names = tuple(ranges)
    partitions = partition_owner_ranges(requirements, ranges, names, AnalysisBudget(), "test")
    for point in product(*(range(low, high + 1) for low, high in ranges.values())):
        exact = {name: (value, value) for name, value in zip(names, point, strict=True)}
        containing = [
            part
            for part in partitions
            if all(low <= exact[name][0] <= high for name, (low, high) in part.intervals)
        ]
        assert len(containing) == 1
        assert containing[0].tails == tuple(
            bounds(expr, exact)[0] == tail for expr, tail in requirements
        )


def test_partition_does_not_split_an_external_device_coordinate():
    with pytest.raises(UnsupportedJaxprError, match="cannot classify"):
        partition_owner_ranges(
            ((axis("device:0"), 1),),
            {"program:0": (0, 3), "device:0": (0, 1)},
            ("program:0",),
            AnalysisBudget(),
            "test",
        )


def test_partition_work_is_charged_to_the_shared_analysis_budget():
    with pytest.raises(ComparisonBudgetError):
        partition_owner_ranges(
            ((axis("program:0"), 999_999),),
            {"program:0": (0, 999_999)},
            ("program:0",),
            AnalysisBudget(8),
            "test",
        )
