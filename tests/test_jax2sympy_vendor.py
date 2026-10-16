"""Keep the audited upstream-style front end reproducible without external checkouts."""

import sys

import jax
from jax._src import compiler, dispatch

from jax_reorder._vendor import jax2sympy
from jax_reorder._vendor.jax2sympy import primitive_mapping
from tools.experiments.equivalence_backends.architecture_reuse_probe import jax2sympy_probe


def test_pinned_fork_retains_the_23_case_explicit_baseline(monkeypatch):
    jax.devices()

    def forbidden(*args, **kwargs):
        raise AssertionError("conversion baseline must not execute JAX")

    monkeypatch.setattr(compiler, "backend_compile_and_load", forbidden)
    monkeypatch.setattr(dispatch, "apply_primitive", forbidden)
    monkeypatch.setitem(sys.modules, "jax2sympy", jax2sympy)
    monkeypatch.setitem(sys.modules, "jax2sympy.primitive_mapping", primitive_mapping)
    report = jax2sympy_probe()
    assert report["default_mapping_repeat"] == "two_calls_succeeded"
    assert len(report["records"]) == 23
    for row in report["records"]:
        if row["case"] == "conditional_region":
            # Regions use the production builder hook, not the explicit baseline.
            assert row["status"] == "error"
            assert "cond" in row["error"]
        else:
            assert row["status"] == "converted", row
            expected = row["case"] not in ("changed_coefficient", "ordered_outputs")
            assert all(q["all_equal"] is expected for q in row["reference_checks"]), row
