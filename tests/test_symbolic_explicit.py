"""Keep the retained explicit converter's formula baseline reproducible."""

import jax
from jax._src import compiler, dispatch

from jax_reorder.algebra_explicit import jaxpr_to_sympy_expressions, primitive_to_sympy_op
from tools.experiments.equivalence_backends.architecture_reuse_probe import jax2sympy_probe


def test_explicit_converter_retains_the_23_case_baseline(monkeypatch):
    jax.devices()

    def forbidden(*args, **kwargs):
        raise AssertionError("conversion baseline must not execute JAX")

    monkeypatch.setattr(compiler, "backend_compile_and_load", forbidden)
    monkeypatch.setattr(dispatch, "apply_primitive", forbidden)
    report = jax2sympy_probe(convert=jaxpr_to_sympy_expressions, mappings=primitive_to_sympy_op)
    assert report["default_mapping_repeat"] == "two_calls_succeeded"
    assert len(report["records"]) == 23
    for row in report["records"]:
        if row["case"] == "conditional_region":
            # Regions use SymbolicBuilder, not the formula-only explicit baseline.
            assert row["status"] == "error"
            assert "cond" in row["error"]
        else:
            assert row["status"] == "converted", row
            expected = row["case"] not in ("changed_coefficient", "ordered_outputs")
            assert all(q["all_equal"] is expected for q in row["reference_checks"]), row
