"""Pinned paired upstream formulas test conversion separately from whole-program comparison."""

import hashlib
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jax._src import compiler, dispatch

from jax_reorder import Status, compare_functions
from jax_reorder.algebra_ir import Budget
from jax_reorder.algebra_symbolic import SymbolicBuilder
from jax_reorder.algebra_transport import freeze
from jax_reorder.algebra_trusted import equal


@pytest.fixture(scope="module")
def pairs(tmp_path_factory):
    destination = tmp_path_factory.mktemp("nerva-sources")
    for package in ("nerva_jax", "nerva_sympy"):
        source = Path(__file__).parent / "fixtures" / package
        manifest = json.loads((source / "manifest.json").read_text())
        for entry in manifest["files"]:
            data = (source / (entry["path"] + ".txt")).read_bytes()
            assert hashlib.sha256(data).hexdigest() == entry["sha256"]
            target = destination / Path(entry["path"]).relative_to("src")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    sys.path.insert(0, str(destination))
    modules = {
        name: importlib.import_module(name)
        for name in (
            "nerva_jax.softmax_functions",
            "nerva_sympy.softmax_functions",
            "nerva_jax.activation_functions",
            "nerva_sympy.activation_functions",
            "nerva_jax.matrix_operations",
            "nerva_sympy.matrix_operations",
            "nerva_jax.loss_functions",
            "nerva_sympy.loss_functions",
        )
    }
    result = {}
    matrix = jax.ShapeDtypeStruct((2, 3), jnp.float32)
    for name in ("softmax", "stable_softmax", "log_softmax", "stable_log_softmax"):
        result[name] = (
            getattr(modules["nerva_jax.softmax_functions"], name),
            getattr(modules["nerva_sympy.softmax_functions"], name),
            (matrix,),
        )
    for name in ("Sigmoid", "Sigmoid_gradient", "Hyperbolic_tangent", "Relu"):
        result[name] = (
            getattr(modules["nerva_jax.activation_functions"], name),
            getattr(modules["nerva_sympy.activation_functions"], name),
            (matrix,),
        )
    for name in ("hadamard", "product"):
        result[name] = (
            getattr(modules["nerva_jax.matrix_operations"], name),
            getattr(modules["nerva_sympy.matrix_operations"], name),
            (matrix, matrix if name == "hadamard" else jax.ShapeDtypeStruct((3, 2), jnp.float32)),
        )
    for name in ("Squared_error_loss", "Squared_error_loss_gradient"):
        result[name] = (
            getattr(modules["nerva_jax.loss_functions"], name),
            getattr(modules["nerva_sympy.loss_functions"], name),
            (matrix, matrix),
        )
    yield result
    sys.path.remove(str(destination))
    for name in list(sys.modules):
        if name.startswith(("nerva_jax", "nerva_sympy")):
            del sys.modules[name]


NAMES = (
    "softmax",
    "stable_softmax",
    "log_softmax",
    "stable_log_softmax",
    "Sigmoid",
    "Sigmoid_gradient",
    "Hyperbolic_tangent",
    "Relu",
    "hadamard",
    "product",
    "Squared_error_loss",
    "Squared_error_loss_gradient",
)


def forbidden(*args, **kwargs):
    raise AssertionError("Nerva verification must not execute JAX arrays")


@pytest.mark.parametrize("name", NAMES)
def test_conversion_against_independent_upstream_formula(pairs, name):
    function, reference, args = pairs[name]
    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        graph = freeze(jax.make_jaxpr(function)(*args))
        builder = SymbolicBuilder()
        output = builder.graph(graph)[0]
        inputs = [builder.input(var.aval, index) for index, var in enumerate(graph.invars)]
        matrices = [
            sp.Matrix(
                value.shape[0],
                value.shape[1],
                lambda i, j, value=value: value.read((sp.Integer(i), sp.Integer(j)))[0],
            )
            for value in inputs
        ]
        expected = reference(*matrices)
        for index in np.ndindex(output.shape):
            expression, domain = output.read(tuple(map(sp.Integer, index)))
            target = expected[index] if output.shape else expected
            assert equal(expression.doit(), target, Budget(10000)) is True
            assert sp.simplify(domain.doit()) == sp.true


@pytest.mark.parametrize("name", NAMES)
def test_whole_jax_pair_uses_public_comparator(pairs, name):
    function, _, args = pairs[name]

    # The oracle is scalar field arithmetic, independent of the converter's mapping.
    def equivalent(*xs):
        return (function(*xs) + 1) - 1

    def wrong(*xs):
        return function(*xs) + 1

    jax.devices()
    with (
        patch.object(compiler, "backend_compile_and_load", forbidden),
        patch.object(dispatch, "apply_primitive", forbidden),
    ):
        result = compare_functions(function, equivalent, *args, algebra_timeout=15)
        assert result.status == Status.EQUIVALENT, result.modes[0].issues
        result = compare_functions(function, wrong, *args, algebra_timeout=15)
        assert result.status == Status.INEQUIVALENT, result.modes[0].issues


@pytest.mark.parametrize(
    "name,rewritten",
    [
        ("Sigmoid", lambda x: jnp.exp(x) / (1 + jnp.exp(x))),
        ("Relu", lambda x: (x + jnp.abs(x)) / 2),
        ("product", lambda x, y: jnp.sum(x[:, :, None] * y[None, :, :], axis=1)),
        ("Squared_error_loss", lambda x, y: jnp.sum(x * x - 2 * x * y + y * y)),
        ("Squared_error_loss_gradient", lambda x, y: 2 * x - 2 * y),
    ],
)
def test_independent_handwritten_formula(pairs, name, rewritten):
    original, _, args = pairs[name]
    result = compare_functions(original, rewritten, *args)
    assert result.status == Status.EQUIVALENT, result.modes[0].issues
