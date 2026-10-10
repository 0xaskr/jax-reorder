"""Structural canonicalization and comparison of CPU JAX computation graphs."""

from .canonical import Canonicalized, Comparison, Status, canonicalize, compare_jaxprs
from .errors import InvalidJaxprError, JaxprError, UnsupportedJaxprError

__all__ = [
    "Canonicalized",
    "Comparison",
    "InvalidJaxprError",
    "JaxprError",
    "Status",
    "UnsupportedJaxprError",
    "canonicalize",
    "compare_jaxprs",
]

__version__ = "0.1.0"
