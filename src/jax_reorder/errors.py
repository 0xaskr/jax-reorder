"""Explicit failures: never discard unknown semantics to obtain a match."""


class JaxprError(ValueError):
    """A graph could not be processed, with a location suitable for a report."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class InvalidJaxprError(JaxprError):
    """The input violates the jaxpr representation or typing rules."""


class UnsupportedJaxprError(JaxprError):
    """Some semantics cannot yet be interpreted completely."""
