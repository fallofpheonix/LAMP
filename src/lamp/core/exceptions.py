"""Custom exception hierarchy for the LAMP package.

Centralises domain-specific error types so callers can catch narrow
exception classes rather than broad built-ins.
"""


class DependencyUnavailableError(RuntimeError):
    """Raised when an optional runtime dependency is missing."""


class DataValidationError(ValueError):
    """Raised when dataset validation cannot proceed due to invalid inputs."""
