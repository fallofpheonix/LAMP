class DependencyUnavailableError(RuntimeError):
    """Raised when an optional runtime dependency is missing."""


class DataValidationError(ValueError):
    """Raised when dataset validation cannot proceed due to invalid inputs."""
