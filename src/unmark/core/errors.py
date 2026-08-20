"""Error taxonomy for Unmarked.

Every error carries the process exit code documented in the CLI specification
(section 8.8). The CLI maps an uncaught :class:`UnmarkedError` to ``error.exit_code``
so command functions never hard-code numbers.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    """Process exit codes defined by the CLI specification."""

    SUCCESS = 0
    USAGE = 2
    ABSTAINED = 3
    UNSUPPORTED = 4
    DEPENDENCY_UNAVAILABLE = 5
    BUDGET_EXHAUSTED = 6
    VALIDATION_OR_WRITE_FAILED = 7
    PARTIAL_BATCH_FAILURE = 8
    INTERRUPTED_WITH_CHECKPOINT = 9


class UnmarkedError(Exception):
    """Base class for all Unmarked errors."""

    exit_code: ExitCode = ExitCode.VALIDATION_OR_WRITE_FAILED

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UsageError(UnmarkedError):
    """Invalid CLI usage or configuration."""

    exit_code = ExitCode.USAGE


class ConfigError(UsageError):
    """Configuration could not be loaded or validated."""


class AbstainedError(UnmarkedError):
    """No candidate met the reduction and fidelity targets."""

    exit_code = ExitCode.ABSTAINED


class UnsupportedError(UnmarkedError):
    """Unsupported media type, language, detector, or capability."""

    exit_code = ExitCode.UNSUPPORTED


class DependencyUnavailableError(UnmarkedError):
    """A required model or detector dependency is unavailable."""

    exit_code = ExitCode.DEPENDENCY_UNAVAILABLE


class BudgetExhaustedError(UnmarkedError):
    """A budget counter was exhausted before the target was met."""

    exit_code = ExitCode.BUDGET_EXHAUSTED


class ValidationError(UnmarkedError):
    """Source/output validation failed."""

    exit_code = ExitCode.VALIDATION_OR_WRITE_FAILED


class AtomicWriteError(UnmarkedError):
    """An atomic write could not be completed."""

    exit_code = ExitCode.VALIDATION_OR_WRITE_FAILED


class OperationError(ValidationError):
    """An operation set is invalid: out of bounds, overlapping, or misordered."""


class ResearchModeRequiredError(UsageError):
    """A research-only capability was requested without acknowledgement."""
