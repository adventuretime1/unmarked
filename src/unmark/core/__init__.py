"""Core contracts.

``core`` depends only on the standard library and Pydantic. It must never import
Typer, Rich, Torch, Transformers, or any vendor SDK.
"""

from unmark.core.budgets import BudgetAccount, BudgetLease, BudgetUsage, RunBudget
from unmark.core.document import Document, MediaType
from unmark.core.errors import ExitCode, UnmarkedError
from unmark.core.events import Event, EventRecorder, JobState
from unmark.core.operations import Operation, apply_operations, invert_operations
from unmark.core.policies import FidelityPolicy, UnicodePolicy
from unmark.core.results import CandidateResult, ResultState, RunResult
from unmark.core.spans import AtomicClaim, Block, Span
from unmark.core.targets import ReductionTarget, ScoreEvidence, StrategyDescriptor

__all__ = [
    "AtomicClaim",
    "Block",
    "BudgetAccount",
    "BudgetLease",
    "BudgetUsage",
    "CandidateResult",
    "Document",
    "Event",
    "EventRecorder",
    "ExitCode",
    "FidelityPolicy",
    "JobState",
    "MediaType",
    "Operation",
    "ReductionTarget",
    "ResultState",
    "RunBudget",
    "RunResult",
    "ScoreEvidence",
    "Span",
    "StrategyDescriptor",
    "UnicodePolicy",
    "UnmarkedError",
    "apply_operations",
    "invert_operations",
]
