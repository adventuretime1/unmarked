"""A replaceable, cheap-first fidelity pipeline for localized edits."""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from unmark.core.budgets import BudgetAccount
from unmark.core.document import Document
from unmark.core.errors import OperationError
from unmark.core.operations import (
    Operation,
    apply_operations,
    char_edit_ratio,
    length_drift_ratio,
    validate_operations,
)
from unmark.core.policies import FidelityPolicy
from unmark.core.spans import StrictModel
from unmark.detectors.localization import TextRegion
from unmark.inspect.structure import parse_blocks

GateStatus = Literal["passed", "failed", "unavailable"]


class GateResult(StrictModel):
    gate_id: str
    status: GateStatus
    hard: bool = True
    reason: str = ""


class FidelityReport(StrictModel):
    passed: bool
    gates: tuple[GateResult, ...]
    candidate_text: str
    char_edit_ratio: float = Field(ge=0.0, le=1.0)
    token_edit_ratio: float = Field(ge=0.0, le=1.0)
    length_drift_ratio: float = Field(ge=0.0, le=1.0)

    @property
    def rejection_reason(self) -> str | None:
        failed = next((gate for gate in self.gates if gate.status == "failed"), None)
        return None if failed is None else f"{failed.gate_id}: {failed.reason}"


@runtime_checkable
class SemanticGate(Protocol):
    id: str

    def check(self, source: str, candidate: str, policy: FidelityPolicy) -> GateResult: ...


def token_edit_ratio(source: str, candidate: str) -> float:
    source_tokens = re.findall(r"\w+|[^\w\s]", source, flags=re.UNICODE)
    candidate_tokens = re.findall(r"\w+|[^\w\s]", candidate, flags=re.UNICODE)
    if not source_tokens:
        return 0.0 if not candidate_tokens else 1.0
    matcher = difflib.SequenceMatcher(a=source_tokens, b=candidate_tokens, autojunk=False)
    unchanged = sum(block.size for block in matcher.get_matching_blocks())
    return min(1.0, 1.0 - unchanged / max(len(source_tokens), len(candidate_tokens), 1))


def _operation_is_local(op: Operation, regions: Sequence[TextRegion]) -> bool:
    return any(region.start <= op.start and op.end <= region.end for region in regions)


def _mapped_offset(source_offset: int, operations: Sequence[Operation]) -> int:
    drift = sum(op.length_delta for op in operations if op.end <= source_offset)
    return source_offset + drift


class BasicFidelityEvaluator:
    """Deterministic gates plus optional semantic/style adapters.

    Missing advanced gates are explicitly reported as unavailable.  They are not
    silently replaced by embedding similarity or treated as factual equivalence.
    """

    def __init__(
        self,
        *,
        semantic_gates: tuple[SemanticGate, ...] = (),
        style_gate: SemanticGate | None = None,
        fluency_gate: SemanticGate | None = None,
    ) -> None:
        self.semantic_gates = semantic_gates
        self.style_gate = style_gate
        self.fluency_gate = fluency_gate

    def missing_required_capabilities(self, policy: FidelityPolicy) -> tuple[str, ...]:
        missing: list[str] = []
        if policy.require_bidirectional_entailment and not self.semantic_gates:
            missing.append("bidirectional entailment / claim equivalence")
        if policy.domain_validator is not None and not any(
            gate.id == policy.domain_validator for gate in self.semantic_gates
        ):
            missing.append(f"domain validator {policy.domain_validator!r}")
        return tuple(missing)

    def evaluate(
        self,
        document: Document,
        operations: tuple[Operation, ...],
        policy: FidelityPolicy,
        budget: BudgetAccount,
        editable_regions: tuple[TextRegion, ...],
    ) -> FidelityReport:
        gates: list[GateResult] = []

        try:
            ordered = validate_operations(operations, document.source_text)
            candidate = apply_operations(document.source_text, ordered)
            candidate.encode("utf-8", errors="strict")
            gates.append(GateResult(gate_id="unicode_materialization", status="passed"))
        except (OperationError, UnicodeError) as error:
            gates.append(
                GateResult(gate_id="unicode_materialization", status="failed", reason=str(error))
            )
            return FidelityReport(
                passed=False,
                gates=tuple(gates),
                candidate_text=document.source_text,
                char_edit_ratio=0,
                token_edit_ratio=0,
                length_drift_ratio=0,
            )

        if all(_operation_is_local(op, editable_regions) for op in ordered):
            gates.append(GateResult(gate_id="operation_locality", status="passed"))
        else:
            gates.append(
                GateResult(
                    gate_id="operation_locality",
                    status="failed",
                    reason="an operation escaped its explicitly editable region",
                )
            )

        char_ratio = char_edit_ratio(document.source_text, candidate, ordered)
        token_ratio = token_edit_ratio(document.source_text, candidate)
        length_ratio = length_drift_ratio(document.source_text, candidate)
        limits = budget.budget
        violations = []
        if char_ratio > limits.max_char_edit_ratio:
            violations.append("character edit ratio")
        if token_ratio > limits.max_token_edit_ratio:
            violations.append("token edit ratio")
        if length_ratio > limits.max_length_drift_ratio:
            violations.append("length drift")
        gates.append(
            GateResult(
                gate_id="edit_budgets",
                status="failed" if violations else "passed",
                reason=", ".join(violations),
            )
        )

        protected_failure = ""
        for span in document.protected_spans:
            if any(span.overlaps(op.start, op.end) for op in ordered):
                protected_failure = (
                    f"operation overlaps protected {span.kind} at {span.start}:{span.end}"
                )
                break
            mapped_start = _mapped_offset(span.start, ordered)
            mapped_end = mapped_start + len(span.value)
            observed = candidate[mapped_start:mapped_end]
            expected = span.value
            if span.policy == "normalized_equal":
                observed = unicodedata.normalize("NFKC", observed)
                expected = unicodedata.normalize("NFKC", expected)
            if observed != expected:
                protected_failure = f"protected {span.kind} changed at {span.start}:{span.end}"
                break
        gates.append(
            GateResult(
                gate_id="protected_spans",
                status="failed" if protected_failure else "passed",
                reason=protected_failure,
            )
        )

        source_shape = tuple(block.kind for block in document.blocks)
        candidate_shape = tuple(
            block.kind for block in parse_blocks(candidate, document.media_type)
        )
        structure_changed = source_shape != candidate_shape
        gates.append(
            GateResult(
                gate_id="document_structure",
                status="failed" if structure_changed else "passed",
                reason=(
                    f"block shape changed from {source_shape!r} to {candidate_shape!r}"
                    if structure_changed
                    else ""
                ),
            )
        )

        for semantic_gate in self.semantic_gates:
            gates.append(semantic_gate.check(document.source_text, candidate, policy))
        if not self.semantic_gates:
            gates.append(
                GateResult(
                    gate_id="claim_and_semantic_equivalence",
                    status="unavailable",
                    hard=policy.require_bidirectional_entailment,
                    reason="no NLI or claim adapter configured",
                )
            )
        gates.append(self._optional_gate("style", self.style_gate, document, candidate, policy))
        gates.append(self._optional_gate("fluency", self.fluency_gate, document, candidate, policy))
        passed = not any(gate.hard and gate.status != "passed" for gate in gates)
        return FidelityReport(
            passed=passed,
            gates=tuple(gates),
            candidate_text=candidate,
            char_edit_ratio=char_ratio,
            token_edit_ratio=token_ratio,
            length_drift_ratio=length_ratio,
        )

    @staticmethod
    def _optional_gate(
        gate_id: str,
        gate: SemanticGate | None,
        document: Document,
        candidate: str,
        policy: FidelityPolicy,
    ) -> GateResult:
        if gate is None:
            return GateResult(
                gate_id=gate_id,
                status="unavailable",
                hard=False,
                reason=f"no {gate_id} adapter configured",
            )
        return gate.check(document.source_text, candidate, policy)
