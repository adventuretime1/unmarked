"""Property tests for operations, hashing, and sanitation invariants."""

from __future__ import annotations

import unicodedata

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from unmark.core.operations import (
    Operation,
    apply_operations,
    char_edit_ratio,
    invert_operations,
    length_drift_ratio,
    normalized_content_hash,
    rollback,
    sha256_text,
)
from unmark.core.policies import UnicodePolicy
from unmark.inspect.unicode_scan import inspect_text, sanitation_operations

# Deliberately includes surrogate-free Unicode across many planes.
text_strategy = st.text(
    alphabet=st.characters(exclude_categories=["Cs"]),
    min_size=0,
    max_size=200,
)


@st.composite
def text_and_operations(draw):
    """A source string plus a valid, non-overlapping operation list."""
    source = draw(st.text(min_size=1, max_size=120))
    count = draw(st.integers(min_value=0, max_value=4))

    cursor = 0
    operations = []
    for _ in range(count):
        if cursor >= len(source):
            break
        start = draw(st.integers(min_value=cursor, max_value=len(source)))
        end = draw(st.integers(min_value=start, max_value=len(source)))
        replacement = draw(st.text(max_size=10))
        if start == end and not replacement:
            continue
        operations.append(
            Operation(
                start=start,
                end=end,
                text=replacement,
                reason="property",
                operator="property",
            )
        )
        cursor = max(end, start + 1)
    return source, operations


class TestOperationProperties:
    @given(text_and_operations())
    def test_rollback_recovers_source_exactly(self, pair):
        source, operations = pair
        assert rollback(source, operations) == source

    @given(text_and_operations())
    def test_inverse_maps_candidate_back(self, pair):
        source, operations = pair
        candidate = apply_operations(source, operations)
        inverse = invert_operations(source, operations)
        assert apply_operations(candidate, inverse) == source

    @given(text_and_operations())
    def test_apply_is_deterministic(self, pair):
        source, operations = pair
        assert apply_operations(source, operations) == apply_operations(source, operations)

    @given(text_and_operations())
    def test_apply_is_order_independent(self, pair):
        source, operations = pair
        assert apply_operations(source, operations) == apply_operations(
            source, list(reversed(operations))
        )

    @given(text_and_operations())
    def test_ratios_stay_in_unit_interval(self, pair):
        source, operations = pair
        candidate = apply_operations(source, operations)
        assert 0.0 <= char_edit_ratio(source, candidate, operations) <= 1.0
        assert 0.0 <= length_drift_ratio(source, candidate) <= 1.0

    @given(text_strategy)
    def test_empty_operation_list_is_identity(self, text):
        assert apply_operations(text, []) == text

    @given(text_and_operations())
    def test_length_matches_sum_of_deltas(self, pair):
        source, operations = pair
        candidate = apply_operations(source, operations)
        expected = len(source) + sum(op.length_delta for op in operations)
        assert len(candidate) == expected


class TestHashProperties:
    @given(text_strategy)
    def test_hash_is_stable(self, text):
        assert sha256_text(text) == sha256_text(text)

    @given(text_strategy)
    def test_normalized_hash_matches_nfc_form(self, text):
        assert normalized_content_hash(text) == normalized_content_hash(
            unicodedata.normalize("NFC", text)
        )

    @given(text_strategy, text_strategy)
    def test_distinct_text_hashes_distinctly(self, first, second):
        assume(unicodedata.normalize("NFC", first) != unicodedata.normalize("NFC", second))
        assert normalized_content_hash(first) != normalized_content_hash(second)


class TestSanitationProperties:
    @settings(max_examples=200)
    @given(text_strategy)
    def test_sanitation_never_raises_and_is_deterministic(self, text):
        policy = UnicodePolicy(name="safe")
        findings = inspect_text(text, policy)
        operations = sanitation_operations(text, findings, policy)
        first = apply_operations(text, operations)
        second = apply_operations(text, operations)
        assert first == second

    @given(text_strategy)
    def test_sanitation_is_exactly_reversible(self, text):
        policy = UnicodePolicy(name="safe")
        findings = inspect_text(text, policy)
        operations = sanitation_operations(text, findings, policy)
        assert rollback(text, operations) == text

    @given(text_strategy)
    def test_sanitation_never_lengthens_text(self, text):
        # Every safe-policy action is a removal or a same-length replacement.
        policy = UnicodePolicy(name="safe")
        findings = inspect_text(text, policy)
        operations = sanitation_operations(text, findings, policy)
        assert len(apply_operations(text, operations)) <= len(text)

    @given(text_strategy)
    def test_report_policy_is_always_identity(self, text):
        policy = UnicodePolicy(name="report")
        findings = inspect_text(text, policy)
        assert sanitation_operations(text, findings, policy) == ()

    @given(text_strategy)
    def test_findings_have_valid_offsets(self, text):
        for finding in inspect_text(text, UnicodePolicy(name="safe")):
            assert 0 <= finding.offset < len(text)
            assert text[finding.offset] == finding.char

    @given(st.text(alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")), max_size=100))
    def test_ordinary_text_is_only_canonically_composed(self, text):
        policy = UnicodePolicy(name="safe")
        findings = inspect_text(text, policy)
        operations = sanitation_operations(text, findings, policy)
        assert apply_operations(text, operations) == unicodedata.normalize("NFC", text)
