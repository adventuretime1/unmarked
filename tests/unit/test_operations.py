"""Operation application, inversion, bounds, overlap, and hashing."""

from __future__ import annotations

import pytest

from unmark.core.errors import OperationError
from unmark.core.operations import (
    Operation,
    apply_operations,
    char_edit_ratio,
    invert_operations,
    length_drift_ratio,
    normalized_content_hash,
    rollback,
    sha256_text,
    validate_operations,
)


def op(start: int, end: int, text: str = "", **kwargs) -> Operation:
    kwargs.setdefault("reason", "test")
    kwargs.setdefault("operator", "test")
    return Operation(start=start, end=end, text=text, **kwargs)


class TestApply:
    def test_delete(self):
        assert apply_operations("hello world", [op(5, 6)]) == "helloworld"

    def test_insert(self):
        assert apply_operations("helloworld", [op(5, 5, " ")]) == "hello world"

    def test_replace(self):
        assert apply_operations("hello world", [op(6, 11, "there")]) == "hello there"

    def test_no_operations_returns_source(self):
        assert apply_operations("unchanged", []) == "unchanged"

    def test_multiple_operations_are_source_anchored(self):
        # Offsets refer to the source, not to the partially edited text.
        text = "aaa bbb ccc"
        result = apply_operations(text, [op(0, 3, "XXXX"), op(8, 11, "Y")])
        assert result == "XXXX bbb Y"

    def test_application_is_order_independent(self):
        text = "one two three"
        forward = [op(0, 3, "1"), op(8, 13, "3")]
        assert apply_operations(text, forward) == apply_operations(text, list(reversed(forward)))

    def test_operations_at_document_boundaries(self):
        assert apply_operations("abc", [op(0, 0, ">")]) == ">abc"
        assert apply_operations("abc", [op(3, 3, "<")]) == "abc<"


class TestValidation:
    def test_out_of_bounds_rejected(self):
        with pytest.raises(OperationError, match="exceeds source length"):
            apply_operations("short", [op(0, 99, "x")])

    def test_overlapping_rejected(self):
        with pytest.raises(OperationError, match="overlap"):
            apply_operations("hello world", [op(0, 5, "a"), op(3, 8, "b")])

    def test_adjacent_operations_allowed(self):
        # [0,3) and [3,6) touch but do not overlap.
        assert apply_operations("abcdef", [op(0, 3, "X"), op(3, 6, "Y")]) == "XY"

    def test_duplicate_insertions_at_same_offset_rejected(self):
        with pytest.raises(OperationError, match="ambiguous"):
            apply_operations("abc", [op(1, 1, "x"), op(1, 1, "y")])

    def test_mismatched_original_rejected(self):
        with pytest.raises(OperationError, match="expected original"):
            apply_operations("hello", [op(0, 5, "x", original="WRONG")])

    def test_matching_original_accepted(self):
        assert apply_operations("hello", [op(0, 5, "x", original="hello")]) == "x"

    def test_end_before_start_rejected(self):
        with pytest.raises(ValueError, match="precedes start"):
            op(5, 2, "x")

    def test_empty_noop_rejected(self):
        with pytest.raises(ValueError, match="no-op"):
            op(3, 3, "")

    def test_negative_offset_rejected(self):
        with pytest.raises(ValueError):
            op(-1, 2, "x")

    def test_validate_returns_sorted(self):
        ordered = validate_operations([op(5, 6, "b"), op(0, 1, "a")], "abcdefg")
        assert [o.start for o in ordered] == [0, 5]


class TestInversion:
    @pytest.mark.parametrize(
        "source,ops",
        [
            ("hello world", [op(5, 6)]),
            ("helloworld", [op(5, 5, " ")]),
            ("hello world", [op(6, 11, "there")]),
            ("aaa bbb ccc", [op(0, 3, "XXXX"), op(8, 11, "Y")]),
            ("abc", [op(0, 0, ">"), op(3, 3, "<")]),
            ("a​b", [op(1, 2)]),
        ],
    )
    def test_rollback_is_exact(self, source, ops):
        assert rollback(source, ops) == source

    def test_adjacent_deletions_invert_without_ambiguity(self):
        # Regression: two source-adjacent deletions invert to two insertions at
        # the same candidate offset, which must be merged rather than rejected.
        source = "a\x1f\x08b"
        ops = [op(1, 2), op(2, 3)]
        assert apply_operations(source, ops) == "ab"
        assert rollback(source, ops) == source

    def test_merged_inverse_preserves_source_order(self):
        source = "XY"
        ops = [op(0, 1), op(1, 2)]
        inverse = invert_operations(source, ops)
        assert len(inverse) == 1
        assert inverse[0].text == "XY"

    def test_inverse_operations_anchor_to_candidate(self):
        source = "hello world"
        ops = [op(0, 5, "goodbye")]
        candidate = apply_operations(source, ops)
        inverse = invert_operations(source, ops)
        assert apply_operations(candidate, inverse) == source


class TestHashing:
    def test_sha256_is_stable(self):
        assert sha256_text("abc") == sha256_text("abc")

    def test_sha256_is_not_normalized(self):
        # Composed vs decomposed forms are different bytes and must hash apart.
        assert sha256_text("é") != sha256_text("é")

    def test_content_hash_is_normalized(self):
        assert normalized_content_hash("é") == normalized_content_hash("é")

    def test_different_text_hashes_differently(self):
        assert normalized_content_hash("a") != normalized_content_hash("b")


class TestRatios:
    def test_char_edit_ratio_in_unit_interval(self):
        source = "a" * 100
        ops = [op(0, 5, "")]
        assert char_edit_ratio(source, apply_operations(source, ops), ops) == pytest.approx(0.05)

    def test_char_edit_ratio_clamped(self):
        source = "abc"
        ops = [op(0, 3, "x" * 100)]
        assert char_edit_ratio(source, apply_operations(source, ops), ops) == 1.0

    def test_empty_source(self):
        assert char_edit_ratio("", "", []) == 0.0
        assert length_drift_ratio("", "") == 0.0

    def test_length_drift(self):
        assert length_drift_ratio("a" * 100, "a" * 105) == pytest.approx(0.05)
        assert length_drift_ratio("a" * 100, "a" * 95) == pytest.approx(0.05)
