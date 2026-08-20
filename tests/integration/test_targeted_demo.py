"""The documented offline demonstration remains executable."""

from unmark.evaluation.targeted_demo import run_demo


def test_targeted_demo() -> None:
    result = run_demo()
    assert result["state"] == "verified_below_threshold"
    assert result["baseline_score"] == 8.0
    assert result["final_score"] == 1.0
    assert result["protected_42_preserved"] is True
    assert "neutral wording" in str(result["selected_text"])
