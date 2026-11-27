import pytest
from pydantic import ValidationError

from holo_playground.agent.evaluate import Contains, Regex, check_answer, normalize

LICENSE = Regex(regex=r"\bapache(?:[- ]license)?(?:,? version)?[- ]?2\.0\b")


def test_normalize_only_touches_case_and_whitespace():
    assert normalize("  Apache-2.0\n licence ") == "apache-2.0 licence"


@pytest.mark.parametrize(
    "answer",
    [
        "apache-2.0",  # trial 2026-11-23 16:12
        "The license for the Holo 3.1 9B model is apache-2.0",  # trial 2026-11-23 13:46
        "Apache 2.0",
        "APACHE-2.0.",
        "The Holo 3.1 9B model is licensed under Apache License 2.0.",  # trial 2026-11-26, H's API
        "Apache License, Version 2.0",
    ],
)
def test_license_regex_accepts_formatting_variants(answer):
    assert check_answer(answer, LICENSE)


@pytest.mark.parametrize("answer", ["MIT", "apache-2.01", "apache-20", "Apache License 1.1", "", None])
def test_license_regex_rejects_wrong_or_missing_answers(answer):
    assert not check_answer(answer, LICENSE)


def test_contains_normalizes_both_sides():
    assert check_answer("License:   APACHE-2.0", Contains(contains="Apache-2.0"))
    assert not check_answer("apache 2.0", Contains(contains="apache-2.0"))


def test_known_false_positive_negation_passes():
    """Documents a known limit: string checks cannot read negation (an LLM judge could)."""
    assert check_answer("The license is MIT, not apache-2.0", LICENSE)


def test_broken_regex_fails_at_task_load():
    with pytest.raises(ValidationError):
        Regex(regex="(")
