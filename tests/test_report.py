"""Sweep report: which budgets are compared, and in what order."""

from holo_playground.report import shared_budgets


def keys(**budgets_by_model):
    return [(m.removeprefix("m"), b) for m, budgets in budgets_by_model.items() for b in budgets]


def test_any_numeric_budget_is_kept_in_numeric_order():
    shared, skipped = shared_budgets(keys(m9b=["default", "1024", "768", "128", "8192"]))
    assert shared == ["128", "768", "1024", "default", "8192"]  # default caps at 4096
    assert skipped == []


def test_a_budget_missing_for_one_model_is_reported_not_silently_dropped():
    shared, skipped = shared_budgets(keys(m4b=["512", "1024"], m9b=["512", "768", "1024"]))
    assert shared == ["512", "1024"]
    assert skipped == ["768"]
