import pytest

from report_summary import extract_report_info


def make_case(labels=None, status="passed"):
    return {
        "labels": labels or [],
        "status": status,
        "name": "case",
        "time": {"start": 1},
    }


def test_parent_suite_preferred():
    case = make_case([
        {"name": "parentSuite", "value": "TeamA"},
        {"name": "suite", "value": "SuiteA"},
    ])
    info = extract_report_info([case])
    assert info["test_suite_name"] == "TeamA"


def test_suite_used_when_no_parent():
    case = make_case([{"name": "suite", "value": "SuiteB"}])
    info = extract_report_info([case])
    assert info["test_suite_name"] == "SuiteB"


def test_only_one_value_per_case():
    case = make_case([
        {"name": "parentSuite", "value": "X"},
        {"name": "parentSuite", "value": "Y"},
    ])
    info = extract_report_info([case])
    # should not join both names
    assert info["test_suite_name"] in {"X", "Y"}


