import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
            "monthly_income": [4000, 6000, 5000, 7000, 8000, 9000],
            "job_satisfaction": [1, 4, 1, 4, 2, 3],
            "overtime": ["Yes", "No", "Yes", "No", "Yes", "No"],
            "attrition": ["Yes", "No", "Yes", "No", "Yes", "No"],
        }
    )


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["Yes", "Yes"]})
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({"employee_id": [1, 2], "attrition": ["No", "No"]})
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns(sample_df):
    result = attrition_by_department(sample_df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_values(sample_df):
    result = attrition_by_department(sample_df)
    sales = result[result["department"] == "Sales"].iloc[0]
    assert sales["employees"] == 2
    assert sales["leavers"] == 1
    assert sales["attrition_rate"] == 50.0


def test_attrition_by_department_sorted_descending(sample_df):
    result = attrition_by_department(sample_df)
    rates = result["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns(sample_df):
    result = attrition_by_overtime(sample_df)
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_values(sample_df):
    result = attrition_by_overtime(sample_df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    # All 3 overtime workers left; none of the non-overtime workers left
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns(sample_df):
    result = average_income_by_attrition(sample_df)
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_values(sample_df):
    result = average_income_by_attrition(sample_df)
    leavers = result[result["attrition"] == "Yes"].iloc[0]["avg_monthly_income"]
    stayers = result[result["attrition"] == "No"].iloc[0]["avg_monthly_income"]
    # Leavers: 4000, 5000, 8000 → mean 5666.67
    # Stayers: 6000, 7000, 9000 → mean 7333.33
    assert leavers == round((4000 + 5000 + 8000) / 3, 2)
    assert stayers == round((6000 + 7000 + 9000) / 3, 2)
    assert leavers < stayers


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns(sample_df):
    result = satisfaction_summary(sample_df)
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_rate_is_within_group(sample_df):
    result = satisfaction_summary(sample_df)
    # satisfaction=1: 2 employees (Sales leaver, HR leaver) → both left → 100%
    row1 = result[result["job_satisfaction"] == 1].iloc[0]
    assert row1["total_employees"] == 2
    assert row1["leavers"] == 2
    assert row1["attrition_rate"] == 100.0


def test_satisfaction_summary_rate_not_share_of_all_leavers(sample_df):
    result = satisfaction_summary(sample_df)
    # If the old (buggy) denominator were used, rates would sum to ~100.
    # With the correct per-group denominator, they need not sum to 100.
    total = result["attrition_rate"].sum()
    assert total != pytest.approx(100.0)


def test_satisfaction_summary_sorted_by_satisfaction(sample_df):
    result = satisfaction_summary(sample_df)
    scores = result["job_satisfaction"].tolist()
    assert scores == sorted(scores)
