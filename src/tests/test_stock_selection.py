"""Formula-based and edge-case tests for selection.stock_selection."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from selection.stock_selection import (
    _assign_volatility_categories,
    select_top_k_stocks,
    select_and_weight_stocks,
    validate_price_data_coverage_full,
    filter_tradeable_stocks,
)


# ---- Helpers ----

def make_probs(quarters, co_names, probs, volatility=None, category=None):
    """Build stock_probabilities DataFrame. probs can be list or dict (quarter -> list)."""
    rows = []
    q_list = list(quarters) if not isinstance(quarters, (int, str)) else [quarters]
    for qi, q in enumerate(q_list):
        names = co_names[qi] if isinstance(co_names[0], (list, tuple)) else co_names
        p = probs[qi] if isinstance(probs[0], (list, np.ndarray)) else probs
        vol = volatility[qi] if volatility is not None and isinstance(volatility[0], (list, np.ndarray)) else volatility
        cat = category[qi] if category is not None and isinstance(category[0], (list, tuple)) else category
        for i, name in enumerate(names):
            row = {"quarter": q, "co_name": name, "prob": p[i] if hasattr(p, "__getitem__") else p}
            if vol is not None:
                row["volatility"] = vol[i] if hasattr(vol, "__getitem__") else vol
            if cat is not None:
                row["category"] = cat[i] if hasattr(cat, "__getitem__") else cat
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# _assign_volatility_categories
# =============================================================================


def test_assign_volatility_categories_terciles():
    """Nine distinct values -> exactly 3 low, 3 medium, 3 high."""
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    out = _assign_volatility_categories(s)
    assert out.value_counts()["low_volatility"] == 3
    assert out.value_counts()["medium_volatility"] == 3
    assert out.value_counts()["high_volatility"] == 3
    # 1,2,3 <= low_cutoff; 7,8,9 > high_cutoff; 4,5,6 in between
    assert list(out.iloc[:3].unique()) == ["low_volatility"]
    assert list(out.iloc[6:].unique()) == ["high_volatility"]


def test_assign_volatility_categories_boundary_equal_groups():
    """Values at 1/3 and 2/3 quantiles -> correct grouping, no NaNs."""
    s = pd.Series([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    out = _assign_volatility_categories(s)
    assert out.isna().sum() == 0
    assert set(out.unique()) <= {"low_volatility", "medium_volatility", "high_volatility"}
    assert len(out) == 9


def test_assign_volatility_categories_constant_series():
    """All same value -> all get same category."""
    s = pd.Series([5.0] * 10)
    out = _assign_volatility_categories(s)
    uniq = out.unique()
    assert len(uniq) == 1
    assert uniq[0] in ("low_volatility", "medium_volatility", "high_volatility")


def test_assign_volatility_categories_empty_series():
    """Empty series -> empty series."""
    s = pd.Series(dtype=float)
    out = _assign_volatility_categories(s)
    assert isinstance(out, pd.Series)
    assert len(out) == 0


# =============================================================================
# select_top_k_stocks
# =============================================================================


def test_select_top_k_stocks_probability_happy_path():
    """One quarter, N >= k; sort by prob; top k selected; weights = 1/k, sum to 1."""
    df = make_probs([202402], [["A", "B", "C", "D", "E"]], [[0.9, 0.7, 0.5, 0.3, 0.1]])
    result = select_top_k_stocks(df, k=3, sort_by="probability")
    assert len(result) == 3
    assert list(result["co_name"]) == ["A", "B", "C"]
    assert result["stock_weight"].iloc[0] == pytest.approx(1.0 / 3)
    assert result["stock_weight"].sum() == pytest.approx(1.0)
    assert "quarter" in result.columns and "stock_weight" in result.columns


def test_select_top_k_stocks_risk_adjusted_happy_path():
    """Input has volatility; risk_adj_score = prob/(vol+eps); sort by it; output includes risk_adj_score."""
    df = pd.DataFrame({
        "quarter": [202402] * 4,
        "co_name": ["A", "B", "C", "D"],
        "prob": [0.4, 0.3, 0.2, 0.1],
        "volatility": [0.1, 0.2, 0.5, 1.0],
    })
    result = select_top_k_stocks(df, k=2, sort_by="risk_adjusted_probability")
    assert len(result) == 2
    assert "risk_adj_score" in result.columns
    # A: 0.4/0.1=4, B: 0.3/0.2=1.5, C: 0.2/0.5=0.4, D: 0.1/1=0.1 -> A, B
    assert list(result["co_name"]) == ["A", "B"]
    assert result["stock_weight"].sum() == pytest.approx(1.0)


def test_select_top_k_stocks_min_prob_threshold():
    """Rows with prob < threshold excluded; quarter with none above skipped (other quarters present)."""
    df = pd.DataFrame({
        "quarter": [202402, 202402, 202405, 202405],
        "co_name": ["A", "B", "C", "D"],
        "prob": [0.1, 0.05, 0.8, 0.6],
    })
    with pytest.warns(UserWarning, match="No stocks above min_prob_threshold"):
        result = select_top_k_stocks(df, k=2, min_prob_threshold=0.5)
    assert result["quarter"].nunique() == 1
    assert 202405 in result["quarter"].values
    assert list(result["co_name"]) == ["C", "D"]
    assert result["stock_weight"].sum() == pytest.approx(1.0)


def test_select_top_k_stocks_fewer_than_k():
    """Fewer than k stocks in quarter -> select all; weights = 1/n, sum to 1."""
    df = make_probs([202402], [["A", "B"]], [[0.9, 0.8]])
    with pytest.warns(UserWarning, match="only 2 available"):
        result = select_top_k_stocks(df, k=5)
    assert len(result) == 2
    assert result["stock_weight"].iloc[0] == pytest.approx(0.5)
    assert result["stock_weight"].sum() == pytest.approx(1.0)


def test_select_top_k_stocks_category_passthrough():
    """Input has category -> output has cat with same values."""
    df = pd.DataFrame({
        "quarter": [202402, 202402, 202402],
        "co_name": ["A", "B", "C"],
        "prob": [0.9, 0.7, 0.5],
        "category": ["largecap", "midcap", "smallcap"],
    })
    result = select_top_k_stocks(df, k=3)
    assert "cat" in result.columns
    assert list(result["cat"]) == ["largecap", "midcap", "smallcap"]


def test_select_top_k_stocks_multiple_quarters():
    """Two quarters; each gets top k; weights sum to 1 per quarter."""
    df = pd.DataFrame({
        "quarter": [202402, 202402, 202402, 202405, 202405, 202405],
        "co_name": ["A", "B", "C", "X", "Y", "Z"],
        "prob": [0.9, 0.5, 0.3, 0.8, 0.6, 0.2],
    })
    result = select_top_k_stocks(df, k=2)
    assert len(result) == 4
    q1 = result[result["quarter"] == 202402]
    q2 = result[result["quarter"] == 202405]
    assert q1["stock_weight"].sum() == pytest.approx(1.0)
    assert q2["stock_weight"].sum() == pytest.approx(1.0)
    assert list(q1["co_name"]) == ["A", "B"]
    assert list(q2["co_name"]) == ["X", "Y"]


def test_select_top_k_stocks_invalid_sort_by_raises():
    """Invalid sort_by -> ValueError."""
    df = make_probs([202402], [["A"]], [[0.5]])
    with pytest.raises(ValueError, match="sort_by must be one of"):
        select_top_k_stocks(df, k=1, sort_by="invalid")


def test_select_top_k_stocks_invalid_weighting_scheme_raises():
    """Invalid weighting_scheme -> ValueError."""
    df = make_probs([202402], [["A"]], [[0.5]])
    with pytest.raises(ValueError, match="weighting_scheme must be one of"):
        select_top_k_stocks(df, k=1, weighting_scheme="invalid")


def test_select_top_k_stocks_risk_adjusted_without_volatility_raises():
    """sort_by risk_adjusted_probability without volatility column -> ValueError."""
    df = make_probs([202402], [["A"]], [[0.5]])
    with pytest.raises(ValueError, match="requires 'volatility' column"):
        select_top_k_stocks(df, k=1, sort_by="risk_adjusted_probability")


def test_select_top_k_stocks_empty_input():
    """Empty input -> empty DataFrame with correct columns."""
    df = pd.DataFrame(columns=["quarter", "co_name", "prob"])
    result = select_top_k_stocks(df, k=5)
    assert result.empty
    assert list(result.columns) == ["quarter", "co_name", "stock_weight"]


# =============================================================================
# select_and_weight_stocks
# =============================================================================


def test_select_and_weight_stocks_invalid_selection_dimension_raises():
    """Invalid selection_dimension -> ValueError."""
    df = make_probs([202402], [["A"]], [[0.5]])
    with pytest.raises(ValueError, match="selection_dimension must be 'volatility' or 'mcap'"):
        select_and_weight_stocks(
            df, selection_dimension="x", weighting_dimension="volatility",
            selection_counts=[1, 1, 1], category_weights=[0.33, 0.33, 0.34],
        )


def test_select_and_weight_stocks_invalid_weighting_dimension_raises():
    """Invalid weighting_dimension -> ValueError."""
    df = make_probs([202402], [["A"]], [[0.5]])
    with pytest.raises(ValueError, match="weighting_dimension must be 'volatility' or 'mcap'"):
        select_and_weight_stocks(
            df, selection_dimension="volatility", weighting_dimension="x",
            selection_counts=[1, 1, 1], category_weights=[0.33, 0.33, 0.34],
        )


def test_select_and_weight_stocks_volatility_requires_volatility_column():
    """selection_dimension='volatility' without volatility column -> ValueError."""
    df = pd.DataFrame({"quarter": [202402], "co_name": ["A"], "prob": [0.5]})
    with pytest.raises(ValueError, match="'volatility' column required"):
        select_and_weight_stocks(
            df, selection_dimension="volatility", weighting_dimension="volatility",
            selection_counts=[1, 1, 1], category_weights=[0.33, 0.33, 0.34],
        )


def test_select_and_weight_stocks_mcap_requires_category_column():
    """selection_dimension='mcap' without category column -> ValueError."""
    df = pd.DataFrame({"quarter": [202402], "co_name": ["A"], "prob": [0.5]})
    with pytest.raises(ValueError, match="'category' column required"):
        select_and_weight_stocks(
            df, selection_dimension="mcap", weighting_dimension="mcap",
            selection_counts=[1, 1, 1], category_weights=[0.33, 0.33, 0.34],
        )


def test_select_and_weight_stocks_selection_counts_length():
    """selection_counts must have length 3."""
    df = pd.DataFrame({
        "quarter": [202402], "co_name": ["A"], "prob": [0.5],
        "volatility": [0.2], "category": ["largecap"],
    })
    with pytest.raises(ValueError, match="selection_counts must be a list of length 3"):
        select_and_weight_stocks(
            df, selection_dimension="volatility", weighting_dimension="volatility",
            selection_counts=[1, 1], category_weights=[0.33, 0.33, 0.34],
        )


def test_select_and_weight_stocks_category_weights_length():
    """category_weights must have length 3 when weighting_scheme is use_category_weights."""
    df = pd.DataFrame({
        "quarter": [202402], "co_name": ["A"], "prob": [0.5], "volatility": [0.2],
    })
    with pytest.raises(ValueError, match="category_weights must be a list of length 3"):
        select_and_weight_stocks(
            df, selection_dimension="volatility", weighting_dimension="volatility",
            selection_counts=[1, 1, 1], category_weights=[0.5, 0.5],
            weighting_scheme="use_category_weights",
        )


def test_select_and_weight_stocks_equal_weighting():
    """weighting_scheme='equal' -> stock_weight = 1/n per quarter; cat = selection_cat."""
    # One quarter, 6 stocks: 2 high_vol, 2 med, 2 low (by volatility values)
    vol = [10.0, 9.0, 5.0, 4.0, 1.0, 0.5]  # high high med med low low
    df = pd.DataFrame({
        "quarter": [202402] * 6,
        "co_name": ["A", "B", "C", "D", "E", "F"],
        "prob": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "volatility": vol,
    })
    result = select_and_weight_stocks(
        df, selection_dimension="volatility", weighting_dimension="volatility",
        selection_counts=[2, 2, 2], category_weights=[0.33, 0.33, 0.34],
        weighting_scheme="equal",
    )
    assert len(result) == 6
    assert result["stock_weight"].sum() == pytest.approx(1.0)
    assert result["stock_weight"].iloc[0] == pytest.approx(1.0 / 6)
    assert "cat" in result.columns
    assert result["cat"].equals(result["selection_cat"])


def test_select_and_weight_stocks_use_category_weights():
    """weighting_scheme use_category_weights: cat_weight from map; stock_weight = cat_weight/count; cat = weight_cat."""
    # 3 stocks in same quarter, one per volatility bucket; select 1 each
    df = pd.DataFrame({
        "quarter": [202402, 202402, 202402],
        "co_name": ["A", "B", "C"],
        "prob": [0.9, 0.8, 0.7],
        "volatility": [10.0, 5.0, 0.5],  # high, medium, low
    })
    result = select_and_weight_stocks(
        df, selection_dimension="volatility", weighting_dimension="volatility",
        selection_counts=[1, 1, 1], category_weights=[0.5, 0.3, 0.2],
        weighting_scheme="use_category_weights",
    )
    assert len(result) == 3
    assert result["stock_weight"].sum() == pytest.approx(1.0)
    assert "cat_weight" in result.columns
    assert "weight_cat" in result.columns
    assert result["cat"].equals(result["weight_cat"])
    # Each stock is alone in its (quarter, cat) -> stock_weight = cat_weight
    assert set(result["stock_weight"]) == {0.5, 0.3, 0.2}


def test_select_and_weight_stocks_sort_by_probability():
    """sort_by probability -> order by prob within each selection category."""
    df = pd.DataFrame({
        "quarter": [202402, 202402, 202402],
        "co_name": ["A", "B", "C"],
        "prob": [0.3, 0.9, 0.6],
        "volatility": [1.0, 1.0, 1.0],  # same vol -> all in same tercile (low_volatility)
    })
    # selection_counts [high, medium, low]; all 3 stocks are low_volatility
    result = select_and_weight_stocks(
        df, selection_dimension="volatility", weighting_dimension="volatility",
        selection_counts=[0, 0, 3], category_weights=[0.0, 0.0, 1.0],
        sort_by="probability", weighting_scheme="use_category_weights",
    )
    assert list(result["co_name"]) == ["B", "C", "A"]


def test_select_and_weight_stocks_sort_by_risk_adjusted():
    """sort_by risk_adjusted_probability -> order by risk_adj_score within category."""
    df = pd.DataFrame({
        "quarter": [202402, 202402, 202402],
        "co_name": ["A", "B", "C"],
        "prob": [0.2, 0.5, 0.8],
        "volatility": [0.1, 0.1, 0.1],  # same vol -> all low_volatility; risk_adj = prob/0.1 -> 2, 5, 8
    })
    result = select_and_weight_stocks(
        df, selection_dimension="volatility", weighting_dimension="volatility",
        selection_counts=[0, 0, 3], category_weights=[0.0, 0.0, 1.0],
        sort_by="risk_adjusted_probability", weighting_scheme="use_category_weights",
    )
    # risk_adj_score descending: C=8, B=5, A=2
    assert list(result["co_name"]) == ["C", "B", "A"]
    assert "risk_adj_score" in result.columns


def test_select_and_weight_stocks_empty_after_threshold():
    """All rows below min_prob_threshold -> empty DataFrame with correct columns."""
    df = pd.DataFrame({
        "quarter": [202402], "co_name": ["A"], "prob": [0.01],
        "volatility": [0.2], "category": ["largecap"],
    })
    with pytest.warns(UserWarning, match="No stocks above min_prob_threshold"):
        result = select_and_weight_stocks(
            df, selection_dimension="volatility", weighting_dimension="volatility",
            selection_counts=[1, 1, 1], category_weights=[0.33, 0.33, 0.34],
            min_prob_threshold=0.5, weighting_scheme="use_category_weights",
        )
    assert result.empty
    assert "quarter" in result.columns and "stock_weight" in result.columns
    assert "selection_cat" in result.columns and "weight_cat" in result.columns


def test_select_and_weight_stocks_invalid_sort_by_raises():
    """Invalid sort_by -> ValueError."""
    df = pd.DataFrame({
        "quarter": [202402], "co_name": ["A"], "prob": [0.5], "volatility": [0.2],
    })
    with pytest.raises(ValueError, match="sort_by must be one of"):
        select_and_weight_stocks(
            df, selection_dimension="volatility", weighting_dimension="volatility",
            selection_counts=[1, 1, 1], category_weights=[0.33, 0.33, 0.34],
            sort_by="invalid",
        )


def test_select_and_weight_stocks_invalid_weighting_scheme_raises():
    """Invalid weighting_scheme -> ValueError."""
    df = pd.DataFrame({
        "quarter": [202402], "co_name": ["A"], "prob": [0.5], "volatility": [0.2],
    })
    with pytest.raises(ValueError, match="weighting_scheme must be one of"):
        select_and_weight_stocks(
            df, selection_dimension="volatility", weighting_dimension="volatility",
            selection_counts=[1, 1, 1], category_weights=[0.33, 0.33, 0.34],
            weighting_scheme="invalid",
        )


def test_select_and_weight_stocks_mcap_dimension():
    """mcap dimension uses category column for selection_cat and weight_cat."""
    df = pd.DataFrame({
        "quarter": [202402, 202402, 202402],
        "co_name": ["A", "B", "C"],
        "prob": [0.9, 0.8, 0.7],
        "category": ["largecap", "midcap", "smallcap"],
    })
    result = select_and_weight_stocks(
        df, selection_dimension="mcap", weighting_dimension="mcap",
        selection_counts=[1, 1, 1], category_weights=[0.5, 0.3, 0.2],
        weighting_scheme="use_category_weights",
    )
    assert len(result) == 3
    assert set(result["selection_cat"]) == {"largecap", "midcap", "smallcap"}
    assert set(result["weight_cat"]) == {"largecap", "midcap", "smallcap"}
    assert result["stock_weight"].sum() == pytest.approx(1.0)


# =============================================================================
# validate_price_data_coverage_full
# =============================================================================


@patch("selection.stock_selection.get_quarter_dates")
def test_validate_price_data_coverage_full_coverage(mock_quarter_dates):
    """Full coverage for each (quarter, co_name) -> empty issues."""
    mock_quarter_dates.return_value = (
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-16"),
    )
    input_data = pd.DataFrame({
        "quarter": [202402, 202402],
        "co_name": ["A", "B"],
    })
    # 2 trading days; A and B each have 2 rows
    price_data = pd.DataFrame({
        "date": ["2024-02-15", "2024-02-15", "2024-02-16", "2024-02-16"],
        "co_name": ["A", "B", "A", "B"],
        "close": [100.0, 200.0, 101.0, 201.0],
    })
    issues = validate_price_data_coverage_full(input_data, price_data)
    assert issues.empty


@patch("selection.stock_selection.get_quarter_dates")
def test_validate_price_data_coverage_full_missing_days(mock_quarter_dates):
    """One stock missing one day -> one issue row with correct detail."""
    mock_quarter_dates.return_value = (
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-16"),
    )
    input_data = pd.DataFrame({
        "quarter": [202402, 202402],
        "co_name": ["A", "B"],
    })
    # 2 market days; A has 2, B has 1
    price_data = pd.DataFrame({
        "date": ["2024-02-15", "2024-02-15", "2024-02-16"],
        "co_name": ["A", "B", "A"],
        "close": [100.0, 200.0, 101.0],
    })
    issues = validate_price_data_coverage_full(input_data, price_data)
    assert len(issues) == 1
    assert issues.iloc[0]["quarter"] == 202402
    assert issues.iloc[0]["co_name"] == "B"
    assert issues.iloc[0]["issue"] == "missing_trading_days"
    assert "Missing 1 day" in issues.iloc[0]["detail"]
    assert "Found 1/2" in issues.iloc[0]["detail"]


@patch("selection.stock_selection.get_quarter_dates")
def test_validate_price_data_coverage_quarter_range(mock_quarter_dates):
    """first_quarter/last_quarter restrict which input rows are checked."""
    def side_effect(q):
        if q == 202402:
            return pd.Timestamp("2024-02-15"), pd.Timestamp("2024-02-16")
        return pd.Timestamp("2024-05-31"), pd.Timestamp("2024-06-01")

    mock_quarter_dates.side_effect = side_effect
    input_data = pd.DataFrame({
        "quarter": [202402, 202405],
        "co_name": ["A", "B"],
    })
    # Only 202402 in range; price data has full coverage for A in 202402
    price_data = pd.DataFrame({
        "date": ["2024-02-15", "2024-02-16"],
        "co_name": ["A", "A"],
        "close": [100.0, 101.0],
    })
    issues = validate_price_data_coverage_full(
        input_data, price_data, first_quarter=202402, last_quarter=202402
    )
    # B is in 202405 which is outside range, so only A (202402) is checked; A has 2/2 -> no issues
    assert issues.empty


@patch("selection.stock_selection.get_quarter_dates")
def test_validate_price_data_coverage_no_trading_days_raises(mock_quarter_dates):
    """Quarter in filtered input but no dates in price_data for that quarter -> ValueError."""
    mock_quarter_dates.return_value = (
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-16"),
    )
    input_data = pd.DataFrame({"quarter": [202402], "co_name": ["A"]})
    price_data = pd.DataFrame({
        "date": ["2024-05-01"],
        "co_name": ["A"],
        "close": [100.0],
    })
    with pytest.raises(ValueError, match="No trading days in 202402"):
        validate_price_data_coverage_full(input_data, price_data)


@patch("selection.stock_selection.get_quarter_dates")
def test_validate_price_data_coverage_multiple_issues(mock_quarter_dates):
    """Several (quarter, co_name) with missing days -> one row per in issues."""
    mock_quarter_dates.return_value = (
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-16"),
    )
    input_data = pd.DataFrame({
        "quarter": [202402, 202402, 202402],
        "co_name": ["A", "B", "C"],
    })
    # Only A has 2 days; B and C have 1 each
    price_data = pd.DataFrame({
        "date": ["2024-02-15", "2024-02-15", "2024-02-15", "2024-02-16"],
        "co_name": ["A", "B", "C", "A"],
        "close": [100.0, 200.0, 300.0, 101.0],
    })
    issues = validate_price_data_coverage_full(input_data, price_data)
    assert len(issues) == 2
    assert set(issues["co_name"]) == {"B", "C"}


# =============================================================================
# filter_tradeable_stocks
# =============================================================================


@patch("selection.stock_selection.get_quarter_dates")
def test_filter_tradeable_stocks_no_issues(mock_quarter_dates):
    """No issues -> (input filtered by range, empty issues_df)."""
    mock_quarter_dates.return_value = (
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-16"),
    )
    input_data = pd.DataFrame({
        "quarter": [202402, 202402],
        "co_name": ["A", "B"],
        "prob": [0.9, 0.8],
    })
    price_data = pd.DataFrame({
        "date": ["2024-02-15", "2024-02-15", "2024-02-16", "2024-02-16"],
        "co_name": ["A", "B", "A", "B"],
        "close": [100.0, 200.0, 101.0, 201.0],
    })
    filtered, issues = filter_tradeable_stocks(input_data, price_data)
    assert issues.empty
    assert len(filtered) == 2
    assert set(filtered["co_name"]) == {"A", "B"}


@patch("selection.stock_selection.get_quarter_dates")
def test_filter_tradeable_stocks_with_issues(mock_quarter_dates):
    """Some (quarter, co_name) flagged -> filtered_data has them removed; issues_df returned."""
    mock_quarter_dates.return_value = (
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-16"),
    )
    input_data = pd.DataFrame({
        "quarter": [202402, 202402],
        "co_name": ["A", "B"],
        "prob": [0.9, 0.8],
    })
    price_data = pd.DataFrame({
        "date": ["2024-02-15", "2024-02-16"],
        "co_name": ["A", "A"],
        "close": [100.0, 101.0],
    })
    filtered, issues = filter_tradeable_stocks(input_data, price_data)
    assert len(issues) == 1
    assert issues.iloc[0]["co_name"] == "B"
    assert len(filtered) == 1
    assert list(filtered["co_name"]) == ["A"]


@patch("selection.stock_selection.get_quarter_dates")
def test_filter_tradeable_stocks_range(mock_quarter_dates):
    """first_quarter/last_quarter applied to input and returned filtered data."""
    mock_quarter_dates.return_value = (
        pd.Timestamp("2024-02-15"),
        pd.Timestamp("2024-02-16"),
    )
    input_data = pd.DataFrame({
        "quarter": [202401, 202402],
        "co_name": ["A", "B"],
        "prob": [0.9, 0.8],
    })
    price_data = pd.DataFrame({
        "date": ["2024-02-15", "2024-02-15", "2024-02-16", "2024-02-16"],
        "co_name": ["A", "B", "A", "B"],
        "close": [100.0, 200.0, 101.0, 201.0],
    })
    filtered, _ = filter_tradeable_stocks(
        input_data, price_data, first_quarter=202402, last_quarter=202402
    )
    assert 202401 not in filtered["quarter"].values
    assert list(filtered["quarter"]) == [202402]
    assert list(filtered["co_name"]) == ["B"]
