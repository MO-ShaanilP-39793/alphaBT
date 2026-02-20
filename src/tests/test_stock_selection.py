"""
Phase 3: Tests for selection/stock_selection.py.

Covers select_top_k_stocks, select_and_weight_stocks (cross-dimensional),
get_entry_start_date, validate_price_data_coverage, filter_tradeable_stocks.
"""

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# Helpers
# =============================================================================

def _make_input_data(n_stocks=9, quarters=None):
    """Create a basic input DataFrame for selection functions."""
    if quarters is None:
        quarters = [202302, 202305]

    names = [f"STOCK_{chr(65 + i)}" for i in range(n_stocks)]
    cats = (["largecap"] * 3 + ["midcap"] * 3 + ["smallcap"] * 3)[:n_stocks]
    vols = np.linspace(0.035, 0.006, n_stocks)
    probs = np.linspace(0.85, 0.45, n_stocks)

    rows = []
    for q in quarters:
        for i, name in enumerate(names):
            rows.append({
                "quarter": q,
                "co_name": name,
                "category": cats[i],
                "prob": probs[i],
                "volatility": vols[i],
            })
    return pd.DataFrame(rows)


# =============================================================================
# select_top_k_stocks
# =============================================================================

class TestSelectTopKStocks:

    def test_selects_correct_count(self):
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data()
        result = select_top_k_stocks(df, k=5, selection_method="probability")
        for q in df["quarter"].unique():
            q_result = result[result["quarter"] == q]
            assert len(q_result) == 5

    def test_equal_weights(self):
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data()
        result = select_top_k_stocks(df, k=5)
        for q in result["quarter"].unique():
            q_rows = result[result["quarter"] == q]
            expected_w = 1.0 / len(q_rows)
            assert all(abs(q_rows["stock_weight"] - expected_w) < 1e-9)

    def test_risk_adjusted_method(self):
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data()
        result = select_top_k_stocks(df, k=3, selection_method="risk_adjusted")
        assert len(result[result["quarter"] == 202302]) == 3

    def test_risk_adjusted_without_volatility_raises(self):
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data().drop(columns=["volatility"])
        with pytest.raises(ValueError):
            select_top_k_stocks(df, k=3, selection_method="risk_adjusted")

    def test_invalid_method_raises(self):
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data()
        with pytest.raises(ValueError):
            select_top_k_stocks(df, k=3, selection_method="invalid")

    def test_category_renamed_to_cat(self):
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data()
        result = select_top_k_stocks(df, k=3)
        assert "cat" in result.columns

    def test_min_prob_threshold(self):
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data()
        result = select_top_k_stocks(df, k=9, min_prob_threshold=0.7)
        # With threshold=0.7, fewer stocks should be selected than without
        for q in result["quarter"].unique():
            q_rows = result[result["quarter"] == q]
            n_eligible = len(df[(df["quarter"] == q) & (df["prob"] >= 0.7)])
            assert len(q_rows) <= min(9, n_eligible)

    def test_k_larger_than_available(self):
        """Requesting more stocks than available should warn and return what's available."""
        from selection.stock_selection import select_top_k_stocks
        df = _make_input_data(n_stocks=3)
        with pytest.warns(UserWarning):
            result = select_top_k_stocks(df, k=10)
        for q in result["quarter"].unique():
            assert len(result[result["quarter"] == q]) == 3


# =============================================================================
# get_entry_start_date (now in utils.quarter)
# =============================================================================

class TestGetEntryStartDate:

    def test_feb_quarter(self):
        from utils.quarter import get_entry_start_date
        result = get_entry_start_date(202302)
        assert result == pd.Timestamp(2023, 2, 15)

    def test_may_quarter(self):
        from utils.quarter import get_entry_start_date
        result = get_entry_start_date(202305)
        assert result == pd.Timestamp(2023, 5, 31)

    def test_aug_quarter(self):
        from utils.quarter import get_entry_start_date
        result = get_entry_start_date(202308)
        assert result == pd.Timestamp(2023, 8, 15)

    def test_nov_quarter(self):
        from utils.quarter import get_entry_start_date
        result = get_entry_start_date(202311)
        assert result == pd.Timestamp(2023, 11, 15)

    def test_invalid_month_raises(self):
        from utils.quarter import get_entry_start_date
        with pytest.raises(ValueError):
            get_entry_start_date(202304)


# =============================================================================
# validate_price_data_coverage / filter_tradeable_stocks
# =============================================================================

class TestPriceDataValidation:

    def _make_price_data(self, stocks, start="2023-02-01", end="2023-08-31"):
        dates = pd.bdate_range(start=start, end=end)
        rows = []
        for name in stocks:
            for d in dates:
                rows.append({
                    "date": d, "co_name": name,
                    "open": 100, "high": 101, "low": 99, "close": 100,
                })
        return pd.DataFrame(rows)

    def test_all_stocks_valid(self):
        from selection.stock_selection import validate_price_data_coverage
        input_df = _make_input_data(n_stocks=3, quarters=[202302])
        price_df = self._make_price_data(input_df["co_name"].unique())
        issues = validate_price_data_coverage(input_df, price_df)
        assert issues.empty

    def test_missing_stock_detected(self):
        from selection.stock_selection import validate_price_data_coverage
        input_df = _make_input_data(n_stocks=3, quarters=[202302])
        # Price data only for first 2 stocks
        price_df = self._make_price_data(input_df["co_name"].unique()[:2])
        issues = validate_price_data_coverage(input_df, price_df)
        assert not issues.empty

    def test_filter_tradeable_removes_bad_stocks(self):
        from selection.stock_selection import filter_tradeable_stocks
        input_df = _make_input_data(n_stocks=3, quarters=[202302])
        # Price data only for first 2 stocks
        price_df = self._make_price_data(input_df["co_name"].unique()[:2])
        filtered, issues = filter_tradeable_stocks(input_df, price_df)
        assert len(filtered) < len(input_df)
        assert not issues.empty
