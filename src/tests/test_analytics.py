"""Formula-based tests for reporting.analytics — in-memory data, expected from first principles."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from reporting.analytics import (
    compute_portfolio_performance,
    compute_rolling_performance,
    compute_monthly_returns_from_daily,
    compute_up_down_months,
    compute_crisis_regime_returns,
    compute_market_regime_returns,
    compute_quarterly_alpha,
    get_stock_counts_by_mcap,
    get_comprehensive_quarter_analysis,
)

# ---- Shared helpers (formula-based) ----


def constant_daily_returns(n_days: int, r: float = 0.001, start: str = "2020-01-01", cols: list = None) -> pd.DataFrame:
    """Build daily returns DataFrame with constant return per period."""
    dates = pd.bdate_range(start, periods=n_days, freq="B")
    col_names = cols or ["strategy_A"]
    data = np.full((n_days, len(col_names)), r)
    df = pd.DataFrame(data, index=dates, columns=col_names)
    df.index.name = "date"
    return df


def cumulative_return(r: float, n: int) -> float:
    """(1 + r)^n - 1."""
    return (1 + r) ** n - 1


# ---- Tier 1: compute_portfolio_performance ----


def test_portfolio_performance_constant_return_matches_formula():
    """G-Rs100 and AReturns match formula for constant daily return."""
    n = 252
    r = 0.001
    df = constant_daily_returns(n, r=r)

    result = compute_portfolio_performance(df, input_frequency="daily")

    cum = cumulative_return(r, n)
    expected_g_rs100 = 100 * (1 + cum)
    expected_ann_ret_pct = (1 + cum) ** (252 / n) - 1
    expected_ann_ret_pct *= 100

    assert "strategy_A" in result.index
    assert result.loc["strategy_A", "G-Rs100"] == pytest.approx(expected_g_rs100, abs=0.1)
    assert result.loc["strategy_A", "AReturns"] == pytest.approx(expected_ann_ret_pct, abs=0.1)
    assert result.loc["strategy_A", "%Up_Periods"] == pytest.approx(100.0, abs=0.01)
    assert result.loc["strategy_A", "%Down_Periods"] == pytest.approx(0.0, abs=0.01)
    assert "ARisk" in result.columns
    assert "Ret/Risk" in result.columns


def test_portfolio_performance_empty_returns_empty_result():
    """Empty DataFrame returns empty DataFrame."""
    df = pd.DataFrame()
    result = compute_portfolio_performance(df, input_frequency="daily")
    assert result.empty


def test_portfolio_performance_invalid_frequency_raises():
    """Invalid input_frequency raises ValueError."""
    df = constant_daily_returns(252)
    with pytest.raises(ValueError, match="daily.*monthly.*yearly"):
        compute_portfolio_performance(df, input_frequency="weekly")


# ---- Tier 1: compute_rolling_performance ----


def test_rolling_performance_constant_return_shape_and_mean():
    """Rolling 1y daily: output shape and mean rolling return match formula."""
    n = 252
    r = 0.001
    df = constant_daily_returns(n, r=r)

    result = compute_rolling_performance(df, input_frequency="daily", roll_period=1, roll_type="yearly")

    expected_roll_return_pct = cumulative_return(r, 252) * 100  # ~28.64
    assert "strategy_A" in result.index
    assert result.loc["strategy_A", "mean"] == pytest.approx(expected_roll_return_pct, abs=0.5)
    assert "count" in result.columns
    assert "min" in result.columns
    assert "max" in result.columns
    assert "<0%P" in result.columns


def test_rolling_performance_short_data_returns_empty():
    """Data shorter than rolling window returns empty DataFrame."""
    df = constant_daily_returns(100)  # 100 days, need 252 for 1y
    result = compute_rolling_performance(df, input_frequency="daily", roll_period=1, roll_type="yearly")
    assert result.empty


# ---- Tier 1: compute_monthly_returns_from_daily ----


def test_monthly_returns_from_daily_constant_return_matches_formula():
    """Monthly compounded return = (1+r)^(days_in_month) - 1 for constant r."""
    # 3 months of 21+ business days each so each month is retained (>15)
    n = 63
    r = 0.001
    df = constant_daily_returns(n, r=r, start="2020-01-01")

    result = compute_monthly_returns_from_daily(df, input_frequency="daily")

    assert len(result) >= 1
    # For constant r, each month's return is (1+r)^(days_in_month) - 1; days vary (20-23)
    for col in result.columns:
        first_ret = result[col].iloc[0]
        assert 0.01 <= first_ret <= 0.03  # (1.001)^20 - 1 ~ 0.020, (1.001)^23 - 1 ~ 0.023
        assert first_ret == pytest.approx(cumulative_return(r, 21), abs=0.003)


def test_monthly_returns_from_daily_monthly_frequency_returns_copy():
    """If input_frequency is monthly, returns copy of input."""
    monthly = pd.DataFrame(
        {"A": [0.01, -0.02, 0.03]},
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-28", "2020-03-31"]),
    )
    monthly.index.name = "Date"
    result = compute_monthly_returns_from_daily(monthly, input_frequency="monthly")
    pd.testing.assert_frame_equal(result, monthly)


def test_monthly_returns_from_daily_drops_short_months():
    """Months with <=15 observations are excluded."""
    # 10 days in one month -> dropped
    df = constant_daily_returns(10, start="2020-01-01")
    result = compute_monthly_returns_from_daily(df, input_frequency="daily")
    # With only 10 days, no month has >15, so result is empty or has no rows
    assert len(result) == 0 or result.empty


# ---- Tier 1: compute_up_down_months ----


def test_up_down_months_known_counts():
    """6 up and 4 down months -> Up_Months=6, Down_Months=4."""
    monthly = pd.DataFrame(
        {"strat": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02, -0.01, 0.01, -0.02]},
        index=pd.date_range("2020-01-31", periods=10, freq="ME"),
    )
    result = compute_up_down_months(monthly)
    assert result.loc["Up_Months", "strat"] == 6
    assert result.loc["Down_Months", "strat"] == 4


def test_up_down_months_multiple_columns():
    """Output has 2 rows and one column per strategy."""
    monthly = pd.DataFrame(
        {"A": [0.01, -0.01], "B": [0.02, -0.02]},
        index=pd.date_range("2020-01-31", periods=2, freq="ME"),
    )
    result = compute_up_down_months(monthly)
    assert result.shape == (2, 2)
    assert list(result.index) == ["Up_Months", "Down_Months"]


# ---- Tier 2: compute_crisis_regime_returns (patched config) ----


FAKE_CRISIS_REGIMES = {
    "TestCrisis": {
        "crisis_start": "01-03-2020",
        "crisis_end": "10-03-2020",
        "recovery_start": "11-03-2020",
        "recovery_end": "20-03-2020",
    }
}


def test_crisis_regime_returns_patched_matches_formula():
    """With patched CRISIS_REGIMES, crisis and recovery returns = (1+r)^n - 1 in %."""
    # 20 business days spanning Mar 1-20 so both crisis and recovery windows have data
    n = 20
    r = 0.001
    df = constant_daily_returns(n, r=r, start="2020-03-01")

    with patch("reporting.analytics.CRISIS_REGIMES", FAKE_CRISIS_REGIMES):
        result = compute_crisis_regime_returns(df)

    assert not result.empty
    assert "Start_Date" in result.columns
    assert "End_Date" in result.columns
    strat_col = result.columns[2]
    crisis_row = result.index[0]
    recovery_row = result.index[1]
    # Slices use calendar dates; business days in window vary. Formula: (1+r)^n - 1 in %
    crisis_pct = result.loc[crisis_row, strat_col]
    recovery_pct = result.loc[recovery_row, strat_col]
    assert 0.5 <= crisis_pct <= 2.0  # ~7-14 business days at r=0.001
    assert 0.5 <= recovery_pct <= 2.0
    assert crisis_pct == pytest.approx(cumulative_return(r, 7) * 100, abs=0.5) or crisis_pct == pytest.approx(cumulative_return(r, 10) * 100, abs=0.5)


def test_crisis_regime_returns_no_overlap_returns_empty():
    """Returns entirely outside patched regime window -> empty."""
    # Data in 2019, regime is March 2020
    df = constant_daily_returns(20, start="2019-06-01")
    with patch("reporting.analytics.CRISIS_REGIMES", FAKE_CRISIS_REGIMES):
        result = compute_crisis_regime_returns(df, data_start_date=pd.Timestamp("2019-06-01"), data_end_date=pd.Timestamp("2019-07-01"))
    assert result.empty


# ---- Tier 2: compute_market_regime_returns (patched config) ----


FAKE_MARKET_REGIMES = {
    "TestRegime": {"start": "01-03-2020", "end": "15-03-2020"}
}


def test_market_regime_returns_patched_matches_formula():
    """With patched MARKET_REGIMES, regime return = (1+r)^n - 1 in %."""
    n = 15
    r = 0.001
    df = constant_daily_returns(n, r=r, start="2020-03-01")

    with patch("reporting.analytics.MARKET_REGIMES", FAKE_MARKET_REGIMES):
        result = compute_market_regime_returns(df)

    assert not result.empty
    assert "Start_Date" in result.columns
    assert "End_Date" in result.columns
    strat_col = result.columns[2]
    reg_pct = result.loc["TestRegime", strat_col]
    # Mar 1-15 has ~10-11 business days; (1+r)^n - 1 in %
    assert 0.5 <= reg_pct <= 2.0


def test_market_regime_returns_no_overlap_returns_empty():
    """Data outside regime window -> empty."""
    df = constant_daily_returns(20, start="2019-06-01")
    with patch("reporting.analytics.MARKET_REGIMES", FAKE_MARKET_REGIMES):
        result = compute_market_regime_returns(df, data_start_date=pd.Timestamp("2019-06-01"), data_end_date=pd.Timestamp("2019-07-01"))
    assert result.empty


# ---- Tier 3: compute_quarterly_alpha ----


def test_quarterly_alpha_two_quarters_returns_and_summary():
    """Two quarters, constant portfolio/benchmark returns -> Quarter, returns, Outperformance, summary rows."""
    # Quarters: 202002 (Feb 15 - May 30), 202005 (May 31 - Aug 14)
    trade_results = pd.DataFrame({"quarter": [202002, 202005]})
    # Daily returns covering both quarters: need dates in [Feb 15, Aug 14]
    dates = pd.bdate_range("2020-02-15", "2020-08-14")
    r_pf = 0.001
    r_bench = 0.0005
    daily_returns = pd.DataFrame(
        {"Portfolio": np.full(len(dates), r_pf), "Benchmark": np.full(len(dates), r_bench)},
        index=dates,
    )
    daily_returns.index.name = "date"

    result = compute_quarterly_alpha(daily_returns, trade_results)

    assert not result.empty
    assert "Quarter" in result.columns
    assert "Portfolio_Return" in result.columns
    assert "Benchmark_Return" in result.columns
    assert "Outperformance" in result.columns
    # Data rows: one per quarter
    data_rows = result[result["Quarter"].astype(str).str.isdigit()]
    assert len(data_rows) == 2
    # Outperformance = portfolio - benchmark (both positive)
    assert (data_rows["Outperformance"] > 0).all()
    # Summary rows
    assert "Total Quarters" in result["Quarter"].values
    assert "Outperformance Rate" in result["Quarter"].values


def test_quarterly_alpha_none_or_missing_quarter_returns_empty():
    """None or missing 'quarter' column returns empty DataFrame."""
    df = constant_daily_returns(252, cols=["Portfolio", "Benchmark"])
    assert compute_quarterly_alpha(df, None).empty
    assert compute_quarterly_alpha(df, pd.DataFrame({"other": [1]})).empty


# ---- Tier 3: get_stock_counts_by_mcap ----


def test_stock_counts_by_mcap_shape_and_counts():
    """Minimal trade_results with quarter, cat, holding_period -> counts and Average row."""
    trade_results = pd.DataFrame({
        "quarter": [202002, 202002, 202005, 202005],
        "cat": ["largecap", "midcap", "largecap", "smallcap"],
        "holding_period": [5.0, 10.0, 7.0, 3.0],
    })

    result = get_stock_counts_by_mcap(trade_results)

    assert "total_stocks" in result.columns
    assert "largecap" in result.columns
    assert "midcap" in result.columns
    assert "smallcap" in result.columns
    assert "quarter" in result.columns
    # 2 quarters + 1 Average row
    assert len(result) == 3
    assert result["quarter"].iloc[-1] == "Average"
    # Q1: 2 stocks (1 large, 1 mid); Q2: 2 stocks (1 large, 1 small)
    assert result["total_stocks"].iloc[0] == 2
    assert result["total_stocks"].iloc[1] == 2


def test_stock_counts_by_mcap_missing_column_raises():
    """Missing required column raises ValueError."""
    df = pd.DataFrame({"quarter": [1], "holding_period": [1.0]})  # no cat/mcap_category
    with pytest.raises(ValueError, match="mcap_category|cat"):
        get_stock_counts_by_mcap(df)
    df2 = pd.DataFrame({"quarter": [1], "cat": ["largecap"]})  # no holding_period
    with pytest.raises(ValueError, match="holding_period"):
        get_stock_counts_by_mcap(df2)


# ---- Tier 3: get_comprehensive_quarter_analysis ----


def test_comprehensive_quarter_analysis_minimal_valid():
    """Minimal valid trade_results -> one row per quarter, key columns present."""
    trade_results = pd.DataFrame({
        "quarter": [202002, 202002, 202005],
        "cat": ["largecap", "midcap", "largecap"],
        "holding_period": [5.0, 10.0, 7.0],
        "stock_return": [0.01, -0.02, 0.03],
        "stock_weight": [0.5, 0.5, 1.0],
    })

    result = get_comprehensive_quarter_analysis(trade_results)

    assert not result.empty
    assert "quarter" in result.columns
    assert "total_stocks" in result.columns
    assert "avg_holding_period" in result.columns
    assert "portfolio_return" in result.columns
    assert len(result) == 2  # 2 quarters
    assert result["total_stocks"].iloc[0] == 2
    assert result["total_stocks"].iloc[1] == 1


def test_comprehensive_quarter_analysis_missing_stock_weight_raises():
    """Missing stock_weight raises ValueError."""
    df = pd.DataFrame({
        "quarter": [202002],
        "cat": ["largecap"],
        "holding_period": [5.0],
        "stock_return": [0.01],
    })
    with pytest.raises(ValueError, match="stock_weight"):
        get_comprehensive_quarter_analysis(df)


def test_comprehensive_quarter_analysis_missing_required_columns_raises():
    """Missing required columns raises ValueError."""
    df = pd.DataFrame({
        "quarter": [202002],
        "stock_weight": [1.0],
        "holding_period": [5.0],
        "stock_return": [0.01],
    })  # missing 'cat'
    with pytest.raises(ValueError, match="Missing required columns"):
        get_comprehensive_quarter_analysis(df)
