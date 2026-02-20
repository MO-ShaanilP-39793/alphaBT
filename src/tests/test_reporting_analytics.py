"""
Phase 1: Tests for reporting analytics functions in reporting/backtest_report.py.

Tests the returns-based analysis functions (compute_portfolio_performance,
compute_rolling_performance, compute_calendar_year_performance, etc.) and
trade-level analysis functions (get_stock_counts_by_mcap,
get_comprehensive_quarter_analysis, compute_quarterly_alpha).
"""

import pytest
import pandas as pd
import numpy as np


# =============================================================================
# compute_portfolio_performance
# =============================================================================

class TestComputePortfolioPerformance:
    """Tests for compute_portfolio_performance (returns-based summary stats)."""

    def test_returns_dataframe_structure(self, daily_returns_df):
        from reporting.analytics import compute_portfolio_performance
        result = compute_portfolio_performance(daily_returns_df, input_frequency="daily")
        assert isinstance(result, pd.DataFrame)
        expected_cols = [
            "G-Rs100", "AReturns", "ARisk", "Sharpe", "DDown",
            "Sortino", "Skewness", "Kurtosis", "%Up_Periods", "%Down_Periods",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_columns_per_strategy(self, daily_returns_df):
        from reporting.analytics import compute_portfolio_performance
        result = compute_portfolio_performance(daily_returns_df)
        assert set(result.index) == {"Portfolio", "Benchmark"}

    def test_up_down_periods_sum_to_100(self, daily_returns_df):
        from reporting.analytics import compute_portfolio_performance
        result = compute_portfolio_performance(daily_returns_df)
        for strategy in result.index:
            total = result.loc[strategy, "%Up_Periods"] + result.loc[strategy, "%Down_Periods"]
            assert abs(total - 100.0) < 0.1, f"Up + Down should be ~100%, got {total}"

    def test_empty_df_returns_empty(self):
        from reporting.analytics import compute_portfolio_performance
        empty_df = pd.DataFrame(columns=["Portfolio"], dtype=float)
        empty_df.index.name = "Date"
        result = compute_portfolio_performance(empty_df)
        assert result.empty

    def test_invalid_frequency_raises(self, daily_returns_df):
        from reporting.analytics import compute_portfolio_performance
        with pytest.raises(ValueError, match="data_timeline"):
            compute_portfolio_performance(daily_returns_df, input_frequency="weekly")

    def test_year_filter(self, daily_returns_df):
        from reporting.analytics import compute_portfolio_performance
        result = compute_portfolio_performance(daily_returns_df, start_year=2022, end_year=2022)
        assert isinstance(result, pd.DataFrame)
        # Should have fewer data points than full dataset
        if not result.empty:
            assert set(result.index) == {"Portfolio", "Benchmark"}

    def test_invalid_year_filter_returns_empty(self, daily_returns_df):
        from reporting.analytics import compute_portfolio_performance
        result = compute_portfolio_performance(daily_returns_df, start_year=2010, end_year=2010)
        assert result.empty


# =============================================================================
# compute_rolling_performance
# =============================================================================

class TestComputeRollingPerformance:
    """Tests for compute_rolling_performance."""

    def test_rolling_returns_structure(self, daily_returns_df):
        from reporting.analytics import compute_rolling_performance
        result = compute_rolling_performance(daily_returns_df, roll_period=1, roll_type="yearly")
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            # Should have probability bucket columns
            expected_in_index = ["<0%P", "0-10%P", "10-20%P", ">20%P"]
            for bucket in expected_in_index:
                assert bucket in result.columns, f"Missing bucket: {bucket}"

    def test_insufficient_data_returns_empty(self):
        """If data is shorter than the rolling window, result should be empty."""
        from reporting.analytics import compute_rolling_performance
        dates = pd.bdate_range("2023-01-02", periods=10)
        df = pd.DataFrame({"Strategy": np.random.normal(0.001, 0.01, 10)}, index=dates)
        df.index.name = "Date"
        result = compute_rolling_performance(df, roll_period=1, roll_type="yearly")
        assert result.empty

    def test_monthly_rolling(self, daily_returns_df):
        from reporting.analytics import compute_rolling_performance
        result = compute_rolling_performance(daily_returns_df, roll_period=3, roll_type="monthly")
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# compute_calendar_year_performance
# =============================================================================

class TestComputeCalendarYearPerformance:
    """Tests for compute_calendar_year_performance."""

    def test_calendar_year_structure(self, daily_returns_df):
        from reporting.analytics import compute_calendar_year_performance
        result = compute_calendar_year_performance(daily_returns_df)
        assert isinstance(result, pd.DataFrame)
        if not result.empty:
            # Index should be period years
            assert "Portfolio" in result.columns

    def test_filters_partial_years(self):
        """Years with < 230 daily observations should be excluded."""
        from reporting.analytics import compute_calendar_year_performance
        # Only 100 days in 2023 — should be filtered out
        dates = pd.bdate_range("2023-06-01", periods=100)
        df = pd.DataFrame({"Strategy": np.random.normal(0.001, 0.01, 100)}, index=dates)
        df.index.name = "Date"
        result = compute_calendar_year_performance(df)
        assert result.empty

    def test_full_year_included(self):
        """A full year should appear in results."""
        from reporting.analytics import compute_calendar_year_performance
        dates = pd.bdate_range("2022-01-03", periods=252)
        df = pd.DataFrame({"Strategy": np.random.normal(0.001, 0.01, 252)}, index=dates)
        df.index.name = "Date"
        result = compute_calendar_year_performance(df)
        assert not result.empty

    def test_invalid_frequency_raises(self, daily_returns_df):
        from reporting.analytics import compute_calendar_year_performance
        with pytest.raises(ValueError, match="data_df_timeline"):
            compute_calendar_year_performance(daily_returns_df, input_frequency="yearly")


# =============================================================================
# compute_trailing_returns
# =============================================================================

class TestComputeTrailingReturns:
    """Tests for compute_trailing_returns."""

    def test_trailing_structure(self, daily_returns_df):
        from reporting.analytics import compute_trailing_returns
        result = compute_trailing_returns(daily_returns_df)
        assert isinstance(result, pd.DataFrame)
        expected_periods = ["1-m", "3-m", "6-m", "1-year"]
        for p in expected_periods:
            assert p in result.columns

    def test_short_data_has_nan_for_long_periods(self):
        """Trailing 3-year, 5-year, 10-year should be NaN when data is only 1 year."""
        from reporting.analytics import compute_trailing_returns
        dates = pd.bdate_range("2023-01-02", periods=252)
        df = pd.DataFrame({"Strategy": np.random.normal(0.001, 0.01, 252)}, index=dates)
        df.index.name = "Date"
        result = compute_trailing_returns(df)
        assert pd.isna(result.loc["Strategy", "3-years"])
        assert pd.isna(result.loc["Strategy", "5-years"])


# =============================================================================
# compute_monthly_returns_from_daily
# =============================================================================

class TestComputeMonthlyReturnsFromDaily:
    """Tests for compute_monthly_returns_from_daily."""

    def test_output_is_monthly(self, daily_returns_df):
        from reporting.analytics import compute_monthly_returns_from_daily
        result = compute_monthly_returns_from_daily(daily_returns_df)
        assert isinstance(result, pd.DataFrame)
        # Should have fewer rows than daily
        assert len(result) < len(daily_returns_df)

    def test_columns_preserved(self, daily_returns_df):
        from reporting.analytics import compute_monthly_returns_from_daily
        result = compute_monthly_returns_from_daily(daily_returns_df)
        assert "Portfolio" in result.columns
        assert "Benchmark" in result.columns

    def test_filters_short_months(self):
        """Months with fewer than 15 trading days should be filtered out."""
        from reporting.analytics import compute_monthly_returns_from_daily
        # Only 10 trading days in one month
        dates = pd.bdate_range("2023-06-15", periods=10)
        df = pd.DataFrame({"Strat": [0.001] * 10}, index=dates)
        df.index.name = "Date"
        result = compute_monthly_returns_from_daily(df)
        assert result.empty

    def test_monthly_passthrough(self, daily_returns_df):
        """When input_frequency='monthly', should return copy unchanged."""
        from reporting.analytics import compute_monthly_returns_from_daily
        result = compute_monthly_returns_from_daily(daily_returns_df, input_frequency="monthly")
        assert len(result) == len(daily_returns_df)


# =============================================================================
# compute_up_down_months
# =============================================================================

class TestComputeUpDownMonths:
    """Tests for compute_up_down_months."""

    def test_structure(self, monthly_returns_df):
        from reporting.analytics import compute_up_down_months
        result = compute_up_down_months(monthly_returns_df)
        assert list(result.index) == ["Up_Months", "Down_Months"]
        assert "Portfolio" in result.columns

    def test_counts_sum_to_total(self, monthly_returns_df):
        from reporting.analytics import compute_up_down_months
        result = compute_up_down_months(monthly_returns_df)
        total = result.loc["Up_Months", "Portfolio"] + result.loc["Down_Months", "Portfolio"]
        assert total == len(monthly_returns_df)

    def test_all_positive(self):
        from reporting.analytics import compute_up_down_months
        dates = pd.date_range("2023-01-31", periods=12, freq="ME")
        df = pd.DataFrame({"Strat": [0.01] * 12}, index=dates)
        df.index.name = "Date"
        result = compute_up_down_months(df)
        assert result.loc["Up_Months", "Strat"] == 12
        assert result.loc["Down_Months", "Strat"] == 0


# =============================================================================
# compute_crisis_regime_returns / compute_market_regime_returns
# =============================================================================

class TestRegimeReturns:
    """Tests for compute_crisis_regime_returns and compute_market_regime_returns."""

    def test_crisis_returns_empty_when_no_overlap(self):
        """Data range doesn't overlap any crisis → empty DataFrame."""
        from reporting.analytics import compute_crisis_regime_returns
        dates = pd.bdate_range("2023-01-02", periods=252)
        df = pd.DataFrame({"Portfolio": [0.001] * 252}, index=dates)
        df.index.name = "Date"
        result = compute_crisis_regime_returns(df)
        assert result.empty

    def test_market_regime_returns_empty_when_no_overlap(self):
        """Data range doesn't overlap any market regime → empty DataFrame."""
        from reporting.analytics import compute_market_regime_returns
        dates = pd.bdate_range("2030-01-02", periods=252)
        df = pd.DataFrame({"Portfolio": [0.001] * 252}, index=dates)
        df.index.name = "Date"
        result = compute_market_regime_returns(df)
        assert result.empty

    def test_crisis_returns_with_overlap(self):
        """Data overlapping Covid crash should return results."""
        from reporting.analytics import compute_crisis_regime_returns
        dates = pd.bdate_range("2020-01-02", periods=504)
        np.random.seed(99)
        df = pd.DataFrame({"Portfolio": np.random.normal(0.0, 0.01, 504)}, index=dates)
        df.index.name = "Date"
        result = compute_crisis_regime_returns(df)
        assert not result.empty
        assert "Start_Date" in result.columns
        assert "End_Date" in result.columns

    def test_market_regime_with_overlap(self):
        """Data overlapping Bull Regime 04 (Aug 2022 - Sep 2024) should return results."""
        from reporting.analytics import compute_market_regime_returns
        dates = pd.bdate_range("2022-08-01", periods=504)
        np.random.seed(99)
        df = pd.DataFrame({"Portfolio": np.random.normal(0.0, 0.01, 504)}, index=dates)
        df.index.name = "Date"
        result = compute_market_regime_returns(df)
        assert not result.empty


# =============================================================================
# compute_quarterly_alpha
# =============================================================================

class TestComputeQuarterlyAlpha:
    """Tests for compute_quarterly_alpha."""

    def test_empty_trade_results(self):
        from reporting.analytics import compute_quarterly_alpha
        dates = pd.bdate_range("2023-02-15", periods=100)
        daily_df = pd.DataFrame(
            {"Portfolio": [0.001] * 100, "Benchmark": [0.0005] * 100},
            index=dates,
        )
        daily_df.index.name = "Date"
        result = compute_quarterly_alpha(daily_df, None)
        assert result.empty

    def test_quarterly_alpha_structure(self):
        from reporting.analytics import compute_quarterly_alpha
        # Create daily returns covering Q202302 (Feb 15 - May 30)
        dates = pd.bdate_range("2023-02-15", "2023-05-30")
        daily_df = pd.DataFrame(
            {"Portfolio": [0.001] * len(dates), "Benchmark": [0.0005] * len(dates)},
            index=dates,
        )
        daily_df.index.name = "Date"
        trade_results = pd.DataFrame({"quarter": [202302, 202302]})
        result = compute_quarterly_alpha(daily_df, trade_results)
        assert not result.empty
        assert "Quarter" in result.columns
        assert "Portfolio_Return" in result.columns
        assert "Outperformance" in result.columns

    def test_outperformance_positive_when_pf_beats_bm(self):
        from reporting.analytics import compute_quarterly_alpha
        dates = pd.bdate_range("2023-02-15", "2023-05-30")
        daily_df = pd.DataFrame(
            {"Portfolio": [0.002] * len(dates), "Benchmark": [0.0005] * len(dates)},
            index=dates,
        )
        daily_df.index.name = "Date"
        trade_results = pd.DataFrame({"quarter": [202302]})
        result = compute_quarterly_alpha(daily_df, trade_results)
        # Filter out summary rows
        data_rows = result[result["Quarter"].apply(lambda x: isinstance(x, (int, np.integer)))]
        assert all(data_rows["Outperformance"] > 0)


# =============================================================================
# get_stock_counts_by_mcap
# =============================================================================

class TestGetStockCountsByMcap:
    """Tests for get_stock_counts_by_mcap."""

    def test_structure(self, trade_results_with_mcap):
        from reporting.analytics import get_stock_counts_by_mcap
        result = get_stock_counts_by_mcap(trade_results_with_mcap)
        assert "total_stocks" in result.columns
        assert "largecap" in result.columns
        assert "midcap" in result.columns
        assert "smallcap" in result.columns

    def test_average_row(self, trade_results_with_mcap):
        from reporting.analytics import get_stock_counts_by_mcap
        result = get_stock_counts_by_mcap(trade_results_with_mcap)
        avg_row = result[result["quarter"] == "Average"]
        assert len(avg_row) == 1

    def test_missing_mcap_column_raises(self):
        from reporting.analytics import get_stock_counts_by_mcap
        df = pd.DataFrame({
            "quarter": [202302], "co_name": ["X"],
            "holding_period": [20], "stock_return": [0.05],
        })
        with pytest.raises(ValueError, match="Missing required column"):
            get_stock_counts_by_mcap(df)

    def test_missing_holding_period_raises(self, trade_results_with_mcap):
        from reporting.analytics import get_stock_counts_by_mcap
        df = trade_results_with_mcap.drop(columns=["holding_period"])
        with pytest.raises(ValueError, match="holding_period"):
            get_stock_counts_by_mcap(df)


# =============================================================================
# get_comprehensive_quarter_analysis
# =============================================================================

class TestGetComprehensiveQuarterAnalysis:
    """Tests for get_comprehensive_quarter_analysis."""

    def test_structure(self, trade_results_with_mcap):
        from reporting.analytics import get_comprehensive_quarter_analysis
        result = get_comprehensive_quarter_analysis(trade_results_with_mcap)
        assert "quarter" in result.columns
        assert "total_stocks" in result.columns
        assert "avg_holding_period" in result.columns
        assert "portfolio_return" in result.columns

    def test_tp_sl_counts(self, trade_results_with_mcap):
        from reporting.analytics import get_comprehensive_quarter_analysis
        result = get_comprehensive_quarter_analysis(trade_results_with_mcap)
        # Should have TP/SL counts since the fixture has those columns
        assert "TP_count" in result.columns
        assert "SL_count" in result.columns

    def test_missing_weight_column_raises(self):
        from reporting.analytics import get_comprehensive_quarter_analysis
        df = pd.DataFrame({
            "quarter": [202302], "co_name": ["X"], "cat": ["largecap"],
            "holding_period": [20], "stock_return": [0.05],
        })
        with pytest.raises(ValueError, match="weight column"):
            get_comprehensive_quarter_analysis(df)

    def test_missing_required_columns_raises(self, trade_results_with_mcap):
        from reporting.analytics import get_comprehensive_quarter_analysis
        df = trade_results_with_mcap.drop(columns=["stock_return"])
        with pytest.raises(ValueError, match="Missing required"):
            get_comprehensive_quarter_analysis(df)

    def test_multiple_quarters(self, trade_results_with_mcap):
        from reporting.analytics import get_comprehensive_quarter_analysis
        result = get_comprehensive_quarter_analysis(trade_results_with_mcap)
        assert len(result) == 2  # Two quarters in fixture
