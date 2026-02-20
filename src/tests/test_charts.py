"""
Phase 6: Smoke tests for chart functions in reporting/backtest_report.py.

Each test verifies the chart function runs without error and returns a BytesIO buffer.
No pixel-level validation — just ensures the chart pipeline doesn't crash.
"""

import pytest
import pandas as pd
import numpy as np
import io


# =============================================================================
# Helpers
# =============================================================================

def _make_comparison_df(n=252):
    dates = pd.bdate_range("2023-01-02", periods=n)
    pf_values = np.linspace(1e9, 1.2e9, n)
    idx_values = np.linspace(1e9, 1.05e9, n)
    pf_ret = ((pf_values - pf_values[0]) / pf_values[0]) * 100
    idx_ret = ((idx_values - idx_values[0]) / idx_values[0]) * 100
    return pd.DataFrame({
        "date": dates, "pf_value": pf_values, "pf_return": pf_ret,
        "index_fund_value": idx_values, "index_return": idx_ret,
        "alpha": pf_ret - idx_ret,
    })


def _make_daily_pf(n=252):
    dates = pd.bdate_range("2023-01-02", periods=n)
    values = np.linspace(1e9, 1.1e9, n)
    return pd.DataFrame({
        "date": dates, "portfolio_value": values, "quarter": 202302,
    })


def _make_monthly_returns(n=24):
    dates = pd.date_range("2022-01-31", periods=n, freq="ME")
    np.random.seed(42)
    df = pd.DataFrame({
        "Portfolio": np.random.normal(0.01, 0.03, n),
        "Benchmark": np.random.normal(0.008, 0.025, n),
    }, index=dates)
    df.index.name = "Date"
    return df


# =============================================================================
# create_pf_vs_index_chart
# =============================================================================

class TestCreatePfVsIndexChart:

    def test_returns_bytesio(self):
        from reporting.charts import create_pf_vs_index_chart
        buf = create_pf_vs_index_chart(_make_comparison_df(), 202302, 202305)
        assert isinstance(buf, io.BytesIO)
        assert buf.getvalue()[:4] == b'\x89PNG'

    def test_none_on_empty(self):
        from reporting.charts import create_pf_vs_index_chart
        empty = pd.DataFrame()
        result = create_pf_vs_index_chart(empty, 202302, 202305)
        assert result is None


# =============================================================================
# create_growth_of_wealth_chart
# =============================================================================

class TestCreateGrowthOfWealthChart:

    def test_returns_bytesio(self):
        from reporting.charts import create_growth_of_wealth_chart
        buf = create_growth_of_wealth_chart(_make_monthly_returns())
        assert isinstance(buf, io.BytesIO)
        assert len(buf.getvalue()) > 100


# =============================================================================
# create_daily_drawdown_chart
# =============================================================================

class TestCreateDailyDrawdownChart:

    def test_returns_bytesio(self):
        from reporting.charts import create_daily_drawdown_chart
        buf = create_daily_drawdown_chart(_make_daily_pf())
        assert isinstance(buf, io.BytesIO)
        assert len(buf.getvalue()) > 100


# =============================================================================
# create_monthly_returns_heatmap
# =============================================================================

class TestCreateMonthlyReturnsHeatmap:

    def test_returns_bytesio(self):
        from reporting.charts import create_monthly_returns_heatmap
        buf = create_monthly_returns_heatmap(_make_daily_pf(n=504))
        assert isinstance(buf, io.BytesIO)


# =============================================================================
# create_calendar_year_heatmap
# =============================================================================

class TestCreateCalendarYearHeatmap:

    def test_returns_bytesio(self):
        from reporting.charts import create_calendar_year_heatmap
        # Build a simple calendar year DataFrame
        df = pd.DataFrame(
            {"Portfolio": [12.5, 8.3], "Benchmark": [10.1, 6.2]},
            index=pd.PeriodIndex([2022, 2023], freq="Y-DEC"),
        )
        buf = create_calendar_year_heatmap(df)
        assert isinstance(buf, io.BytesIO)

    def test_none_on_empty(self):
        from reporting.charts import create_calendar_year_heatmap
        result = create_calendar_year_heatmap(pd.DataFrame())
        assert result is None


# =============================================================================
# create_correlation_heatmap
# =============================================================================

class TestCreateCorrelationHeatmap:

    def test_returns_bytesio(self):
        from reporting.charts import create_correlation_heatmap
        buf = create_correlation_heatmap(_make_monthly_returns())
        assert isinstance(buf, io.BytesIO)


# =============================================================================
# create_distribution_chart
# =============================================================================

class TestCreateDistributionChart:

    def test_returns_bytesio(self):
        from reporting.charts import create_distribution_chart
        buf = create_distribution_chart(_make_monthly_returns())
        assert isinstance(buf, io.BytesIO)


# =============================================================================
# create_box_plot
# =============================================================================

class TestCreateBoxPlot:

    def test_returns_bytesio(self):
        from reporting.charts import create_box_plot
        buf = create_box_plot(_make_monthly_returns())
        assert isinstance(buf, io.BytesIO)
