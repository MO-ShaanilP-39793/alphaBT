"""
Tests for key reporting metrics.

Tests compute_portfolio_metrics, compute_drawdown_series, and
compute_benchmark_metrics with hand-verifiable synthetic portfolios.

Structured so additional metric functions can be added incrementally.
"""

import pytest
import numpy as np
import pandas as pd

from reporting.metrics import (
    compute_portfolio_metrics,
    compute_drawdown_series,
    compute_benchmark_metrics,
)
from config.defaults import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR, DAYS_PER_YEAR


# =============================================================================
# compute_portfolio_metrics — CAGR
# =============================================================================

class TestCAGR:

    def test_cagr_constant_portfolio(self, constant_portfolio):
        """Flat portfolio → CAGR ≈ 0%."""
        metrics = compute_portfolio_metrics(constant_portfolio)
        assert abs(metrics["cagr_pct"]) < 0.01, f"CAGR should be ~0%, got {metrics['cagr_pct']}%"

    def test_cagr_known_growth(self, known_cagr_portfolio):
        """
        Portfolio grows 1B → 1.5B over ~2 years.
        CAGR = (1.5/1.0)^(1/years) - 1.
        years ≈ (503 bdays) / 365.25 ≈ 1.377
        """
        metrics = compute_portfolio_metrics(known_cagr_portfolio)
        df = known_cagr_portfolio
        initial = df["portfolio_value"].iloc[0]
        final = df["portfolio_value"].iloc[-1]
        start = pd.to_datetime(df["date"].iloc[0])
        end = pd.to_datetime(df["date"].iloc[-1])
        years = (end - start).days / DAYS_PER_YEAR
        expected_cagr = ((final / initial) ** (1 / years) - 1) * 100

        assert abs(metrics["cagr_pct"] - expected_cagr) < 0.1, (
            f"CAGR mismatch: expected ~{expected_cagr:.2f}%, got {metrics['cagr_pct']}%"
        )

    def test_total_return(self, linear_growth_portfolio):
        """total_return_pct = (final - initial) / initial * 100."""
        metrics = compute_portfolio_metrics(linear_growth_portfolio)
        expected = (1.1e9 - 1e9) / 1e9 * 100  # 10%
        assert abs(metrics["total_return_pct"] - expected) < 0.1


# =============================================================================
# compute_portfolio_metrics — Sharpe Ratio
# =============================================================================

class TestSharpeRatio:

    def test_sharpe_zero_vol(self, constant_portfolio):
        """Constant portfolio has zero volatility → Sharpe = 0."""
        metrics = compute_portfolio_metrics(constant_portfolio)
        assert metrics["sharpe_ratio"] == 0.0

    def test_sharpe_positive_when_beating_rfr(self):
        """Portfolio with CAGR > risk-free rate → positive Sharpe."""
        # Create portfolio with ~20% annualized return (well above 6.5% RFR)
        n = 252
        daily_r = (1.20) ** (1 / 252) - 1  # ~20% annualized
        values = [1e9 * (1 + daily_r) ** i for i in range(n)]
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "portfolio_value": values,
            "quarter": 202302,
        })
        metrics = compute_portfolio_metrics(df)
        assert metrics["sharpe_ratio"] > 0, f"Sharpe should be positive, got {metrics['sharpe_ratio']}"

    def test_sharpe_negative_when_below_rfr(self):
        """Portfolio with CAGR < risk-free rate → negative Sharpe."""
        n = 252
        daily_r = (1.02) ** (1 / 252) - 1  # ~2% annualized (below 6.5% RFR)
        values = [1e9 * (1 + daily_r) ** i for i in range(n)]
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "portfolio_value": values,
            "quarter": 202302,
        })
        metrics = compute_portfolio_metrics(df)
        assert metrics["sharpe_ratio"] < 0, f"Sharpe should be negative, got {metrics['sharpe_ratio']}"


# =============================================================================
# compute_portfolio_metrics — Sortino Ratio
# =============================================================================

class TestSortinoRatio:

    def test_sortino_is_finite(self, linear_growth_portfolio):
        """Linear growth portfolio → Sortino should be finite and calculable."""
        metrics = compute_portfolio_metrics(linear_growth_portfolio)
        assert np.isfinite(metrics["sortino_ratio"])

    def test_sortino_zero_when_no_downside(self):
        """
        For a purely exponential upward portfolio where every daily return
        exceeds the daily risk-free rate, downside deviation = 0
        and Sortino = 0 (guarded division).
        """
        n = 252
        daily_r = (1.25) ** (1 / 252) - 1  # ~25% annualized
        values = [1e9 * (1 + daily_r) ** i for i in range(n)]
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "portfolio_value": values,
            "quarter": 202302,
        })
        metrics = compute_portfolio_metrics(df)
        # Pure upward: all daily returns > daily RFR → zero downside dev → Sortino = 0
        assert metrics["sortino_ratio"] == 0

    def test_sortino_nonzero_with_drawdowns(self, drawdown_portfolio):
        """Portfolio with real drawdowns → Sortino is nonzero and finite."""
        metrics = compute_portfolio_metrics(drawdown_portfolio)
        assert metrics["sortino_ratio"] != 0
        assert np.isfinite(metrics["sortino_ratio"])


# =============================================================================
# compute_portfolio_metrics — Max Drawdown
# =============================================================================

class TestMaxDrawdown:

    def test_max_drawdown_known(self, drawdown_portfolio):
        """
        Portfolio peaks at 1.2B, drops to 0.9B → drawdown = (0.9 - 1.2)/1.2 = -25%.
        """
        metrics = compute_portfolio_metrics(drawdown_portfolio)
        assert abs(metrics["max_drawdown_pct"] - (-25.0)) < 1.0, (
            f"Max drawdown should be ~-25%, got {metrics['max_drawdown_pct']}%"
        )

    def test_max_drawdown_constant(self, constant_portfolio):
        """Constant portfolio → zero drawdown."""
        metrics = compute_portfolio_metrics(constant_portfolio)
        assert metrics["max_drawdown_pct"] == 0.0

    def test_max_drawdown_linear_growth(self, linear_growth_portfolio):
        """Monotonically increasing portfolio → zero drawdown."""
        metrics = compute_portfolio_metrics(linear_growth_portfolio)
        assert metrics["max_drawdown_pct"] == 0.0


# =============================================================================
# compute_portfolio_metrics — Calmar Ratio
# =============================================================================

class TestCalmarRatio:

    def test_calmar_ratio_calculation(self, drawdown_portfolio):
        """Calmar = CAGR / |max_drawdown|."""
        metrics = compute_portfolio_metrics(drawdown_portfolio)
        if metrics["max_drawdown_pct"] != 0:
            expected_calmar = metrics["cagr_pct"] / abs(metrics["max_drawdown_pct"])
            assert abs(metrics["calmar_ratio"] - expected_calmar) < 0.01

    def test_calmar_zero_drawdown(self, constant_portfolio):
        """Zero drawdown → Calmar capped at max (excellent performance)."""
        metrics = compute_portfolio_metrics(constant_portfolio)
        assert metrics["calmar_ratio"] == 10.0


# =============================================================================
# compute_portfolio_metrics — VaR & day statistics
# =============================================================================

class TestVaRAndDayStats:

    def test_var_95(self, drawdown_portfolio):
        """VaR(95%) is the 5th percentile of daily returns."""
        metrics = compute_portfolio_metrics(drawdown_portfolio)
        df = drawdown_portfolio.copy()
        df["daily_return"] = df["portfolio_value"].pct_change()
        expected_var = df["daily_return"].quantile(0.05) * 100
        assert abs(metrics["var_95_pct"] - expected_var) < 0.01

    def test_positive_days_linear_growth(self, linear_growth_portfolio):
        """Linear growth → nearly all days positive (close to 100%)."""
        metrics = compute_portfolio_metrics(linear_growth_portfolio)
        assert metrics["positive_days_pct"] > 95.0


# =============================================================================
# compute_portfolio_metrics — All keys present
# =============================================================================

class TestMetricsKeys:

    def test_all_keys_present(self, linear_growth_portfolio):
        """Verify all 17 expected keys are returned."""
        metrics = compute_portfolio_metrics(linear_growth_portfolio)
        expected_keys = {
            "start_date", "end_date", "trading_days", "years",
            "initial_value", "final_value", "total_return_pct",
            "cagr_pct", "volatility_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "calmar_ratio", "var_95_pct",
            "best_day_pct", "worst_day_pct", "positive_days_pct",
        }
        assert set(metrics.keys()) == expected_keys


# =============================================================================
# compute_drawdown_series
# =============================================================================

class TestDrawdownSeries:

    def test_drawdown_columns(self, drawdown_portfolio):
        """Output has the expected columns."""
        dd = compute_drawdown_series(drawdown_portfolio)
        expected_cols = {"date", "portfolio_value", "cummax", "drawdown", "drawdown_pct"}
        assert set(dd.columns) == expected_cols

    def test_drawdown_values(self, drawdown_portfolio):
        """Peak at 1.2B, trough at 0.9B → drawdown = -300M, drawdown_pct = -25%."""
        dd = compute_drawdown_series(drawdown_portfolio)
        min_dd_pct = dd["drawdown_pct"].min()
        assert abs(min_dd_pct - (-25.0)) < 1.0, f"Expected ~-25%, got {min_dd_pct}%"

        min_dd_abs = dd["drawdown"].min()
        assert min_dd_abs < 0, "Drawdown should be negative at trough"

    def test_drawdown_linear_growth(self, linear_growth_portfolio):
        """Monotonically increasing → drawdown always 0."""
        dd = compute_drawdown_series(linear_growth_portfolio)
        assert (dd["drawdown"] == 0).all()
        assert (dd["drawdown_pct"] == 0).all()


# =============================================================================
# compute_benchmark_metrics
# =============================================================================

class TestBenchmarkMetrics:

    def test_benchmark_metrics_keys(self, comparison_df_identical):
        """All 10 expected keys present."""
        metrics = compute_benchmark_metrics(comparison_df_identical)
        expected_keys = {
            "portfolio_return_pct", "index_return_pct", "alpha_pct",
            "beta", "tracking_error_pct", "information_ratio",
            "up_capture_pct", "down_capture_pct",
            "correlation", "outperformance_days_pct",
        }
        assert set(metrics.keys()) == expected_keys

    def test_beta_identical_series(self, comparison_df_identical):
        """Portfolio = index → beta ≈ 1.0."""
        metrics = compute_benchmark_metrics(comparison_df_identical)
        assert abs(metrics["beta"] - 1.0) < 0.05, f"Beta should be ~1.0, got {metrics['beta']}"

    def test_tracking_error_identical(self, comparison_df_identical):
        """Portfolio = index → tracking error ≈ 0."""
        metrics = compute_benchmark_metrics(comparison_df_identical)
        assert abs(metrics["tracking_error_pct"]) < 0.1, (
            f"Tracking error should be ~0, got {metrics['tracking_error_pct']}%"
        )

    def test_correlation_identical(self, comparison_df_identical):
        """Portfolio = index → correlation ≈ 1.0."""
        metrics = compute_benchmark_metrics(comparison_df_identical)
        assert abs(metrics["correlation"] - 1.0) < 0.01

    def test_alpha_outperformance(self, comparison_df_outperforming):
        """Portfolio beats index → positive alpha."""
        metrics = compute_benchmark_metrics(comparison_df_outperforming)
        assert metrics["alpha_pct"] > 0, f"Alpha should be positive, got {metrics['alpha_pct']}"

    def test_information_ratio_outperforming(self, comparison_df_outperforming):
        """Consistently outperforming → positive information ratio."""
        metrics = compute_benchmark_metrics(comparison_df_outperforming)
        assert metrics["information_ratio"] > 0, (
            f"IR should be positive for outperforming portfolio, got {metrics['information_ratio']}"
        )
